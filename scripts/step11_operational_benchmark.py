"""Step 11: the operational benchmark that replaces Table 3.

Table 3 reported 1.92% nMAE and +9.2%/+25.3% skill. Those were computed with
MEASURED irradiance from the site's own pyranometer. That number is not
available when a forecast is issued, so Table 3 is a model-quality diagnostic
and cannot be quoted as forecasting performance. On identical hours the
day-ahead error is 3.03x larger.

This script recomputes the whole baseline ladder on inputs that genuinely exist
at forecast time, and reports uncertainty by resampling whole DAYS so temporal
dependence is preserved.

REPRODUCIBILITY -- everything needed to repeat this run is printed at the top:
evaluation window, forecast lead and issue convention, weather source, model
artefact, normalisation, split, and sample counts.

    python scripts/step11_operational_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import conformal, features, metrics, model, physics, pipeline  # noqa: E402
from dustcast.config import ARTIFACTS, DKASC_SITE_03, PROCESSED  # noqa: E402

WINDOW = ("2024-01-01", "2025-08-20")
LEAD_DAY = 1                      # day-ahead: the horizon reserve is procured on
CAL_FRACTION = 0.50               # first half fits the non-ML derates
BOOTSTRAP = 1000
SEED = 11

BASE_VARS = {
    "shortwave_radiation": "ghi", "diffuse_radiation": "dhi",
    "direct_normal_irradiance": "dni", "temperature_2m": "temp_air",
    "wind_speed_10m": "wind_speed", "relative_humidity_2m": "relative_humidity",
    "precipitation": "precipitation",
}


def rule(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def lead_weather(raw: pd.DataFrame, lead: int) -> pd.DataFrame:
    cols = {}
    for src, dst in BASE_VARS.items():
        name = src if lead == 0 else f"{src}_previous_day{lead}"
        if name in raw:
            cols[dst] = raw[name]
    df = pd.DataFrame(cols)
    # Open-Meteo reports radiation and precipitation as preceding-hour means.
    return df.set_axis(df.index - pd.Timedelta("30min"), axis=0).dropna(how="all")


def day_block_bootstrap(actual, preds, ref_name, ml_name, n=BOOTSTRAP, seed=SEED):
    """Resample whole days to get a CI on the skill score.

    Hourly PV errors are strongly autocorrelated -- a cloudy afternoon is one
    event, not five independent ones -- so an hour-level bootstrap would badly
    understate the interval. Resampling days preserves that dependence.
    """
    rng = np.random.default_rng(seed)
    days = pd.Index(sorted(set(actual.index.normalize())))
    by_day = {d: actual.index.normalize() == d for d in days}
    out_mae, out_rmse = [], []
    for _ in range(n):
        pick = rng.choice(len(days), size=len(days), replace=True)
        mask = np.zeros(len(actual), dtype=bool)
        for i in pick:
            mask |= by_day[days[i]]
        a = actual[mask]
        r_mae = np.mean(np.abs(preds[ref_name][mask] - a))
        m_mae = np.mean(np.abs(preds[ml_name][mask] - a))
        r_rms = np.sqrt(np.mean((preds[ref_name][mask] - a) ** 2))
        m_rms = np.sqrt(np.mean((preds[ml_name][mask] - a) ** 2))
        out_mae.append(1 - m_mae / r_mae)
        out_rmse.append(1 - m_rms / r_rms)
    q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return q(out_mae), q(out_rmse)


def main() -> int:
    site = DKASC_SITE_03
    cap = site.ac_capacity_w / 1000.0
    rm = model.ResidualModel.load(ARTIFACTS / "residual_model.joblib")
    ref = pipeline.prepare(site)
    raw = pd.read_parquet(PROCESSED / "dkasc_leadtime_nwp.parquet")
    raw.index = pd.to_datetime(raw.index)
    measured = pipeline.hourly_frame(site)

    rule("BENCHMARK SPECIFICATION")
    print(f"  site                 {site.name}")
    print(f"  evaluation window    {WINDOW[0]} .. {WINDOW[1]}")
    print(f"  weather source       Open-Meteo Historical Forecast API")
    print(f"  forecast lead        {LEAD_DAY} day  (`*_previous_day{LEAD_DAY}`:")
    print(f"                       the forecast issued {LEAD_DAY} day before each")
    print(f"                       timestamp, verified to diverge with lead)")
    print(f"  target               measured AC power, DKASC array 3, hourly means")
    print(f"                       labelled at interval midpoints")
    print(f"  normalisation        nMAE = MAE / {cap:.1f} kW AC nameplate")
    print(f"  model artefact       residual_model.joblib, trained <= 2021-12-31")
    print(f"                       on measured weather, never refitted here")
    print(f"  ageing / clear-sky   fitted on the pre-2022 training split")
    print(f"  baseline derates     fitted on the first {CAL_FRACTION:.0%} of this window,")
    print(f"                       evaluated on the remainder")
    print(f"  scored hours         daylight only (elevation > 5 deg), identical")
    print(f"                       across every model")
    print(f"  uncertainty          {BOOTSTRAP} day-block bootstrap resamples, seed {SEED}")

    w = lead_weather(raw, LEAD_DAY).join(measured[["ac_power_kw"]], how="inner")
    w = w.loc[WINDOW[0]:WINDOW[1]].dropna(subset=["ghi", "temp_air", "ac_power_kw"])

    frame = physics.build_physics_frame(site, w, clearsky_scale=ref.clearsky_scale)
    frame = frame[frame["expected_physics_kw"].notna() & frame["ac_power_kw"].notna()]
    frame = model.apply_degradation(frame, site, *ref.ageing)
    X = features.make_features(frame, site)[features.feature_names()]
    ok = X.notna().all(axis=1) & frame["residual_kw"].notna() & metrics.daylight_mask(frame)
    frame, X = frame[ok], X[ok]

    cut = int(len(frame) * CAL_FRACTION)
    fit_idx, test_idx = frame.index[:cut], frame.index[cut:]
    ft = frame.loc[test_idx]
    factor = model.fit_physics_calibration(frame, fit_idx)

    preds = {
        "smart persistence": model.baseline_smart_persistence(frame, site).loc[test_idx],
        "physics, raw": model.baseline_physics(ft, site),
        "physics + PVWatts losses": model.baseline_physics_derated(ft, site),
        "physics + fitted derate": model.baseline_physics_calibrated(ft, site, factor),
        "physics + ageing trend": model.baseline_physics_trend(ft, site),
        "physics + diurnal bias": model.baseline_physics_diurnal(
            ft, site, model.fit_diurnal_bias(frame, fit_idx)),
        "physics + ML (DustCast)": model.clamp_power(
            ft["expected_kw"] + rm.predict_residual(X.loc[test_idx, rm.feature_cols]),
            ft, site),
    }
    actual = ft["ac_power_kw"]
    scored = actual.notna()
    for p in preds.values():
        scored &= p.reindex(actual.index).notna()
    actual = actual[scored]
    preds = {k: v.reindex(actual.index) for k, v in preds.items()}

    rule(f"RESULTS  ({len(actual):,} daylight hours, day-ahead forecast inputs)")
    rows = {k: metrics.deterministic(actual, v, capacity_kw=cap) for k, v in preds.items()}
    non_ml = [k for k in rows if "DustCast" not in k]
    best = min(non_ml, key=lambda k: rows[k]["mae"])
    table = metrics.compare(rows, baseline=best)
    pd.set_option("display.width", 140, "display.float_format", lambda v: f"{v:,.4f}")
    print(table[["n", "mae", "rmse", "mbe", "nmae_pct", "skill_mae", "skill_rmse"]].to_string())
    print(f"\n  strongest non-ML baseline: {best}")

    ml = "physics + ML (DustCast)"
    arr = {k: v.to_numpy() for k, v in preds.items()}
    (lo_m, hi_m), (lo_r, hi_r) = day_block_bootstrap(actual, arr, best, ml)
    print(f"\n  skill MAE   {table.loc[ml,'skill_mae']:+.1%}   95% CI [{lo_m:+.1%}, {hi_m:+.1%}]")
    print(f"  skill RMSE  {table.loc[ml,'skill_rmse']:+.1%}   95% CI [{lo_r:+.1%}, {hi_r:+.1%}]")
    sig = lo_m > 0 and lo_r > 0
    print(f"  -> improvement {'excludes zero' if sig else 'DOES NOT exclude zero'} on both metrics")

    rule("WHAT THIS REPLACES")
    print(f"  Table 3 (measured irradiance)   nMAE 1.92%   skill +9.2% / +25.3%")
    print(f"  This benchmark (day-ahead NWP)  nMAE {rows[ml]['nmae_pct']:.2f}%   "
          f"skill {table.loc[ml,'skill_mae']:+.1%} / {table.loc[ml,'skill_rmse']:+.1%}")
    print("\n  The first is a diagnostic of the residual model given perfect")
    print("  irradiance. Only the second is a forecasting claim.")
    table.to_csv(ARTIFACTS / "step11_operational_benchmark.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
