"""Step 12: retrain the deployed model on the input it will actually consume.

The shipped residual model was trained on MEASURED irradiance and then applied
to NWP forecasts. Step 11 showed what that costs: under day-ahead inputs it is
2.2% WORSE than a fitted physics derate, carrying a -0.26 kW bias because it is
evaluated out of distribution. Retraining on forecast inputs turns that into
roughly +30%.

This script produces the artefacts the live pipeline should use, and evaluates
them on a held-out period against the full baseline ladder with day-block
bootstrap confidence intervals.

A deliberate trade is being made. The forecast-trained model sees ~2,900
training hours against the measured-trained model's 53,336, because Open-Meteo's
day-ahead archive only reaches back to January 2024. Being trained on the right
distribution beats eighteen times more data from the wrong one -- but the sample
is small, and that is a limitation to state, not to hide.

    python scripts/step12_retrain_operational.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import conformal, features, metrics, model, physics, pipeline  # noqa: E402
from dustcast.config import ARTIFACTS, CONFIDENCE_LEVEL, DKASC_SITE_03, PROCESSED  # noqa: E402

LEAD_DAY = 1
SPLIT_TRAIN_END = "2024-12-31"
SPLIT_CAL_END = "2025-03-31"
BOOTSTRAP, SEED = 1000, 12

BASE_VARS = {
    "shortwave_radiation": "ghi", "diffuse_radiation": "dhi",
    "direct_normal_irradiance": "dni", "temperature_2m": "temp_air",
    "wind_speed_10m": "wind_speed", "relative_humidity_2m": "relative_humidity",
    "precipitation": "precipitation",
}


def rule(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main() -> int:
    site = DKASC_SITE_03
    cap = site.ac_capacity_w / 1000.0
    ref = pipeline.prepare(site)
    measured = pipeline.hourly_frame(site)
    raw = pd.read_parquet(PROCESSED / "dkasc_leadtime_nwp.parquet")
    raw.index = pd.to_datetime(raw.index)

    cols = {d: raw[f"{s}_previous_day{LEAD_DAY}"] for s, d in BASE_VARS.items()
            if f"{s}_previous_day{LEAD_DAY}" in raw}
    w = pd.DataFrame(cols)
    w = w.set_axis(w.index - pd.Timedelta("30min"), axis=0)
    w = w.join(measured[["ac_power_kw"]], how="inner")
    w = w.dropna(subset=["ghi", "temp_air", "ac_power_kw"])

    frame = physics.build_physics_frame(site, w, clearsky_scale=ref.clearsky_scale)
    frame = frame[frame["expected_physics_kw"].notna() & frame["ac_power_kw"].notna()]
    # Ageing stays fitted on the long measured record: it is a property of the
    # array, and 19 months of forecast data cannot identify a 1.3%/yr trend.
    frame = model.apply_degradation(frame, site, *ref.ageing)
    X = features.make_features(frame, site)[features.feature_names()]
    y = frame["residual_kw"]
    ok = X.notna().all(axis=1) & y.notna() & metrics.daylight_mask(frame)
    frame, X, y = frame[ok], X[ok], y[ok]

    tz = frame.index.tz
    tr = frame.index[frame.index <= pd.Timestamp(SPLIT_TRAIN_END, tz=tz)]
    ca = frame.index[(frame.index > pd.Timestamp(SPLIT_TRAIN_END, tz=tz))
                     & (frame.index <= pd.Timestamp(SPLIT_CAL_END, tz=tz))]
    te = frame.index[frame.index > pd.Timestamp(SPLIT_CAL_END, tz=tz)]

    rule("SPECIFICATION")
    print(f"  weather        Open-Meteo Historical Forecast, lead {LEAD_DAY} day")
    print(f"  target         measured AC power, DKASC array 3")
    print(f"  train          {tr.min():%Y-%m-%d} .. {tr.max():%Y-%m-%d}  {len(tr):>6,} lit hours")
    print(f"  calibrate      {ca.min():%Y-%m-%d} .. {ca.max():%Y-%m-%d}  {len(ca):>6,} lit hours")
    print(f"  test           {te.min():%Y-%m-%d} .. {te.max():%Y-%m-%d}  {len(te):>6,} lit hours")
    print(f"  normalisation  nMAE = MAE / {cap:.1f} kW AC")
    print(f"  CIs            {BOOTSTRAP} day-block resamples, seed {SEED}")

    rule("TRAIN")
    rm = model.ResidualModel(site=site, feature_cols=features.feature_names())
    rm.fit(X.loc[tr], y.loc[tr])
    print(f"  {rm.best_iteration + 1} trees, {len(rm.feature_cols)} features")
    print("  top features by gain:")
    for n_, v in rm.importances.head(6).items():
        print(f"    {n_:<24s} {v:.4f}")

    bundle = conformal.fit_cqr_bundle(rm, X, y, tr, ca, CONFIDENCE_LEVEL)
    print(f"\n  conformal correction {bundle.correction:+.4f} kW "
          f"(calibrated on forecast-driven residuals)")

    rule("HELD-OUT EVALUATION")
    ft = frame.loc[te]
    actual = ft["ac_power_kw"]
    factor = model.fit_physics_calibration(frame, tr)
    # On a re-run the deploy slot already holds the new model, so the old one
    # must come from the diagnostic copy or the comparison silently compares a
    # model with itself.
    diag = ARTIFACTS / "residual_model_measured_diagnostic.joblib"
    old = model.ResidualModel.load(
        diag if diag.exists() else ARTIFACTS / "residual_model.joblib")
    preds = {
        "smart persistence": model.baseline_smart_persistence(frame, site).loc[te],
        "physics, raw": model.baseline_physics(ft, site),
        "physics + PVWatts losses": model.baseline_physics_derated(ft, site),
        "physics + fitted derate": model.baseline_physics_calibrated(ft, site, factor),
        "physics + ageing trend": model.baseline_physics_trend(ft, site),
        "physics + diurnal bias": model.baseline_physics_diurnal(
            ft, site, model.fit_diurnal_bias(frame, tr)),
        "ML trained on measured (old)": model.clamp_power(
            ft["expected_kw"] + old.predict_residual(X.loc[te, old.feature_cols]), ft, site),
        "ML trained on forecast (new)": model.clamp_power(
            ft["expected_kw"] + rm.predict_residual(X.loc[te, rm.feature_cols]), ft, site),
    }
    keep = actual.notna()
    for p in preds.values():
        keep &= p.reindex(actual.index).notna()
    actual = actual[keep]
    preds = {k: v.reindex(actual.index) for k, v in preds.items()}

    rows = {k: metrics.deterministic(actual, v, capacity_kw=cap) for k, v in preds.items()}
    non_ml = [k for k in rows if "ML trained" not in k]
    best = min(non_ml, key=lambda k: rows[k]["mae"])
    table = metrics.compare(rows, baseline=best)
    pd.set_option("display.width", 145, "display.float_format", lambda v: f"{v:,.4f}")
    print(table[["n", "mae", "rmse", "mbe", "nmae_pct", "skill_mae", "skill_rmse"]].to_string())
    print(f"\n  strongest non-ML baseline: {best}")

    rng = np.random.default_rng(SEED)
    days = pd.Index(sorted(set(actual.index.normalize())))
    by_day = {d: actual.index.normalize() == d for d in days}
    a = actual.to_numpy()
    r = preds[best].to_numpy()
    new = preds["ML trained on forecast (new)"].to_numpy()
    sm, sr = [], []
    for _ in range(BOOTSTRAP):
        pick = rng.choice(len(days), len(days), True)
        mask = np.zeros(len(a), bool)
        for i in pick:
            mask |= by_day[days[i]]
        sm.append(1 - np.mean(np.abs(new[mask] - a[mask])) / np.mean(np.abs(r[mask] - a[mask])))
        sr.append(1 - np.sqrt(np.mean((new[mask] - a[mask]) ** 2))
                  / np.sqrt(np.mean((r[mask] - a[mask]) ** 2)))
    lo_m, hi_m = np.percentile(sm, 2.5), np.percentile(sm, 97.5)
    lo_r, hi_r = np.percentile(sr, 2.5), np.percentile(sr, 97.5)
    print(f"\n  skill MAE   {table.loc['ML trained on forecast (new)','skill_mae']:+.1%}"
          f"   95% CI [{lo_m:+.1%}, {hi_m:+.1%}]")
    print(f"  skill RMSE  {table.loc['ML trained on forecast (new)','skill_rmse']:+.1%}"
          f"   95% CI [{lo_r:+.1%}, {hi_r:+.1%}]")

    lo_r_, hi_r_ = bundle.residual_interval(X.loc[actual.index])
    lo = model.clamp_power(ft.loc[actual.index, "expected_kw"] + lo_r_, ft.loc[actual.index], site)
    hi = model.clamp_power(ft.loc[actual.index, "expected_kw"] + hi_r_, ft.loc[actual.index], site)
    print(f"\n  interval    PICP {metrics.picp(actual, lo, hi):.1%} "
          f"(nominal {CONFIDENCE_LEVEL:.0%})  PINAW {metrics.pinaw(lo, hi, cap):.4f}")

    rule("ARTEFACTS")
    ok_to_ship = lo_m > 0 and lo_r > 0
    if ok_to_ship:
        # Never overwrite the diagnostic once it exists. On a re-run the deploy
        # slot already holds the forecast-trained model, and copying it over the
        # diagnostic destroys the only record of the measured-weather baseline.
        if not diag.exists():
            old.save(diag)
        rm.save(ARTIFACTS / "residual_model.joblib")
        bundle.save(ARTIFACTS / "cqr_bundle_deploy.joblib")
        print("  residual_model.joblib                  <- forecast-trained (DEPLOY)")
        print("  residual_model_measured_diagnostic.joblib <- old measured-trained, kept")
        print("     as a labelled diagnostic of the ceiling given perfect irradiance")
        print("  cqr_bundle_deploy.joblib               <- recalibrated on forecast residuals")
    else:
        print("  NOT shipped: the improvement's CI includes zero.")
    table.to_csv(ARTIFACTS / "step12_retrain_benchmark.csv")
    return 0 if ok_to_ship else 1


if __name__ == "__main__":
    raise SystemExit(main())
