"""Step 15: capacity-normalised, multi-array training -- one bounded attempt.

Step 14 showed the residual model transferring badly to an unseen array: -48.3%
MAE skill against a thirteen-number diurnal baseline. Two causes were identified
by inspection of the code and of the error structure, BEFORE any retraining:

  1. UNIT BUG. `residual_kw = actual - expected` is in kW, but physics.py and
     CLAUDE.md both describe the target as "independent of installed capacity".
     The physics removes the diurnal and seasonal SHAPE, not the SCALE. Trained
     on a 6.0 kW AC array and spent on a 2.5 kW one, the model over-corrects by
     the capacity ratio -- which is a positive bias at EVERY hour, which is
     exactly what step 14 measured (+0.07..+0.22 kW, never negative).

  2. NO AZIMUTH VARIANCE. Training used one array at one orientation, so
     `azimuth_offset` is constant in the training set and cannot have taught the
     model anything transferable about orientation.

Fix 1 is `model.normalise_residual`. Fix 2 needs more than one training array.

TWO TESTS, DELIBERATELY SEPARATED:

  A  ISOLATES THE UNIT FIX. Train on array 3 alone, normalised, and re-score the
     step-14 holdout (16C, east). This is a SECOND look at 16C, so it is a
     diagnostic, not independent evidence, and is labelled as such.

  B  LEAST-CONTAMINATED TEST. Train on arrays 3 (north, 0 deg) and 16C (east,
     90 deg), normalised, and score array 16D (west, 280 deg). Unseen array and
     a period after the training cut. It is NOT fully untouched: 16D's geometry
     was fitted before this run (on 2015-2020, outside the evaluation window)
     and its capacity was assumed from a full-record power comparison that did
     read the evaluation period in aggregate. See config.DKASC_SITE_16D.

  C  DEVELOPMENT EXPERIMENT, NOT INDEPENDENT. C reuses B's exact evaluation
     window, and it was written after B's result had been seen. It asks a
     different question -- what happens when the target array is in the training
     pool -- but it carries no independent evidential weight and must not be
     reported as confirmation.

NONE of these establish zero-history performance. Every baseline here, including
the one the ML is measured against, adapts using the target array's own past
production. They speak to forecasting where local history exists. The no-local-
data case is a separate capability needing separate evidence.

    python scripts/step15_multiarray.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import features, metrics, model, physics, pipeline  # noqa: E402
from dustcast.config import (ARTIFACTS, DKASC_SITE_03, DKASC_SITE_16C,  # noqa: E402
                             DKASC_SITE_16D, PROCESSED, Site)

BOOTSTRAP, SEED = 1000, 15
TRAIN_CUT_B = "2025-03-30"      # arrays 3 + 16C train up to here for test B
TEST_B_START = "2025-03-31"     # 16D scored after it
HOLDOUT_A = "2025-08-21"        # step 14's window, unchanged
BASE_VARS = {
    "shortwave_radiation": "ghi", "diffuse_radiation": "dhi",
    "direct_normal_irradiance": "dni", "temperature_2m": "temp_air",
    "wind_speed_10m": "wind_speed", "relative_humidity_2m": "relative_humidity",
    "precipitation": "precipitation",
}


def rule(t: str) -> None:
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}")


def lead1(df: pd.DataFrame) -> pd.DataFrame:
    """Day-ahead forecast weather, timestamped at the hour midpoint."""
    cols = {d: df[f"{s}_previous_day1"] for s, d in BASE_VARS.items()
            if f"{s}_previous_day1" in df}
    out = pd.DataFrame(cols)
    return out.set_axis(out.index - pd.Timedelta("30min"), axis=0).dropna(how="all")


def nwp_archive() -> pd.DataFrame:
    """All arrays share one location, so one NWP series serves all of them."""
    a = pd.read_parquet(PROCESSED / "dkasc_leadtime_nwp.parquet")
    a.index = pd.to_datetime(a.index)
    b = pd.read_parquet(PROCESSED / "holdout_16c_nwp_lead1.parquet")
    w = pd.concat([lead1(a), lead1(b)]).sort_index()
    return w[~w.index.duplicated(keep="last")]


def build(site: Site, nwp: pd.DataFrame) -> pd.DataFrame:
    """Forecast-driven physics frame for one array, with its own site physics.

    Clear-sky scale and ageing are fitted on the array's own LONG measured
    record. Ageing especially: the NWP archive starts in 2024, and 19 months
    cannot identify a ~1%/yr trend -- attempting it on the joined frame returned
    a POSITIVE ageing slope.
    """
    power = pipeline.hourly_frame(site)
    cs = physics.fit_clearsky_scale(site, power)
    meas = physics.build_physics_frame(site, power, clearsky_scale=cs)
    meas = meas[meas["expected_physics_kw"].notna() & meas["ac_power_kw"].notna()]
    ageing = model.fit_degradation(meas, meas.index, site)

    w = nwp.join(power[["ac_power_kw"]], how="inner")
    w = w.dropna(subset=["ghi", "temp_air", "ac_power_kw"])
    f = physics.build_physics_frame(site, w, clearsky_scale=cs)
    f = f[f["expected_physics_kw"].notna() & f["ac_power_kw"].notna()]
    f = model.apply_degradation(f, site, *ageing)
    f = f[metrics.daylight_mask(f)]
    f.attrs["site"], f.attrs["ageing"] = site, ageing
    return f


def xy(frame: pd.DataFrame, site: Site) -> tuple[pd.DataFrame, pd.Series]:
    X = features.make_features(frame, site)[features.feature_names()]
    y = model.normalise_residual(frame, site)      # fraction of AC capacity
    ok = X.notna().all(axis=1) & y.notna()
    return X[ok], y[ok]


def train_on(parts: list[tuple[pd.DataFrame, Site]], cut: str | None) -> model.ResidualModel:
    """Fit one booster on the pooled, capacity-normalised residuals."""
    blocks = []
    for f, s in parts:
        X, y = xy(f, s)
        if cut is not None:
            keep = X.index <= pd.Timestamp(cut, tz=X.index.tz)
            X, y = X[keep], y[keep]
        blocks.append(X.assign(_y=y.to_numpy()))
    # Carry the target as a COLUMN. The arrays share a location and therefore
    # share timestamps, so concatenating and then realigning y by index label
    # would cross-join the duplicates instead of pairing them.
    # Interleave by time so the early-stopping tail is not one single array.
    both = pd.concat(blocks).sort_index()
    X, y = both.drop(columns="_y"), both["_y"]
    rm = model.ResidualModel(site=parts[0][1], feature_cols=features.feature_names(),
                             capacity_normalised=True)
    rm.fit(X, y)
    return rm


def score(frame: pd.DataFrame, site: Site, rm: model.ResidualModel,
          fit_idx: pd.DatetimeIndex, test_idx: pd.DatetimeIndex,
          label: str, old: model.ResidualModel | None = None) -> pd.DataFrame:
    """Score every model on identical hours, with day-block bootstrap CIs."""
    X, _ = xy(frame, site)
    frame = frame.loc[X.index]
    fit_idx = fit_idx.intersection(X.index)
    test_idx = test_idx.intersection(X.index)

    hod = model.fit_diurnal_bias(frame, fit_idx)
    factor = model.fit_physics_calibration(frame, fit_idx)

    def preds(idx):
        f = frame.loc[idx]
        out = {
            "physics + fitted derate": model.baseline_physics_calibrated(f, site, factor),
            "physics + diurnal bias": model.baseline_physics_diurnal(f, site, hod),
            "DustCast (normalised)": rm.predict_power(X.loc[idx], f, site),
        }
        if old is not None:
            out["DustCast (step 14, kW target)"] = model.clamp_power(
                f["expected_kw"] + old.predict_residual(X.loc[idx, old.feature_cols]),
                f, site)
        return out

    # Same site adaptation the diurnal baseline gets: hour-of-day mean of the
    # model's own error on the fit window. Nothing from the test period.
    fit_p = preds(fit_idx)
    act_fit = frame.loc[fit_idx, "ac_power_kw"]
    ph = preds(test_idx)
    for name in [k for k in ph if k.startswith("DustCast")]:
        e = fit_p[name] - act_fit
        corr = pd.Series(test_idx.hour, index=test_idx).map(
            e.groupby(e.index.hour).mean()).fillna(0.0)
        ph[f"{name} + site bias"] = model.clamp_power(ph[name] - corr, frame.loc[test_idx], site)

    act = frame.loc[test_idx, "ac_power_kw"]
    cap = site.ac_capacity_w / 1000.0
    rows = {k: metrics.deterministic(act, v, capacity_kw=cap) for k, v in ph.items()}
    best = min((k for k in rows if not k.startswith("DustCast")),
               key=lambda k: rows[k]["mae"])
    tbl = metrics.compare(rows, baseline=best)

    rule(label)
    print(f"  test array   {site.name}")
    print(f"  azimuth      {site.azimuth:.0f} deg")
    print(f"  fit window   {fit_idx.min():%Y-%m-%d} .. {fit_idx.max():%Y-%m-%d} ({len(fit_idx):,} h)")
    print(f"  TEST window  {test_idx.min():%Y-%m-%d} .. {test_idx.max():%Y-%m-%d} ({len(test_idx):,} h)")
    pd.set_option("display.width", 170, "display.float_format", lambda v: f"{v:,.4f}")
    print()
    print(tbl[["n", "mae", "rmse", "mbe", "nmae_pct", "skill_mae", "skill_rmse"]].to_string())
    print(f"\n  strongest non-ML baseline: {best}")

    rng = np.random.default_rng(SEED)
    days = pd.Index(sorted(set(act.index.normalize())))
    bd = {d: act.index.normalize() == d for d in days}
    A, R = act.to_numpy(), ph[best].to_numpy()
    picks = [rng.choice(len(days), len(days), True) for _ in range(BOOTSTRAP)]
    for name in [k for k in ph if k.startswith("DustCast")]:
        M = ph[name].to_numpy()
        sm = []
        for pk in picks:
            mk = np.zeros(len(A), bool)
            for i in pk:
                mk |= bd[days[i]]
            sm.append(1 - np.mean(np.abs(M[mk]-A[mk])) / np.mean(np.abs(R[mk]-A[mk])))
        print(f"  {name:<34s} skill MAE {tbl.loc[name,'skill_mae']:+7.1%}  "
              f"95% CI [{np.percentile(sm,2.5):+.1%}, {np.percentile(sm,97.5):+.1%}]")
    return tbl


def main() -> int:
    nwp = nwp_archive()
    f3 = build(DKASC_SITE_03, nwp)
    f16c = build(DKASC_SITE_16C, nwp)
    f16d = build(DKASC_SITE_16D, nwp)

    rule("ARRAYS")
    for f, s in ((f3, DKASC_SITE_03), (f16c, DKASC_SITE_16C), (f16d, DKASC_SITE_16D)):
        a = f.attrs["ageing"]
        print(f"  {s.key:<14s} az {s.azimuth:>3.0f} deg  {s.ac_capacity_w/1000:.1f} kW AC  "
              f"{len(f):>6,} lit h  {f.index.min():%Y-%m-%d}..{f.index.max():%Y-%m-%d}  "
              f"ageing {-a[1]/a[0]*100:.2f} %/yr")

    # ---- TEST A: unit fix alone, second look at 16C (diagnostic) ----
    rm_a = train_on([(f3, DKASC_SITE_03)], cut=None)
    old = model.ResidualModel.load(ARTIFACTS / "residual_model.joblib")
    tz = f16c.index.tz
    cut_a = pd.Timestamp(HOLDOUT_A, tz=tz)
    score(f16c, DKASC_SITE_16C, rm_a,
          f16c.index[f16c.index < cut_a], f16c.index[f16c.index >= cut_a],
          "TEST A -- UNIT FIX ONLY, re-scoring step 14's 16C holdout "
          "(SECOND LOOK: diagnostic, not independent)", old=old)

    # ---- TEST B: multi-array, fresh array 16D ----
    rm_b = train_on([(f3, DKASC_SITE_03), (f16c, DKASC_SITE_16C)], cut=TRAIN_CUT_B)
    tz = f16d.index.tz
    cut_b = pd.Timestamp(TEST_B_START, tz=tz)
    print(f"\n  test B training pool: arrays 3 (az 0) + 16C (az 90), "
          f"through {TRAIN_CUT_B}, {rm_b.best_iteration + 1} trees")
    score(f16d, DKASC_SITE_16D, rm_b,
          f16d.index[f16d.index < cut_b], f16d.index[f16d.index >= cut_b],
          "TEST B -- MULTI-ARRAY + UNIT FIX on array 16D (west), "
          "LEAST-CONTAMINATED (geometry fitted pre-run; capacity assumed)")

    # ---- TEST C: the monitored-site case ----
    # Tests A and B ask whether a model trained elsewhere can be dropped onto an
    # array it has never seen. A deployed platform does not stay in that state:
    # it accumulates local measurements. Test C adds the target array's OWN
    # pre-cut history to the training pool and scores the same later period, so
    # the only thing that changes is whether the model has seen this array.
    rm_c = train_on([(f3, DKASC_SITE_03), (f16c, DKASC_SITE_16C),
                     (f16d, DKASC_SITE_16D)], cut=TRAIN_CUT_B)
    print(f"\n  test C training pool: arrays 3 + 16C + 16D's OWN history "
          f"through {TRAIN_CUT_B}, {rm_c.best_iteration + 1} trees")
    score(f16d, DKASC_SITE_16D, rm_c,
          f16d.index[f16d.index < cut_b], f16d.index[f16d.index >= cut_b],
          "TEST C -- DEVELOPMENT EXPERIMENT (reuses B's window, written after "
          "seeing B): target array now IN the training pool")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
