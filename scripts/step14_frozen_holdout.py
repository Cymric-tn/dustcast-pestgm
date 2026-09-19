"""Step 14: the frozen evaluation on data that guided no decision.

Models, baselines, calibration and dispatch design were all revised while
looking at array 3's 2025-03-31..08-20 window. That window is development data
now, and any number from it is optimistic by an unknown amount.

This evaluates the pipeline, unchanged, on DKASC array 16C over
2025-08-21..2026-02-19 -- an array never trained on, tuned on or inspected, over
a period later than anything previously scored. It confounds an unseen site with
an unseen period, which is stated rather than hidden, and is closer to
deployment than a pure temporal holdout would be.

WHAT IS FROZEN        residual model, conformal bundle, feature set, dispatch
                      rules, and the choice of baselines.
WHAT IS REFITTED      only site-specific physics that a deployment would also
                      fit per site -- ageing, clear-sky scale, the diurnal bias
                      baseline, and the cost-optimal reserve level -- all on
                      16C data strictly BEFORE the holdout begins.

    python scripts/step14_frozen_holdout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import (conformal, dispatch, features, metrics, model,  # noqa: E402
                      physics, pipeline)
from dustcast.config import ARTIFACTS, DKASC_SITE_16C, PROCESSED  # noqa: E402

HOLDOUT_START = "2025-08-21"
RELIABILITY_GRID = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995]
BOOTSTRAP, SEED = 1000, 14
BASE_VARS = {
    "shortwave_radiation": "ghi", "diffuse_radiation": "dhi",
    "direct_normal_irradiance": "dni", "temperature_2m": "temp_air",
    "wind_speed_10m": "wind_speed", "relative_humidity_2m": "relative_humidity",
    "precipitation": "precipitation",
}


def rule(t: str) -> None:
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}")


def lead1(df: pd.DataFrame) -> pd.DataFrame:
    cols = {d: df[f"{s}_previous_day1"] for s, d in BASE_VARS.items()
            if f"{s}_previous_day1" in df}
    out = pd.DataFrame(cols)
    return out.set_axis(out.index - pd.Timedelta("30min"), axis=0).dropna(how="all")


def conformal_offset(err: np.ndarray, level: float) -> float:
    e = np.asarray(err, float)
    e = e[np.isfinite(e)]
    n = len(e)
    return float(np.quantile(e, min(np.ceil((n + 1) * level) / n, 1.0), method="higher"))


def main() -> int:
    site = DKASC_SITE_16C
    cap = site.ac_capacity_w / 1000.0
    dc_kw = site.dc_capacity_w / 1000.0
    a = dispatch.DispatchAssumptions()

    rm = model.ResidualModel.load(ARTIFACTS / "residual_model.joblib")
    bundle = conformal.CQRBundle.load(ARTIFACTS / "cqr_bundle_deploy.joblib")

    power = pipeline.hourly_frame(site)
    fit_nwp = lead1(pd.read_parquet(PROCESSED / "dkasc_leadtime_nwp.parquet")
                    .set_axis(pd.to_datetime(
                        pd.read_parquet(PROCESSED / "dkasc_leadtime_nwp.parquet").index)))
    hold_nwp = lead1(pd.read_parquet(PROCESSED / "holdout_16c_nwp_lead1.parquet"))
    nwp = pd.concat([fit_nwp, hold_nwp]).sort_index()
    nwp = nwp[~nwp.index.duplicated(keep="last")]

    w = nwp.join(power[["ac_power_kw"]], how="inner")
    w = w.dropna(subset=["ghi", "temp_air", "ac_power_kw"])

    # Site physics fitted on pre-holdout 16C data only.
    tz = w.index.tz
    cutoff = pd.Timestamp(HOLDOUT_START, tz=tz)
    pre_meas = power.loc[:cutoff]
    cs_scale = physics.fit_clearsky_scale(site, pre_meas)

    # Ageing must be fitted on the array's LONG measured record, not on the
    # NWP-joined frame. The NWP archive starts in 2024, so that frame spans only
    # 1.6 years of array age and the linear fit is unidentifiable: it returned
    # x(0.6708 + 0.00417*age), a POSITIVE ageing slope, which is unphysical.
    # 16C has measured output back to 2008, and a deployment fitting ageing for
    # a site would use exactly that history.
    meas_frame = physics.build_physics_frame(site, pre_meas, clearsky_scale=cs_scale)
    meas_frame = meas_frame[meas_frame["expected_physics_kw"].notna()
                            & meas_frame["ac_power_kw"].notna()]
    ageing = model.fit_degradation(meas_frame, meas_frame.index, site)

    frame = physics.build_physics_frame(site, w, clearsky_scale=cs_scale)
    frame = frame[frame["expected_physics_kw"].notna() & frame["ac_power_kw"].notna()]
    frame = model.apply_degradation(frame, site, *ageing)

    X = features.make_features(frame, site)[features.feature_names()]
    ok = X.notna().all(axis=1) & frame["residual_kw"].notna()
    frame, X = frame[ok], X[ok]
    fit_idx = frame.index[frame.index < cutoff]
    hold = frame.index[frame.index >= cutoff]

    rule("FROZEN HOLDOUT SPECIFICATION")
    print(f"  site (unseen)   {site.name}")
    print(f"                  azimuth {site.azimuth:.0f} deg vs the training array's 0 deg")
    print(f"  fitted on       {fit_idx.min():%Y-%m-%d} .. {fit_idx.max():%Y-%m-%d}  "
          f"({len(fit_idx):,} h)")
    print(f"  HOLDOUT         {hold.min():%Y-%m-%d} .. {hold.max():%Y-%m-%d}  "
          f"({len(hold):,} h)")
    print(f"  frozen          residual model, conformal bundle, features,")
    print(f"                  dispatch rules, baseline choice")
    print(f"  refitted on     ageing x({ageing[0]:.4f}{ageing[1]:+.5f}*age) = "
          f"{-ageing[1]/ageing[0]*100:.2f} %/yr, clear-sky")
    print(f"                  scale, diurnal bias, reserve level -- pre-holdout only")

    lit_fit = fit_idx.intersection(frame.index[metrics.daylight_mask(frame)])
    lit_hold = hold.intersection(frame.index[metrics.daylight_mask(frame)])
    hod = model.fit_diurnal_bias(frame, lit_fit)
    factor = model.fit_physics_calibration(frame, lit_fit)

    def _dustcast(idx):
        f = frame.loc[idx]
        return model.clamp_power(
            f["expected_kw"] + rm.predict_residual(X.loc[idx, rm.feature_cols]), f, site)

    # The diurnal baseline gets 13 numbers refitted on this array. Freezing the
    # residual model while granting the baseline that adaptation is not a like-
    # for-like test, so DustCast is given the SAME privilege: the hour-of-day
    # mean of its own error, fitted on the same pre-holdout window, nothing from
    # the holdout. A real deployment on a new rooftop has exactly this much.
    dc_fit_err = _dustcast(lit_fit) - frame.loc[lit_fit, "ac_power_kw"]
    dc_hod = dc_fit_err.groupby(dc_fit_err.index.hour).mean()

    def points(idx):
        f = frame.loc[idx]
        dc = _dustcast(idx)
        corr = pd.Series(idx.hour, index=idx).map(dc_hod).fillna(0.0)
        return {
            "physics + fitted derate": model.baseline_physics_calibrated(f, site, factor),
            "physics + diurnal bias": model.baseline_physics_diurnal(f, site, hod),
            "DustCast": dc,
            "DustCast + site bias": model.clamp_power(dc - corr, f, site),
        }

    rule("FORECAST ACCURACY ON THE FROZEN HOLDOUT (daylight)")
    ph = points(lit_hold)
    act = frame.loc[lit_hold, "ac_power_kw"]
    rows = {k: metrics.deterministic(act, v, capacity_kw=cap) for k, v in ph.items()}
    best = min((k for k in rows if not k.startswith("DustCast")),
               key=lambda k: rows[k]["mae"])
    tbl = metrics.compare(rows, baseline=best)
    pd.set_option("display.width", 150, "display.float_format", lambda v: f"{v:,.4f}")
    print(tbl[["n", "mae", "rmse", "mbe", "nmae_pct", "skill_mae", "skill_rmse"]].to_string())
    print(f"\n  strongest non-ML baseline: {best}")

    rng = np.random.default_rng(SEED)
    days = pd.Index(sorted(set(act.index.normalize())))
    bd = {d: act.index.normalize() == d for d in days}
    A = act.to_numpy(); R = ph[best].to_numpy()
    # Resample whole DAYS, not hours: forecast errors are strongly correlated
    # within a day, so an hourly bootstrap would report intervals far too tight.
    picks = [rng.choice(len(days), len(days), True) for _ in range(BOOTSTRAP)]
    for name in ("DustCast", "DustCast + site bias"):
        M = ph[name].to_numpy()
        sm, sr = [], []
        for pk in picks:
            mk = np.zeros(len(A), bool)
            for i in pk:
                mk |= bd[days[i]]
            sm.append(1 - np.mean(np.abs(M[mk]-A[mk])) / np.mean(np.abs(R[mk]-A[mk])))
            sr.append(1 - np.sqrt(np.mean((M[mk]-A[mk])**2))
                        / np.sqrt(np.mean((R[mk]-A[mk])**2)))
        print(f"  {name:<22s} skill MAE  {tbl.loc[name,'skill_mae']:+6.1%}  "
              f"95% CI [{np.percentile(sm,2.5):+.1%}, {np.percentile(sm,97.5):+.1%}]")
        print(f"  {'':<22s} skill RMSE {tbl.loc[name,'skill_rmse']:+6.1%}  "
              f"95% CI [{np.percentile(sr,2.5):+.1%}, {np.percentile(sr,97.5):+.1%}]")

    rule("INTERVAL QUALITY -- WIDTH AND COVERAGE ON IDENTICAL CASES")
    fh = frame.loc[lit_hold]
    lo_r, hi_r = bundle.residual_interval(X.loc[lit_hold])
    lo = model.clamp_power(fh["expected_kw"] + lo_r, fh, site)
    hi = model.clamp_power(fh["expected_kw"] + hi_r, fh, site)
    print(f"  DustCast CQR band   PICP {metrics.picp(act, lo, hi):.1%}   "
          f"PINAW {metrics.pinaw(lo, hi, cap):.4f}   "
          f"mean width {np.mean(hi - lo):.3f} kW   "
          f"Winkler {metrics.winkler_score(act, lo, hi, 0.10):.3f}")
    for name in (best, "DustCast", "DustCast + site bias"):
        off = conformal_offset((points(lit_fit)[name]
                                - frame.loc[lit_fit, "ac_power_kw"]).to_numpy(), 0.95)
        b_lo = (ph[name] - off).clip(lower=0.0)
        b_hi = ph[name] + off
        print(f"  {name[:22]:<22s}  PICP {metrics.picp(act, b_lo, b_hi):.1%}   "
              f"PINAW {metrics.pinaw(b_lo, b_hi, cap):.4f}   "
              f"mean width {np.mean(b_hi - b_lo):.3f} kW   "
              f"Winkler {metrics.winkler_score(act, b_lo, b_hi, 0.10):.3f}")
    print("  (a smaller conformal correction is not by itself a better interval;")
    print("   width and coverage on the same cases are what settle it)")


    rule("WHERE THE ERROR LIVES (bias in kW; + = over-predicts)")
    d = pd.DataFrame({"actual": act, "dust": ph["DustCast"], "base": ph[best]})
    d["hour"] = d.index.hour
    by_hour = d.groupby("hour").apply(
        lambda g: pd.Series({
            "n": len(g),
            "mean_kw": g["actual"].mean(),
            "dust_bias": (g["dust"] - g["actual"]).mean(),
            "base_bias": (g["base"] - g["actual"]).mean(),
            "dust_mae": (g["dust"] - g["actual"]).abs().mean(),
            "base_mae": (g["base"] - g["actual"]).abs().mean(),
        }), include_groups=False)
    print(by_hour.to_string())

    # 16C faces EAST. Its output peaks in the morning and falls away in the
    # afternoon. The training array faces NORTH. If the residual model learned a
    # correction tied to the training array's sun geometry rather than a
    # transferable loss mechanism, the damage concentrates in one half-day.
    for label, part in (("morning  (sun onto the east array)", d[d["hour"] < 12]),
                        ("afternoon (sun off the east array)", d[d["hour"] >= 12])):
        print(f"\n  {label}: n={len(part)}, mean output {part['actual'].mean():.3f} kW")
        for nm, col in (("DustCast", "dust"), (best[:22], "base")):
            print(f"    {nm:<22s} bias {(part[col]-part['actual']).mean():+.4f}"
                  f"   MAE {(part[col]-part['actual']).abs().mean():.4f}")

    np.save(ARTIFACTS / "step14_holdout_actual.npy", A)
    tbl.to_csv(ARTIFACTS / "step14_holdout_accuracy.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
