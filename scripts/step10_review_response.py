"""Step 10: experiments demanded by external review.

Four gaps, each one a claim the report made that the evidence did not carry.

  1  POWER error by forecast horizon. The lead-time figure showed GHI error and
     the text drew conclusions about PV forecasting. Different quantity.
  2  Ramp skill against a TRIVIAL baseline. A POD of 0.95 means nothing if the
     ramps are sunrise and sunset, which a clear-sky model already knows.
  3  The aerosol correction ablated on its own, separately from the ML dust
     features. The two were conflated, and the features are the weak half.
  4  Forecast-ERROR correlation between two real sites, versus the clear-sky
     index correlation used as its proxy in the national aggregation.

Every experiment prints its evaluation window, model, and sample count, because
the previous version did not and that was the first thing the review caught.

    python scripts/step10_review_response.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import (conformal, features, live, metrics, model,  # noqa: E402
                      physics, pipeline)
from dustcast.config import ARTIFACTS, DKASC_SITE_03, PROCESSED  # noqa: E402

LEAD_NWP = PROCESSED / "dkasc_leadtime_nwp.parquet"
LEADS = (0, 1, 2, 3, 5, 7)
BASE_VARS = {
    "shortwave_radiation": "ghi", "diffuse_radiation": "dhi",
    "direct_normal_irradiance": "dni", "temperature_2m": "temp_air",
    "wind_speed_10m": "wind_speed", "relative_humidity_2m": "relative_humidity",
    "precipitation": "precipitation",
}
#: The lead-time archive was fetched over this window. Stated here because the
#: report previously implied seventeen years, which is the TRAINING span.
LEAD_WINDOW = ("2024-01-01", "2025-08-20")


def rule(t: str) -> None:
    print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


def lead_weather(raw: pd.DataFrame, lead: int) -> pd.DataFrame:
    cols = {}
    for src, dst in BASE_VARS.items():
        name = src if lead == 0 else f"{src}_previous_day{lead}"
        if name in raw:
            cols[dst] = raw[name]
    df = pd.DataFrame(cols)
    return df.set_axis(df.index - pd.Timedelta("30min"), axis=0).dropna(how="all")


def build(site, weather, ageing, cs_scale):
    frame = physics.build_physics_frame(site, weather, clearsky_scale=cs_scale)
    frame = frame[frame["expected_physics_kw"].notna() & frame["ac_power_kw"].notna()]
    frame = model.apply_degradation(frame, site, *ageing)
    X = features.make_features(frame, site)[features.feature_names()]
    y = frame["residual_kw"]
    ok = X.notna().all(axis=1) & y.notna() & metrics.daylight_mask(frame)
    return frame[ok], X[ok], y[ok]


def main() -> int:
    site = DKASC_SITE_03
    cap = site.ac_capacity_w / 1000.0
    rm = model.ResidualModel.load(ARTIFACTS / "residual_model.joblib")
    bundle_path = (ARTIFACTS / "cqr_bundle_deploy.joblib"
                   if (ARTIFACTS / "cqr_bundle_deploy.joblib").exists()
                   else ARTIFACTS / "cqr_bundle.joblib")
    bundle = conformal.CQRBundle.load(bundle_path)
    ref = pipeline.prepare(site)
    raw = pd.read_parquet(LEAD_NWP)
    raw.index = pd.to_datetime(raw.index)
    measured = pipeline.hourly_frame(site)

    # ---------------------------------------------------------------- 1
    rule("1. POWER FORECAST ERROR BY LEAD TIME  (not irradiance)")
    print(f"site        {site.name}")
    print(f"window      {LEAD_WINDOW[0]} .. {LEAD_WINDOW[1]}  (NOT the 17-year training span)")
    print(f"weather     Open-Meteo Historical Forecast API; lead 0 is the stitched")
    print(f"            first-hours analysis, leads 1-7 are `previous_dayN` forecasts")
    print(f"target      measured AC power, DKASC array 3\n")

    rows = []
    for L in LEADS:
        w = lead_weather(raw, L).join(measured[["ac_power_kw"]], how="inner")
        w = w.dropna(subset=["ghi", "temp_air", "ac_power_kw"])
        frame, X, y = build(site, w, ref.ageing, ref.clearsky_scale)
        if frame.empty:
            continue
        actual = frame["ac_power_kw"]
        point = model.clamp_power(
            frame["expected_kw"] + rm.predict_residual(X[rm.feature_cols]), frame, site)
        lo_r, hi_r = bundle.residual_interval(X)
        lo = model.clamp_power(frame["expected_kw"] + lo_r, frame, site)
        hi = model.clamp_power(frame["expected_kw"] + hi_r, frame, site)
        det = metrics.deterministic(actual, point, capacity_kw=cap)
        rows.append({"lead_d": L, "n": det["n"], "MAE_kW": det["mae"],
                     "RMSE_kW": det["rmse"], "nMAE_%": det["nmae_pct"],
                     "PICP": metrics.picp(actual, lo, hi),
                     "PINAW": metrics.pinaw(lo, hi, cap)})
    tbl = pd.DataFrame(rows)
    pd.set_option("display.width", 130, "display.float_format", lambda v: f"{v:,.4f}")
    print(tbl.to_string(index=False))
    d0, d7 = tbl.iloc[0], tbl.iloc[-1]
    print(f"\npower nMAE {d0['nMAE_%']:.2f}% at lead 0 -> {d7['nMAE_%']:.2f}% at lead 7 "
          f"({d7['nMAE_%'] / d0['nMAE_%'] - 1:+.0%})")
    print("The band is calibrated on lead-1 data, so its coverage degrades with")
    print("lead -- which is the honest reason to calibrate per horizon, and is")
    print("visible here in a way the irradiance-only figure could not show.")
    tbl.to_csv(ARTIFACTS / "step10_power_by_lead.csv", index=False)

    # ---------------------------------------------------------------- 2
    rule("2. RAMP SKILL ABOVE A TRIVIAL BASELINE")
    print("A high POD is worthless if the ramps are sunrise and sunset. Scored")
    print("against two baselines that already know the time of day:\n")
    w = lead_weather(raw, 1).join(measured[["ac_power_kw"]], how="inner")
    w = w.dropna(subset=["ghi", "temp_air", "ac_power_kw"])
    frame, X, y = build(site, w, ref.ageing, ref.clearsky_scale)
    actual = frame["ac_power_kw"]
    cands = {
        "clear-sky physics only": model.baseline_physics_trend(frame, site),
        "smart persistence": model.baseline_smart_persistence(frame, site),
        "DustCast (physics+ML)": model.clamp_power(
            frame["expected_kw"] + rm.predict_residual(X[rm.feature_cols]), frame, site),
    }
    out = []
    for name, pred in cands.items():
        r = metrics.ramp_detection(actual, pred.reindex(actual.index),
                                   window="3h", capacity_kw=cap)
        out.append({"forecast": name, "down_POD": r["down"]["POD"],
                    "down_FAR": r["down"]["FAR"], "down_CSI": r["down"]["CSI"],
                    "up_CSI": r["up"]["CSI"],
                    "amp_MAE_kW": r["amplitude_mae_kw"]})
    rt = pd.DataFrame(out)
    print(rt.to_string(index=False))
    base_csi = rt.loc[rt.forecast == "clear-sky physics only", "down_CSI"].iloc[0]
    ours = rt.loc[rt.forecast == "DustCast (physics+ML)", "down_CSI"].iloc[0]
    print(f"\ndown-ramp CSI: physics alone {base_csi:.3f} -> DustCast {ours:.3f} "
          f"({ours - base_csi:+.3f})")
    print("Most ramp detection IS the diurnal cycle. The ML's contribution is the")
    print("difference between these rows, not the absolute score.")
    rt.to_csv(ARTIFACTS / "step10_ramp_skill.csv", index=False)

    # ---------------------------------------------------------------- 3
    rule("3. THE AEROSOL CORRECTION, ABLATED ON ITS OWN")
    print("Previously the dust FEATURES and the aerosol CORRECTION were reported")
    print("together, and the features are the weak half. Here the pipeline is")
    print("identical except that forecast GHI/DHI are scaled by the fitted")
    print("aerosol bias. Lead-1 forecasts, measured power as truth.\n")
    dust = live.fetch_dust_history(site, "2022-08-04", "2025-08-23")
    bias_a, bias_b = physics.fit_nwp_aod_bias(
        lead_weather(raw, 1)["ghi"], measured["ghi"],
        dust["aerosol_optical_depth"])
    print(f"fitted bias: rel = {bias_a:+.4f} {bias_b:+.4f} * log10(AOD)")

    base_w = lead_weather(raw, 1).join(measured[["ac_power_kw"]], how="inner")
    base_w = base_w.dropna(subset=["ghi", "temp_air", "ac_power_kw"])
    aod = dust["aerosol_optical_depth"].reindex(base_w.index).interpolate(limit=3)
    factor = 1.0 / (1.0 + bias_a + bias_b * np.log10(
        aod.clip(lower=physics.AOD_FLOOR)))
    corr_w = base_w.copy()
    corr_w["ghi"] = corr_w["ghi"] * factor.fillna(1.0)
    corr_w["dhi"] = corr_w["dhi"] * factor.fillna(1.0)

    res = {}
    for label, wx in (("aerosol correction OFF", base_w),
                      ("aerosol correction ON", corr_w)):
        fr, Xx, _ = build(site, wx, ref.ageing, ref.clearsky_scale)
        act = fr["ac_power_kw"]
        pt = model.clamp_power(
            fr["expected_kw"] + rm.predict_residual(Xx[rm.feature_cols]), fr, site)
        lo_r, hi_r = bundle.residual_interval(Xx)
        lo = model.clamp_power(fr["expected_kw"] + lo_r, fr, site)
        hi = model.clamp_power(fr["expected_kw"] + hi_r, fr, site)
        a_al = aod.reindex(fr.index)
        heavy = a_al >= a_al.quantile(0.90)
        res[label] = {
            "n": len(fr),
            "MAE_all": metrics.deterministic(act, pt)["mae"],
            "MAE_hiAOD": metrics.deterministic(act[heavy], pt[heavy])["mae"],
            "PICP_all": metrics.picp(act, lo, hi),
            "PICP_hiAOD": metrics.picp(act[heavy], lo[heavy], hi[heavy]),
            "Winkler": metrics.winkler_score(act, lo, hi, 0.10),
        }
    at = pd.DataFrame(res).T
    print(at.to_string())
    dm = (1 - at.loc["aerosol correction ON", "MAE_hiAOD"]
          / at.loc["aerosol correction OFF", "MAE_hiAOD"])
    print(f"\non the dustiest 10% of hours the correction changes MAE by {dm:+.1%}")
    at.to_csv(ARTIFACTS / "step10_aerosol_ablation.csv")

    # ---------------------------------------------------------------- 4
    rule("4. FORECAST-ERROR CORRELATION vs ITS CLEAR-SKY-INDEX PROXY")
    print("The national interval narrows by assuming regional forecast ERRORS")
    print("are only partly correlated, but rho was estimated from clear-sky")
    print("INDEX. Those are different quantities. Measured on two real sites:\n")
    yul = pd.read_csv("data/raw/fleet/yulara_total_pv.csv",
                      usecols=["timestamp", "Global_Horizontal_Radiation"],
                      on_bad_lines="warn")
    yul["timestamp"] = pd.to_datetime(yul["timestamp"], errors="coerce")
    yul = yul.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    yul = yul[~yul.index.duplicated()]
    yul.index = yul.index.tz_localize(site.tz, ambiguous="NaT", nonexistent="NaT")
    yul = yul[yul.index.notna()]
    ygh = pd.to_numeric(yul["Global_Horizontal_Radiation"],
                        errors="coerce").resample("1h").mean()
    ygh = ygh.set_axis(ygh.index + pd.Timedelta("30min"))

    ynwp = live.fetch_forecast_history(
        type(site)(**{**site.__dict__, "key": "yulara", "latitude": -25.2406,
                      "longitude": 130.9889, "altitude": 492.0}),
        LEAD_WINDOW[0], LEAD_WINDOW[1])["ghi"]

    anwp = lead_weather(raw, 0)["ghi"]
    pair = pd.DataFrame({"a_m": measured["ghi"], "a_f": anwp,
                         "y_m": ygh, "y_f": ynwp}).dropna()
    pair = pair[(pair[["a_m", "y_m"]] > 100).all(axis=1)]
    err_a, err_y = pair["a_f"] - pair["a_m"], pair["y_f"] - pair["y_m"]
    rho_err = float(err_a.corr(err_y))
    kt_a = pair["a_m"] / pair["a_m"].rolling(24 * 30, min_periods=50).max()
    kt_y = pair["y_m"] / pair["y_m"].rolling(24 * 30, min_periods=50).max()
    rho_kt = float(kt_a.corr(kt_y))
    print(f"  overlapping well-lit hours     {len(pair):,}")
    print(f"  clear-sky-index proxy   rho =  {rho_kt:.3f}")
    print(f"  ACTUAL forecast error   rho =  {rho_err:.3f}")
    n = 24
    for label, r in (("kt proxy", rho_kt), ("true error", rho_err)):
        w = np.sqrt((1 + (n - 1) * r) / n)
        print(f"  -> national band factor using {label:<11s}: {w:.3f} of the naive sum")
    print("\nIf these differ, the reserve reduction claimed in the report is")
    print("wrong by that ratio, and the proxy must be stated as an assumption.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
