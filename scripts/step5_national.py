"""Step 5: virtual fleet, governorate and national upscaling, grid impact.

Gate (CLAUDE.md section 8): the national number is plausible against installed
capacity.

    Open-Meteo + CAMS for 24 governorates  (two requests, not forty-eight)
      -> 9 sampled rooftop archetypes each, real orientation spread
      -> pvlib + DKASC-trained residual + conformal band, per archetype
      -> specific yield (kW per kW installed) per governorate
      -> x governorate capacity -> x 24 -> national MW
      -> reserve, ramp, and a dust ablation

    python scripts/step5_national.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import (conformal, decisions, features, fleet, live, model,  # noqa: E402
                      physics, pipeline, tunisia)
from dustcast.config import ARTIFACTS, DKASC_SITE_03, PROCESSED  # noqa: E402

TZ = "Africa/Tunis"
HORIZON_HOURS = 48
TN_DEGRADATION_PER_YEAR = 0.005
DUST_FIT_WINDOW = ("2022-08-04", "2025-08-23")


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def forecast_archetype(arch, weather, rm, bundle, derate_a, derate_b):
    """One rooftop: physics -> residual -> band, all in kW."""
    frame = physics.build_physics_frame(arch.site, weather, clearsky_scale=None)
    frame = model.apply_degradation(frame, arch.site, derate_a, derate_b)
    frame = frame[frame["expected_kw"].notna()]
    if frame.empty:
        return None

    X = features.make_features(frame, arch.site)
    lo, hi = bundle.residual_interval(X)
    point = frame["expected_kw"] + rm.predict_residual(X[rm.feature_cols])
    return pd.DataFrame({
        "point": model.clamp_power(point, frame, arch.site),
        "lower": model.clamp_power(frame["expected_kw"] + lo, frame, arch.site),
        "upper": model.clamp_power(frame["expected_kw"] + hi, frame, arch.site),
    })


def forecast_governorate(row, weather, rm, bundle, derate_a, derate_b, seed):
    """Sample yield for one governorate, in kW produced per kW installed."""
    archetypes = fleet.sample_archetypes(
        row.name, row["latitude"], row["longitude"], 50.0, TZ, seed=seed)

    parts, capacity_kw = [], 0.0
    for arch in archetypes:
        out = forecast_archetype(arch, weather, rm, bundle, derate_a, derate_b)
        if out is None:
            continue
        parts.append(out)
        capacity_kw += arch.site.dc_capacity_w / 1000.0

    if not parts:
        return None
    total = sum(parts)
    return total / capacity_kw            # specific yield, dimensionless


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    rm = model.ResidualModel.load(ARTIFACTS / "residual_model.joblib")
    bundle_path = (ARTIFACTS / "cqr_bundle_deploy.joblib"
                   if (ARTIFACTS / "cqr_bundle_deploy.joblib").exists()
                   else ARTIFACTS / "cqr_bundle.joblib")
    bundle = conformal.CQRBundle.load(bundle_path)

    rule("1. NWP AEROSOL BIAS, MEASURED AT DKASC")
    hourly = pipeline.hourly_frame(DKASC_SITE_03)
    nwp_hist = live.fetch_forecast_history(DKASC_SITE_03, *DUST_FIT_WINDOW)
    dust_hist = live.fetch_dust_history(DKASC_SITE_03, *DUST_FIT_WINDOW)
    bias_a, bias_b = physics.fit_nwp_aod_bias(
        nwp_hist["ghi"], hourly["ghi"], dust_hist["aerosol_optical_depth"])
    print(f"rel_bias = {bias_a:+.4f} {bias_b:+.4f} * log10(AOD)")
    for aod in (0.05, 0.20, 0.50):
        print(f"   AOD {aod:.2f} -> NWP GHI is "
              f"{bias_a + bias_b * np.log10(aod):+.1%}")
    p95 = dust_hist["aerosol_optical_depth"].quantile(0.95)
    print(f"fitted over DKASC AOD up to p95 = {p95:.3f}; Tunisian values above "
          f"that are EXTRAPOLATED")

    rule("2. FLEET AND CAPACITY")
    caps = tunisia.capacity_allocation()
    print(f"{len(caps)} governorates, {caps['capacity_mw'].sum():.0f} MW "
          f"residential rooftop (STEG end-June 2026), allocated by population")
    print(f"{fleet.ARCHETYPES_PER_GOVERNORATE} archetypes each: "
          f"tilt ~N({fleet.TILT_MEAN:.0f},{fleet.TILT_SD:.0f}), "
          f"azimuth ~N({fleet.AZIMUTH_MEAN:.0f},{fleet.AZIMUTH_SD:.0f}), "
          f"capacity ~lognormal(3 kW)")
    print("SIMULATED FLEET -- no Tunisian PV registry exists. "
          "The upscaling method is real; the sample is not.")

    rule("3. LIVE DATA FOR 24 GOVERNORATES")
    lats, lons = tunisia.latitudes(), tunisia.longitudes()
    # Seven forecast days, not three. Step 8 established that day-ahead and
    # week-ahead are one accuracy regime, so the extra horizon is real product
    # rather than padding -- and the dashboard needs something to scrub into.
    weathers = live.fetch_weather_multi(lats, lons, TZ, past_days=60,
                                        forecast_days=7, refresh=args.refresh,
                                        tag="tn")
    dusts = live.fetch_dust_multi(lats, lons, TZ, past_days=60,
                                  forecast_days=5, refresh=args.refresh,
                                  tag="tn")
    now = pd.Timestamp.now(tz=TZ)
    print(f"weather {len(weathers)} x {len(weathers[0]):,} h, "
          f"dust {len(dusts)} x {len(dusts[0]):,} h   (2 requests total)")
    aod_now = pd.Series(
        {g.key: dusts[i]["aerosol_optical_depth"].loc[now:now + pd.Timedelta(
            hours=HORIZON_HOURS)].mean()
         for i, g in enumerate(tunisia.GOVERNORATES)})
    print(f"AOD over the next {HORIZON_HOURS} h: "
          f"{aod_now.min():.3f} ({caps.loc[aod_now.idxmin(), 'name']}) .. "
          f"{aod_now.max():.3f} ({caps.loc[aod_now.idxmax(), 'name']})")

    rule("4. GOVERNORATE FORECASTS")
    # Clear-sky index per governorate, daylight only: the spatial-correlation
    # proxy used when aggregating the national interval.
    kt_cols, sky = {}, {}
    for i, gov in enumerate(tunisia.GOVERNORATES):
        probe = fleet.sample_archetypes(gov.key, gov.latitude, gov.longitude,
                                        50.0, TZ, n=1, seed=i)[0].site
        w = physics.complete_weather(probe, weathers[i])
        kt_cols[gov.key] = w["clearsky_index"].where(w["solar_elevation"] > 10.0)
        # Kept for the dashboard: the gap between forecast GHI and the
        # clear-sky envelope IS the cloud, which is what makes a dip in the
        # power curve legible rather than mysterious.
        sky[gov.key] = w[["ghi", "clearsky_ghi", "temp_air"]]
    kt_frame = pd.DataFrame(kt_cols).dropna(how="all")
    print(f"clear-sky index computed for {kt_frame.shape[1]} governorates "
          f"over {kt_frame.notna().any(axis=1).sum():,} daylight hours")

    derate_a = model.pvwatts_loss_factor()
    derate_b = -TN_DEGRADATION_PER_YEAR * derate_a

    results = {}
    for scenario in ("dust_on", "dust_off"):
        per_gov = {}
        for i, gov in enumerate(tunisia.GOVERNORATES):
            weather = weathers[i].copy()
            if scenario == "dust_on":
                aod = dusts[i]["aerosol_optical_depth"].reindex(
                    weather.index).interpolate(limit=3)
                factor = 1.0 / (1.0 + bias_a + bias_b * np.log10(
                    aod.clip(lower=physics.AOD_FLOOR)))
                # Scale GHI and DHI together so the closure stays consistent;
                # complete_weather rebuilds DNI from them. A finer model would
                # raise the diffuse FRACTION under aerosol rather than holding
                # it fixed -- scattering redistributes rather than only removes.
                weather["ghi"] = weather["ghi"] * factor.fillna(1.0)
                weather["dhi"] = weather["dhi"] * factor.fillna(1.0)
            yield_kw = forecast_governorate(
                caps.loc[gov.key], weather, rm, bundle, derate_a, derate_b,
                seed=i)
            per_gov[gov.key] = yield_kw
        results[scenario] = per_gov
        print(f"  {scenario}: {len(per_gov)} governorates x "
              f"{fleet.ARCHETYPES_PER_GOVERNORATE} archetypes modelled")

    rule("5. NATIONAL TOTAL")
    national = {}
    for scenario, per_gov in results.items():
        mw = pd.DataFrame({
            key: fleet.upscale(y["point"], caps.loc[key, "capacity_mw"])
            for key, y in per_gov.items() if y is not None})
        widths = pd.DataFrame({
            key: fleet.upscale(y["upper"] - y["lower"],
                               caps.loc[key, "capacity_mw"])
            for key, y in per_gov.items() if y is not None})

        # Correlate the CLEAR-SKY INDEX, not the power forecast. Power is
        # dominated by the diurnal cycle, which every governorate shares, so
        # correlating it returns ~0.99 no matter what the weather is doing and
        # the whole refinement collapses to the naive sum. kt divides that
        # cycle out, leaving the cloud field -- which is what actually makes
        # forecast errors move together across the country.
        rho = fleet.estimate_spatial_correlation(kt_frame)
        total = mw.sum(axis=1)
        band = fleet.aggregate_interval_width(widths, rho)
        national[scenario] = pd.DataFrame({
            "point_mw": total,
            "lower_mw": (total - band / 2).clip(lower=0),
            "upper_mw": total + band / 2,
            "naive_band_mw": widths.sum(axis=1),
            "band_mw": band,
        })
        national[scenario].attrs["rho"] = rho

    base = national["dust_on"]
    rho = base.attrs["rho"]
    window = base.loc[now:now + pd.Timedelta(hours=HORIZON_HOURS)]
    peak_t = window["point_mw"].idxmax()
    print(f"inter-governorate forecast correlation rho = {rho:.3f}")
    print(f"  national band is {base['band_mw'].mean() / base['naive_band_mw'].mean():.1%} "
          f"of the naive sum-of-bands "
          f"(summing assumes perfect correlation and overstates it)")
    print(f"\npeak over the next {HORIZON_HOURS} h: "
          f"{window['point_mw'].max():.0f} MW "
          f"[{window.loc[peak_t, 'lower_mw']:.0f} .. "
          f"{window.loc[peak_t, 'upper_mw']:.0f}] at {peak_t:%a %H:%M}")
    print(f"energy: {window['point_mw'].sum():.0f} MWh over the window")
    cf = window["point_mw"].max() / caps["capacity_mw"].sum()
    print(f"peak as a fraction of the 500 MW fleet: {cf:.1%}")

    rule("6. DUST ABLATION AT NATIONAL SCALE")
    on = national["dust_on"].loc[window.index, "point_mw"]
    off = national["dust_off"].loc[window.index, "point_mw"]
    diff = on - off
    print(f"dust-aware vs dust-blind over the next {HORIZON_HOURS} h:")
    print(f"  mean shift {diff.mean():+.1f} MW, "
          f"largest {diff.abs().max():.1f} MW at "
          f"{diff.abs().idxmax():%a %H:%M}")
    print(f"  energy    {on.sum():.0f} vs {off.sum():.0f} MWh "
          f"({(on.sum() / off.sum() - 1):+.1%})")

    rule("7. GRID IMPACT")
    reserve = decisions.reserve_requirement(window["point_mw"],
                                            window["lower_mw"])
    ramp = decisions.ramp_risk(window["point_mw"], window="3h")
    print(f"reserve to cover the downside of the band: {reserve:.0f} MW")
    print(f"  (sized off the pessimistic edge, not the point forecast)")
    print(f"worst 3 h ramp: down {ramp['max_down_kw']:.0f} MW, "
          f"up {ramp['max_up_kw']:.0f} MW")

    gov_now = pd.Series({
        key: fleet.upscale(y["point"], caps.loc[key, "capacity_mw"]).loc[
            window.index].max()
        for key, y in results["dust_on"].items() if y is not None})
    caps["peak_mw"] = gov_now
    print(f"\ntop governorates by forecast peak:")
    print(caps.sort_values("peak_mw", ascending=False)[
        ["name", "region", "capacity_mw", "peak_mw"]].head(6).to_string(
        float_format=lambda v: f"{v:8.2f}"))

    out = base.copy()
    out["dust_off_mw"] = national["dust_off"]["point_mw"]
    out.to_parquet(PROCESSED / "step5_national.parquet")
    caps.to_csv(ARTIFACTS / "step5_governorates.csv")

    # Per-governorate series in long form, so the map UI (step 6) does not have
    # to re-run 432 pvlib chains just to draw a chart.
    rows = []
    for key, y in results["dust_on"].items():
        if y is None:
            continue
        mw = caps.loc[key, "capacity_mw"]
        off = results["dust_off"][key]
        idx = [g.key for g in tunisia.GOVERNORATES].index(key)
        aod_series = dusts[idx]["aerosol_optical_depth"].reindex(
            y.index).interpolate(limit=3)
        dust_series = dusts[idx]["dust"].reindex(y.index).interpolate(limit=3)
        # Real forecast precipitation, not a fabricated zero. Passing zeros
        # made the soiling balance accumulate unchecked across the whole 60-day
        # history and fired the cleaning trigger in all 24 governorates.
        precip_series = weathers[idx]["precipitation"].reindex(
            y.index).fillna(0.0)
        w_gov = sky[key].reindex(y.index)
        part = pd.DataFrame({
            "governorate": key,
            "ghi": w_gov["ghi"],
            "clearsky_ghi": w_gov["clearsky_ghi"],
            "temp_air": w_gov["temp_air"],
            "point_mw": y["point"] * mw,
            "lower_mw": y["lower"] * mw,
            "upper_mw": y["upper"] * mw,
            "dust_off_mw": off["point"] * mw,
            "aod": aod_series,
            "dust": dust_series,
            "precipitation": precip_series,
        })
        rows.append(part)
    pd.concat(rows).to_parquet(PROCESSED / "step5_governorate_series.parquet")

    rule("GATE: is the national number plausible against installed capacity?")
    peak_mw = window["point_mw"].max()
    installed = caps["capacity_mw"].sum()
    plausible = 0.35 * installed <= peak_mw <= 0.90 * installed
    print(f"  installed rooftop      {installed:.0f} MW")
    print(f"  forecast peak          {peak_mw:.0f} MW  ({cf:.1%} of installed)")
    print(f"  plausible band for a mixed-orientation fleet in September: "
          f"35-90% of installed")
    print(f"\n  {'PASS' if plausible else 'FAIL'}")
    print(f"\n  The 500 MW total is STEG's. Its split across governorates is "
          f"ours,\n  allocated by population because no registry exists. "
          f"Say that out loud.")

    return 0 if plausible else 1


if __name__ == "__main__":
    raise SystemExit(main())
