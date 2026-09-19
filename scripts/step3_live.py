"""Step 3: live forecast for one Tunisian city, end to end.

Gate (CLAUDE.md section 8): the pipeline runs end-to-end on real forecast data.

    Open-Meteo forecast + CAMS dust
        -> pvlib on forecast weather      (Tunisian array geometry)
        -> XGBoost residual                (trained at DKASC)
        -> conformal interval              (calibrated at DKASC)
        -> 48 h forecast with a 90% band

The transfer is the whole point and also the whole risk, so the assumptions are
printed at run time rather than buried.

    python scripts/step3_live.py
    python scripts/step3_live.py --refresh --retrain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import conformal, features, live, model, physics, pipeline  # noqa: E402
from dustcast.config import (  # noqa: E402
    ARTIFACTS, CONFIDENCE_LEVEL, DKASC_SITE_03, PROCESSED, SFAX_ROOFTOP,
)

# Prefer the NWP-calibrated bundle. The measured-weather one is 21x too narrow
# for a deployment driven by forecast irradiance (see step 8).
BUNDLE_PATH = (ARTIFACTS / "cqr_bundle_deploy.joblib"
               if (ARTIFACTS / "cqr_bundle_deploy.joblib").exists()
               else ARTIFACTS / "cqr_bundle.joblib")
MODEL_PATH = ARTIFACTS / "residual_model.joblib"

#: Nominal degradation for a modern Tunisian rooftop. DKASC's measured
#: -1.55 %/yr belongs to a 2008 poly-Si array in a harsh desert with an unknown
#: cleaning regime and must not be transferred; this is the industry rule of
#: thumb, and it is an ASSUMPTION.
TN_DEGRADATION_PER_YEAR = 0.005


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def load_or_train(retrain: bool):
    if not retrain and BUNDLE_PATH.exists() and MODEL_PATH.exists():
        return model.ResidualModel.load(MODEL_PATH), conformal.CQRBundle.load(BUNDLE_PATH)

    print("training on DKASC (cached ingest, ~2 min)...")
    prep = pipeline.prepare(DKASC_SITE_03)
    rm = pipeline.fit_residual_model(prep)
    lit = prep.frame.index[prep.daylight()]
    bundle = conformal.fit_cqr_bundle(
        rm, prep.X, prep.y,
        prep.split.train.intersection(lit),
        prep.split.calibrate.intersection(lit),
        CONFIDENCE_LEVEL,
    )
    rm.save(MODEL_PATH)
    bundle.save(BUNDLE_PATH)
    return rm, bundle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="bypass the API cache")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--hours", type=int, default=48)
    args = ap.parse_args()

    site = SFAX_ROOFTOP

    rule("1. MODEL (trained at DKASC Alice Springs)")
    rm, bundle = load_or_train(args.retrain)
    print(f"residual model  {len(bundle.feature_cols)} features, "
          f"{rm.best_iteration + 1} trees")
    print(f"conformal       {bundle.confidence_level:.0%} band, "
          f"correction {bundle.correction:+.4f} kW, "
          f"calibrated {bundle.calibrated_on}")

    rule("2. LIVE DATA (Open-Meteo)")
    weather = live.fetch_weather(site, past_days=60, forecast_days=3,
                                 refresh=args.refresh)
    dust = live.fetch_dust(site, past_days=60, forecast_days=5,
                           refresh=args.refresh)
    now = pd.Timestamp.now(tz=site.tz)
    print(f"site     {site.name}")
    print(f"weather  {len(weather):,} h  "
          f"{weather.index.min():%Y-%m-%d %H:%M} .. {weather.index.max():%Y-%m-%d %H:%M}")
    print(f"dust     {len(dust):,} h  "
          f"{dust.index.min():%Y-%m-%d} .. {dust.index.max():%Y-%m-%d}")
    print(f"now      {now:%Y-%m-%d %H:%M %Z}")

    ahead = dust.loc[now:now + pd.Timedelta(hours=args.hours)]
    print(f"\ndust next {args.hours} h: "
          f"mean {ahead['dust'].mean():.1f} ug/m3, peak {ahead['dust'].max():.1f}")
    print(f"AOD  next {args.hours} h: "
          f"mean {ahead['aerosol_optical_depth'].mean():.3f}, "
          f"peak {ahead['aerosol_optical_depth'].max():.3f}")

    rule("3. PHYSICS ON FORECAST WEATHER")
    # No clear-sky scale: the DKASC factor absorbed that site's pyranometer and
    # is meaningless here. Deriving turbidity from the CAMS AOD already fetched
    # above is the physical route, and belongs to step 4.
    frame = physics.build_physics_frame(site, weather, clearsky_scale=None)

    # The derate must be built for Tunisia, not inherited. DKASC's fitted
    # ageing (-1.55 %/yr off a 0.974 intercept) describes a specific 2008 array
    # and would impose three times the real degradation on a five-year-old
    # rooftop.
    a = model.pvwatts_loss_factor()
    b = -TN_DEGRADATION_PER_YEAR * a
    frame = model.apply_degradation(frame, site, a, b)
    age = model.age_years(frame.index, site).mean()
    print(f"derate   PVWatts x{a:.4f}, ageing {TN_DEGRADATION_PER_YEAR:.1%}/yr "
          f"(ASSUMED), array age {age:.1f} yr")
    print(f"expected peak over window: "
          f"{frame['expected_kw'].max():.3f} kW of {site.ac_capacity_w / 1000:.1f} kW")

    rule("4. FORECAST WITH CALIBRATED BAND")
    X = features.make_features(frame, site)
    # Require only the physics driver. Demanding every feature be present is
    # right when assembling clean training rows but wrong at forecast time: an
    # operator needs a number for every hour, and XGBoost splits on missingness
    # natively. Insisting on completeness here discarded the entire forecast
    # because one feature (days since rain) was legitimately unknown.
    usable = frame["expected_kw"].notna()
    X, frame = X[usable], frame[usable]
    incomplete = X[bundle.feature_cols].isna().any(axis=1).mean()
    print(f"{len(X):,} forecast hours; {incomplete:.1%} have at least one "
          f"feature missing (handled natively by the booster)")

    resid_lo, resid_hi = bundle.residual_interval(X)
    point = model.clamp_power(
        frame["expected_kw"] + rm.predict_residual(X[rm.feature_cols]), frame, site)
    lower = model.clamp_power(frame["expected_kw"] + resid_lo, frame, site)
    upper = model.clamp_power(frame["expected_kw"] + resid_hi, frame, site)

    out = pd.DataFrame({
        "expected_kw": frame["expected_kw"], "point_kw": point,
        "lower_kw": lower, "upper_kw": upper,
        "ghi": frame["ghi"], "temp_air": frame["temp_air"],
        "clearsky_index": frame["clearsky_index"],
        "solar_elevation": frame["solar_elevation"],
    }).join(dust.reindex(frame.index).interpolate(limit=3))

    window = out.loc[now:now + pd.Timedelta(hours=args.hours)]
    lit = window["solar_elevation"] > 0
    print(f"forecast horizon {len(window)} h, {lit.sum()} of them daylight")
    print(f"energy   point {window['point_kw'].sum():.1f} kWh  "
          f"[{window['lower_kw'].sum():.1f} .. {window['upper_kw'].sum():.1f}]")
    print(f"peak     {window['point_kw'].max():.3f} kW  "
          f"[{window.loc[window['point_kw'].idxmax(), 'lower_kw']:.3f} .. "
          f"{window.loc[window['point_kw'].idxmax(), 'upper_kw']:.3f}]")
    band = (window.loc[lit, "upper_kw"] - window.loc[lit, "lower_kw"])
    print(f"band     mean {band.mean():.3f} kW "
          f"({band.mean() / (site.ac_capacity_w / 1000):.1%} of capacity) "
          f"over daylight hours")

    print("\nnext 12 daylight hours:")
    head = window[lit].head(12)
    show = head[["ghi", "temp_air", "dust", "expected_kw",
                 "point_kw", "lower_kw", "upper_kw"]]
    pd.set_option("display.width", 130, "display.float_format",
                  lambda v: f"{v:,.2f}")
    print(show.to_string())

    out.to_parquet(PROCESSED / "step3_live_forecast.parquet")

    rule("GATE: does the pipeline run end-to-end on real forecast data?")
    ok = len(window) > 0 and window["point_kw"].notna().all() and lit.sum() > 0
    print(f"  live weather fetched      {len(weather):,} hours")
    print(f"  live CAMS dust fetched    {len(dust):,} hours")
    print(f"  physics + ML + interval   {len(out):,} hours produced")
    print(f"\n  {'PASS' if ok else 'FAIL'}")

    print("\n  UNVERIFIABLE, and it must be said out loud: there is no measured")
    print("  Tunisian output to score this against. The 90% band inherits its")
    print("  calibration from DKASC, and its coverage here is an assumption,")
    print("  not a result. Step 2 additionally showed that even at DKASC a")
    print("  static calibration under-covers once the error distribution")
    print("  drifts -- and drift across a continent is larger than drift")
    print("  across two years.")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
