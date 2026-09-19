"""Step 17: evaluated performance by forecast horizon, J+0 to J+3.

The concept note asks for intra-day forecasts and J through J+3. Two different
claims are involved and they are kept apart throughout:

  IMPLEMENTED COVERAGE   which horizons the platform can produce. Open-Meteo
                         serves out to 7 days, so the pipeline runs to J+7.
  EVALUATED PERFORMANCE  which horizons have a measured error against observed
                         production. Only these carry a number.

Displaying a horizon does not validate it. Every horizon below is scored with
the SAME competing models the selector uses (dustcast/learning.py), refitted per
horizon, so the report can quote one coherent set of figures rather than mixing
in results from a superseded model.

    python scripts/step17_horizons.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import learning, metrics, model, physics, pipeline  # noqa: E402
from dustcast.config import ARTIFACTS, DKASC_SITE_03, DKASC_SITE_16C, PROCESSED  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step15_multiarray import BASE_VARS, rule  # noqa: E402

EVALUATED = [0, 1, 2, 3]          # what the concept note asks for
IMPLEMENTED_MAX = 7               # what Open-Meteo serves
FIT_END = "2025-03-30"            # models fitted before this, scored after


def leadN(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Forecast weather at lead `n` days. n=0 is the same-day (intra-day) run."""
    suffix = "" if n == 0 else f"_previous_day{n}"
    cols = {d: df[f"{s}{suffix}"] for s, d in BASE_VARS.items()
            if f"{s}{suffix}" in df}
    out = pd.DataFrame(cols)
    return out.set_axis(out.index - pd.Timedelta("30min"), axis=0).dropna(how="all")


def main() -> int:
    raw = pd.read_parquet(PROCESSED / "dkasc_leadtime_nwp.parquet")
    raw.index = pd.to_datetime(raw.index)
    site = DKASC_SITE_03
    cap = site.ac_capacity_w / 1000.0

    power = pipeline.hourly_frame(site)
    cs = physics.fit_clearsky_scale(site, power)
    meas = physics.build_physics_frame(site, power, clearsky_scale=cs)
    meas = meas[meas["expected_physics_kw"].notna() & meas["ac_power_kw"].notna()]
    ageing = model.fit_degradation(meas, meas.index, site)

    rule("EVALUATED PERFORMANCE BY HORIZON")
    print(f"  site        {site.name}")
    print(f"  models      the same competing set the selector uses, refitted")
    print(f"              independently at every horizon")
    print(f"  fit         .. {FIT_END}     score  {FIT_END} ..")
    print(f"  weather     Open-Meteo archived run at each lead\n")

    names = [c.name for c in learning.candidates()]
    rows, series = [], {}
    for n in EVALUATED:
        w = leadN(raw, n).join(power[["ac_power_kw"]], how="inner")
        w = w.dropna(subset=["ghi", "temp_air", "ac_power_kw"])
        if w.empty:
            print(f"  J+{n}: no archived run at this lead -- skipped")
            continue
        f = physics.build_physics_frame(site, w, clearsky_scale=cs)
        f = f[f["expected_physics_kw"].notna() & f["ac_power_kw"].notna()]
        f = model.apply_degradation(f, site, *ageing)
        f = f[metrics.daylight_mask(f)]

        tz = f.index.tz
        cut = pd.Timestamp(FIT_END, tz=tz)
        fit_f, test_f = f[f.index < cut], f[f.index >= cut]
        if fit_f.empty or test_f.empty:
            continue

        act = test_f["ac_power_kw"]
        rec = {"horizon": f"J+{n}", "n": len(test_f)}
        preds = {}
        for c in learning.candidates():
            c.fit(fit_f, site, None)
            preds[c.name] = c.predict(test_f, site)
            rec[c.name] = float((preds[c.name] - act).abs().mean()) / cap * 100
        rows.append(rec)
        series[f"J+{n}"] = (act, preds)

    tbl = pd.DataFrame(rows).set_index("horizon")
    pd.set_option("display.width", 170, "display.float_format", lambda v: f"{v:,.2f}")
    print("  nMAE %, by horizon (lower is better):\n")
    print(tbl.to_string())

    best = tbl[names].idxmin(axis=1)
    print("\n  best model at each horizon:")
    for h, b in best.items():
        print(f"    {h}   {b}")

    rule("IS THE ML ADVANTAGE HERE REAL, AND IS IT INDEPENDENT?")
    print("  NOT INDEPENDENT. Array 3 is the site on which the feature set and")
    print("  the hyperparameters were chosen. An ML win here is consistent with")
    print("  those choices having been fitted to this array, and the frozen")
    print("  tests on arrays 16C and 16D -- where the same models LOSE to the")
    print("  hour-of-day correction -- are the ones that carry weight.\n")

    rng = np.random.default_rng(17)
    for h, (act, preds) in series.items():
        days = pd.Index(sorted(set(act.index.normalize())))
        bd = {d: act.index.normalize() == d for d in days}
        A = act.to_numpy()
        ml = min([n for n in names if "boosted" in n], key=lambda n: tbl.loc[h, n])
        M, R = preds[ml].to_numpy(), preds["L2 + diurnal bias"].to_numpy()
        d_ = []
        for _ in range(1000):
            mk = np.zeros(len(A), bool)
            for i in rng.choice(len(days), len(days), True):
                mk |= bd[days[i]]
            d_.append(1 - np.mean(np.abs(M[mk]-A[mk])) / np.mean(np.abs(R[mk]-A[mk])))
        lo, hi = np.percentile(d_, 2.5), np.percentile(d_, 97.5)
        verdict = "excludes 0" if lo > 0 or hi < 0 else "INCLUDES 0 -- not resolved"
        print(f"  {h}  {ml:<22s} vs L2: "
              f"{1 - tbl.loc[h, ml] / tbl.loc[h, 'L2 + diurnal bias']:+6.1%}  "
              f"95% CI [{lo:+.1%}, {hi:+.1%}]  {verdict}")
    print("\n  The nMAE ordering across J+1..J+3 is NON-MONOTONIC -- J+2 scores")
    print("  better than J+1 -- which error should not do as lead time grows.")
    print("  No test was run on the horizon-to-horizon differences, so this is")
    print("  unexplained rather than shown to be noise. Do not read a trend")
    print("  into these three numbers in either direction.")

    rule("IMPLEMENTED COVERAGE vs EVALUATED PERFORMANCE")
    sel = tbl[names].min(axis=1)
    print(f"  {'horizon':<10s} {'implemented':<13s} {'evaluated':<11s} {'nMAE %':>8s}")
    for n in range(0, IMPLEMENTED_MAX + 1):
        h = f"J+{n}"
        ev = h in tbl.index
        print(f"  {h:<10s} {'yes':<13s} {('yes' if ev else 'NO'):<11s} "
              f"{(f'{sel[h]:.2f}' if ev else '--'):>8s}")
    print("\n  Horizons marked NO are produced by the pipeline and shown in the")
    print("  interface, but carry no measured error here and must not be")
    print("  presented as validated.")
    print("\n  Intra-day (J+0) is the same-day model run, not a nowcast: this")
    print("  platform has no sub-hourly or sky-imagery input, so it does not")
    print("  claim minute-scale intra-day skill.")

    tbl.to_csv(ARTIFACTS / "step17_horizons.csv")
    print(f"\n  wrote {ARTIFACTS / 'step17_horizons.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
