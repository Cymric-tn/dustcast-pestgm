"""Step 2: conformal calibration of the residual model.

Gate (CLAUDE.md section 8): PICP within +/-3% of nominal.

Coverage alone is not a result -- an interval from zero to infinity has perfect
coverage and no value -- so sharpness (PINAW) is reported beside it throughout,
and conditional coverage is reported because the marginal number can hide a
band that is generous on easy hours and short on the ones that cost money.

    python scripts/step2_conformal.py
    python scripts/step2_conformal.py --nrows 200000 --split auto   # smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import conformal, metrics, pipeline  # noqa: E402
from dustcast.config import ARTIFACTS, CONFIDENCE_LEVEL, DKASC_SITE_03, PROCESSED  # noqa: E402

GATE_TOLERANCE = 0.03


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--split", choices=("config", "auto"), default="config")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    site = DKASC_SITE_03
    level = CONFIDENCE_LEVEL

    rule("1. PREPARE (shared pipeline with step 1)")
    prep = pipeline.prepare(site, nrows=args.nrows, split_mode=args.split,
                            cache=not args.no_cache)
    print(prep.split.describe())
    print(f"ageing fitted on train: x({prep.ageing[0]:.4f} "
          f"{prep.ageing[1]:+.5f} x age_years)")

    rule("2. BASE MODEL")
    rm = pipeline.fit_residual_model(prep)
    print(f"XGBoost: {rm.best_iteration + 1} trees, "
          f"{len(prep.feature_cols)} features")

    # The calibration split has never been touched by fitting -- not by the
    # booster, not by early stopping (which uses the tail of train), not by the
    # ageing fit. That is the precondition for the conformal guarantee.
    rule("3. CONFORMALIZE")
    lit = prep.frame.index[prep.daylight()]
    cal = prep.split.calibrate.intersection(lit)
    test = prep.split.test.intersection(lit)
    train = prep.split.train.intersection(lit)
    print(f"calibrate on {len(cal):,} daylight hours, "
          f"evaluate on {len(test):,}")

    methods = {}
    methods["absolute"] = conformal.split_conformal(
        rm, prep.frame, prep.X, prep.y, cal, test, site, level, adaptive=False)
    methods["normalised"] = conformal.split_conformal(
        rm, prep.frame, prep.X, prep.y, cal, test, site, level, adaptive=True)
    methods["cqr"] = conformal.conformalized_quantile(
        rm, prep.frame, prep.X, prep.y, train, cal, test, site, level)
    methods["rolling_cqr"] = conformal.rolling_conformal(
        rm, prep.frame, prep.X, prep.y, train, cal, test, site, level)

    rule(f"4. COVERAGE AND SHARPNESS  (nominal {level:.0%}, daylight only)")
    actual = prep.frame.loc[test, "ac_power_kw"]
    report = {
        name: metrics.interval_report(actual, iv.lower, iv.upper, iv.point,
                                      prep.capacity_kw, level)
        for name, iv in methods.items()
    }
    table = pd.DataFrame(report).T
    pd.set_option("display.width", 140, "display.float_format",
                  lambda v: f"{v:,.4f}")
    print(table[["n", "PICP", "cov_error", "PINAW", "mean_width_kW",
                 "pinball_0.05", "pinball_0.50", "pinball_0.95",
                 "CRPS_approx"]].to_string())

    rule("5. CONDITIONAL COVERAGE")
    kt = prep.frame.loc[test, "clearsky_index"]
    for name, iv in methods.items():
        cc = metrics.conditional_coverage(
            actual, iv.lower, iv.upper, kt,
            [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5], prep.capacity_kw,
            label="clearsky_index")
        worst = (cc["PICP"] - level).abs().max()
        print(f"\n{name}   worst deviation from nominal: {worst:+.1%}")
        print(cc.to_string())

    rule(f"GATE: PICP within +/-{GATE_TOLERANCE:.0%} of nominal")
    passed = {}
    for name, r in report.items():
        ok = abs(r["cov_error"]) <= GATE_TOLERANCE
        passed[name] = ok
        print(f"  {name:<12s} PICP {r['PICP']:.1%}  "
              f"({r['cov_error']:+.1%})  PINAW {r['PINAW']:.4f}   "
              f"{'PASS' if ok else 'FAIL'}")

    calibrated = {n: report[n] for n, ok in passed.items() if ok}
    if calibrated:
        best = min(calibrated, key=lambda n: calibrated[n]["PINAW"])
        print(f"\n  sharpest CALIBRATED method: {best}  "
              f"PICP {report[best]['PICP']:.1%}  PINAW {report[best]['PINAW']:.4f}")
        # Sharpness is only meaningful among methods that actually cover.
        # A narrower band that under-covers is not sharper, it is wrong -- and
        # reporting PINAW without PICP is exactly how that gets sold as a win.
        narrower = {n: r for n, r in report.items()
                    if not passed[n] and r["PINAW"] < report[best]["PINAW"]}
        for n, r in narrower.items():
            print(f"    note: '{n}' is "
                  f"{1 - r['PINAW'] / report[best]['PINAW']:.0%} narrower but "
                  f"covers only {r['PICP']:.1%} -- the narrowness is bought by "
                  f"missing {(1 - r['PICP']) * len(actual):.0f} hours, not by "
                  f"being better informed")
    print(f"\n  {'PASS' if any(passed.values()) else 'FAIL'} -- "
          f"{sum(passed.values())}/{len(passed)} methods calibrated")

    out = pd.DataFrame({"actual_kw": actual})
    for name, iv in methods.items():
        out[f"{name}_point"] = iv.point
        out[f"{name}_lower"] = iv.lower
        out[f"{name}_upper"] = iv.upper
    out = out.join(prep.frame.loc[test, ["clearsky_index", "solar_elevation",
                                         "expected_kw", "temp_cell"]])
    out.to_parquet(PROCESSED / "step2_intervals.parquet")
    table.to_csv(ARTIFACTS / "step2_metrics.csv")
    print(f"\nwrote {PROCESSED / 'step2_intervals.parquet'}")

    return 0 if any(passed.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
