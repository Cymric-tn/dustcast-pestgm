"""Step 1: DKASC ingest -> pvlib baseline -> XGBoost residual, one site, no UI.

Gate (CLAUDE.md section 8): the residual model must beat clear-sky physics
alone. If it does not, the ML layer is not earning its place and the physics
should ship on its own.

    python scripts/step1_train.py                # full run
    python scripts/step1_train.py --nrows 200000 # smoke test on a prefix
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import metrics, model, pipeline  # noqa: E402
from dustcast.config import ARTIFACTS, DKASC_SITE_03, PROCESSED  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrows", type=int, default=None,
                    help="Read only the first N raw rows (smoke test).")
    ap.add_argument("--split", choices=("config", "auto"), default="config")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    site = DKASC_SITE_03

    rule("1. PREPARE")
    prep = pipeline.prepare(site, nrows=args.nrows, split_mode=args.split,
                            cache=not args.no_cache)
    day = prep.daylight()
    print(f"site      {site.name}")
    print(f"modelled  {len(prep.frame):,} hours, {day.sum():,} daylight  "
          f"{prep.frame.index.min():%Y-%m-%d} .. {prep.frame.index.max():%Y-%m-%d}")

    scale = prep.clearsky_scale
    print(f"clear-sky scale (fitted on train): "
          f"{scale.min():.3f}-{scale.max():.3f} by month")
    kt = prep.frame.loc[day, "clearsky_index"]
    print(f"clear-sky index: median {kt.median():.3f}, "
          f"{(kt > 1).mean():.1%} above 1.0")

    a, b = prep.ageing
    print(f"ageing    actual ~ expected x ({a:.4f} {b:+.5f} x age_years)  "
          f"-> {-b / a * 100:.2f} %/yr")

    resid = prep.frame.loc[day, "residual_kw"]
    print(f"residual  mean {resid.mean():+.3f} kW, sd {resid.std():.3f} kW")

    rule("2. SPLIT (chronological)")
    print(prep.split.describe())

    rule("3. TRAIN RESIDUAL MODEL")
    rm = pipeline.fit_residual_model(prep)
    train_idx = prep.split.train.intersection(prep.frame.index[day])
    print(f"XGBoost fitted on {len(train_idx):,} daylight hours, "
          f"early stopping chose {rm.best_iteration + 1} trees")
    print("\ntop features by gain:")
    for name, imp in rm.importances.head(10).items():
        print(f"  {name:<24s} {imp:.4f}")

    rule("4. EVALUATE ON HELD-OUT TEST (daylight hours only)")
    test = prep.frame.loc[prep.split.test]
    Xt = prep.X.loc[prep.split.test]
    day_t = metrics.daylight_mask(test)

    factor = model.fit_physics_calibration(prep.frame, prep.split.train)
    print(f"calibrated physics derate fitted on train: x{factor:.4f} "
          f"({(1 - factor) * 100:.1f}% loss)   "
          f"[PVWatts standard: x{model.pvwatts_loss_factor():.4f}]")

    preds = {
        "persistence": model.baseline_smart_persistence(
            prep.frame, site).loc[prep.split.test],
        "physics_raw": model.baseline_physics(test, site),
        "physics_pvwatts": model.baseline_physics_derated(test, site),
        "physics_calibrated": model.baseline_physics_calibrated(test, site, factor),
        "physics_trend": model.baseline_physics_trend(test, site),
        "physics+ML": rm.predict_power(Xt, test),
    }

    actual = test["ac_power_kw"]
    # Every model scored on identical hours: a baseline that is undefined on
    # the hardest hours would otherwise look better than it is.
    scored = day_t & actual.notna()
    for p in preds.values():
        scored &= p.notna()
    print(f"scoring all models on the same {scored.sum():,} daylight hours "
          f"({int(day_t.sum() - scored.sum()):,} dropped where any is undefined)")

    results = {
        name: metrics.deterministic(actual[scored], p[scored],
                                    capacity_kw=prep.capacity_kw)
        for name, p in preds.items()
    }

    # Gate against whichever non-ML baseline actually wins, chosen at run time.
    ref = min((k for k in results if k != "physics+ML"),
              key=lambda k: results[k]["mae"])
    table = metrics.compare(results, baseline=ref)
    pd.set_option("display.width", 140, "display.float_format", lambda v: f"{v:,.4f}")
    print()
    print(table[["n", "mae", "rmse", "mbe", "nmae_pct", "nrmse_pct",
                 "skill_mae", "skill_rmse"]].to_string())

    rule("GATE: does the residual model beat clear-sky physics alone?")
    skill_mae = table.loc["physics+ML", "skill_mae"]
    skill_rmse = table.loc["physics+ML", "skill_rmse"]
    passed = skill_mae > 0 and skill_rmse > 0
    print(f"  reference: {ref} (best non-ML baseline)")
    print(f"  MAE  {results[ref]['mae']:.4f} -> "
          f"{results['physics+ML']['mae']:.4f} kW   skill {skill_mae:+.1%}")
    print(f"  RMSE {results[ref]['rmse']:.4f} -> "
          f"{results['physics+ML']['rmse']:.4f} kW   skill {skill_rmse:+.1%}")
    print(f"\n  {'PASS' if passed else 'FAIL'} -- "
          f"{'ML earns its place' if passed else 'physics should ship alone'}")

    context = ["solar_elevation", "expected_kw", "temp_cell", "temp_air",
               "clearsky_index", "ghi", "residual_kw"]
    out = pd.DataFrame({"actual_kw": actual, **preds}).join(test[context])
    out.to_parquet(PROCESSED / "step1_test_predictions.parquet")
    rm.save(ARTIFACTS / "step1_residual_model.joblib")
    table.to_csv(ARTIFACTS / "step1_metrics.csv")
    rm.importances.to_csv(ARTIFACTS / "step1_feature_importance.csv")
    print(f"\nwrote {ARTIFACTS / 'step1_residual_model.joblib'}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
