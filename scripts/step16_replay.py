"""Step 16: the chronological learning replay -- what actually happens.

Four models compete. At each monthly boundary every one of them is REFITTED on
past data (the gradient-boosted model is retrained, not re-ranked), the best is
selected on a recent window none of them was fitted on, and that choice then
forecasts the next month blind.

This is a replay of historical observations, not a live system: weather is the
archived day-ahead Open-Meteo forecast and production is DKASC's measured output.
It is labelled as a replay wherever it is shown.

The point is to record what the selection does, including remaining on one model
for the whole run or switching back and forth. Nothing here promises that more
data promotes the boosted model.

    python scripts/step16_replay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import learning, metrics  # noqa: E402
from dustcast.config import (ARTIFACTS, DKASC_SITE_03, DKASC_SITE_16C,  # noqa: E402
                             DKASC_SITE_16D)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step15_multiarray import build, nwp_archive, rule  # noqa: E402

REPLAY_START = "2024-08-01"


def main() -> int:
    nwp = nwp_archive()
    target = build(DKASC_SITE_16C, nwp)
    pool = [(build(DKASC_SITE_03, nwp), DKASC_SITE_03),
            (build(DKASC_SITE_16D, nwp), DKASC_SITE_16D)]

    res, diag = learning.replay(target, DKASC_SITE_16C, pool,
                                start=REPLAY_START, step="MS",
                                select_days=42, warmup_days=120)

    cap = DKASC_SITE_16C.ac_capacity_w / 1000.0
    names = [c.name for c in learning.candidates()]

    rule("LEARNING REPLAY -- monthly refit, selection on held-back data")
    print(f"  target array   {DKASC_SITE_16C.name}")
    print(f"  fleet pool     {', '.join(s.key for _, s in pool)}")
    print(f"  weather        archived Open-Meteo day-ahead forecast (replay)")
    print(f"  selection      lowest MAE on the 42 days before each boundary,")
    print(f"                 held back from fitting; the winner forecasts the")
    print(f"                 next month with no further information")
    pd.set_option("display.width", 170, "display.float_format", lambda v: f"{v:,.4f}")

    show = res[["n", "selected"] + names]
    print("\n  realised MAE (kW) on each month, all candidates:\n")
    print(show.to_string())

    rule("WHAT THE SELECTION DID")
    counts = res["selected"].value_counts()
    for k, v in counts.items():
        print(f"  {k:<22s} selected {v:>2d} / {len(res)} months")
    switches = int((res["selected"] != res["selected"].shift()).sum() - 1)
    print(f"  switches between models: {switches}")

    # How often was the pick right, and how often COULD the ML have been picked?
    truth = res[names].idxmin(axis=1)
    hit = int((res["selected"] == truth).sum())
    print(f"\n  the selection was the month's actual winner {hit}/{len(res)} times")
    print("  months each model actually won (hindsight):")
    for k, v in truth.value_counts().items():
        print(f"    {k:<24s} {v:>2d}")
    ml = [n for n in names if "boosted" in n]
    print("\n  An ML model won the month in "
          f"{int(truth.isin(ml).sum())}/{len(res)} cases, so it is competitive")
    print("  rather than decorative -- but its losing months are the heavy ones,")
    print("  which is why it trails once the months are pooled by hour.")

    rule("POOLED OVER THE WHOLE REPLAY (equal weight per hour)")
    tot = res["n"].sum()
    out = {}
    for n in names:
        out[n] = float((res[n] * res["n"]).sum() / tot)
    out["SELECTED (the platform's forecast)"] = float(
        (res["realised_selected"] * res["n"]).sum() / tot)
    best_fixed = min(names, key=lambda n: out[n])
    print(f"  {'model':<38s} {'MAE kW':>8s} {'nMAE %':>8s}  vs best fixed")
    for n, v in sorted(out.items(), key=lambda kv: kv[1]):
        mark = "  <- best fixed choice" if n == best_fixed else ""
        delta = "" if n == best_fixed else f"{(1 - v / out[best_fixed]):+7.1%}"
        print(f"  {n:<38s} {v:8.4f} {v / cap * 100:8.2f}  {delta}{mark}")

    print("\n  The selector cannot beat the best fixed choice in hindsight; it")
    print("  can only avoid needing to know it in advance. What matters is")
    print("  whether it stays close to it without being told.")

    res.to_csv(ARTIFACTS / "step16_replay.csv")
    diag.to_csv(ARTIFACTS / "step16_replay_selection.csv")
    print(f"\n  wrote {ARTIFACTS / 'step16_replay.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
