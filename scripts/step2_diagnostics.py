"""Diagnostic figure for step 2: are the intervals calibrated, and are they sharp?"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast.config import ARTIFACTS, CONFIDENCE_LEVEL, DKASC_SITE_03, PROCESSED  # noqa: E402
from dustcast.metrics import picp  # noqa: E402

INK, GRID = "#1b1b1f", "#d8d8de"
COLOURS = {"absolute": "#9aa0a6", "normalised": "#e07b39",
           "cqr": "#5c9e5c", "rolling_cqr": "#2d7dd2"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8, length=3, color=GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def main() -> int:
    site = DKASC_SITE_03
    cap = site.ac_capacity_w / 1000.0
    df = pd.read_parquet(PROCESSED / "step2_intervals.parquet")
    methods = [m for m in COLOURS if f"{m}_lower" in df]
    lit = df["solar_elevation"] > 5.0
    d = df[lit]

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    fig.patch.set_facecolor("white")

    # --- 1: fan chart, a representative week ----------------------------
    ax = axes[0, 0]
    daily = d["clearsky_index"].resample("1D").mean()
    stamp = daily.rolling(7).mean().idxmin()  # the most disturbed week
    wk = df.loc[stamp - pd.Timedelta(days=7): stamp]
    ax.fill_between(wk.index, wk["rolling_cqr_lower"], wk["rolling_cqr_upper"],
                    color="#2d7dd2", alpha=0.22, linewidth=0,
                    label="rolling CQR 90% band")
    ax.plot(wk.index, wk["rolling_cqr_point"], color="#2d7dd2", linewidth=1.2,
            label="point forecast")
    ax.plot(wk.index, wk["actual_kw"], color="#111318", linewidth=1.3,
            label="measured")
    ax.set_title("Most disturbed week — calibrated band", fontsize=10, loc="left")
    ax.set_ylabel("AC power (kW)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    style(ax); ax.tick_params(axis="x", rotation=30, labelsize=7)

    # --- 2: coverage by clear-sky index ---------------------------------
    ax = axes[0, 1]
    bins = pd.cut(d["clearsky_index"], [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5])
    for m in methods:
        cov = d.groupby(bins, observed=True).apply(
            lambda g, m=m: picp(g["actual_kw"], g[f"{m}_lower"], g[f"{m}_upper"]),
            include_groups=False)
        ax.plot([i.mid for i in cov.index], cov.values, "o-",
                color=COLOURS[m], linewidth=1.4, markersize=4, label=m)
    ax.axhline(CONFIDENCE_LEVEL, color=INK, linestyle="--", linewidth=1)
    ax.text(0.02, CONFIDENCE_LEVEL + 0.01, "nominal 90%", fontsize=8, color=INK)
    ax.set_title("Conditional coverage by clear-sky index", fontsize=10, loc="left")
    ax.set_xlabel("clear-sky index", fontsize=9); ax.set_ylabel("PICP", fontsize=9)
    ax.legend(frameon=False, fontsize=8); style(ax)

    # --- 3: sharpness by clear-sky index --------------------------------
    ax = axes[1, 0]
    for m in methods:
        w = ((d[f"{m}_upper"] - d[f"{m}_lower"]) / cap).groupby(
            bins, observed=True).mean()
        ax.plot([i.mid for i in w.index], w.values, "o-", color=COLOURS[m],
                linewidth=1.4, markersize=4, label=m)
    ax.set_title("Sharpness by clear-sky index  (band width / capacity)",
                 fontsize=10, loc="left")
    ax.set_xlabel("clear-sky index", fontsize=9); ax.set_ylabel("PINAW", fontsize=9)
    ax.legend(frameon=False, fontsize=8); style(ax)

    # --- 4: THE DRIFT -- rolling coverage through the test period -------
    ax = axes[1, 1]
    for m in methods:
        inside = ((d["actual_kw"] >= d[f"{m}_lower"])
                  & (d["actual_kw"] <= d[f"{m}_upper"])).astype(float)
        ax.plot(inside.index, inside.rolling("30D").mean(), color=COLOURS[m],
                linewidth=1.3, label=m)
    ax.axhline(CONFIDENCE_LEVEL, color=INK, linestyle="--", linewidth=1)
    ax.set_title("Rolling 30-day coverage — why static calibration fails",
                 fontsize=10, loc="left")
    ax.set_ylabel("PICP (30-day)", fontsize=9)
    ax.legend(frameon=False, fontsize=8); style(ax)
    ax.tick_params(axis="x", rotation=30, labelsize=7)

    # --- 5: width vs absolute error, the sharpness/coverage trade -------
    ax = axes[2, 0]
    for m in methods:
        w = (d[f"{m}_upper"] - d[f"{m}_lower"])
        err = (d["actual_kw"] - d[f"{m}_point"]).abs()
        q = pd.qcut(w, 8, duplicates="drop")
        ax.plot(w.groupby(q, observed=True).mean(),
                err.groupby(q, observed=True).mean(), "o-",
                color=COLOURS[m], linewidth=1.4, markersize=4, label=m)
    # Bound this to the range the data actually occupies. Drawing the
    # reference line out to the maximum band width stretched the axis so far
    # that every real point collapsed into the bottom-left corner.
    top = max((d[f"{m}_upper"] - d[f"{m}_lower"]).quantile(0.99) for m in methods)
    lim = np.linspace(0, float(top), 10)
    ax.plot(lim, lim / 2, color=INK, linestyle=":", linewidth=1,
            label="width = 2x error")
    ax.set_xlim(0, float(top))
    ax.set_title("Does the band widen where the model is actually wrong?",
                 fontsize=10, loc="left")
    ax.set_xlabel("mean band width (kW)", fontsize=9)
    ax.set_ylabel("mean |error| (kW)", fontsize=9)
    ax.legend(frameon=False, fontsize=8); style(ax)

    # --- 6: coverage vs sharpness summary -------------------------------
    ax = axes[2, 1]
    for m in methods:
        p = picp(d["actual_kw"], d[f"{m}_lower"], d[f"{m}_upper"])
        w = ((d[f"{m}_upper"] - d[f"{m}_lower"]) / cap).mean()
        ax.scatter(w, p, s=90, color=COLOURS[m], zorder=3, label=m)
        ax.annotate(m, (w, p), textcoords="offset points", xytext=(8, -3),
                    fontsize=8, color=INK)
    ax.axhline(CONFIDENCE_LEVEL, color=INK, linestyle="--", linewidth=1)
    ax.axhspan(CONFIDENCE_LEVEL - 0.03, CONFIDENCE_LEVEL + 0.03,
               color="#2d7dd2", alpha=0.10, zorder=0)
    ax.text(0.001, CONFIDENCE_LEVEL + 0.035, "gate: 90% +/- 3%",
            fontsize=8, color=INK)
    ax.set_title("The only trade that matters: coverage vs sharpness",
                 fontsize=10, loc="left")
    ax.set_xlabel("PINAW (narrower is better)", fontsize=9)
    ax.set_ylabel("PICP (must hit 90%)", fontsize=9)
    style(ax)

    fig.suptitle(f"DustCast step 2 — conformal intervals\n{site.name}",
                 fontsize=12, color=INK, x=0.01, ha="left")
    out = ARTIFACTS / "step2_diagnostics.png"
    fig.savefig(out, dpi=140, facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
