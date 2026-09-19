"""Figure for steps 8 and 9: temporal scales, and upscaling validated for real."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast.config import ARTIFACTS, PROCESSED  # noqa: E402

INK, GRID = "#1b1b1f", "#d8d8de"
TEAL, OCHRE, RED = "#0a8f9c", "#c07a18", "#a3271d"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8, length=3, color=GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def main() -> int:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.patch.set_facecolor("white")

    # --- 1: lead-time plateau --------------------------------------------
    ax = axes[0, 0]
    lead = pd.read_csv(ARTIFACTS / "step8_leadtime.csv")
    ax.plot(lead["lead_day"], lead["rMAE_pct"], "o-", color=TEAL,
            linewidth=2, markersize=7)
    ax.annotate("nowcast regime", (0, lead["rMAE_pct"].iloc[0]),
                xytext=(12, -22), textcoords="offset points", fontsize=9,
                color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=.9))
    ax.annotate("everything beyond today\nis one regime",
                (4, lead["rMAE_pct"].iloc[-2]), xytext=(-8, -46),
                textcoords="offset points", fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=.9))
    ax.set_ylim(0, lead["rMAE_pct"].max() * 1.35)
    ax.set_xlabel("forecast lead time (days)", fontsize=9)
    ax.set_ylabel("NWP GHI error, rMAE (%)", fontsize=9)
    ax.set_title("Error jumps once, then flattens\n"
                 "(DKASC, archived forecasts vs 17 years of measured output)",
                 fontsize=10, loc="left", color=INK)
    style(ax)

    # --- 2: per-scale calibration ----------------------------------------
    ax = axes[0, 1]
    cov = pd.read_csv(ARTIFACTS / "step8_scale_coverage.csv")
    scales = cov["scale"].unique()
    x = np.arange(len(scales))
    for i, (mode, colour) in enumerate((("shared", OCHRE), ("own", TEAL))):
        vals = [cov[(cov.scale == s) & (cov.correction == mode)]["PICP"].iloc[0]
                for s in scales]
        ax.bar(x + (i - 0.5) * 0.34, vals, width=0.32, color=colour,
               label=f"{mode} correction", zorder=3)
    ax.axhline(0.90, color=INK, linestyle="--", linewidth=1.2, zorder=4)
    ax.text(len(scales) - 0.45, 0.905, "nominal 90%", fontsize=8, color=INK,
            ha="right")
    ax.axhspan(0.87, 0.93, color=TEAL, alpha=0.09, zorder=0)
    ax.set_xticks(x, scales, fontsize=9)
    ax.set_ylim(0.80, 1.06)
    ax.set_ylabel("coverage (PICP)", fontsize=9)
    ax.set_title("One correction does not serve all three\n"
                 "the nowcast is a different regime and over-covers by 9 points",
                 fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncols=2)
    style(ax)

    # --- 3: upscaling vs the measured fleet -------------------------------
    ax = axes[1, 0]
    up = pd.read_parquet(PROCESSED / "step9_upscaling.parquet")
    week = up.loc[up.index[-24 * 7:]].copy()
    # The saved frame holds daylight hours only, so consecutive days are not
    # adjacent in time. Reindexing onto a continuous grid leaves NaNs at night
    # and matplotlib breaks the line there, instead of drawing a false ramp
    # straight through the small hours.
    week = week.reindex(pd.date_range(week.index[0], week.index[-1],
                                      freq="1h", tz=week.index.tz))
    ax.plot(week.index, week["fleet_kw"], color=INK, linewidth=1.6,
            label="measured fleet total")
    ax.plot(week.index, week["upscaled_kw"], color=TEAL, linewidth=1.5,
            linestyle="--", label="upscaled from ONE 4.95 kW array")
    ax.set_ylabel("site PV output (kW)", fontsize=9)
    ax.set_title("A sample that is 1.6% of the fleet predicts the whole:\n"
                 "5.9% rMAE, r = 0.985 on held-out data",
                 fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    ax.tick_params(axis="x", rotation=25, labelsize=7)

    # --- 4: the stale register -------------------------------------------
    ax = axes[1, 1]
    lit = up["fleet_kw"] > 0.05 * up["fleet_kw"].max()
    d = up[lit]
    ax.scatter(d["fleet_kw"], d["upscaled_stale_kw"], s=5, color=RED,
               alpha=0.30, linewidth=0, label="stale capacity register (92% rMAE)")
    ax.scatter(d["fleet_kw"], d["upscaled_kw"], s=5, color=TEAL,
               alpha=0.45, linewidth=0, label="current register (6.8% rMAE)")
    top = float(d["fleet_kw"].max())
    ax.plot([0, top], [0, top], color=INK, linestyle=":", linewidth=1.2,
            label="perfect")
    ax.set_xlabel("measured fleet output (kW)", fontsize=9)
    ax.set_ylabel("upscaled estimate (kW)", fontsize=9)
    ax.set_title("Same forecast, one number changed\n"
                 "upscaling is bounded by the capacity register, not the model",
                 fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left", markerscale=2.5)
    style(ax)

    fig.suptitle("DustCast steps 8-9 — temporal scales, and upscaling checked "
                 "against a real metered fleet",
                 fontsize=12, color=INK, x=0.01, ha="left")
    out = ARTIFACTS / "step89_validation.png"
    fig.savefig(out, dpi=140, facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
