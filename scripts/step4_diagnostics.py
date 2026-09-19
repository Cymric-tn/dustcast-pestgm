"""Figure for step 4: does the dust layer earn its place?"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import live  # noqa: E402
from dustcast.config import (  # noqa: E402
    ARTIFACTS, DKASC_SITE_03, PROCESSED, SFAX_ROOFTOP,
)

INK, GRID = "#1b1b1f", "#d8d8de"
GREY, BLUE, SAND = "#9aa0a6", "#2d7dd2", "#c8862f"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8, length=3, color=GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def main() -> int:
    d = pd.read_parquet(PROCESSED / "step4_dust_ablation.parquet")
    lit = d["expected_kw"] > 0.2 * d["expected_kw"].max()
    clear = d["clearsky_index"] >= d["clearsky_index"].median()
    sub = d[lit & clear]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.patch.set_facecolor("white")

    # --- 1: the gate -- band width vs AOD --------------------------------
    ax = axes[0, 0]
    bins = pd.qcut(sub["aod"], 6, duplicates="drop")
    for name, colour in (("no_dust", GREY), ("with_dust", BLUE)):
        rel = (sub[f"{name}_upper"] - sub[f"{name}_lower"]) / sub["expected_kw"]
        g = rel.groupby(bins, observed=True).mean()
        ax.plot([i.mid for i in g.index], g.values, "o-", color=colour,
                linewidth=1.6, markersize=5, label=name)
    ax.set_title("THE GATE — band width vs aerosol optical depth\n"
                 "(clear-sky hours only, normalised by expected output)",
                 fontsize=10, loc="left", color=INK)
    ax.set_xlabel("AOD (column)", fontsize=9)
    ax.set_ylabel("band width / expected kW", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    style(ax)

    # --- 2: the attribution ----------------------------------------------
    ax = axes[0, 1]
    for name, colour in (("no_dust", GREY), ("with_dust", BLUE)):
        err = (sub["actual_kw"] - sub[f"{name}_point"]).abs()
        g = err.groupby(bins, observed=True).mean()
        ax.plot([i.mid for i in g.index], g.values, "o-", color=colour,
                linewidth=1.6, markersize=5, label=name)
    ax.set_title("Point error vs AOD — the dust features barely separate",
                 fontsize=10, loc="left", color=INK)
    ax.set_xlabel("AOD (column)", fontsize=9)
    ax.set_ylabel("MAE (kW)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    style(ax)

    # --- 3: soiling index over the test period ---------------------------
    ax = axes[1, 0]
    ax.plot(d.index, d["soiling_index"] * 100, color=SAND, linewidth=1.3,
            label="modelled soiling index")
    ax.set_ylabel("soiling index (%)", fontsize=9, color=SAND)
    rain = d["precipitation"].resample("1D").sum()
    ax2 = ax.twinx()
    ax2.bar(rain.index, rain.values, width=1.0, color=BLUE, alpha=0.45,
            label="daily rain")
    ax2.set_ylabel("rain (mm/day)", fontsize=9, color=BLUE)
    ax2.spines["top"].set_visible(False)
    ax.set_title("Soiling accumulates, rain resets it  "
                 "(relative index — no ground truth)",
                 fontsize=10, loc="left", color=INK)
    style(ax)
    ax.tick_params(axis="x", rotation=30, labelsize=7)

    # --- 4: THE REGIME GAP ------------------------------------------------
    ax = axes[1, 1]
    au = live.fetch_dust_history(DKASC_SITE_03, "2022-08-04",
                                 "2025-08-23")["aerosol_optical_depth"]
    tn = live.fetch_dust(SFAX_ROOFTOP, past_days=92,
                         forecast_days=5)["aerosol_optical_depth"]
    edges = np.linspace(0, 0.8, 41)
    ax.hist(au.dropna(), bins=edges, density=True, color=GREY, alpha=0.65,
            label=f"Alice Springs (train)  mean {au.mean():.3f}")
    ax.hist(tn.dropna(), bins=edges, density=True, color=SAND, alpha=0.65,
            label=f"Sfax (deploy)  mean {tn.mean():.3f}")
    ax.axvline(au.quantile(0.95), color=GREY, linestyle="--", linewidth=1.2)
    ax.text(au.quantile(0.95), ax.get_ylim()[1] * 0.9, " Alice p95",
            fontsize=8, color=INK)
    ax.axvline(tn.median(), color=SAND, linestyle="--", linewidth=1.2)
    ax.text(tn.median(), ax.get_ylim()[1] * 0.75, " Sfax median",
            fontsize=8, color=INK)
    ax.set_title("Why DKASC cannot prove the dust case:\n"
                 "Sfax's MEDIAN aerosol exceeds Alice Springs' 95th percentile",
                 fontsize=10, loc="left", color=INK)
    ax.set_xlabel("AOD (column)", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    style(ax)

    fig.suptitle("DustCast step 4 — dust ablation, soiling, and the regime gap",
                 fontsize=12, color=INK, x=0.01, ha="left")
    out = ARTIFACTS / "step4_dust.png"
    fig.savefig(out, dpi=140, facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
