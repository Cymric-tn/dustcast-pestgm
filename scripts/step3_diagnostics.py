"""Figure for step 3: the live Tunisian forecast with its calibrated band."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast.config import ARTIFACTS, PROCESSED, SFAX_ROOFTOP  # noqa: E402

INK, GRID = "#1b1b1f", "#d8d8de"
BLUE, SAND = "#2d7dd2", "#c8862f"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8, length=3, color=GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def main() -> int:
    site = SFAX_ROOFTOP
    df = pd.read_parquet(PROCESSED / "step3_live_forecast.parquet")
    now = pd.Timestamp.now(tz=site.tz)
    w = df.loc[now - pd.Timedelta(hours=12): now + pd.Timedelta(hours=60)]

    fig, axes = plt.subplots(
        3, 1, figsize=(13, 10), sharex=True, constrained_layout=True,
        height_ratios=[3, 1.4, 1.4])
    fig.patch.set_facecolor("white")

    # --- forecast with band ---------------------------------------------
    ax = axes[0]
    ax.fill_between(w.index, w["lower_kw"], w["upper_kw"], color=BLUE,
                    alpha=0.20, linewidth=0, label="90% conformal band")
    ax.plot(w.index, w["point_kw"], color=BLUE, linewidth=1.8,
            label="physics + ML point forecast")
    ax.plot(w.index, w["expected_kw"], color=INK, linewidth=1.0,
            linestyle="--", alpha=0.7, label="physics alone")
    ax.axvline(now, color="#b3261e", linewidth=1.2)
    ax.text(now, ax.get_ylim()[1] * 0.97, " now", color="#b3261e", fontsize=8,
            va="top")
    ax.set_ylabel("AC power (kW)", fontsize=9)
    ax.set_title(
        f"Live 48 h rooftop PV forecast — Sfax, Tunisia   "
        f"({site.dc_capacity_w / 1000:.0f} kW archetype)",
        fontsize=11, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, ncols=3)
    style(ax)

    # --- dust ------------------------------------------------------------
    ax = axes[1]
    ax.fill_between(w.index, 0, w["dust"], color=SAND, alpha=0.35, linewidth=0)
    ax.plot(w.index, w["dust"], color=SAND, linewidth=1.3)
    ax.set_ylabel("dust\n(µg/m³)", fontsize=9)
    ax.axvline(now, color="#b3261e", linewidth=1.2)
    ax.set_title("CAMS surface dust — drives soiling, integrates over weeks",
                 fontsize=9, loc="left", color=INK)
    style(ax)

    # --- AOD -------------------------------------------------------------
    ax = axes[2]
    ax.plot(w.index, w["aerosol_optical_depth"], color="#7a4bbf", linewidth=1.4)
    ax.set_ylabel("AOD\n(column)", fontsize=9)
    ax.axvline(now, color="#b3261e", linewidth=1.2)
    ax.set_title("CAMS aerosol optical depth — drives attenuation, acts today",
                 fontsize=9, loc="left", color=INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d\n%H:%M"))
    style(ax)

    fig.savefig(ARTIFACTS / "step3_live_forecast.png", dpi=140,
                facecolor="white")
    print(f"wrote {ARTIFACTS / 'step3_live_forecast.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
