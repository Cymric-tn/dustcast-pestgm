"""Figure for step 5: the national picture."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import live, tunisia  # noqa: E402
from dustcast.config import ARTIFACTS, PROCESSED  # noqa: E402

INK, GRID = "#1b1b1f", "#d8d8de"
BLUE, SAND = "#2d7dd2", "#c8862f"
TZ = "Africa/Tunis"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8, length=3, color=GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def main() -> int:
    nat = pd.read_parquet(PROCESSED / "step5_national.parquet")
    govs = pd.read_csv(ARTIFACTS / "step5_governorates.csv", index_col=0)
    now = pd.Timestamp.now(tz=TZ)
    w = nat.loc[now - pd.Timedelta(hours=6): now + pd.Timedelta(hours=54)]

    fig = plt.figure(figsize=(15, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.5, 1.2])
    fig.patch.set_facecolor("white")

    # --- national forecast ------------------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    ax.fill_between(w.index, w["lower_mw"], w["upper_mw"], color=BLUE,
                    alpha=0.20, linewidth=0, label="90% band (correlation-aware)")
    ax.plot(w.index, w["point_mw"], color=BLUE, linewidth=2.0,
            label="national rooftop PV forecast")
    ax.plot(w.index, w["dust_off_mw"], color=SAND, linewidth=1.3,
            linestyle="--", label="dust model OFF")
    ax.axvline(now, color="#b3261e", linewidth=1.2)
    ax.set_ylabel("MW", fontsize=9)
    ax.set_title(f"Tunisia — national rooftop PV, next 48 h   "
                 f"(peak {w['point_mw'].max():.0f} MW of "
                 f"{govs['capacity_mw'].sum():.0f} MW installed)",
                 fontsize=11, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, ncols=3)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
    style(ax)

    # --- proto-map --------------------------------------------------------
    ax = fig.add_subplot(gs[:, 2])
    lats = govs["latitude"].to_numpy()
    lons = govs["longitude"].to_numpy()
    aod = pd.Series({g.key: live.fetch_dust_multi(
        tunisia.latitudes(), tunisia.longitudes(), TZ, past_days=60,
        forecast_days=5, tag="tn")[i]["aerosol_optical_depth"].loc[
        now:now + pd.Timedelta(hours=48)].mean()
        for i, g in enumerate(tunisia.GOVERNORATES)}).reindex(govs.index)
    sizes = 40 + 900 * govs["peak_mw"] / govs["peak_mw"].max()
    sc = ax.scatter(lons, lats, s=sizes, c=aod, cmap="YlOrBr",
                    vmin=float(aod.min()), vmax=float(aod.max()),
                    edgecolor=INK, linewidth=0.5, alpha=0.92, zorder=3)
    # Four governorates share greater Tunis and their labels collide. Fan the
    # crowded ones out rather than letting them overprint into mush.
    offsets = {"tunis": (26, 8), "ariana": (-26, 10), "ben_arous": (30, -6),
               "manouba": (-34, -4), "zaghouan": (0, -12)}
    for key, row in govs.iterrows():
        dx, dy = offsets.get(key, (0, -12))
        ax.annotate(row["name"], (row["longitude"], row["latitude"]),
                    fontsize=6.5, color=INK, xytext=(dx, dy),
                    textcoords="offset points",
                    ha="center" if dx == 0 else ("left" if dx > 0 else "right"))
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("mean AOD, next 48 h", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("Bubble = forecast peak MW\nColour = Saharan dust load",
                 fontsize=10, loc="left", color=INK)
    ax.set_xlabel("longitude", fontsize=9)
    ax.set_ylabel("latitude", fontsize=9)
    ax.set_aspect(1.2)
    style(ax)

    # --- ramp -------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ramp = w["point_mw"].diff().rolling("3h").sum()
    ax.fill_between(ramp.index, 0, ramp, where=ramp >= 0, color=BLUE,
                    alpha=0.5, linewidth=0, label="up-ramp")
    ax.fill_between(ramp.index, 0, ramp, where=ramp < 0, color="#b3261e",
                    alpha=0.45, linewidth=0, label="down-ramp")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_ylabel("MW per 3 h", fontsize=9)
    ax.set_title(f"Ramp risk — worst 3 h swing "
                 f"{ramp.min():.0f} / +{ramp.max():.0f} MW",
                 fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
    style(ax)
    ax.tick_params(axis="x", rotation=20, labelsize=7)

    # --- band aggregation -------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    lit = w["point_mw"] > 0.05 * w["point_mw"].max()
    ax.plot(w.index[lit], w.loc[lit, "naive_band_mw"], color=SAND,
            linewidth=1.5, label="naive: sum of governorate bands")
    ax.plot(w.index[lit], w.loc[lit, "band_mw"], color=BLUE, linewidth=1.8,
            label="correlation-aware")
    saved = 1 - w.loc[lit, "band_mw"].mean() / w.loc[lit, "naive_band_mw"].mean()
    ax.set_ylabel("national band width (MW)", fontsize=9)
    ax.set_title(f"Summing regional bands assumes perfect correlation\n"
                 f"and overstates the national band by {saved:.0%}",
                 fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
    style(ax)
    ax.tick_params(axis="x", rotation=20, labelsize=7)

    fig.suptitle("DustCast step 5 — virtual fleet and national upscaling  "
                 "(SIMULATED fleet: no Tunisian PV registry exists)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    out = ARTIFACTS / "step5_national.png"
    fig.savefig(out, dpi=140, facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
