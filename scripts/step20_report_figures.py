"""Step 20: the two figures for the technical report.

Both are drawn from artefacts written by the evaluation scripts, never from
numbers retyped by hand, so the figures cannot drift from the tables.

    python scripts/step20_report_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import dispatch  # noqa: E402
from dustcast.config import ARTIFACTS  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "report"
OUT.mkdir(exist_ok=True)
PEAK_LOAD_MW = 4600.0


def fig_replay() -> None:
    """Realised monthly error of every candidate, with the selection marked."""
    r = pd.read_csv(ARTIFACTS / "step16_replay.csv", index_col=0, parse_dates=True)
    names = [c for c in r.columns
             if c.startswith(("L0", "L1", "L2", "L3", "L4"))]

    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    styles = {"L0 physics": ("#b0b0b0", "-"), "L1 + derate": ("#c9a227", "-"),
              "L2 + diurnal bias": ("#1f77b4", "-"),
              "L3 + boosted": ("#d62728", "--"),
              "L4 diurnal + boosted": ("#2ca02c", "-")}
    for n in names:
        c, ls = styles.get(n, ("#333", "-"))
        ax.plot(r.index, r[n], ls, color=c, lw=1.7, label=n, alpha=0.95)

    # Mark which model the loop actually selected each month.
    for t, row in r.iterrows():
        ax.plot(t, row[row["selected"]], "o", ms=7, mfc="none",
                mec="black", mew=1.4, zorder=5)
    ax.plot([], [], "o", ms=7, mfc="none", mec="black", mew=1.4,
            label="selected that month")

    ax.set_ylabel("realised MAE (kW)")
    ax.set_title("Monthly replay: every candidate refitted on past data, "
                 "selected on a held-back window", fontsize=10)
    ax.legend(fontsize=7.5, ncol=3, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, lw=0.6)
    ax.margins(x=0.01)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_replay.pdf")
    print(f"  wrote {OUT / 'fig_replay.pdf'}")


def fig_netload() -> None:
    """Where rooftop PV actually bites: the evening ramp, not a midday belly."""
    n = pd.read_csv(ARTIFACTS / "step18_national.csv", index_col=0,
                    parse_dates=True)["national_mw"]
    n.index = pd.DatetimeIndex(n.index)
    demand = dispatch.demand_series(n.index, PEAK_LOAD_MW)
    net = demand - n

    # Ramp statistics are computed over the FULL horizon, exactly as the
    # consumer reports them, and only the two days containing the worst ramp are
    # plotted. Computing them over the plotted window instead would put a
    # different number on the figure from the one in the text.
    ramp3_full = (demand - n).diff().rolling(3).sum()
    worst = ramp3_full.idxmax()
    ramp_mw = float(ramp3_full.max())
    pv_share = float(n.loc[worst - pd.Timedelta("3h")] - n.loc[worst]) / ramp_mw

    lo = (worst - pd.Timedelta("1D")).normalize()
    hi = lo + pd.Timedelta("2D")
    win = (n.index >= lo) & (n.index < hi)
    n, demand, net = n[win], demand[win], net[win]

    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.plot(demand.index, demand, color="#444", lw=1.6, label="demand (assumed)")
    ax.plot(net.index, net, color="#1f77b4", lw=2.0, label="net load = demand - PV")
    ax.fill_between(n.index, net, demand, color="#f5b301", alpha=0.45,
                    label="rooftop PV")

    ax.axvspan(worst - pd.Timedelta("3h"), worst, color="#d62728", alpha=0.13)
    ax.annotate(f"steepest 3h ramp over the horizon\n"
                f"{ramp_mw:,.0f} MW, of which {pv_share:.0%} is PV",
                xy=(worst, net.loc[worst]),
                xytext=(0.46, 0.60), textcoords="axes fraction",
                fontsize=8, ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", lw=0.8),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#d62728"))

    ax.set_ylabel("MW")
    ax.set_title("At this penetration rooftop PV shaves the midday shoulder; "
                 "it does not dig a belly", fontsize=10, pad=12)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=3)
    ax.set_ylim(net.min() - 230, demand.max() + 430)
    ax.grid(alpha=0.25, lw=0.6)
    ax.margins(x=0.01)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "fig_netload.pdf")
    print(f"  wrote {OUT / 'fig_netload.pdf'}")


def fig_forecast() -> None:
    """The product: national forecast with its interval, over J+0..J+3."""
    d = pd.read_csv(ARTIFACTS / "step21_national_band.csv", index_col=0,
                    parse_dates=True)
    d.index = pd.DatetimeIndex(d.index)
    # J+0 must be TODAY. The series carries two days of hindcast (the service
    # requests past_days=2), and starting the window at the series minimum
    # would label those past days J+0..J+1 -- presenting hindcast as forecast.
    now = pd.Timestamp.now(tz=d.index.tz)
    start = now.normalize()
    d = d[(d.index >= start) & (d.index < start + pd.Timedelta("4D"))]

    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    half = d["width_mw"] / 2.0
    ax.fill_between(d.index, (d["point_mw"] - half).clip(lower=0),
                    d["point_mw"] + half, color="#1f77b4", alpha=0.22,
                    label="90% interval (correlation-aware aggregation)")
    ax.plot(d.index, d["point_mw"], color="#1f77b4", lw=1.9,
            label="national point forecast")

    for k in range(4):
        t = start + pd.Timedelta(days=k)
        ax.axvline(t, color="#999", lw=0.7, ls=":")
        ax.text(t + pd.Timedelta("1h"), ax.get_ylim()[1] * 0.02, f"J+{k}",
                fontsize=7.5, color="#666")

    ax.set_ylabel("MW")
    ax.set_title("National rooftop PV forecast, J+0 to J+3, with aggregated "
                 "uncertainty", fontsize=10, pad=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, lw=0.6)
    ax.margins(x=0.01)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "fig_forecast.pdf")
    print(f"  wrote {OUT / 'fig_forecast.pdf'}")


if __name__ == "__main__":
    fig_replay()
    fig_netload()
    fig_forecast()
