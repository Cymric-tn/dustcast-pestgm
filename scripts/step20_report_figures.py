"""Step 20: the two figures for the technical report.

Both are drawn from artefacts written by the evaluation scripts, never from
numbers retyped by hand, so the figures cannot drift from the tables.

    python scripts/step20_report_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import json

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


#: Numbers the report prose quotes from the scenario figure. Written as LaTeX
#: macros so the text cannot drift from the chart: both come from this run.
MACROS: dict[str, str] = {}


def fig_netload() -> None:
    """Where rooftop PV actually bites: the evening ramp, not a midday belly."""
    # Same snapshot as the forecast chart, so the ramp numbers quoted in the
    # prose describe the forecast the report actually shows.
    snap = json.loads((ARTIFACTS / "step22_snapshot.json").read_text())
    d = pd.DataFrame(snap["series"])
    d["time"] = pd.DatetimeIndex(d["time"])
    n = d.set_index("time")["mw"]
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

    midday = net.index.hour.isin(range(11, 15))
    MACROS["RampMW"] = f"{ramp_mw:,.0f}"
    MACROS["RampPVMW"] = f"{n.loc[worst - pd.Timedelta('3h')] - n.loc[worst]:,.0f}"
    MACROS["RampPVPct"] = f"{pv_share * 100:.0f}"
    MACROS["MiddayShavePct"] = f"{n[midday].mean() / demand[midday].mean() * 100:.1f}"

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
    """The product, drawn from the frozen snapshot -- never from a fresh run.

    The snapshot carries its own issue time. Hours before it are CONTEXT: values
    recomputed from archived weather, not operational forecasts of those hours.
    They are drawn in grey and separated by the issue-time marker, because a
    newly generated estimate for an elapsed hour is not a forecast of it.
    """
    snap = json.loads((ARTIFACTS / "step22_snapshot.json").read_text())
    issued = pd.Timestamp(snap["issued_at"])
    d = pd.DataFrame(snap["series"])
    d["time"] = pd.DatetimeIndex(d["time"])
    d = d.set_index("time")

    lo = issued.normalize() - pd.Timedelta("1D")
    d = d[(d.index >= lo) & (d.index < issued.normalize() + pd.Timedelta("4D"))]
    fc = d[d["is_forecast"]]
    ctx = d[~d["is_forecast"]]

    fig, ax = plt.subplots(figsize=(9.2, 3.3))
    if not ctx.empty:
        ax.plot(ctx.index, ctx["mw"], color="#9a9a9a", lw=1.5,
                label="context (recomputed from archived weather, not a forecast)")
    ax.fill_between(fc.index, fc["low_illustrative"], fc["high_illustrative"],
                    color="#1f77b4", alpha=0.22,
                    label="illustrative range (aggregated, NOT a validated interval)")
    ax.plot(fc.index, fc["mw"], color="#1f77b4", lw=1.9, label="forecast")

    ax.axvline(issued, color="#d62728", lw=1.4)
    ax.annotate(f"forecast issued\n{issued:%Y-%m-%d %H:%M}",
                xy=(issued, ax.get_ylim()[1] * 0.92), xytext=(7, 0),
                textcoords="offset points", fontsize=7.5, color="#d62728",
                va="top")

    # J+n counted from the issue time, not from the start of the series.
    for k in range(4):
        t = issued.normalize() + pd.Timedelta(days=k)
        if t < d.index.min():
            continue
        ax.axvline(t, color="#bbb", lw=0.7, ls=":")
        ax.text(t + pd.Timedelta("1h"), ax.get_ylim()[1] * 0.04, f"J+{k}",
                fontsize=7.5, color="#666")

    p = snap["peak"]
    ax.set_ylabel("MW")
    ax.set_title(f"National rooftop PV, J+0 to J+3 \u2014 snapshot issued "
                 f"{issued:%Y-%m-%d %H:%M %Z}", fontsize=10, pad=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=1)
    ax.grid(alpha=0.25, lw=0.6)
    ax.margins(x=0.01)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "fig_forecast.pdf")

    MACROS["SnapIssued"] = f"{issued:%Y-%m-%d %H:%M}"
    MACROS["SnapPeakMW"] = f"{p['mw']:,.1f}"
    MACROS["SnapPeakHalf"] = f"{p['half_width_mw']:,.1f}"
    MACROS["SnapPeakRange"] = f"{p['range_mw']:,.1f}"
    MACROS["SnapRho"] = f"{snap['config']['spatial_rho']:.3f}"
    MACROS["SnapFleetMW"] = f"{snap['config']['fleet_mw']:.0f}"
    MACROS["SnapUnits"] = f"{snap['config']['districts']}"
    MACROS["SnapContextH"] = f"{snap['context_hours']}"
    MACROS["SnapForecastH"] = f"{snap['forecast_hours']}"
    print(f"  wrote {OUT / 'fig_forecast.pdf'}")


def write_macros() -> None:
    """Emit the figure-derived numbers as LaTeX macros.

    The scenario figures are built from live weather, so their values move
    between runs. Quoting them by hand in the prose guaranteed the text and the
    chart would eventually disagree -- and they did. The report now \\input{}s
    this file and uses the macros, so one run produces both.
    """
    out = OUT / "figures.tex"
    lines = ["% generated by scripts/step20_report_figures.py -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}: " + ", ".join(f"{k}={v}" for k, v in sorted(MACROS.items())))


if __name__ == "__main__":
    fig_replay()
    fig_netload()
    fig_forecast()
    write_macros()
