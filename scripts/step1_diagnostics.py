"""Diagnostic figure for step 1.

Reads what step1_train.py wrote and asks the questions a judge will ask:
where is the error, when is it worst, and is the residual model correcting a
real conditional loss or just re-fitting a constant?
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast.config import ARTIFACTS, DKASC_SITE_03, PROCESSED  # noqa: E402
from dustcast.metrics import daylight_mask  # noqa: E402

INK = "#1b1b1f"
GRID = "#d8d8de"
SERIES = {
    "actual_kw": ("#111318", "measured"),
    "physics_trend": ("#e07b39", "physics + ageing trend"),
    "physics+ML": ("#2d7dd2", "physics + ML"),
}


def style(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8, length=3, color=GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def pick_week(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Pick the clearest or most broken week in the test set."""
    daily_kt = df["clearsky_index"].where(daylight_mask(df)).resample("1D").mean()
    weekly = daily_kt.rolling(7).mean()
    stamp = weekly.idxmax() if kind == "clear" else weekly.idxmin()
    if pd.isna(stamp):
        stamp = df.index[len(df) // 2]
    return df.loc[stamp - pd.Timedelta(days=7): stamp]


def main() -> int:
    site = DKASC_SITE_03
    df = pd.read_parquet(PROCESSED / "step1_test_predictions.parquet")
    imp = pd.read_csv(ARTIFACTS / "step1_feature_importance.csv",
                      index_col=0).squeeze("columns")
    day = daylight_mask(df)

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig.patch.set_facecolor("white")

    # --- 1&2: a clear week and a disturbed week -------------------------
    for ax, kind, title in (
        (axes[0, 0], "clear", "Clearest week in the test set"),
        (axes[0, 1], "cloudy", "Most disturbed week in the test set"),
    ):
        wk = pick_week(df, kind)
        for col, (colour, label) in SERIES.items():
            if col in wk:
                ax.plot(wk.index, wk[col], color=colour, linewidth=1.4,
                        label=label, alpha=0.9 if col == "actual_kw" else 0.85)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_ylabel("AC power (kW)", fontsize=9, color=INK)
        ax.legend(frameon=False, fontsize=8)
        style(ax)
        ax.tick_params(axis="x", rotation=30, labelsize=7)

    # --- 3: error vs cell temperature -----------------------------------
    ax = axes[1, 0]
    d = df[day]
    bins = pd.cut(d["temp_cell"], np.arange(0, 80, 5))
    for col, (colour, label) in SERIES.items():
        if col == "actual_kw" or col not in d:
            continue
        err = (d[col] - d["actual_kw"]).groupby(bins, observed=True).mean()
        ax.plot([i.mid for i in err.index], err.values, "o-", color=colour,
                linewidth=1.4, markersize=4, label=label)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_title("Mean error vs cell temperature  (does heat derating get captured?)",
                 fontsize=10, color=INK, loc="left")
    ax.set_xlabel("modelled cell temperature (degC)", fontsize=9)
    ax.set_ylabel("mean error (kW)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    style(ax)

    # --- 4: error vs clear-sky index ------------------------------------
    ax = axes[1, 1]
    bins = pd.cut(d["clearsky_index"], np.arange(0, 1.3, 0.1))
    for col, (colour, label) in SERIES.items():
        if col == "actual_kw" or col not in d:
            continue
        err = (d[col] - d["actual_kw"]).abs().groupby(bins, observed=True).mean()
        ax.plot([i.mid for i in err.index], err.values, "o-", color=colour,
                linewidth=1.4, markersize=4, label=label)
    ax.set_title("MAE vs clear-sky index  (1.0 = cloudless, low = overcast)",
                 fontsize=10, color=INK, loc="left")
    ax.set_xlabel("clear-sky index", fontsize=9)
    ax.set_ylabel("MAE (kW)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    style(ax)

    # --- 5: monthly MAE over the test period ----------------------------
    ax = axes[2, 0]
    for col, (colour, label) in SERIES.items():
        if col == "actual_kw" or col not in d:
            continue
        monthly = (d[col] - d["actual_kw"]).abs().resample("1MS").mean()
        ax.plot(monthly.index, monthly.values, "o-", color=colour,
                linewidth=1.4, markersize=3.5, label=label)
    ax.set_title("Monthly MAE across the test period  (is the gain stable?)",
                 fontsize=10, color=INK, loc="left")
    ax.set_ylabel("MAE (kW)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    ax.tick_params(axis="x", rotation=30, labelsize=7)

    # --- 6: feature importance ------------------------------------------
    ax = axes[2, 1]
    top = imp.head(12).iloc[::-1]
    ax.barh(top.index, top.values, color="#2d7dd2", alpha=0.85, height=0.7)
    ax.set_title("Residual model — feature importance by gain",
                 fontsize=10, color=INK, loc="left")
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(labelsize=8)

    fig.suptitle(
        f"DustCast step 1 — residual model diagnostics\n{site.name}",
        fontsize=12, color=INK, x=0.01, ha="left", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = ARTIFACTS / "step1_diagnostics.png"
    fig.savefig(out, dpi=140, facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
