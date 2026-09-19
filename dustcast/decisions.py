"""Turning a forecast into something an operator can act on.

The distinction this module exists to enforce: a forecast answers "how much
power?", a decision answers "what should I do on Thursday?". Reserve is sized
off the pessimistic edge of the interval, never off the point forecast, because
the operator's exposure is asymmetric -- being short costs far more than being
long.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Modelled soiling loss above which cleaning is worth considering at all.
SOILING_ACTION_THRESHOLD = 0.08

#: Rain expected to wash the panels, making a cleaning visit wasted money.
RAIN_WILL_CLEAN_MM = 3.0


def reserve_requirement(point_kw: pd.Series, lower_kw: pd.Series) -> float:
    """Reserve to cover the downside of the interval, in kW.

    Sized off the gap between the point forecast and the *lower* bound: that is
    the generation the operator might not get and must be able to replace.
    Using the point forecast alone implicitly assumes the forecast is right,
    which is the assumption the whole uncertainty layer exists to avoid.
    """
    return float((point_kw - lower_kw).max())


def ramp_risk(power_kw: pd.Series, window: str = "3h",
              limit_kw_per_window: float | None = None) -> dict:
    """Largest sustained ramp over a rolling window, up and down.

    Down-ramps and up-ramps are reported separately because they cost
    differently: a sunset down-ramp must be met by dispatchable plant, while an
    up-ramp mostly threatens minimum generation limits and curtailment.
    """
    change = power_kw.diff().rolling(window).sum()
    worst_down = float(change.min())
    worst_up = float(change.max())
    out = {"window": window, "max_down_kw": worst_down, "max_up_kw": worst_up}
    if limit_kw_per_window is not None:
        out["breach"] = bool(max(abs(worst_down), worst_up) > limit_kw_per_window)
    return out


@dataclass
class CleaningAdvice:
    recommend: bool
    reason: str
    modelled_loss: float
    rain_expected_mm: float

    def __str__(self) -> str:
        verdict = "CLEAN" if self.recommend else "hold"
        return (f"{verdict}: {self.reason} "
                f"(modelled loss {self.modelled_loss:.1%}, "
                f"rain ahead {self.rain_expected_mm:.1f} mm)")


def cleaning_recommendation(
    soiling_index: pd.Series,
    rain_forecast_mm: pd.Series,
    threshold: float = SOILING_ACTION_THRESHOLD,
    rain_cleans_mm: float = RAIN_WILL_CLEAN_MM,
) -> CleaningAdvice:
    """Recommend cleaning only if the loss is material AND no rain is coming.

    The second condition is what makes this a decision rather than an alarm.
    Dispatching a crew the day before rain wastes the visit entirely, and in
    Tunisia's climate a soiled panel in late autumn is often a week away from
    being washed for free.

    IMPORTANT: `soiling_index` is a modelled relative index with no ground
    truth anywhere in this project (CLAUDE.md section 10). It is a maintenance
    *trigger*, and the percentage must never be presented as a measured loss.
    """
    loss = float(soiling_index.iloc[-1]) if len(soiling_index) else 0.0
    rain = float(rain_forecast_mm.sum()) if len(rain_forecast_mm) else 0.0

    if loss < threshold:
        return CleaningAdvice(False, "modelled soiling below action threshold",
                              loss, rain)
    if rain >= rain_cleans_mm:
        return CleaningAdvice(False, "defer - forecast rain will wash the array",
                              loss, rain)
    return CleaningAdvice(True, "soiling above threshold and no rain forecast",
                          loss, rain)


def dust_impact_summary(power_with: pd.Series, power_without: pd.Series,
                        dust: pd.Series, high_dust_percentile: float = 90) -> dict:
    """How much the dust model changes the answer, on the hours that matter.

    Reported on high-dust hours specifically. Averaged over a whole year the
    dust layer looks unimportant simply because most hours are clean, which is
    exactly the averaging that hides the events an operator needs warning of.
    """
    cut = float(np.nanpercentile(dust, high_dust_percentile))
    heavy = dust >= cut
    return {
        "dust_threshold": cut,
        "n_heavy_hours": int(heavy.sum()),
        "mean_shift_kw": float((power_with[heavy] - power_without[heavy]).mean()),
        "max_shift_kw": float((power_with[heavy] - power_without[heavy]).abs().max()),
    }
