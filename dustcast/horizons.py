"""Forecasting at more than one temporal scale, each calibrated on its own.

The brief asks for production forecast "at multiple spatial and temporal
scales". The spatial half is the governorate/national upscaling; this is the
temporal half, and it is not just the same model printed at three resolutions.

Two measurements drive the design, both taken at DKASC against seventeen years
of real output.

FIRST: NWP irradiance error jumps between the analysis and the first forecast
day and then flattens out. Energy-weighted relative MAE runs 18.2% at lead 0,
21.2% at lead 1, and only 21.5% at lead 7. So the meaningful accuracy regimes
are *nowcast* versus *everything beyond today* -- not a smooth decay with lead
time, and not a reason to publish a different band for every hour of horizon.

SECOND: a conformal correction calibrated at one scale does not transfer to
another. Fifteen-minute output is more variable than its hourly mean, so an
hourly-calibrated band under-covers it; a week-ahead forecast is driven by a
worse input than a nowcast, so a nowcast-calibrated band under-covers that too.
Each scale therefore gets its own conformal correction, measured rather than
assumed. That is what makes this three products instead of one rescaled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scale:
    key: str
    label: str
    freq: str            # resolution the product is published at
    horizon_hours: int   # how far ahead it runs
    lead_day: int        # archived-forecast lead used to calibrate it
    rationale: str


#: Lead 0 is the analysis, so the nowcast is calibrated against measured
#: weather at 15 minutes -- the closest available stand-in for "the weather
#: input is essentially current". Beyond that the archived forecasts made 1 and
#: 5 days earlier are used directly, which is exactly the input the live
#: product will consume at those horizons.
SCALES: tuple[Scale, ...] = (
    Scale("nowcast", "Nowcast", "15min", 6, 0,
          "sub-hourly ramps; weather input is effectively current"),
    Scale("day_ahead", "Day-ahead", "1h", 48, 1,
          "unit commitment and reserve procurement"),
    Scale("week_ahead", "Week-ahead", "1h", 168, 5,
          "maintenance planning and cleaning-crew scheduling"),
)

SCALE_BY_KEY = {s.key: s for s in SCALES}


def conformal_correction(scores: np.ndarray, confidence_level: float) -> float:
    """The finite-sample conformal quantile of a set of conformity scores."""
    scores = np.asarray(scores, float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("nan")
    n = scores.size
    level = min(np.ceil((n + 1) * confidence_level) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def calibrate(bundle, X: pd.DataFrame, y: pd.Series,
              index: pd.DatetimeIndex, confidence_level: float = 0.90) -> float:
    """Recompute the conformal correction for one scale's calibration set.

    The quantile models are reused unchanged -- they describe the *shape* of the
    uncertainty, which is a property of the weather and the array, not of the
    publication cadence. Only the scalar correction is refitted, because that is
    the part that carries the coverage guarantee and the part that is specific
    to the resolution and lead time being published.
    """
    cols = bundle.feature_cols
    q_lo = bundle.lo.predict(X.loc[index, cols])
    q_hi = bundle.hi.predict(X.loc[index, cols])
    truth = y.loc[index].to_numpy()
    return conformal_correction(
        np.maximum(q_lo - truth, truth - q_hi), confidence_level)


def apply_correction(bundle, X: pd.DataFrame, correction: float
                     ) -> tuple[pd.Series, pd.Series]:
    """Residual interval using a correction other than the bundle's own."""
    cols = X[bundle.feature_cols]
    return (pd.Series(bundle.lo.predict(cols), index=X.index) - correction,
            pd.Series(bundle.hi.predict(cols), index=X.index) + correction)


def resample_dust(dust: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Put CAMS hourly aerosol onto an arbitrary forecast grid.

    Interpolated rather than forward-filled: dust concentration is a smooth
    physical field, and a step function through it would put artificial
    discontinuities into the soiling integral at every hour boundary.
    """
    return dust.reindex(dust.index.union(index)).interpolate(
        method="time", limit=8).reindex(index)
