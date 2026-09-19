"""Prediction intervals for the model the national layer actually runs.

The interval results reported elsewhere belong to LOCALLY ADAPTED models --- they
use a site's own observed production. The Tunisian fleet has none, so the
national layer runs L0 physics, and an interval for L0 has to be built
differently: calibrated where observations exist (DKASC), carried across as a
FRACTION OF AC CAPACITY, and aggregated across regions.

Three things this does not do, stated here so they are not inferred:

  * It does not establish national coverage. Coverage is measured per site at
    DKASC. Aggregating calibrated site bands does not calibrate the aggregate,
    because the aggregation depends on a spatial correlation that is estimated,
    not verified against a measured national series. No such series exists for
    Tunisia.
  * It does not transfer a Tunisian-validated number. The calibration is
    Australian.
  * It is not conditional on dust load, only on sun elevation and clear-sky
    index. Dust widens the interval only insofar as it moves those.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics
from .config import Site

#: Sun-elevation bin edges (degrees). Error scales strongly with how much power
#: the array is making, and elevation is the cleanest available proxy that is
#: known at forecast time.
ELEVATION_BINS = [0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0]

#: Clear-sky index bins. A forecast that says "overcast" is wrong in a different
#: way from one that says "clear", so the band should not be the same width.
KT_BINS = [0.0, 0.4, 0.7, 0.9, 2.0]


def _conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Split-conformal quantile with the finite-sample correction."""
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    n = s.size
    q = min(np.ceil((n + 1) * level) / n, 1.0)
    return float(np.quantile(s, q, method="higher"))


def _bin_index(values: pd.Series, edges: list[float]) -> pd.Series:
    """Bin to an INTEGER index, not a pandas Interval.

    Interval categories do not survive a CSV round-trip: reloading them gives
    strings, a merge against freshly cut Intervals matches nothing, and every
    row silently falls back to the overall quantile -- which made the served
    band flat and ~1.75x too wide. Integers round-trip exactly.
    """
    return pd.Series(np.digitize(values.to_numpy(float), edges[1:-1]),
                     index=values.index).where(values.notna())


def fit_l0_band(parts: list[tuple[pd.DataFrame, Site]],
                level: float = 0.90) -> pd.DataFrame:
    """Calibrate an L0 interval on arrays that HAVE measured production.

    The score is the absolute L0 error divided by AC capacity, so it is
    dimensionless and can be spent on an array of any size. Binned by sun
    elevation and clear-sky index; bins that never fill fall back to the
    all-data quantile.
    """
    rows = []
    for frame, site in parts:
        f = frame[metrics.daylight_mask(frame)]
        cap = site.ac_capacity_w / 1000.0
        err = (f["expected_kw"].clip(lower=0.0) - f["ac_power_kw"]).abs() / cap
        rows.append(pd.DataFrame({
            "score": err,
            "elev": _bin_index(f["solar_elevation"], ELEVATION_BINS),
            "kt": _bin_index(
                f.get("clearsky_index", pd.Series(1.0, index=f.index)), KT_BINS),
        }))
    d = pd.concat(rows).dropna(subset=["score"])

    overall = _conformal_quantile(d["score"].to_numpy(), level)
    out = (d.dropna(subset=["elev", "kt"])
             .groupby(["elev", "kt"], observed=True)["score"]
             .agg(n="size", offset=lambda s: _conformal_quantile(s.to_numpy(), level))
             .reset_index())
    # A bin calibrated on a handful of hours is noise, not calibration.
    out.loc[out["n"] < 50, "offset"] = overall
    out["offset"] = out["offset"].fillna(overall)
    # Carried as a COLUMN as well as an attr: DataFrame.attrs does not survive
    # to_csv, and the fallback must round-trip with the table.
    out["overall"] = overall
    out["level"] = level
    out.attrs["overall"] = overall
    out.attrs["level"] = level
    return out


def apply_l0_band(frame: pd.DataFrame, site: Site,
                  band: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Half-width in kW for each hour, as (lower, upper) power bounds."""
    cap = site.ac_capacity_w / 1000.0
    lookup = {(int(r.elev), int(r.kt)): float(r.offset)
              for r in band.itertuples()}
    fallback = float(band.attrs.get("overall",
                                    band["overall"].iloc[0] if "overall" in band
                                    else 0.0))
    elev = _bin_index(frame["solar_elevation"], ELEVATION_BINS)
    kt = _bin_index(
        frame.get("clearsky_index", pd.Series(1.0, index=frame.index)), KT_BINS)
    off = pd.Series(
        [lookup.get((int(e), int(k)), fallback) if pd.notna(e) and pd.notna(k)
         else fallback for e, k in zip(elev, kt)],
        index=frame.index) * cap

    point = frame["expected_kw"].clip(lower=0.0)
    night = frame["solar_elevation"] <= 0.0
    lo = (point - off).clip(lower=0.0).where(~night, 0.0)
    hi = (point + off).clip(upper=cap).where(~night, 0.0)
    return lo, hi
