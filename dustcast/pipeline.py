"""Shared preparation: raw CSV -> model matrix, split, ageing-corrected target.

Both the step 1 and step 2 scripts go through here. Duplicating this would let
the two steps drift apart, and a conformal interval calibrated on a slightly
different preprocessing than the model it wraps is silently invalid.

The expensive part (ingest + pvlib over 17 years) is cached to parquet. The
ageing fit is not cached: it depends on the training split, and folding a
split-dependent quantity into a cache is how leakage gets in.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import features, model, physics
from .config import PROCESSED, SPLIT_CAL_END, SPLIT_TRAIN_END, Site
from .ingest import load_clean


@dataclass
class Prepared:
    site: Site
    frame: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    feature_cols: list[str]
    split: model.Split
    ageing: tuple[float, float]
    clearsky_scale: pd.Series

    @property
    def capacity_kw(self) -> float:
        return self.site.ac_capacity_w / 1000.0

    def daylight(self, index=None) -> pd.Series:
        from .metrics import daylight_mask

        frame = self.frame if index is None else self.frame.loc[index]
        return daylight_mask(frame)


def hourly_frame(site: Site, nrows: int | None = None,
                 cache: bool = True, freq: str = "1h") -> pd.DataFrame:
    """Ingest only, cached. Reading the 313 MB CSV is the expensive step."""
    tag = "full" if nrows is None else f"n{nrows}"
    suffix = "" if freq == "1h" else f"_{freq}"
    path = PROCESSED / f"{site.key}_hourly_{tag}{suffix}.parquet"

    if cache and path.exists():
        return pd.read_parquet(path)
    frame = load_clean(site, nrows=nrows, freq=freq)
    if cache:
        frame.to_parquet(path)
    return frame


def physics_frame(site: Site, hourly: pd.DataFrame,
                  clearsky_scale: pd.Series | None = None) -> pd.DataFrame:
    """pvlib over the ingested weather. Not cached: it depends on the
    clear-sky calibration, which depends on the split."""
    frame = physics.build_physics_frame(site, hourly, clearsky_scale=clearsky_scale)
    return frame[frame["expected_physics_kw"].notna()
                 & frame["ac_power_kw"].notna()]


def auto_split(index: pd.DatetimeIndex, fracs=(0.70, 0.15)) -> model.Split:
    """Chronological 70/15/15 fallback when configured dates do not partition."""
    n = len(index)
    a, b = int(n * fracs[0]), int(n * (fracs[0] + fracs[1]))
    return model.Split(train=index[:a], calibrate=index[a:b], test=index[b:])


def prepare(site: Site, nrows: int | None = None, split_mode: str = "config",
            cache: bool = True, with_dust: bool = False,
            dust: pd.DataFrame | None = None,
            window: tuple[str, str] | None = None,
            split_dates: tuple[str, str] | None = None,
            weather: pd.DataFrame | None = None) -> Prepared:
    # `weather` overrides the measured record entirely. Step 4 uses it to feed
    # archived NWP irradiance while keeping measured power as the target, which
    # is the deployment configuration rather than the hindcast one.
    hourly = hourly_frame(site, nrows=nrows, cache=cache) if weather is None else weather

    # `window` restricts the whole pipeline to a date range -- used by step 4,
    # where the CAMS archive only reaches back to August 2022 and the dust and
    # no-dust models must be compared on exactly the same hours.
    if window is not None:
        lo = pd.Timestamp(window[0], tz=site.tz)
        hi = pd.Timestamp(window[1], tz=site.tz)
        hourly = hourly.loc[lo:hi]

    train_end, cal_end = split_dates or (SPLIT_TRAIN_END, SPLIT_CAL_END)

    def make_split(index: pd.DatetimeIndex) -> model.Split:
        if split_mode == "config":
            s = model.chronological_split(index, train_end, cal_end)
            if min(len(s.train), len(s.calibrate), len(s.test)) > 0:
                return s
        return auto_split(index)

    # The clear-sky calibration is fitted on the training window only. It uses
    # measured GHI, which is an input rather than the target, but it is still a
    # quantity estimated from data and so must not see the test period.
    scale = physics.fit_clearsky_scale(site, hourly.loc[make_split(hourly.index).train])

    frame = physics_frame(site, hourly, clearsky_scale=scale)
    split = make_split(frame.index)

    # Ageing lives in the physics layer -- a tree cannot extrapolate age_years
    # past its training range. Fitted on train only.
    a, b = model.fit_degradation(frame, split.train, site)
    frame = model.apply_degradation(frame, site, a, b)

    X = features.make_features(frame, site, dust=dust)
    cols = features.feature_names(with_dust=with_dust)
    X, y = X[cols], frame["residual_kw"]

    usable = X.notna().all(axis=1) & y.notna()
    X, y, frame = X[usable], y[usable], frame[usable]
    split = model.Split(
        train=split.train.intersection(X.index),
        calibrate=split.calibrate.intersection(X.index),
        test=split.test.intersection(X.index),
    )

    return Prepared(site=site, frame=frame, X=X, y=y, feature_cols=cols,
                    split=split, ageing=(a, b), clearsky_scale=scale)


def fit_residual_model(prep: Prepared) -> model.ResidualModel:
    """Train on daylight hours of the training window only.

    Night rows have a target of exactly zero and are ~half the record, so
    including them spends half the model's capacity on a constant that
    clamp_power enforces anyway.
    """
    lit = prep.frame.index[prep.daylight()]
    train_idx = prep.split.train.intersection(lit)
    rm = model.ResidualModel(site=prep.site, feature_cols=prep.feature_cols)
    return rm.fit(prep.X.loc[train_idx], prep.y.loc[train_idx])
