"""Conformal prediction intervals on the residual.

The point of this layer is not to add error bars for decoration. An operator
sizes reserve off the pessimistic edge of the interval, not the point forecast,
so a band that is *calibrated* (90% means 90%) and *sharp* (as narrow as
honesty allows) is directly worth money in avoided spinning reserve.

Three methods are implemented because the obvious one is not good enough:

`absolute`  -- textbook split conformal. One constant width for every hour.
               Marginally calibrated and conditionally useless: the same band
               at midnight, at midday, and in a dust storm.
`normalised` -- split conformal with the width scaled by a learned estimate of
               |error|. Adaptive.
`cqr`        -- conformalized quantile regression (Romano et al. 2019).
               Quantile regressors give an adaptive shape, and the conformal
               step corrects whatever coverage they miss.

Step 4's gate is "intervals visibly widen on dust events", which a constant
width cannot do at all. Constant width is kept as the calibrated baseline that
the adaptive methods have to beat on sharpness at equal coverage.

NOTE on the MAPIE version: CLAUDE.md section 6.5 uses `MapieRegressor(...,
method="plus", cv="prefit")`, which is the 0.8.x API and does not exist in the
installed MAPIE 1.5. The 1.x equivalent is SplitConformalRegressor with
prefit=True, then .conformalize(), then .predict_interval().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from mapie.conformity_scores import ResidualNormalisedScore
from mapie.regression import SplitConformalRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor

from .config import Site
from .model import ResidualModel, clamp_power

#: Fraction of the calibration window used to fit the error-magnitude model in
#: the `normalised` method. Taken chronologically from the front, never
#: shuffled: MAPIE's own default would take a random 20%, which for a time
#: series lets the magnitude model see hours adjacent to the ones it is later
#: judged on.
MAGNITUDE_FIT_FRACTION = 0.40

QUANTILE_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    n_jobs=-1,
    random_state=42,
)


class ExponentiatedMagnitude(RegressorMixin, BaseEstimator):
    """Learn log|error|, predict |error|.

    MAPIE's ResidualNormalisedScore divides by this model's output, and with
    `prefit=True` it requires `predict` to return the magnitude itself while
    the model is *trained* on the log. Handing it raw log predictions is
    catastrophic rather than merely wrong: the logs are negative, MAPIE floors
    them at a ~1e-8 epsilon, and dividing by that produced intervals of
    +/-1e8 kW with the bounds inverted on 100% of rows -- which then clipped to
    a nonsensical negative width instead of failing loudly.
    """

    def __init__(self, inner=None, floor: float = 1e-3):
        self.inner = inner
        self.floor = floor

    def fit(self, X, y):
        self.inner.fit(X, np.log(np.clip(np.asarray(y, float), self.floor, None)))
        return self

    def predict(self, X):
        return np.exp(self.inner.predict(X))


@dataclass
class IntervalForecast:
    """Power forecast with a calibrated band, in kW."""

    point: pd.Series
    lower: pd.Series
    upper: pd.Series
    method: str

    @property
    def width(self) -> pd.Series:
        return self.upper - self.lower


def _to_power(residual_point, residual_lower, residual_upper,
              frame: pd.DataFrame, site: Site, method: str) -> IntervalForecast:
    """Shift a residual interval onto the physics baseline and enforce limits.

    Clipping the bounds to [0, capacity] cannot damage coverage: the measured
    power obeys the same limits, so any observation inside the raw band is
    still inside the clipped band. It does tighten PINAW, which is why
    sharpness must always be read alongside coverage.

    At night both bounds collapse to zero along with the point forecast. A
    band around a value that is known to be exactly zero is not uncertainty,
    and leaving it open would inflate apparent coverage on thousands of free
    hours.
    """
    inverted = float((residual_upper < residual_lower).mean())
    if inverted > 0:
        raise ValueError(
            f"{method}: upper bound below lower on {inverted:.1%} of rows. "
            "The interval is malformed; clipping it would hide the fault "
            "behind a plausible-looking coverage number."
        )

    base = frame["expected_kw"]
    return IntervalForecast(
        point=clamp_power(base + residual_point, frame, site),
        lower=clamp_power(base + residual_lower, frame, site),
        upper=clamp_power(base + residual_upper, frame, site),
        method=method,
    )


def split_conformal(rm: ResidualModel, frame: pd.DataFrame, X: pd.DataFrame,
                    y: pd.Series, cal_index: pd.DatetimeIndex,
                    test_index: pd.DatetimeIndex, site: Site,
                    confidence_level: float = 0.90,
                    adaptive: bool = False) -> IntervalForecast:
    """MAPIE split conformal, optionally with a residual-normalised score."""
    cols = rm.feature_cols
    X_cal, y_cal = X.loc[cal_index, cols], y.loc[cal_index]
    X_test = X.loc[test_index, cols]

    if adaptive:
        # Fit the error-magnitude model on the front of the calibration window,
        # conformalize on the back. Both are out-of-sample for the base model,
        # and the ordering is preserved.
        cut = int(len(cal_index) * MAGNITUDE_FIT_FRACTION)
        fit_idx, conf_idx = cal_index[:cut], cal_index[cut:]

        magnitude = ExponentiatedMagnitude(inner=XGBRegressor(**QUANTILE_PARAMS))
        base_err = np.abs(y.loc[fit_idx] - rm.predict_residual(X.loc[fit_idx, cols]))
        magnitude.fit(X.loc[fit_idx, cols], base_err)

        score = ResidualNormalisedScore(
            residual_estimator=magnitude, prefit=True, sym=True
        )
        X_cal, y_cal = X.loc[conf_idx, cols], y.loc[conf_idx]
    else:
        score = "absolute"

    scr = SplitConformalRegressor(
        estimator=rm.booster,
        confidence_level=confidence_level,
        prefit=True,
        conformity_score=score,
    )
    scr.conformalize(X_cal, y_cal)
    point, interval = scr.predict_interval(X_test)

    lower = pd.Series(interval[:, 0, 0], index=test_index)
    upper = pd.Series(interval[:, 1, 0], index=test_index)
    return _to_power(pd.Series(point, index=test_index), lower, upper,
                     frame.loc[test_index], site,
                     "normalised" if adaptive else "absolute")


def conformalized_quantile(rm: ResidualModel, frame: pd.DataFrame,
                           X: pd.DataFrame, y: pd.Series,
                           train_index: pd.DatetimeIndex,
                           cal_index: pd.DatetimeIndex,
                           test_index: pd.DatetimeIndex, site: Site,
                           confidence_level: float = 0.90) -> IntervalForecast:
    """Conformalized quantile regression on the residual (Romano et al. 2019).

    Implemented directly rather than through MAPIE's ConformalizedQuantileRegressor
    because that class drives the quantile level through a `set_params(alpha=...)`
    convention that XGBoost does not follow -- its parameter is `quantile_alpha`.

    Two quantile regressors give a band whose *shape* already varies with the
    weather. Conformalization then fixes its *level*: whatever coverage the
    quantile models actually achieve on held-out data, one scalar correction
    Q brings it to nominal. That is the appeal -- the adaptivity comes from the
    model, the guarantee comes from the calibration, and neither has to be
    trusted on its own.
    """
    alpha = 1.0 - confidence_level
    cols = rm.feature_cols

    lo = XGBRegressor(objective="reg:quantileerror",
                      quantile_alpha=alpha / 2, **QUANTILE_PARAMS)
    hi = XGBRegressor(objective="reg:quantileerror",
                      quantile_alpha=1 - alpha / 2, **QUANTILE_PARAMS)
    lo.fit(X.loc[train_index, cols], y.loc[train_index])
    hi.fit(X.loc[train_index, cols], y.loc[train_index])

    q_lo_cal = lo.predict(X.loc[cal_index, cols])
    q_hi_cal = hi.predict(X.loc[cal_index, cols])
    y_cal = y.loc[cal_index].to_numpy()

    # Conformity score: how far outside its own band did the truth fall?
    # Negative when the truth was comfortably inside.
    scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)

    # The finite-sample correction. Using the plain (1-alpha) empirical
    # quantile instead of this (n+1)/n adjustment under-covers slightly, and
    # the whole point of conformal prediction is the guarantee.
    n = len(scores)
    level = min(np.ceil((n + 1) * confidence_level) / n, 1.0)
    correction = float(np.quantile(scores, level, method="higher"))

    lower = pd.Series(lo.predict(X.loc[test_index, cols]) - correction,
                      index=test_index)
    upper = pd.Series(hi.predict(X.loc[test_index, cols]) + correction,
                      index=test_index)
    point = rm.predict_residual(X.loc[test_index, cols])

    return _to_power(point, lower, upper, frame.loc[test_index], site, "cqr")


@dataclass
class CQRBundle:
    """A trained, calibrated interval forecaster, portable to another site.

    This is the artefact step 3 deploys. It carries the two quantile models and
    the single conformal correction, and nothing else -- no data, no site
    geometry -- because the residual it predicts is deliberately
    capacity-independent and expressed in physical units the target site
    supplies for itself.
    """

    feature_cols: list[str]
    lo: XGBRegressor
    hi: XGBRegressor
    correction: float
    confidence_level: float
    trained_site: str
    calibrated_on: str

    def residual_interval(self, X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        cols = X[self.feature_cols]
        return (
            pd.Series(self.lo.predict(cols), index=X.index) - self.correction,
            pd.Series(self.hi.predict(cols), index=X.index) + self.correction,
        )

    def save(self, path) -> None:
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "CQRBundle":
        import joblib

        return joblib.load(path)


def fit_cqr_bundle(rm: ResidualModel, X: pd.DataFrame, y: pd.Series,
                   train_index: pd.DatetimeIndex, cal_index: pd.DatetimeIndex,
                   confidence_level: float = 0.90) -> CQRBundle:
    """Fit quantile models on train and conformalize on the calibration split."""
    alpha = 1.0 - confidence_level
    cols = rm.feature_cols
    lo, hi = _quantile_models(rm, X, y, train_index, alpha)

    q_lo = lo.predict(X.loc[cal_index, cols])
    q_hi = hi.predict(X.loc[cal_index, cols])
    y_cal = y.loc[cal_index].to_numpy()
    scores = np.maximum(q_lo - y_cal, y_cal - q_hi)

    n = len(scores)
    level = min(np.ceil((n + 1) * confidence_level) / n, 1.0)
    correction = float(np.quantile(scores, level, method="higher"))

    return CQRBundle(
        feature_cols=list(cols), lo=lo, hi=hi, correction=correction,
        confidence_level=confidence_level, trained_site=rm.site.key,
        calibrated_on=f"{cal_index.min():%Y-%m-%d}..{cal_index.max():%Y-%m-%d}",
    )


def _quantile_models(rm: ResidualModel, X: pd.DataFrame, y: pd.Series,
                     train_index: pd.DatetimeIndex, alpha: float):
    cols = rm.feature_cols
    lo = XGBRegressor(objective="reg:quantileerror",
                      quantile_alpha=alpha / 2, **QUANTILE_PARAMS)
    hi = XGBRegressor(objective="reg:quantileerror",
                      quantile_alpha=1 - alpha / 2, **QUANTILE_PARAMS)
    lo.fit(X.loc[train_index, cols], y.loc[train_index])
    hi.fit(X.loc[train_index, cols], y.loc[train_index])
    return lo, hi


def rolling_conformal(rm: ResidualModel, frame: pd.DataFrame, X: pd.DataFrame,
                      y: pd.Series, train_index: pd.DatetimeIndex,
                      cal_index: pd.DatetimeIndex, test_index: pd.DatetimeIndex,
                      site: Site, confidence_level: float = 0.90,
                      window_days: int = 60,
                      feedback_lag_hours: int = 24,
                      min_scores: int = 200) -> IntervalForecast:
    """CQR whose correction is recomputed from a trailing window of observed error.

    Split conformal assumes the calibration and test hours are exchangeable.
    On this array they are demonstrably not: the base model's error standard
    deviation is 0.0885 kW over the 2022-23 calibration window and 0.1654 kW
    over 2024-25, so a band sized on the former under-covers the latter by
    about fourteen points. That is not a defect of conformal prediction, it is
    conformal prediction correctly reporting that the world moved.

    The standard remedy is to stop treating calibration as a one-off. At each
    hour the correction is the conformal quantile of the errors actually
    observed in the preceding `window_days`, so the band tracks drift instead
    of assuming it away.

    DEPLOYMENT ASSUMPTION, and it is a real one: this needs measured output
    back within `feedback_lag_hours`. STEG cannot see behind-the-meter rooftop
    generation -- which is the entire problem this project exists to solve --
    so this is only available for the *monitored sample* sites that the
    upscaling layer already depends on. The correction learned there is then
    applied to the unmonitored fleet. Every score used for an hour is lagged by
    `feedback_lag_hours`, so nothing is used before it could have been known.
    """
    alpha = 1.0 - confidence_level
    cols = rm.feature_cols
    lo, hi = _quantile_models(rm, X, y, train_index, alpha)

    # Hours whose truth is eventually observable: the calibration window plus
    # the test period itself, the latter only ever consulted with a lag.
    observed = cal_index.union(test_index)
    q_lo = pd.Series(lo.predict(X.loc[observed, cols]), index=observed)
    q_hi = pd.Series(hi.predict(X.loc[observed, cols]), index=observed)
    scores = np.maximum(q_lo - y.loc[observed], y.loc[observed] - q_hi)

    def conformal_quantile(window: np.ndarray) -> float:
        n = len(window)
        level = min(np.ceil((n + 1) * confidence_level) / n, 1.0)
        return float(np.quantile(window, level, method="higher"))

    correction = scores.rolling(f"{window_days}D",
                                min_periods=min_scores).apply(
        conformal_quantile, raw=True)

    # Shift the whole curve forward in time so an hour can only ever use a
    # correction computed from data at least `feedback_lag_hours` old. A
    # positional shift would be wrong here: the series is daylight-only and
    # full of gaps, so N rows back is not N hours back.
    lagged = correction.copy()
    lagged.index = lagged.index + pd.Timedelta(hours=feedback_lag_hours)
    applied = lagged.reindex(test_index, method="ffill")

    # Before the window has filled, fall back to the static split-conformal
    # correction from the calibration set.
    static = conformal_quantile(scores.loc[cal_index].to_numpy())
    applied = applied.fillna(static)

    lower = pd.Series(lo.predict(X.loc[test_index, cols]), index=test_index) - applied
    upper = pd.Series(hi.predict(X.loc[test_index, cols]), index=test_index) + applied
    point = rm.predict_residual(X.loc[test_index, cols])

    return _to_power(point, lower, upper, frame.loc[test_index], site,
                     "rolling_cqr")
