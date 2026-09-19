"""The residual regressor and the baselines it has to beat.

The model never predicts power directly. It predicts the *gap* between what
physics expects and what the array actually delivered, and the gap is added
back to the physics prediction at the end. That keeps the learned quantity
small, roughly zero-centred and independent of installed capacity -- which is
what makes an Australian-trained model defensible on a Tunisian rooftop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .config import Site

#: Below this solar elevation the array produces nothing. Predictions are
#: forced to exactly zero rather than left to the model, which would otherwise
#: emit small non-zero values at night and (worse) be rewarded for it.
ZERO_OUTPUT_ELEVATION_DEG = 0.0

DEFAULT_PARAMS = dict(
    # Deliberately more trees than needed, with early stopping to choose the
    # actual count. A fixed 600 either underfits or overfits and there is no
    # way to tell which from the training loss alone.
    n_estimators=2000,
    early_stopping_rounds=50,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42,
)

#: Fraction of the *end* of the training period held back to early-stop on.
#: It has to come out of train, not out of the calibration split: MAPIE needs
#: the calibration set to be untouched by fitting or the conformal coverage
#: guarantee is void. Reusing it for early stopping would quietly break the
#: one property step 2 exists to provide.
EARLY_STOP_FRACTION = 0.15


@dataclass
class Split:
    """A chronological train/calibrate/test partition.

    Chronological, never random. A shuffled split lets the model see hours
    either side of the hour it is scoring and reports an accuracy that no
    forecast could ever achieve.
    """

    train: pd.DatetimeIndex
    calibrate: pd.DatetimeIndex
    test: pd.DatetimeIndex

    def describe(self) -> str:
        def rng(ix):
            return f"{ix.min():%Y-%m-%d} .. {ix.max():%Y-%m-%d}  ({len(ix):,} h)"

        return (
            f"  train     {rng(self.train)}\n"
            f"  calibrate {rng(self.calibrate)}\n"
            f"  test      {rng(self.test)}"
        )


def chronological_split(index: pd.DatetimeIndex, train_end: str, cal_end: str) -> Split:
    tz = index.tz
    t_end = pd.Timestamp(train_end, tz=tz)
    c_end = pd.Timestamp(cal_end, tz=tz)
    return Split(
        train=index[index <= t_end],
        calibrate=index[(index > t_end) & (index <= c_end)],
        test=index[index > c_end],
    )


@dataclass
class ResidualModel:
    site: Site
    feature_cols: list[str]
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    booster: XGBRegressor | None = None

    best_iteration: int | None = None

    #: Whether the booster was trained on residual / AC capacity rather than on
    #: residual in kW. Old artefacts predate the flag and default to False.
    capacity_normalised: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ResidualModel":
        """Fit on the training window, early-stopping on its final slice.

        The validation slice is the chronological *tail* of the training data,
        never a random sample: early stopping judged on interleaved hours would
        pick a tree count that suits interpolation rather than forecasting.
        """
        cut = int(len(X) * (1.0 - EARLY_STOP_FRACTION))
        X_fit, y_fit = X.iloc[:cut], y.iloc[:cut]
        X_val, y_val = X.iloc[cut:], y.iloc[cut:]

        self.booster = XGBRegressor(**self.params)
        self.booster.fit(
            X_fit[self.feature_cols], y_fit,
            eval_set=[(X_val[self.feature_cols], y_val)],
            verbose=False,
        )
        self.best_iteration = getattr(self.booster, "best_iteration", None)
        return self

    def predict_residual(self, X: pd.DataFrame,
                         site: Site | None = None) -> pd.Series:
        """Predicted residual in kW, for the site the forecast is FOR.

        `site` matters only for a capacity-normalised model, where the booster
        emits a fraction of AC capacity and the caller's array sets the scale.
        It defaults to the training site, which is correct for in-sample use and
        wrong the moment the model is applied to a different array -- so any
        cross-site caller must pass it explicitly.
        """
        pred = pd.Series(self.booster.predict(X[self.feature_cols]),
                         index=X.index, name="residual_pred_kw")
        if self.capacity_normalised:
            pred = pred * ac_capacity_kw(site or self.site)
        return pred

    def predict_power(self, X: pd.DataFrame, frame: pd.DataFrame,
                      site: Site | None = None) -> pd.Series:
        """Physics baseline + learned residual, with night and sign enforced."""
        site = site or self.site
        power = frame["expected_kw"] + self.predict_residual(X, site)
        return clamp_power(power, frame, site)

    @property
    def importances(self) -> pd.Series:
        return pd.Series(
            self.booster.feature_importances_, index=self.feature_cols
        ).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "ResidualModel":
        import joblib

        return joblib.load(path)


def ac_capacity_kw(site: Site) -> float:
    return site.ac_capacity_w / 1000.0


def normalise_residual(frame: pd.DataFrame, site: Site) -> pd.Series:
    """The residual as a FRACTION of AC capacity -- the transferable target.

    physics.py and CLAUDE.md both describe the residual as "independent of
    installed capacity", but `actual - expected` is in kW and therefore scales
    with array size: the physics removes the diurnal and seasonal SHAPE, not the
    SCALE. A model trained on a 6 kW array and applied to a 2.5 kW one emits
    corrections ~2.4x too large, which is a uniform positive bias at every hour
    -- exactly the frozen-holdout failure on array 16C.

    Dividing by AC capacity makes the target genuinely dimensionless, so it can
    be learned on one array and spent on another.
    """
    return frame["residual_kw"] / ac_capacity_kw(site)


def clamp_power(power: pd.Series, frame: pd.DataFrame, site: Site) -> pd.Series:
    """Apply the two constraints physics guarantees and ML does not."""
    night = frame["solar_elevation"] <= ZERO_OUTPUT_ELEVATION_DEG
    power = power.where(~night, 0.0)
    return power.clip(lower=0.0, upper=site.ac_capacity_w / 1000.0)


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


#: PVWatts' standard loss stack, applied multiplicatively. These are the
#: default derates any engineer would put on a system with no measurements:
#: soiling 2%, mismatch 2%, wiring 2%, connections 0.5%, light-induced
#: degradation 1.5%, nameplate tolerance 1%. Shading and snow are zero for this
#: array; availability is excluded because inverter downtime is an outage, not
#: a power derate.
PVWATTS_LOSS_FACTORS = {
    "soiling": 0.02,
    "mismatch": 0.02,
    "wiring": 0.02,
    "connections": 0.005,
    "lid": 0.015,
    "nameplate_rating": 0.01,
}


def pvwatts_loss_factor() -> float:
    factor = 1.0
    for loss in PVWATTS_LOSS_FACTORS.values():
        factor *= 1.0 - loss
    return factor


def baseline_physics(frame: pd.DataFrame, site: Site) -> pd.Series:
    """Raw clear-sky physics, no derates: the pure physical ceiling."""
    return clamp_power(frame["expected_physics_kw"], frame, site).rename("physics_kw")


def baseline_physics_derated(frame: pd.DataFrame, site: Site) -> pd.Series:
    """Physics with the standard PVWatts loss stack -- the textbook baseline."""
    power = frame["expected_physics_kw"] * pvwatts_loss_factor()
    return clamp_power(power, frame, site).rename("physics_derated_kw")


def fit_physics_calibration(frame: pd.DataFrame, train_index: pd.DatetimeIndex,
                            elevation_deg: float = 0.0) -> float:
    """Fit the single scalar derate that minimises MAE on the training period.

    This is the strongest baseline available without machine learning: an
    engineer with the same data would tune one loss factor and stop. Gating
    against it, rather than against raw physics, is what stops the skill score
    from being an artefact of an under-derated comparator -- the ML has to earn
    its place by capturing *conditional* losses (heat, cloud, soiling history),
    not the constant one anybody can fit by eye.

    Fitted by direct search on MAE rather than as a median ratio. A median
    ratio optimises the wrong thing: it weights a 0.1 kW dawn hour and a 4 kW
    midday hour equally, and in testing it produced a baseline *worse* than
    PVWatts' untuned default -- which would have handed the ML a free win.
    """
    sub = frame.loc[frame.index.intersection(train_index)]
    sub = sub[sub["solar_elevation"] > elevation_deg]
    # Must read the same column baseline_physics_calibrated applies the factor
    # to -- untouched physics -- or the factor is fitted against one expectation
    # and applied to a different one.
    expected = sub["expected_physics_kw"].to_numpy(float)
    actual = sub["ac_power_kw"].to_numpy(float)
    ok = np.isfinite(expected) & np.isfinite(actual)
    expected, actual = expected[ok], actual[ok]

    if expected.size == 0:
        return 1.0

    candidates = np.linspace(0.70, 1.05, 351)
    errors = [np.mean(np.abs(expected * f - actual)) for f in candidates]
    return float(candidates[int(np.argmin(errors))])


def baseline_physics_calibrated(frame: pd.DataFrame, site: Site,
                                factor: float) -> pd.Series:
    power = frame["expected_physics_kw"] * factor
    return clamp_power(power, frame, site).rename("physics_calibrated_kw")


def fit_degradation(frame: pd.DataFrame, train_index: pd.DatetimeIndex,
                    site: Site, elevation_deg: float = 0.0) -> tuple[float, float]:
    """Fit `actual ~ expected * (a + b * age_years)` on the training period.

    Why this belongs in the physics layer and not in the features:

    The array loses roughly 1%/yr, so across 17 years the physics baseline
    drifts ~16% high. A gradient-boosted tree cannot extrapolate a trending
    feature -- trained on ages 0-13 and asked about age 16, every split sends
    the row to the same terminal leaf and it silently applies the age-13
    correction. The trend would survive into the test period as a systematic
    over-prediction that no amount of tree depth can fix.

    A straight line in age is exactly the kind of known, smooth, physical
    effect the deterministic layer is supposed to absorb, leaving the ML the
    conditional losses it is actually good at. This is the project's own
    "physics first, ML on the residual" principle applied to ageing.

    Fitted in power space (not on the ratio) so midday hours dominate, which is
    where the degradation is observable and where forecast error costs money.
    Returns (a, b): `a` is the intercept derate, `b` the per-year slope.
    """
    sub = frame.loc[frame.index.intersection(train_index)]
    sub = sub[sub["solar_elevation"] > elevation_deg]

    age = age_years(sub.index, site)
    expected = sub["expected_physics_kw"].to_numpy(float)
    actual = sub["ac_power_kw"].to_numpy(float)

    ok = np.isfinite(expected) & np.isfinite(actual) & np.isfinite(age)
    expected, actual, age = expected[ok], actual[ok], age[ok]
    if expected.size == 0:
        return 1.0, 0.0

    design = np.column_stack([expected, expected * age])
    coef, *_ = np.linalg.lstsq(design, actual, rcond=None)
    return float(coef[0]), float(coef[1])


def age_years(index: pd.DatetimeIndex, site: Site) -> np.ndarray:
    commissioned = pd.Timestamp(site.commissioned, tz=site.tz)
    return ((index - commissioned).days / 365.25).to_numpy(float)


def apply_degradation(frame: pd.DataFrame, site: Site,
                      a: float, b: float) -> pd.DataFrame:
    """Fold the fitted ageing trend into the physics baseline.

    Keeps the untouched physics in `expected_physics_kw` so the raw baseline
    stays reportable, and recomputes the residual target against the corrected
    expectation.
    """
    frame = frame.copy()
    age = age_years(frame.index, site)
    frame["expected_kw"] = frame["expected_physics_kw"] * (a + b * age)
    if "ac_power_kw" in frame:
        frame["residual_kw"] = frame["ac_power_kw"] - frame["expected_kw"]
    return frame


def fit_diurnal_bias(frame: pd.DataFrame, train_index: pd.DatetimeIndex) -> pd.Series:
    """Mean residual by local hour, fitted on the training window.

    The single most important baseline once the physics is driven by FORECAST
    weather, and it was missing. The NWP carries a large, systematic
    time-of-day bias at this site -- the mean residual runs -0.71 kW at 08:00
    and +0.49 kW at 15:00, a 1.2 kW swing on a 6 kW array -- so a lookup table
    of thirteen numbers recovers most of what a gradient-boosted tree does.

    Reporting ML skill against a derate that does NOT know the hour inflates
    the model's apparent contribution roughly threefold: +29.7% against the
    fitted derate collapses to +9.9% against this. Anything the model cannot
    beat here, it has not earned.
    """
    sub = frame.loc[frame.index.intersection(train_index)]
    return sub["residual_kw"].groupby(sub.index.hour).mean()


def baseline_physics_diurnal(frame: pd.DataFrame, site: Site,
                             hourly_bias: pd.Series) -> pd.Series:
    """Physics plus a per-hour mean bias correction."""
    offset = pd.Series(frame.index.hour, index=frame.index).map(hourly_bias)
    power = frame["expected_kw"] + offset.fillna(0.0)
    return clamp_power(power, frame, site).rename("physics_diurnal_kw")


def baseline_physics_trend(frame: pd.DataFrame, site: Site) -> pd.Series:
    """Physics with the fitted ageing trend -- the toughest non-ML baseline."""
    return clamp_power(frame["expected_kw"], frame, site).rename("physics_trend_kw")


def baseline_smart_persistence(
    frame: pd.DataFrame, site: Site, lag_hours: int = 24
) -> pd.Series:
    """Yesterday's performance ratio applied to today's physics.

    Naive persistence (copy yesterday's power) is a straw man for solar because
    it ignores the known change in sun geometry. Smart persistence carries over
    only the *unexplained* part -- the ratio of actual to expected -- and is the
    baseline the forecasting literature actually uses.

    Yesterday's ratio is missing wherever the record has a gap -- and the 2024
    to 2025 test window is only ~76% complete, so a strict version is undefined
    for over a third of it. Since every model has to be scored on the same
    hours, that would silently shrink the whole evaluation to the subset where
    this one baseline happens to be defined. An operator would not give up:
    they would carry forward the most recent day they do have. So we forward
    fill up to a week, then fall back to the causal expanding median, which
    uses only ratios already observed at that point in time.
    """
    ratio = (frame["ac_power_kw"] / frame["expected_kw"].replace(0, np.nan)).clip(0, 1.5)
    carried = ratio.shift(lag_hours)
    carried = carried.ffill(limit=lag_hours * 7)
    carried = carried.fillna(ratio.shift(lag_hours).expanding().median())
    power = frame["expected_kw"] * carried
    return clamp_power(power, frame, site).rename("persistence_kw")
