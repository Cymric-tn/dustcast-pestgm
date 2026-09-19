"""Evaluation.

Two rules govern this file.

1. Night is excluded from every accuracy metric. Both the model and every
   baseline predict exactly zero at night, so including those hours adds tens
   of thousands of free correct answers and inflates every score. A model
   evaluated over 24 h looks roughly twice as good as the same model evaluated
   over daylight, and the difference is entirely fictitious.

2. Coverage and sharpness are always reported together. PICP alone is trivially
   gamed -- an interval from zero to infinity has perfect coverage and no value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import NIGHT_ELEVATION_DEG


def daylight_mask(frame: pd.DataFrame, threshold: float = NIGHT_ELEVATION_DEG) -> pd.Series:
    """True where the sun is high enough for the hour to be informative."""
    return frame["solar_elevation"] > threshold


# --------------------------------------------------------------------------
# Deterministic
# --------------------------------------------------------------------------


def deterministic(y_true, y_pred, capacity_kw: float | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]

    err = y_pred - y_true
    out = {
        "n": int(ok.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mbe": float(np.mean(err)),  # signed: is the model systematically high?
    }
    if capacity_kw:
        out["nmae_pct"] = 100.0 * out["mae"] / capacity_kw
        out["nrmse_pct"] = 100.0 * out["rmse"] / capacity_kw
    return out


def skill_score(model_error: float, baseline_error: float) -> float:
    """Fraction of the baseline's error removed. Negative means worse.

    This is the number that decides whether the ML layer earns its place. If
    the residual model cannot beat clear-sky physics alone, the physics should
    ship on its own and the ML should be cut.
    """
    if baseline_error == 0:
        return float("nan")
    return 1.0 - model_error / baseline_error


# --------------------------------------------------------------------------
# Probabilistic (step 2)
# --------------------------------------------------------------------------


def picp(y_true, lower, upper) -> float:
    """Prediction Interval Coverage Probability -- does 90% mean 90%?"""
    y_true, lower, upper = map(lambda a: np.asarray(a, float), (y_true, lower, upper))
    ok = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    return float(np.mean((y_true[ok] >= lower[ok]) & (y_true[ok] <= upper[ok])))


def pinaw(lower, upper, capacity_kw: float) -> float:
    """Normalised interval width -- sharpness. Lower is better, given coverage."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    ok = np.isfinite(lower) & np.isfinite(upper)
    return float(np.mean(upper[ok] - lower[ok]) / capacity_kw)


def pinball_loss(y_true, y_quantile, quantile: float) -> float:
    """Asymmetric quantile loss: matches the operator's asymmetric cost."""
    y_true, y_quantile = np.asarray(y_true, float), np.asarray(y_quantile, float)
    ok = np.isfinite(y_true) & np.isfinite(y_quantile)
    diff = y_true[ok] - y_quantile[ok]
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def crps_from_quantiles(y_true, quantile_preds: dict[float, np.ndarray]) -> float:
    """CRPS approximated by averaging pinball loss over a quantile grid.

    Exact in the limit of a dense grid; with the handful of quantiles a
    conformal wrapper gives us it is an approximation, and should be labelled
    as one.
    """
    qs = sorted(quantile_preds)
    return float(2.0 * np.mean([pinball_loss(y_true, quantile_preds[q], q) for q in qs]))


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def interval_report(y_true, lower, upper, point, capacity_kw: float,
                    confidence_level: float = 0.90) -> dict:
    """Coverage, sharpness and quantile loss for one interval method."""
    alpha = 1.0 - confidence_level
    q_lo, q_hi = alpha / 2, 1 - alpha / 2
    return {
        "n": int(np.isfinite(np.asarray(y_true, float)).sum()),
        "PICP": picp(y_true, lower, upper),
        "target": confidence_level,
        "cov_error": picp(y_true, lower, upper) - confidence_level,
        "PINAW": pinaw(lower, upper, capacity_kw),
        "mean_width_kW": float(np.nanmean(np.asarray(upper, float)
                                          - np.asarray(lower, float))),
        f"pinball_{q_lo:.2f}": pinball_loss(y_true, lower, q_lo),
        "pinball_0.50": pinball_loss(y_true, point, 0.50),
        f"pinball_{q_hi:.2f}": pinball_loss(y_true, upper, q_hi),
        # Labelled an approximation deliberately: a true CRPS integrates over
        # the whole predictive distribution, and three quantiles is a coarse
        # grid. Useful for ranking methods, not for quoting as CRPS.
        "CRPS_approx": crps_from_quantiles(
            y_true, {q_lo: np.asarray(lower, float),
                     0.50: np.asarray(point, float),
                     q_hi: np.asarray(upper, float)}
        ),
    }


def conditional_coverage(y_true: pd.Series, lower: pd.Series, upper: pd.Series,
                         by: pd.Series, bins, capacity_kw: float,
                         label: str = "bin") -> pd.DataFrame:
    """PICP and PINAW within bins of some conditioning variable.

    Marginal coverage is the weakest possible guarantee. A band can cover 90%
    of all hours while covering 99% of easy overcast hours and 70% of the
    clear, high-output hours where the operator is actually exposed. Split
    conformal guarantees only the marginal number, so the conditional table is
    where an adaptive method earns -- or fails to earn -- its complexity.
    """
    grp = pd.cut(by, bins) if not isinstance(bins, int) else pd.qcut(by, bins)
    frame = pd.DataFrame({"y": y_true, "lo": lower, "hi": upper, "g": grp})
    out = frame.groupby("g", observed=True).apply(
        lambda d: pd.Series({
            "n": len(d),
            "PICP": picp(d["y"], d["lo"], d["hi"]),
            "PINAW": pinaw(d["lo"], d["hi"], capacity_kw),
            "mean_kW": float(d["y"].mean()),
        }),
        include_groups=False,
    )
    out.index.name = label
    return out


def compare(results: dict[str, dict], baseline: str) -> pd.DataFrame:
    """Table of every model against a named baseline."""
    table = pd.DataFrame(results).T
    for metric in ("mae", "rmse"):
        if metric in table:
            table[f"skill_{metric}"] = [
                skill_score(v, table.loc[baseline, metric]) for v in table[metric]
            ]
    return table


# --------------------------------------------------------------------------
# Operator metrics
# --------------------------------------------------------------------------
#
# CLAUDE.md section 7 asks for these by name and they were missing: a band the
# operator cannot be caught outside of, and a forecast that sees the ramps
# coming, are the two things that decide whether any of this is grid-ready.
# RMSE says nothing about either.


def winkler_score(y_true, lower, upper, alpha: float = 0.10) -> float:
    """Interval score: width, plus a penalty for every observation outside it.

    The proper scoring rule for an interval forecast, and the reason it is worth
    having alongside PICP and PINAW: those two can each be gamed alone, and the
    Winkler score cannot -- widening to buy coverage costs width, and narrowing
    to buy sharpness costs penalties. Lower is better.
    """
    y, lo, up = (np.asarray(a, float) for a in (y_true, lower, upper))
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(up)
    y, lo, up = y[ok], lo[ok], up[ok]

    score = up - lo
    score = score + np.where(y < lo, (2.0 / alpha) * (lo - y), 0.0)
    score = score + np.where(y > up, (2.0 / alpha) * (y - up), 0.0)
    return float(np.mean(score))


def reserve_shortfall(y_true, lower, capacity_kw: float | None = None) -> dict:
    """How often, and by how much, the operator would have been caught short.

    Reserve is procured to cover generation down to the interval's lower edge.
    An hour where actual output falls BELOW that edge is an hour the reserve did
    not cover -- the operator is short and must find energy somewhere else at
    whatever the balancing market charges.

    This is the asymmetric half of coverage, and the half that costs money.
    A 90% interval is expected to miss 10% of hours, but an operator does not
    care about the 5% that overshoot; overshooting is a curtailment problem, not
    a security one. Reported separately for that reason.
    """
    y, lo = (np.asarray(a, float) for a in (y_true, lower))
    ok = np.isfinite(y) & np.isfinite(lo)
    y, lo = y[ok], lo[ok]
    if y.size == 0:
        return {"n": 0, "frequency": float("nan")}

    short = lo - y
    breached = short > 0
    out = {
        "n": int(y.size),
        "n_short": int(breached.sum()),
        "frequency": float(breached.mean()),
        "worst_kw": float(short[breached].max()) if breached.any() else 0.0,
        "mean_when_short_kw": float(short[breached].mean()) if breached.any() else 0.0,
        # Energy the reserve failed to cover, over the scored hours.
        "unserved_kwh": float(short[breached].sum()) if breached.any() else 0.0,
    }
    if capacity_kw:
        out["worst_pct_capacity"] = 100.0 * out["worst_kw"] / capacity_kw
    return out


def ramp_detection(y_true: pd.Series, y_pred: pd.Series, window: str = "3h",
                   threshold_kw: float | None = None,
                   threshold_frac: float = 0.15,
                   capacity_kw: float | None = None) -> dict:
    """Did the forecast see the ramps coming?

    A ramp is a sustained change over `window` exceeding a threshold. Scored as
    a detection problem rather than a regression one, because that is how the
    consequence works: an operator either has dispatchable plant warming up or
    does not.

    Reported as POD (of the ramps that happened, how many were forecast), FAR
    (of the ramps forecast, how many did not happen) and CSI, which combines
    them. A forecast can score a fine RMSE while missing every ramp -- the
    errors are concentrated in a handful of hours and averaged away.

    Down-ramps are separated out because they are the expensive direction.
    """
    if threshold_kw is None:
        base = capacity_kw if capacity_kw else float(np.nanmax(y_true))
        threshold_kw = threshold_frac * base

    actual = y_true.diff().rolling(window).sum()
    pred = y_pred.diff().rolling(window).sum()
    valid = actual.notna() & pred.notna()
    actual, pred = actual[valid], pred[valid]

    def score(a_event, p_event) -> dict:
        hits = int((a_event & p_event).sum())
        misses = int((a_event & ~p_event).sum())
        false_alarms = int((~a_event & p_event).sum())
        pod = hits / (hits + misses) if (hits + misses) else float("nan")
        far = false_alarms / (hits + false_alarms) if (hits + false_alarms) else 0.0
        denom = hits + misses + false_alarms
        return {"events": int(a_event.sum()), "hits": hits, "misses": misses,
                "false_alarms": false_alarms, "POD": pod, "FAR": far,
                "CSI": hits / denom if denom else float("nan")}

    return {
        "window": window,
        "threshold_kw": float(threshold_kw),
        "down": score(actual <= -threshold_kw, pred <= -threshold_kw),
        "up": score(actual >= threshold_kw, pred >= threshold_kw),
        # Magnitude error on the hours that were genuinely ramping.
        "amplitude_mae_kw": float(
            (pred - actual)[actual.abs() >= threshold_kw].abs().mean())
        if (actual.abs() >= threshold_kw).any() else float("nan"),
    }
