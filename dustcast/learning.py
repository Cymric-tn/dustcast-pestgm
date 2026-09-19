"""Competing forecast models and the chronological learning loop.

Four models forecast the same hours. They COMPETE -- this is not a progression
in which more data eventually promotes the gradient-boosted model. Measured
results say the hour-of-day correction wins at day-ahead, and the replay is built
to show whatever actually happens, including the selection staying put or
switching back.

    L0  physics          pvlib on forecast weather, site geometry only
    L1  + derate         one scalar, actual/expected on the fit window
    L2  + diurnal bias   thirteen numbers, mean residual by local hour
    L3  + boosted        XGBoost on the capacity-normalised residual, trained on
                         the pooled fleet plus the target array's own past

Every model is refitted from past data at each step -- including L3, which is
retrained, not merely re-ranked. Selection happens on a held-back recent window
that no model was fitted on, and the chosen model then forecasts the next period
blind. Updating the parameters is the learning loop; reordering a leaderboard
would not be.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import features, metrics, model
from .config import Site


@dataclass
class Candidate:
    """One competing forecast model."""

    name: str
    level: str

    def fit(self, frame: pd.DataFrame, site: Site,
            pool: list[tuple[pd.DataFrame, Site]] | None = None) -> "Candidate":
        return self

    def predict(self, frame: pd.DataFrame, site: Site) -> pd.Series:
        raise NotImplementedError


@dataclass
class PhysicsOnly(Candidate):
    name: str = "L0 physics"
    level: str = "L0"

    def predict(self, frame, site):
        return model.clamp_power(frame["expected_kw"], frame, site)


@dataclass
class FittedDerate(Candidate):
    name: str = "L1 + derate"
    level: str = "L1"
    factor: float = 1.0

    def fit(self, frame, site, pool=None):
        self.factor = model.fit_physics_calibration(frame, frame.index)
        return self

    def predict(self, frame, site):
        return model.baseline_physics_calibrated(frame, site, self.factor)


@dataclass
class DiurnalBias(Candidate):
    name: str = "L2 + diurnal bias"
    level: str = "L2"
    hourly: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def fit(self, frame, site, pool=None):
        self.hourly = model.fit_diurnal_bias(frame, frame.index)
        return self

    def predict(self, frame, site):
        return model.baseline_physics_diurnal(frame, site, self.hourly)


@dataclass
class GradientBoosted(Candidate):
    """XGBoost on the capacity-normalised residual, retrained at every step.

    Pools the target array's own past with the rest of the fleet. The residual is
    divided by AC capacity so arrays of different sizes contribute a comparable
    target -- without that the model over-corrects small arrays by the capacity
    ratio, which is a positive bias at every hour.
    """

    name: str = "L3 + boosted"
    level: str = "L3"
    rm: model.ResidualModel | None = None

    def fit(self, frame, site, pool=None):
        blocks = []
        for f, s in [(frame, site)] + list(pool or []):
            X = features.make_features(f, s)[features.feature_names()]
            y = model.normalise_residual(f, s)
            ok = X.notna().all(axis=1) & y.notna()
            blocks.append(X[ok].assign(_y=y[ok].to_numpy()))
        both = pd.concat(blocks).sort_index()      # interleave; see step 15
        self.rm = model.ResidualModel(
            site=site, feature_cols=features.feature_names(),
            capacity_normalised=True,
        ).fit(both.drop(columns="_y"), both["_y"])
        return self

    def predict(self, frame, site):
        X = features.make_features(frame, site)[features.feature_names()]
        ok = X.notna().all(axis=1)
        out = pd.Series(np.nan, index=frame.index)
        if ok.any():
            out.loc[ok[ok].index] = self.rm.predict_power(
                X[ok], frame.loc[ok[ok].index], site)
        # Where features are unavailable the model abstains and physics stands.
        return out.fillna(model.clamp_power(frame["expected_kw"], frame, site))


@dataclass
class BoostedOnDiurnal(Candidate):
    """L2 first, then boost on what the hour-of-day correction leaves behind.

    L3 beats raw physics but loses to L2, so the question that decides whether
    the ML earns a place is not "physics or ML" but "does the ML add anything on
    top of the correction that wins". This stacks them: fit the thirteen-number
    diurnal bias, subtract it, and train the booster on THAT residual.

    It is one principled composition given the measured ordering, not a search
    over architectures, and it competes on exactly the same protocol as the
    others -- refitted on past data, selected on a held-back window.
    """

    name: str = "L4 diurnal + boosted"
    level: str = "L4"
    hourly: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    rm: model.ResidualModel | None = None

    @staticmethod
    def _offset(frame: pd.DataFrame, hourly: pd.Series) -> pd.Series:
        return pd.Series(frame.index.hour, index=frame.index).map(hourly).fillna(0.0)

    def fit(self, frame, site, pool=None):
        self.hourly = model.fit_diurnal_bias(frame, frame.index)
        blocks = []
        for f, s in [(frame, site)] + list(pool or []):
            # Each array gets its OWN diurnal correction before pooling; the
            # bias is site-specific, the leftover structure is what may transfer.
            h = model.fit_diurnal_bias(f, f.index)
            X = features.make_features(f, s)[features.feature_names()]
            y = (f["residual_kw"] - self._offset(f, h)) / model.ac_capacity_kw(s)
            ok = X.notna().all(axis=1) & y.notna()
            blocks.append(X[ok].assign(_y=y[ok].to_numpy()))
        both = pd.concat(blocks).sort_index()
        self.rm = model.ResidualModel(
            site=site, feature_cols=features.feature_names(),
            capacity_normalised=True,
        ).fit(both.drop(columns="_y"), both["_y"])
        return self

    def predict(self, frame, site):
        base = frame["expected_kw"] + self._offset(frame, self.hourly)
        X = features.make_features(frame, site)[features.feature_names()]
        ok = X.notna().all(axis=1)
        add = pd.Series(0.0, index=frame.index)
        if ok.any():
            idx = ok[ok].index
            add.loc[idx] = self.rm.predict_residual(X.loc[idx], site)
        return model.clamp_power(base + add, frame, site)


def candidates() -> list[Candidate]:
    return [PhysicsOnly(), FittedDerate(), DiurnalBias(), GradientBoosted(),
            BoostedOnDiurnal()]


def replay(frame: pd.DataFrame, site: Site,
           pool: list[tuple[pd.DataFrame, Site]] | None = None,
           *, start: str, step: str = "MS", select_days: int = 42,
           warmup_days: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk forward in time, refitting and reselecting from past data only.

    At each boundary t:
      1. past      = everything strictly before t
      2. selection = the last `select_days` of past, held back from fitting
      3. fit       = past minus selection; every model is refitted on it
      4. select    = lowest MAE on the selection window, which no model saw
      5. score     = the selected model, and every other, on [t, t+step)

    Returns (per-period results, per-period selection diagnostics).
    """
    tz = frame.index.tz
    bounds = pd.date_range(pd.Timestamp(start, tz=tz),
                           frame.index.max(), freq=step)
    rows, diag = [], []

    for t, nxt in zip(bounds, list(bounds[1:]) + [frame.index.max() + pd.Timedelta("1D")]):
        past = frame[frame.index < t]
        future = frame[(frame.index >= t) & (frame.index < nxt)]
        if len(past) < warmup_days * 6 or future.empty:
            continue

        sel_start = t - pd.Timedelta(days=select_days)
        fit_f = past[past.index < sel_start]
        sel_f = past[past.index >= sel_start]
        if fit_f.empty or sel_f.empty:
            continue

        sub_pool = [(f[f.index < sel_start], s) for f, s in (pool or [])]
        sub_pool = [(f, s) for f, s in sub_pool if not f.empty]

        fitted, sel_mae = [], {}
        for c in candidates():
            c.fit(fit_f, site, sub_pool)
            p = c.predict(sel_f, site)
            sel_mae[c.name] = float((p - sel_f["ac_power_kw"]).abs().mean())
            fitted.append(c)

        chosen = min(sel_mae, key=sel_mae.get)
        act = future["ac_power_kw"]
        rec = {"period": t, "n": len(future), "selected": chosen,
               "selected_on_mae": sel_mae[chosen]}
        for c in fitted:
            rec[c.name] = float((c.predict(future, site) - act).abs().mean())
        rec["realised_selected"] = rec[chosen]
        rows.append(rec)
        diag.append({"period": t, "fit_h": len(fit_f), "select_h": len(sel_f),
                     **{f"sel_{k}": v for k, v in sel_mae.items()}})

    return pd.DataFrame(rows).set_index("period"), pd.DataFrame(diag).set_index("period")
