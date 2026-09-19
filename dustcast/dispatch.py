"""A simplified day-ahead dispatch and reserve-procurement simulation.

The forecasting metrics answer "how wrong is the number". This answers the only
question an operator actually has: "does using this forecast instead of that one
leave me better off". Those can diverge -- a forecast can win on MAE and lose on
cost, because cost is asymmetric and MAE is not.

THE MODEL, stated plainly. One node, hourly, no network, no unit commitment
integrality, no storage, perfect conventional availability.

    day ahead   schedule conventional generation for the forecast net load,
                and procure upward reserve equal to the distance from the
                point forecast down to the interval's lower edge.
    real time   PV turns out to be something else.
                below the reserved floor -> the shortfall is unserved energy;
                above the schedule       -> conventional backs down, and if it
                                            hits its minimum, PV is curtailed.

Reserve is procured off the interval, so a forecast with an honest band buys
less reserve than one with a pessimistic band, and pays for it in unserved
energy when the band was too tight. That trade is the whole point of the
exercise, and it is why both forecasts must carry a CALIBRATED interval: give
one of them a sloppy band and the comparison measures the band, not the
forecast.

This is a simulation under assumptions, not a Tunisian operational result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DispatchAssumptions:
    """Everything the answer depends on. All of it is an assumption."""

    energy_cost: float = 80.0        # USD/MWh, conventional (gas) energy
    reserve_cost: float = 12.0       # USD per MW held per hour
    voll: float = 3000.0             # USD/MWh, value of lost load
    curtailment_cost: float = 5.0    # USD/MWh, opportunity cost of spilt PV
    gen_min_mw: float = 1800.0       # conventional minimum stable generation
    ramp_limit_mw_h: float = 450.0   # conventional fleet ramp capability, MW/h
    peak_demand_mw: float = 4000.0   # system peak
    pv_capacity_mw: float = 500.0    # installed rooftop PV

    def describe(self) -> str:
        return (f"energy {self.energy_cost:.0f} $/MWh · reserve "
                f"{self.reserve_cost:.0f} $/MW/h · VoLL {self.voll:.0f} $/MWh · "
                f"curtail {self.curtailment_cost:.0f} $/MWh\n"
                f"    Gmin {self.gen_min_mw:.0f} MW · ramp "
                f"{self.ramp_limit_mw_h:.0f} MW/h · peak demand "
                f"{self.peak_demand_mw:.0f} MW · PV {self.pv_capacity_mw:.0f} MW")


#: A stated synthetic Tunisian daily load shape, normalised to the daily peak.
#: Summer-evening peaking, as air-conditioning drives the Tunisian system.
#: THIS IS ASSUMED. No STEG load data is public, and the shape matters: a
#: midday-peaking system would value solar forecasting differently.
LOAD_SHAPE = np.array([
    0.58, 0.55, 0.53, 0.52, 0.53, 0.56, 0.62, 0.68,   # 00-07
    0.74, 0.79, 0.83, 0.85, 0.86, 0.85, 0.84, 0.84,   # 08-15
    0.86, 0.90, 0.95, 0.99, 1.00, 0.94, 0.80, 0.67,   # 16-23
])


def demand_series(index: pd.DatetimeIndex, peak_mw: float) -> pd.Series:
    """Deterministic demand from the stated shape.

    Deliberately deterministic: the experiment isolates the effect of PV
    forecast error, and adding demand noise would blur exactly the signal it is
    trying to measure. Real dispatch of course faces both.
    """
    return pd.Series(LOAD_SHAPE[index.hour] * peak_mw, index=index, name="demand_mw")


@dataclass
class DispatchResult:
    name: str
    hours: int
    energy_cost: float
    reserve_cost: float
    unserved_cost: float
    curtail_cost: float
    reserve_mwh: float
    unserved_mwh: float
    unserved_hours: int
    curtailed_mwh: float
    ramp_violations: int
    detail: pd.DataFrame = field(repr=False, default=None)

    @property
    def total_cost(self) -> float:
        return (self.energy_cost + self.reserve_cost
                + self.unserved_cost + self.curtail_cost)

    @property
    def decision_cost(self) -> float:
        """The part of the bill the forecast can actually change.

        Fuel for energy that was genuinely delivered is burnt whatever the
        forecast said -- it depends on what the sun did, not on what was
        predicted. Including it makes every candidate look identical to within
        a rounding error: in the first run of this experiment it was 99.4% of
        the total and buried a 40% difference in the costs that matter.

        Reserve held, energy left unserved, and PV spilt are the three the
        operator's forecast choice moves.
        """
        return self.reserve_cost + self.unserved_cost + self.curtail_cost


def simulate(actual_pv_mw: pd.Series, point_mw: pd.Series, lower_mw: pd.Series,
             a: DispatchAssumptions, name: str) -> DispatchResult:
    """Run the dispatch for one forecast."""
    idx = actual_pv_mw.index
    demand = demand_series(idx, a.peak_demand_mw)

    # Day-ahead: schedule for the forecast net load, reserve down to the band.
    scheduled = (demand - point_mw).clip(lower=a.gen_min_mw)
    reserve = (point_mw - lower_mw).clip(lower=0.0)

    # Real time.
    need = demand - actual_pv_mw
    available = scheduled + reserve
    shortfall = (need - available).clip(lower=0.0)

    # Conventional backs down to its minimum; PV above that is spilt.
    room_for_pv = (demand - a.gen_min_mw).clip(lower=0.0)
    curtailed = (actual_pv_mw - room_for_pv).clip(lower=0.0)
    delivered_pv = actual_pv_mw - curtailed
    generation = (demand - delivered_pv).clip(lower=a.gen_min_mw)

    ramp = scheduled.diff().abs()
    violations = int((ramp > a.ramp_limit_mw_h).sum())

    detail = pd.DataFrame({
        "demand_mw": demand, "pv_actual_mw": actual_pv_mw,
        "pv_point_mw": point_mw, "pv_lower_mw": lower_mw,
        "scheduled_mw": scheduled, "reserve_mw": reserve,
        "unserved_mwh": shortfall, "curtailed_mwh": curtailed,
        "generation_mw": generation,
    })
    return DispatchResult(
        name=name, hours=len(idx),
        energy_cost=float(generation.sum() * a.energy_cost),
        reserve_cost=float(reserve.sum() * a.reserve_cost),
        unserved_cost=float(shortfall.sum() * a.voll),
        curtail_cost=float(curtailed.sum() * a.curtailment_cost),
        reserve_mwh=float(reserve.sum()),
        unserved_mwh=float(shortfall.sum()),
        unserved_hours=int((shortfall > 1e-6).sum()),
        curtailed_mwh=float(curtailed.sum()),
        ramp_violations=violations, detail=detail,
    )


def optimal_reliability(actual_pv_mw: pd.Series, point_mw: pd.Series,
                        offsets: dict[float, float], a: DispatchAssumptions,
                        scale: float = 1.0) -> tuple[float, float]:
    """Pick the reserve level that minimises decision cost, given the costs.

    Fixing reliability at a constant makes a cost sensitivity meaningless: no
    cost assumption can change a decision that was never a function of cost.
    A real operator equates the marginal cost of holding reserve against the
    marginal expected cost of falling short, so the reliability target must be
    chosen per cost scenario -- and chosen on CALIBRATION data, never on the
    period being scored.
    """
    best_level, best_cost = None, np.inf
    for level, off in offsets.items():
        lo = (point_mw - off).clip(lower=0.0)
        r = simulate(actual_pv_mw * scale, point_mw * scale, lo * scale, a, "trial")
        if r.decision_cost < best_cost:
            best_level, best_cost = level, r.decision_cost
    return best_level, best_cost


def compare(results: list[DispatchResult], reference: str) -> pd.DataFrame:
    rows = []
    ref = next(r for r in results if r.name == reference)
    for r in results:
        rows.append({
            "forecast": r.name,
            "decision_$": r.decision_cost,
            "reserve_$": r.reserve_cost,
            "unserved_$": r.unserved_cost,
            "curtail_$": r.curtail_cost,
            "reserve_MWh": r.reserve_mwh,
            "unserved_MWh": r.unserved_mwh,
            "unserved_h": r.unserved_hours,
            "curtailed_MWh": r.curtailed_mwh,
            "ramp_viol": r.ramp_violations,
            "saving_vs_ref_%": 100.0 * (1 - r.decision_cost / ref.decision_cost),
        })
    return pd.DataFrame(rows)
