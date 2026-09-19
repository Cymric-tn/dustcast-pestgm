"""The physical layer: what a panel *should* produce.

This is the deterministic half of the model. Given sun geometry, array
geometry and the measured (or forecast) weather, pvlib computes expected AC
power. Everything it cannot know -- soiling, real orientation error, ageing,
inverter quirks, thermal model error, NWP bias -- is left in the residual for
the ML layer to learn.

Stripping the diurnal and seasonal cycles here is what makes the residual a
small, roughly zero-centred, capacity-independent signal that transfers across
sites and climates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import Array, FixedMount, PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

from .config import Site

#: Below this solar elevation the beam component recovered from the GHI/DHI
#: closure is numerically meaningless, so it is discarded and the irradiance is
#: treated as entirely diffuse.
MIN_BEAM_ELEVATION_DEG = 3.0

#: Headroom above clear-sky DNI, to allow real cloud-edge enhancement.
BEAM_ENHANCEMENT_TOLERANCE = 0.10

#: The clear-sky ceiling is applied only *below* this elevation. Above it the
#: closure relation is well conditioned and needs no help -- the
#: extraterrestrial bound is never once exceeded above 20 degrees in 17 years
#: of this record.
#:
#: Scoping matters here. Applying the clear-sky ceiling at every elevation
#: clipped 24.6% of high-sun hours, because Ineichen's default Linke turbidity
#: climatology under-states clear-sky DNI at a site as exceptionally clear as
#: Alice Springs. That destroyed real beam irradiance on a quarter of the best
#: hours and made the physics baseline worse, while *raising* the ML's apparent
#: skill -- the model was being credited for repairing damage the physics layer
#: had just inflicted.
BEAM_CEILING_MAX_ELEVATION_DEG = 10.0


def build_location(site: Site) -> Location:
    return Location(
        latitude=site.latitude,
        longitude=site.longitude,
        altitude=site.altitude,
        tz=site.tz,
        name=site.name,
    )


def build_system(site: Site) -> PVSystem:
    """A PVWatts representation of the array.

    PVWatts rather than a CEC/SAPM module model on purpose: the array's exact
    module is a 2008 BP 3165 that is not reliably present in the current CEC
    database, and a wrong database entry is worse than an honest two-parameter
    model. PVWatts needs only nameplate DC power and a temperature
    coefficient, both of which are published for this array.
    """
    mount = FixedMount(surface_tilt=site.tilt, surface_azimuth=site.azimuth)
    array = Array(
        mount=mount,
        module_parameters={
            "pdc0": site.dc_capacity_w,
            "gamma_pdc": site.gamma_pdc,
        },
        temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS["sapm"][
            site.temperature_model
        ],
    )
    return PVSystem(
        arrays=[array],
        inverter_parameters={
            "pdc0": site.ac_capacity_w,
            "eta_inv_nom": site.eta_inv_nom,
        },
    )


def build_chain(site: Site) -> ModelChain:
    return ModelChain(
        system=build_system(site),
        location=build_location(site),
        clearsky_model="ineichen",
        transposition_model="haydavies",
        aoi_model="physical",
        spectral_model="no_loss",
        dc_model="pvwatts",
        ac_model="pvwatts",
        losses_model="no_loss",
    )


#: Percentile of the measured/modelled ratio taken to represent a genuinely
#: clear sky. Not the maximum: cloud-edge enhancement pushes a handful of hours
#: well above the true clear-sky curve, and fitting to those would over-correct.
CLEAR_ENVELOPE_PERCENTILE = 95

#: Only well-lit hours are informative; at low sun the airmass term dominates
#: and the ratio becomes unstable.
CLEARSKY_FIT_MIN_ELEVATION_DEG = 20.0


def fit_clearsky_scale(site: Site, weather: pd.DataFrame) -> pd.Series:
    """Fit a per-month multiplicative correction to the clear-sky reference.

    After the interval-centring fix, measured GHI still sits a few percent above
    Ineichen's clear-sky curve on the clearest hours, and `kt` has a median of
    1.03. A clear-sky index whose "clear" value is not 1.0 is mis-scaled, and
    step 4 reads the dust signal as a drop in exactly this variable.

    This is deliberately NOT expressed as a Linke turbidity. Fitting turbidity
    was tried first and pinned to the bottom of a plausible grid (TL=1.5) in
    every month while still under-predicting -- meaning no physically sensible
    atmosphere explains the gap. Forcing an unphysical TL to absorb it would
    dress up an instrument-or-model calibration as an atmospheric measurement.

    So: an honest empirical scale factor, which absorbs pyranometer calibration
    and Ineichen's bias at a high-altitude desert site together, and which is
    not to be quoted as a property of the sky. Monthly, because both the
    seasonal atmosphere and the sun's declination range vary through the year.
    """
    loc = build_location(site)
    df = weather[weather["ghi"].notna()]
    sun = solar_frame(site, df.index)
    lit = sun["solar_elevation"] > CLEARSKY_FIT_MIN_ELEVATION_DEG
    df, sun = df[lit], sun[lit]

    usable = sun["clearsky_ghi"] > 50.0
    ratio = (df.loc[usable, "ghi"] / sun.loc[usable, "clearsky_ghi"])
    scale = ratio.groupby(ratio.index.month).quantile(
        CLEAR_ENVELOPE_PERCENTILE / 100.0)
    scale.name = "clearsky_scale"
    return scale


def solar_frame(site: Site, times: pd.DatetimeIndex,
                clearsky_scale: pd.Series | None = None) -> pd.DataFrame:
    """Solar geometry and the clear-sky reference for a set of timestamps."""
    loc = build_location(site)
    solpos = loc.get_solarposition(times)
    clearsky = loc.get_clearsky(times, model="ineichen")

    if clearsky_scale is not None:
        factor = pd.Series(times.month, index=times).map(clearsky_scale)
        clearsky = clearsky.mul(factor, axis=0)

    out = pd.DataFrame(index=times)
    out["solar_zenith"] = solpos["apparent_zenith"]
    out["solar_elevation"] = solpos["apparent_elevation"]
    out["solar_azimuth"] = solpos["azimuth"]
    out["clearsky_ghi"] = clearsky["ghi"]
    out["clearsky_dni"] = clearsky["dni"]
    out["clearsky_dhi"] = clearsky["dhi"]
    return out


def complete_weather(site: Site, weather: pd.DataFrame,
                     clearsky_scale: pd.Series | None = None) -> pd.DataFrame:
    """Add DNI (which DKASC does not measure) and the clear-sky index.

    DNI is recovered from the closure relation ghi = dhi + dni*cos(zenith).
    This is numerically unstable near sunrise and sunset, where cos(zenith)
    approaches zero and a small GHI error explodes into a huge DNI.

    Bounding by the extraterrestrial constant is not enough, and measurably
    made things worse: below 2 degrees of elevation the closure exceeded the
    ceiling almost every hour, so the clip *parked* DNI at ~1361 W/m2 instead
    of rejecting it, and the physics baseline over-predicted those hours by
    5.2x. The physically meaningful ceiling is clear-sky DNI, which already
    collapses at low sun because the beam crosses a huge airmass. A 10% margin
    allows for genuine cloud-edge enhancement without readmitting the blow-up.

    Below MIN_BEAM_ELEVATION_DEG the beam component is discarded entirely and
    the irradiance is treated as diffuse, which is very nearly true at grazing
    incidence anyway.
    """
    sun = solar_frame(site, weather.index, clearsky_scale=clearsky_scale)
    w = weather.join(sun)

    filled = pvlib.irradiance.complete_irradiance(
        solar_zenith=w["solar_zenith"],
        ghi=w["ghi"],
        dhi=w["dhi"],
        dni=None,
    )
    dni = filled["dni"].clip(lower=0)

    ceiling = pvlib.irradiance.get_extra_radiation(w.index)
    low_sun = w["solar_elevation"] <= BEAM_CEILING_MAX_ELEVATION_DEG
    ceiling = ceiling.where(
        ~low_sun,
        np.minimum(ceiling, w["clearsky_dni"] * (1.0 + BEAM_ENHANCEMENT_TOLERANCE)),
    )
    dni = np.minimum(dni, ceiling)
    w["dni"] = dni.where(w["solar_elevation"] > MIN_BEAM_ELEVATION_DEG, 0.0)

    # Clear-sky index: the single most informative cloud feature. Bounded
    # because the clear-sky denominator collapses at low sun.
    w["clearsky_index"] = (
        w["ghi"] / w["clearsky_ghi"].clip(lower=1.0)
    ).clip(0.0, 1.5)

    return w


def run_physics(site: Site, weather: pd.DataFrame) -> pd.DataFrame:
    """Run the ModelChain and return expected power plus its internal state.

    `weather` must already carry ghi, dhi, dni, temp_air and wind_speed.
    Rows with any missing driver are returned as NaN rather than being filled,
    so that a gap in the weather record never masquerades as a forecast.

    Beyond expected power we surface two intermediates the ML layer genuinely
    needs and cannot cheaply rediscover:

    `temp_cell` -- modelled cell temperature. Heat derating is the single
    largest predictable loss on a 45 degC Tunisian afternoon, and it is driven
    by cell temperature, not air temperature. Handing the model temp_air and
    wind_speed and expecting it to re-derive the SAPM thermal relation wastes
    capacity on physics we already know exactly.

    `poa_global` -- plane-of-array irradiance. This, not GHI, is what actually
    strikes the panel; it already contains the transposition and the array's
    orientation.

    Both are computed from forecast weather at deployment time, so neither
    leaks anything a live forecast would not have.
    """
    required = ["ghi", "dhi", "dni", "temp_air", "wind_speed"]
    complete = weather[required].dropna()

    empty = pd.DataFrame(
        np.nan, index=weather.index,
        columns=["expected_kw", "temp_cell", "poa_global"],
    )
    if complete.empty:
        return empty

    chain = build_chain(site)
    chain.run_model(complete)

    ac_w = chain.results.ac
    if isinstance(ac_w, pd.DataFrame):  # multi-array systems
        ac_w = ac_w.sum(axis=1)

    def _first(result):
        return result[0] if isinstance(result, tuple) else result

    poa = _first(chain.results.total_irrad)
    cell = _first(chain.results.cell_temperature)

    out = pd.DataFrame(index=complete.index)
    # PVWatts' inverter model returns negative power at night (tare loss).
    # That is real for the hardware but it is not generation, and a negative
    # baseline would put a spurious offset into every night-time residual.
    out["expected_kw"] = ac_w.clip(lower=0.0) / 1000.0
    out["temp_cell"] = cell
    out["poa_global"] = poa["poa_global"] if isinstance(poa, pd.DataFrame) else poa

    return out.reindex(weather.index)


def expected_power(site: Site, weather: pd.DataFrame) -> pd.Series:
    """Expected AC power in kW (convenience wrapper around run_physics)."""
    return run_physics(site, weather)["expected_kw"].rename("expected_kw")


def build_physics_frame(site: Site, hourly: pd.DataFrame,
                        clearsky_scale: pd.Series | None = None) -> pd.DataFrame:
    """Weather + solar geometry + physics baseline + residual target."""
    w = complete_weather(site, hourly, clearsky_scale=clearsky_scale)
    w = w.join(
        run_physics(site, w).rename(columns={"expected_kw": "expected_physics_kw"})
    )

    # `expected_physics_kw` is untouched physics and stays that way, so the raw
    # baseline remains reportable. `expected_kw` is the expectation the residual
    # is measured against; until an ageing trend is fitted (which needs a train
    # split, so it happens in the training script) the two are identical.
    w["expected_kw"] = w["expected_physics_kw"]

    # The training target. Physics has already removed the diurnal cycle, the
    # seasonal cycle and the dependence on installed capacity; what remains is
    # the loss signal. A deployment site has no measured output at all -- that
    # is the entire premise of the project -- so this is conditional.
    if "ac_power_kw" in w:
        w["residual_kw"] = w["ac_power_kw"] - w["expected_kw"]

    return w


# --------------------------------------------------------------------------
# Aerosol correction to forecast irradiance
# --------------------------------------------------------------------------

#: AOD floor before taking a logarithm. CAMS reports exact zeros over clean
#: ocean air and log(0) would poison the whole correction.
AOD_FLOOR = 0.005


def fit_nwp_aod_bias(nwp_ghi: pd.Series, measured_ghi: pd.Series,
                     aod: pd.Series, min_ghi: float = 100.0) -> tuple[float, float]:
    """Fit the NWP's aerosol-dependent GHI bias: rel_bias ~ a + b*log10(AOD).

    This is the dust layer's physical channel, and it is *measured* rather than
    assumed. Step 4 showed the residual model barely uses the dust features --
    only +0.3% of the interval widening was attributable to them -- but the
    same analysis showed the numerical weather model itself carries a large,
    orderly, aerosol-dependent error: at DKASC its GHI runs ~10% low in clean
    air and ~7% high once AOD passes 0.12.

    Correcting a measured bias in the input is a stronger move than hoping a
    gradient-boosted tree rediscovers it from five weakly-informative columns,
    and it transfers on the mechanism rather than on the statistics.

    Linear in log10(AOD) because aerosol attenuation follows Beer-Lambert in
    the exponent: equal *ratios* of optical depth, not equal increments, shift
    the transmitted beam by comparable amounts.

    Fitted through per-decile ENERGY-WEIGHTED bias -- sum(forecast) over
    sum(measured) within each bin -- rather than by regressing the raw hourly
    ratio. Two failure modes are being avoided at once:

    Least squares on the raw ratio chases heavy tails. An hour with 110 W/m2
    measured against 190 forecast is a +73% "bias" that describes a passing
    cloud, not aerosol; that fit put the bias at +15.4% at AOD 0.185 where the
    binned evidence says +6.8%, which would have doubled the national dust
    correction.

    A median of ratios is robust but weights a 110 W/m2 dawn hour the same as a
    900 W/m2 noon hour, and it is the noon hours that carry the power error the
    correction exists to fix. It flattened the slope to near nothing.

    Aggregating energy first and dividing second is the quantity that actually
    governs aggregate output error.
    """
    df = pd.DataFrame({"nwp": nwp_ghi, "measured": measured_ghi,
                       "aod": aod}).dropna()
    df = df[df["measured"] > min_ghi]
    if len(df) < 100:
        return 0.0, 0.0

    df["rel_bias"] = (df["nwp"] - df["measured"]) / df["measured"]
    df["log_aod"] = np.log10(df["aod"].clip(lower=AOD_FLOOR))

    bins = pd.qcut(df["log_aod"], 10, duplicates="drop")
    grouped = df.groupby(bins, observed=True)
    summary = pd.DataFrame({
        "x": grouped["log_aod"].median(),
        "y": grouped["nwp"].sum() / grouped["measured"].sum() - 1.0,
        "n": grouped.size(),
    }).dropna()
    if len(summary) < 3:
        return 0.0, 0.0

    b, a = np.polyfit(summary["x"], summary["y"], 1, w=np.sqrt(summary["n"]))
    return float(a), float(b)


def apply_aod_correction(ghi: pd.Series, aod: pd.Series,
                         a: float, b: float) -> pd.Series:
    """Remove the fitted aerosol-dependent bias from a forecast GHI series.

    Dividing rather than subtracting: the fitted quantity is a *relative*
    bias, so a forecast that is 7% high must be scaled down by 1.07, not
    reduced by a fixed number of W/m2 that would be meaningless at dawn.
    """
    bias = a + b * np.log10(aod.clip(lower=AOD_FLOOR))
    return ghi / (1.0 + bias.reindex(ghi.index).fillna(0.0))
