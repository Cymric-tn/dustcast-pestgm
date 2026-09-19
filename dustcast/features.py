"""Feature construction for the residual model.

Every feature here must be computable *at forecast time* from an NWP product.
Nothing may depend on a measurement that only exists after the fact -- that is
the difference between a forecast and a hindcast, and it is the easiest way to
report a score the operator will never see.

The dust block is deliberately optional. Step 1 trains without it (DKASC gives
us no aerosol record); step 4 supplies a `dust` frame and the same function
produces the full feature set with no other change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from .config import Site

#: Rain heavy enough to wash a panel. Below this, dew and drizzle can actually
#: make soiling worse by cementing dust to the glass.
RAIN_CLEAN_MM = 3.0

BASE_FEATURES = [
    # physics baseline -- the residual scales with it, so the model needs it
    "expected_kw",
    "poa_global",
    # sky state
    "ghi",
    "clearsky_index",
    "clearsky_index_std_3h",
    "clearsky_index_lag_1h",
    # thermal drivers. temp_cell_excess is the quantity heat derating is
    # actually proportional to, so the model gets it directly rather than
    # re-deriving the thermal model from air temperature and wind.
    "temp_air",
    "temp_cell",
    "temp_cell_excess",
    "wind_speed",
    "relative_humidity",
    # sun geometry, expressed relative to the array rather than to the compass
    "solar_elevation",
    "solar_zenith",
    "azimuth_offset",
    "aoi",
    # calendar
    "hour_sin",
    "hour_cos",
    "declination_signed",
    # site state
    "age_years",
    # surface wetness / cleaning history
    "days_since_rain",
    "precip_7d",
]

DUST_FEATURES = [
    "aod",
    "dust",
    "dust_7d",
    "dust_30d",
    "soiling_index",
]


def days_since_rain(precip: pd.Series, threshold: float = RAIN_CLEAN_MM) -> pd.Series:
    """Days since the last panel-washing rain, broadcast back to hourly.

    The daily counter is lagged by one day before being broadcast. Without the
    lag, an hour at 09:00 would see rain that fell at 18:00 the same day --
    a look-ahead that the live pipeline could never reproduce from measured
    data.

    Before the first qualifying rain the counter runs from the start of the
    record, making it a LOWER BOUND on the true days since rain rather than a
    missing value. Returning NaN there was the original design and it broke the
    live pipeline outright: a 17-day September window at Sfax contains no rain
    at all, so every row was missing and the whole forecast was discarded. It
    was also wrong on the merits -- "no rain for the last 17 days" is the most
    informative soiling state there is, not an absence of information. The
    lower bound under-states soiling in a long drought, which is the safe
    direction; fetch a longer history to loosen it.
    """
    daily_total = precip.resample("1D").sum()
    washed = daily_total >= threshold

    counter = 0.0
    counts = []
    for was_washed in washed:
        counter = 0.0 if was_washed else counter + 1.0
        counts.append(counter)

    daily_count = pd.Series(counts, index=daily_total.index).shift(1)
    return daily_count.reindex(precip.index, method="ffill")


def soiling_accumulation(
    dust: pd.Series,
    precip: pd.Series,
    rain_reset_mm: float = RAIN_CLEAN_MM,
    k: float = 4e-6,
    cap: float = 0.30,
) -> pd.Series:
    """A deposition/cleaning balance producing a relative soiling index.

    CAUTION: `k` and `rain_reset_mm` are tuned, not measured. We have no
    soiling ground truth anywhere in this project. This output is a relative
    index and a maintenance trigger -- never quote it as a measured loss
    percentage. See CLAUDE.md section 10.

    `k` is applied per HOUR, against a dust concentration in ug/m3. CLAUDE.md
    section 6.4 suggests 1e-4, which saturates: at Alice Springs' typical
    ~20 ug/m3 that reaches the 30% cap in six days and stays pinned there, so
    the index carried no information and the cleaning trigger fired
    permanently. 4e-6 corresponds to roughly 0.2%/day of loss at 20 ug/m3,
    which reaches a few percent over a dry fortnight -- the order of magnitude
    the soiling literature reports for arid sites. Still a calibration choice,
    not a measurement.

    Deposition SATURATES asymptotically rather than being clipped. A hard
    `min(s + k*d, cap)` pins at the ceiling and loses all ordering once it gets
    there: four Tunisian governorates sat at exactly 30.00% after a 60-day dry
    spell and became indistinguishable from each other. Scaling the increment
    by the remaining headroom is also the more physical statement -- dust
    landing on already-dirty glass shades less additional cell area than dust
    landing on clean glass -- and it keeps the index strictly monotonic in
    accumulated deposition, which is what a maintenance ranking needs.
    """
    s, out = 0.0, []
    for d, p in zip(dust.fillna(0.0), precip.fillna(0.0)):
        if p >= rain_reset_mm:
            s = 0.0
        else:
            s += k * d * (1.0 - s / cap)
        out.append(s)
    return pd.Series(out, index=dust.index, name="soiling_index")


def make_features(
    frame: pd.DataFrame,
    site: Site,
    dust: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the model matrix from a physics frame.

    `frame` is the output of physics.build_physics_frame.
    `dust` is an optional hourly frame with `aerosol_optical_depth` and `dust`.
    """
    X = pd.DataFrame(index=frame.index)

    X["expected_kw"] = frame["expected_kw"]
    X["poa_global"] = frame["poa_global"]
    X["ghi"] = frame["ghi"]
    X["temp_air"] = frame["temp_air"]
    X["temp_cell"] = frame["temp_cell"]
    # Degrees above the 25 degC STC reference: derating is ~0.4%/degC of this,
    # so it is the linear driver of the thermal loss the residual must capture.
    X["temp_cell_excess"] = frame["temp_cell"] - 25.0
    X["wind_speed"] = frame["wind_speed"]
    X["relative_humidity"] = frame["relative_humidity"]

    kt = frame["clearsky_index"]
    X["clearsky_index"] = kt
    # Cloud *variability*, not cloud amount. A broken sky is where forecasts
    # fail, so this is the feature that should drive interval width in step 2.
    X["clearsky_index_std_3h"] = kt.rolling("3h").std()
    X["clearsky_index_lag_1h"] = kt.shift(1)

    X["solar_elevation"] = frame["solar_elevation"]
    X["solar_zenith"] = frame["solar_zenith"]

    # Bearing of the sun *relative to where the panel points*, wrapped to
    # [-180, 180]. Raw compass azimuth does not transfer between hemispheres:
    # at Alice Springs the sun sits near 0 deg (north) and the array faces
    # north; in Tunisia the sun sits near 180 deg and the array faces south. A
    # tree trained on the former has simply never seen the latter's values and
    # would send every Tunisian midday hour down a branch built for Australian
    # dawn. Expressed relative to the array, both read 0 at "sun straight
    # ahead", which is the quantity that actually drives the physics.
    X["azimuth_offset"] = _wrap180(frame["solar_azimuth"] - site.azimuth)

    # Angle of incidence on the array plane: the geometric mismatch between the
    # sun and this specific array's orientation.
    X["aoi"] = _aoi(site, frame["solar_zenith"], frame["solar_azimuth"])

    hour = frame.index.hour + frame.index.minute / 60.0
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # Season, signed by hemisphere. Day-of-year is worse than useless across
    # the equator: DKASC's summer is December and Tunisia's is June, so
    # doy_sin/doy_cos are anti-correlated between the training site and the
    # deployment site. Solar declination multiplied by the sign of the latitude
    # means the same thing everywhere -- positive is "high summer sun here".
    declination = np.degrees(
        pvlib.solarposition.declination_spencer71(frame.index.dayofyear)
    )
    X["declination_signed"] = declination * np.sign(site.latitude)

    commissioned = pd.Timestamp(site.commissioned, tz=site.tz)
    X["age_years"] = (frame.index - commissioned).days / 365.25

    precip = frame.get("precipitation", pd.Series(0.0, index=frame.index))
    X["days_since_rain"] = days_since_rain(precip)
    X["precip_7d"] = precip.rolling("7D").sum()

    if dust is not None:
        d = dust.reindex(frame.index).interpolate(limit=3)
        X["aod"] = d["aerosol_optical_depth"]
        X["dust"] = d["dust"]
        # Two different physical mechanisms with two different time constants:
        # AOD attenuates today's beam, accumulated dust soils the glass over
        # weeks. Both are needed; neither substitutes for the other.
        X["dust_7d"] = d["dust"].rolling("7D").mean()
        X["dust_30d"] = d["dust"].rolling("30D").mean()
        X["soiling_index"] = soiling_accumulation(d["dust"], precip)

    return X


def feature_names(with_dust: bool = False) -> list[str]:
    return BASE_FEATURES + (DUST_FEATURES if with_dust else [])


def _wrap180(degrees: pd.Series) -> pd.Series:
    """Fold an angular difference into [-180, 180]."""
    return ((degrees + 180.0) % 360.0) - 180.0


def _aoi(site: Site, zenith: pd.Series, azimuth: pd.Series) -> pd.Series:
    return pvlib.irradiance.aoi(
        surface_tilt=site.tilt,
        surface_azimuth=site.azimuth,
        solar_zenith=zenith,
        solar_azimuth=azimuth,
    )
