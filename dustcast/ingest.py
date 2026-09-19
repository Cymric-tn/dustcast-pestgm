"""Load and clean the DKASC Alice Springs array export.

The raw file is a ~313 MB CSV at 5-minute resolution covering 2008-09 to
2025-08. This module turns it into a clean, timezone-aware hourly frame with
the columns the physics layer expects.

Source (static export, no query params -- the provider's parameterised
download endpoint currently returns HTTP 500):
    https://solarcentre.spinifexvalley.com.au/export/70-Site_DKA-M5_A-Phase.csv
"""

from __future__ import annotations

import pandas as pd

from .config import TARGET_FREQ, Site

# Raw provider column -> our canonical name. Anything not listed is dropped.
COLUMN_MAP = {
    "timestamp": "timestamp",
    "Active_Power": "ac_power_kw",
    "Wind_Speed": "wind_speed",
    "Weather_Temperature_Celsius": "temp_air",
    "Weather_Relative_Humidity": "relative_humidity",
    "Global_Horizontal_Radiation": "ghi",
    "Diffuse_Horizontal_Radiation": "dhi",
    "Weather_Daily_Rainfall": "rain_cumulative_mm",
    "Performance_Ratio": "performance_ratio",
}

# Physically admissible ranges. Anything outside becomes NaN rather than being
# clipped: a GHI of 3000 W/m2 is a broken sensor, and clipping it to 1500 would
# quietly feed the model a fabricated clear-sky hour.
VALID_RANGES = {
    "ghi": (0.0, 1500.0),
    "dhi": (0.0, 1200.0),
    "temp_air": (-10.0, 60.0),
    "wind_speed": (0.0, 40.0),
    "relative_humidity": (0.0, 100.0),
}

#: A resampled hour is only kept if at least this fraction of its 5-minute
#: samples survived quality control. A partially-observed hour averages to a
#: value that is not the hourly mean, which would bias both the target and the
#: irradiance driving the physics model.
MIN_HOUR_COVERAGE = 0.75


def load_raw(site: Site, path=None, nrows: int | None = None) -> pd.DataFrame:
    """Read the raw 5-minute CSV into a tz-aware frame."""
    from .config import RAW

    path = path or (RAW / site.raw_file)
    usecols = list(COLUMN_MAP)

    df = pd.read_csv(
        path,
        usecols=lambda c: c in usecols,
        nrows=nrows,
        on_bad_lines="warn",
    )
    df = df.rename(columns=COLUMN_MAP)

    # The provider's export contains occasional malformed records -- at least
    # one row near the end of the 17-year file is missing its newline, gluing a
    # value to the next timestamp ('49.32"2025-08-22 07:00:00"'). Coercing
    # rather than casting turns those into NaN so quality_control can drop
    # them. A plain astype would leave the whole column as object dtype and the
    # failure would surface much later as an unreadable aggregation error.
    for col in df.columns:
        if col != "timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # The provider records local wall-clock time. The Northern Territory has no
    # daylight saving, so localisation is unambiguous -- but we still guard
    # against duplicate/missing wall-clock times rather than assuming.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index = df.index.tz_localize(
        site.tz, ambiguous="NaT", nonexistent="NaT"
    )
    return df[df.index.notna()]


def quality_control(df: pd.DataFrame, site: Site) -> pd.DataFrame:
    """Null out physically impossible readings. Never silently repair them."""
    df = df.copy()

    for col, (lo, hi) in VALID_RANGES.items():
        if col in df:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = pd.NA

    # Diffuse cannot exceed global. Where it does, both sensors are suspect.
    bad_split = df["dhi"] > df["ghi"] * 1.05
    df.loc[bad_split, ["ghi", "dhi"]] = pd.NA

    # Negative AC power is night-time inverter self-consumption, which is real
    # but not generation; floor it at zero. Power above the inverter's AC
    # rating plus headroom is a metering fault, not clipping.
    df.loc[df["ac_power_kw"] < 0, "ac_power_kw"] = 0.0
    ac_limit_kw = site.ac_capacity_w / 1000.0 * 1.2
    df.loc[df["ac_power_kw"] > ac_limit_kw, "ac_power_kw"] = pd.NA

    # Data before commissioning is not this array.
    df = df[df.index >= pd.Timestamp(site.commissioned, tz=site.tz)]

    return df


def to_hourly(df: pd.DataFrame, freq: str = TARGET_FREQ) -> pd.DataFrame:
    """Resample 5-minute records to `freq` means with a coverage guard.

    `freq` is a parameter rather than a constant because the platform forecasts
    at more than one temporal scale: a 15-minute nowcast and an hourly
    day-ahead are different products with different error distributions, and
    each has to be calibrated on data at its own resolution.
    """
    means = ["ac_power_kw", "ghi", "dhi", "temp_air", "wind_speed",
             "relative_humidity", "performance_ratio"]
    means = [c for c in means if c in df]

    res = df[means].resample(freq).mean()

    # Expected samples per bin at the native 5-minute cadence.
    expected = max(1, int(pd.Timedelta(freq) / pd.Timedelta("5min")))
    counts = df[means].resample(freq).count()
    res = res.mask(counts < expected * MIN_HOUR_COVERAGE)

    # Daily cumulative rainfall -> per-hour increment. The counter resets each
    # local midnight, so a negative difference is a reset, not negative rain.
    if "rain_cumulative_mm" in df:
        daily_max = df["rain_cumulative_mm"].resample(freq).max()
        increment = daily_max.diff()
        increment[increment < 0] = daily_max[increment < 0]
        res["precipitation"] = increment.clip(lower=0).fillna(0.0)

    return recentre_index(res, freq=freq)


def recentre_index(df: pd.DataFrame, freq: str = TARGET_FREQ) -> pd.DataFrame:
    """Relabel each averaging interval by its midpoint rather than its start.

    pandas labels a resampled bin with its LEFT edge, so the row marked 07:00
    holds the mean of 07:00-07:55. Solar position, clear-sky irradiance and the
    whole ModelChain are then evaluated at the *instant* 07:00 -- half an hour
    before the average they are being compared against.

    The damage is not subtle. Morning irradiance rises steeply, so the interval
    mean sits well above the instant at its start, and the clear-sky index came
    out at 1.26 at 07:00 falling monotonically to 0.45 at 18:00. That is a pure
    artefact of the labelling: it inflated kt above 1.0 on 47% of daylight
    hours, biased `expected_kw` by half an hour of sun movement, and made
    `hour_sin`/`hour_cos` the residual model's most important features -- the ML
    was spending its capacity undoing this.

    Shifting once here, at the point the averaging happens, means every
    downstream consumer is correct without having to know the convention. The
    resulting timestamps fall on the half hour, which is the honest label for
    an hourly mean.

    Sources with the opposite convention need the opposite sign: Open-Meteo
    reports radiation as the *preceding* hour's mean, so its 12:00 value is
    centred on 11:30.
    """
    return df.set_axis(df.index + pd.Timedelta(freq) / 2, axis=0)


def impute_wind(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing wind speed from a month-by-hour climatology.

    The site's anemometer stops reporting in this export after 2016 (DKASC
    moved to Class A instrumentation under the NTSR project in April 2019).
    Because the SAPM thermal model needs wind, leaving the gap would silently
    discard nine of the seventeen years -- including every year in which the
    array is most degraded.

    Measured against the years where wind *is* recorded, substituting a
    month-by-hour climatology changes the physics baseline by 0.0122 kW MAE
    (0.54% of mean output) with a bias of -0.0003 kW. That is an order of
    magnitude smaller than the ~0.15 kW error the residual model exists to
    correct, so the trade is heavily in favour of keeping the data.

    Deliberately no `wind_imputed` flag: it would be False before 2017 and True
    after, giving the model a clean proxy for "recent years" and therefore a
    shortcut to the ageing trend that has nothing to do with weather.
    """
    wind = df["wind_speed"]
    if wind.notna().all() or wind.isna().all():
        return df

    climatology = wind.groupby([wind.index.month, wind.index.hour]).mean()
    key = pd.MultiIndex.from_arrays([df.index.month, df.index.hour])
    fallback = pd.Series(climatology.reindex(key).to_numpy(), index=df.index)

    df = df.copy()
    df["wind_speed"] = wind.fillna(fallback)
    return df


def load_clean(site: Site, path=None, nrows: int | None = None,
               freq: str = TARGET_FREQ) -> pd.DataFrame:
    """Full ingest: raw CSV -> QC'd, gap-filled frame at `freq`."""
    df = load_raw(site, path=path, nrows=nrows)
    df = quality_control(df, site)
    return impute_wind(to_hourly(df, freq=freq))
