"""Live weather and dust from Open-Meteo.

Two endpoints, two jobs, and they must not be confused:

  forecast API      -- irradiance, temperature, wind, rain. What the panel sees.
  air-quality API   -- CAMS dust and aerosol optical depth. What is in the way.

`domains=cams_global` is mandatory on the air-quality call. The default `auto`
can resolve to the European CAMS domain, which does not properly cover Tunisia
and will silently return a differently-sourced series.

Responses are cached on disk. The free tier is non-commercial and rate limited,
and a demo loop that refetches on every render will get itself blocked.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from .config import PROCESSED, Site

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

CACHE = PROCESSED / "live_cache"
CACHE.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SECONDS = 3600

#: Open-Meteo names -> ours.
WEATHER_VARS = {
    "shortwave_radiation": "ghi",
    "diffuse_radiation": "dhi",
    "direct_normal_irradiance": "dni",
    "temperature_2m": "temp_air",
    "relative_humidity_2m": "relative_humidity",
    "wind_speed_10m": "wind_speed",
    "precipitation": "precipitation",
}

#: Variables Open-Meteo reports as the *preceding* hour's mean or sum, rather
#: than an instantaneous reading at the timestamp.
AGGREGATED_VARS = {"ghi", "dhi", "dni", "precipitation"}


def _get(url: str, params: dict, cache_key: str, ttl: int = CACHE_TTL_SECONDS) -> dict:
    path = CACHE / f"{cache_key}.json"
    if path.exists() and (time.time() - path.stat().st_mtime) < ttl:
        return json.loads(path.read_text())

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    path.write_text(json.dumps(payload))
    return payload


def _to_frame(payload: dict, rename: dict, tz: str,
              block: str = "hourly") -> pd.DataFrame:
    hourly = payload[block]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(tz)
    df = df.set_index("time").rename(columns=rename)
    return df[[c for c in rename.values() if c in df]].astype("float64")


def fetch_weather(site: Site, past_days: int = 14, forecast_days: int = 3,
                  refresh: bool = False, freq: str = "1h") -> pd.DataFrame:
    """Hourly forecast weather, on the same interval-centred convention as training.

    `past_days` is not decoration: `days_since_rain` and `precip_7d` need real
    recent history, and a forecast that starts today would have to invent it.

    Timestamp handling is the subtle part. Open-Meteo reports radiation and
    precipitation as the *preceding* hour (its 12:00 value covers 11:00-12:00,
    centred on 11:30) but temperature, humidity and wind as instantaneous
    readings at the timestamp. The training data was recentred so that every
    row is labelled by the midpoint of the interval it describes, so the two
    families need opposite treatment: shift the aggregated ones back half an
    hour, and interpolate the instantaneous ones forward onto the same grid.

    Getting this wrong is not cosmetic. The equivalent mistake on the training
    side put the clear-sky index at 1.26 in the morning and 0.45 in the evening
    and made time-of-day the model's most important feature.
    """
    block = "minutely_15" if freq == "15min" else "hourly"
    params = {
        "latitude": site.latitude,
        "longitude": site.longitude,
        block: ",".join(WEATHER_VARS),
        "wind_speed_unit": "ms",  # default is km/h; the model was trained on m/s
        "timezone": site.tz,
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    key = f"weather_{site.key}_{past_days}_{forecast_days}_{freq}"
    payload = _get(FORECAST_URL, params, key,
                   0 if refresh else CACHE_TTL_SECONDS)
    df = _to_frame(payload, WEATHER_VARS, site.tz, block=block)
    return _centre_mixed(df, freq=freq)


HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def fetch_forecast_history(site: Site, start: str, end: str,
                           refresh: bool = False) -> pd.DataFrame:
    """Archived NWP forecasts for a past date range.

    This is what makes an honest dust experiment possible. Training the
    residual model on *measured* irradiance hides the entire effect the dust
    layer exists to correct: a pyranometer reading already contains the
    aerosol attenuation, so dust features can add nothing beyond slow soiling,
    and the ablation comes out near-null by construction.

    The forecast the live pipeline actually consumes is a different object.
    Measured against DKASC over 9,041 well-lit hours, Open-Meteo's GHI runs
    9.9% LOW on the cleanest hours (AOD < 0.03) and 6.8% HIGH once AOD passes
    0.12 -- a swing of roughly seventeen points that tracks aerosol load. That
    is exactly the error a dust-aware residual should learn, and it is
    invisible unless the physics is driven by forecast irradiance.

    So: NWP weather in, measured power as the target. The deployment
    configuration, with a known answer.
    """
    params = {
        "latitude": site.latitude,
        "longitude": site.longitude,
        "hourly": ",".join(WEATHER_VARS),
        "wind_speed_unit": "ms",
        "timezone": site.tz,
        "start_date": start,
        "end_date": end,
    }
    key = f"nwphist_{site.key}_{start}_{end}"
    payload = _get(HISTORICAL_FORECAST_URL, params, key,
                   0 if refresh else 30 * 24 * 3600)
    df = _to_frame(payload, WEATHER_VARS, site.tz)
    return _centre_mixed(df)


def _centre_mixed(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Put Open-Meteo's two timestamp conventions onto one centred grid.

    Radiation and precipitation are the *preceding* hour's mean or sum, so the
    12:00 value is centred on 11:30 and only needs relabelling. Temperature,
    humidity and wind are instantaneous at the timestamp and are interpolated
    onto the same half-hour grid.
    """
    half = pd.Timedelta(freq) / 2
    aggregated = [c for c in df if c in AGGREGATED_VARS]
    instantaneous = [c for c in df if c not in AGGREGATED_VARS]
    centred = df[aggregated].set_axis(df.index - half, axis=0)
    instant = df[instantaneous].rolling(2).mean().set_axis(df.index - half, axis=0)
    return centred.join(instant).dropna(how="all")


def fetch_dust(site: Site, past_days: int = 60, forecast_days: int = 5,
               refresh: bool = False) -> pd.DataFrame:
    """CAMS dust concentration and aerosol optical depth.

    Two different physical mechanisms, both needed:

    `aerosol_optical_depth` is a whole-column quantity and drives *attenuation*
    -- sunlight blocked before it reaches the glass, today.

    `dust` is a surface concentration and drives *soiling* -- particles
    settling on the glass, integrating over weeks. Hence the long `past_days`:
    the soiling state depends on deposition history, not on today's plume.
    """
    params = {
        "latitude": site.latitude,
        "longitude": site.longitude,
        "hourly": "dust,aerosol_optical_depth",
        # Mandatory. `auto` may resolve to the European CAMS domain, which does
        # not properly cover Tunisia.
        "domains": "cams_global",
        "timezone": site.tz,
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    key = f"dust_{site.key}_{past_days}_{forecast_days}"
    payload = _get(AIR_QUALITY_URL, params, key,
                   0 if refresh else CACHE_TTL_SECONDS)
    return _centre_instantaneous(
        _to_frame(payload, {"dust": "dust",
                            "aerosol_optical_depth": "aerosol_optical_depth"},
                  site.tz))


def _centre_instantaneous(df: pd.DataFrame) -> pd.DataFrame:
    """Put instantaneous hourly readings onto the half-hour centred grid.

    CAMS concentrations are instantaneous values at the timestamp, not period
    means, so they are interpolated to the interval midpoint rather than
    shifted. Averaging consecutive hours is exactly linear interpolation to the
    half hour, and dust varies slowly enough for that to be safe.
    """
    return df.rolling(2).mean().set_axis(df.index - pd.Timedelta("30min"), axis=0)


#: The CAMS global archive behind Open-Meteo begins here. Earlier dates return
#: HTTP 200 with a full time grid and every value null, which is a far more
#: dangerous failure than an error: a naive fetch silently yields an all-NaN
#: dust column and the dust model quietly becomes a no-op.
CAMS_ARCHIVE_START = "2022-08-04"


def fetch_dust_history(site: Site, start: str, end: str,
                       refresh: bool = False) -> pd.DataFrame:
    """Historical CAMS dust and AOD for a date range.

    This is what makes step 4 possible at all, and also what limits it: the
    archive starts in August 2022, so there is no aerosol record covering
    DKASC's seventeen-year power history. The dust model therefore cannot be
    trained on the same window as the step 1 model, and the ablation is run on
    the dust era alone -- which has the compensating virtue of being a properly
    controlled comparison: identical data and split, features the only change.
    """
    start = max(pd.Timestamp(start), pd.Timestamp(CAMS_ARCHIVE_START)).strftime(
        "%Y-%m-%d")
    params = {
        "latitude": site.latitude,
        "longitude": site.longitude,
        "hourly": "dust,aerosol_optical_depth",
        "domains": "cams_global",
        "timezone": site.tz,
        "start_date": start,
        "end_date": end,
    }
    key = f"dusthist_{site.key}_{start}_{end}"
    payload = _get(AIR_QUALITY_URL, params, key,
                   0 if refresh else 30 * 24 * 3600)
    df = _to_frame(payload, {"dust": "dust",
                             "aerosol_optical_depth": "aerosol_optical_depth"},
                   site.tz)
    df = df.dropna(how="all")
    if df.empty:
        raise RuntimeError(
            f"CAMS returned no dust data for {start}..{end}. The archive "
            f"starts {CAMS_ARCHIVE_START}; earlier requests succeed with all "
            f"values null."
        )
    return _centre_instantaneous(df)


def cache_status() -> pd.DataFrame:
    rows = []
    for path in sorted(CACHE.glob("*.json")):
        age = time.time() - path.stat().st_mtime
        rows.append({"file": path.name, "age_min": age / 60,
                     "kB": path.stat().st_size / 1024})
    return pd.DataFrame(rows)


def _to_frames_multi(payload, rename: dict, tz: str) -> list[pd.DataFrame]:
    """Open-Meteo returns a JSON list when several locations are requested."""
    entries = payload if isinstance(payload, list) else [payload]
    return [_to_frame(entry, rename, tz) for entry in entries]


def fetch_weather_multi(latitudes: list[float], longitudes: list[float],
                        tz: str, past_days: int = 60, forecast_days: int = 3,
                        refresh: bool = False, tag: str = "multi"
                        ) -> list[pd.DataFrame]:
    """Forecast weather for many locations in ONE request.

    Twenty-four governorates times two endpoints is forty-eight calls, which is
    both slow and an easy way to get rate limited off a free non-commercial
    tier. Open-Meteo accepts comma-separated coordinates and returns a list in
    the same order, so the whole country costs one request per endpoint.
    """
    params = {
        "latitude": ",".join(f"{v:.4f}" for v in latitudes),
        "longitude": ",".join(f"{v:.4f}" for v in longitudes),
        "hourly": ",".join(WEATHER_VARS),
        "wind_speed_unit": "ms",
        "timezone": tz,
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    key = f"weather_multi_{tag}_{len(latitudes)}_{past_days}_{forecast_days}"
    payload = _get(FORECAST_URL, params, key,
                   0 if refresh else CACHE_TTL_SECONDS)
    return [_centre_mixed(df)
            for df in _to_frames_multi(payload, WEATHER_VARS, tz)]


def fetch_dust_multi(latitudes: list[float], longitudes: list[float], tz: str,
                     past_days: int = 60, forecast_days: int = 5,
                     refresh: bool = False, tag: str = "multi"
                     ) -> list[pd.DataFrame]:
    """CAMS dust and AOD for many locations in one request."""
    params = {
        "latitude": ",".join(f"{v:.4f}" for v in latitudes),
        "longitude": ",".join(f"{v:.4f}" for v in longitudes),
        "hourly": "dust,aerosol_optical_depth",
        "domains": "cams_global",
        "timezone": tz,
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    key = f"dust_multi_{tag}_{len(latitudes)}_{past_days}_{forecast_days}"
    payload = _get(AIR_QUALITY_URL, params, key,
                   0 if refresh else CACHE_TTL_SECONDS)
    rename = {"dust": "dust", "aerosol_optical_depth": "aerosol_optical_depth"}
    return [_centre_instantaneous(df)
            for df in _to_frames_multi(payload, rename, tz)]
