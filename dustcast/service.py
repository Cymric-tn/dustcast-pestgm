"""Forecast service: builds the multi-scale forecast and keeps it fresh.

One code path serves the scripts, the API and the dashboard, so the number in
the report and the number on the endpoint cannot drift apart.

REFRESH POLICY. Open-Meteo publishes a new model run roughly hourly. The service
does not poll on a timer and hope: it watches the mtime of the weather cache
that `live` writes, and rebuilds when that file is newer than the forecast built
from it, or when the forecast is older than MAX_AGE regardless. So a refresh is
triggered by weather actually arriving, not by the clock alone.

MODEL CHOICE. The Tunisian fleet is simulated and has no measured production, so
no model that learns from observed output can be fitted to it -- the service runs
L0 physics nationally and says so in every payload. The learning loop and the
competing models run where there ARE observations (DKASC), and their evaluated
performance is served alongside, never merged into the national numbers.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import fleet, learning, live, physics, tunisia
from .config import ARTIFACTS

TZ = "Africa/Tunis"
ARCHETYPES_PER_SEGMENT = 5
MAX_AGE_SECONDS = 3600
MODEL_NATIONAL = "L0 physics"

_lock = threading.Lock()
_cached: "ForecastBundle | None" = None


@dataclass
class ForecastBundle:
    generated_at: pd.Timestamp
    weather_stamp: float           # mtime of the weather cache it was built from
    districts: pd.DataFrame        # hours x district
    governorates: pd.DataFrame     # hours x governorate
    national: pd.Series            # hours
    model: str = MODEL_NATIONAL
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def horizon_hours(self) -> int:
        return len(self.national)

    def age_seconds(self) -> float:
        return (pd.Timestamp.now(tz=TZ) - self.generated_at).total_seconds()


def _weather_cache_stamp() -> float:
    """Newest mtime among the cached weather payloads."""
    cache = Path(live.CACHE)
    stamps = [p.stat().st_mtime for p in cache.glob("weather_multi_tn*.json")]
    return max(stamps) if stamps else 0.0


def _segment_yield(gov, segment, weather, seed) -> pd.Series | None:
    archs = fleet.sample_segment_archetypes(
        gov.key, segment, gov.latitude, gov.longitude, 50.0, TZ,
        n=ARCHETYPES_PER_SEGMENT, seed=seed)
    l0 = learning.PhysicsOnly()
    parts, cap_kw = [], 0.0
    for a in archs:
        f = physics.build_physics_frame(a.site, weather, clearsky_scale=None)
        f = f[f["expected_kw"].notna()]
        if f.empty:
            continue
        parts.append(l0.predict(f, a.site))
        cap_kw += a.site.dc_capacity_w / 1000.0
    return None if not parts else sum(parts) / cap_kw


def build_forecast(refresh: bool = False, forecast_days: int = 7) -> ForecastBundle:
    """Compute district, governorate and national MW from current weather."""
    dists = tunisia.districts()
    weathers = live.fetch_weather_multi(
        tunisia.latitudes(), tunisia.longitudes(), TZ,
        past_days=2, forecast_days=forecast_days, refresh=refresh, tag="tn")

    yields: dict[tuple[str, str], pd.Series] = {}
    for i, gov in enumerate(tunisia.GOVERNORATES):
        for j, seg in enumerate(fleet.SEGMENTS):
            y = _segment_yield(gov, seg, weathers[i], seed=i * 10 + j)
            if y is not None:
                yields[(gov.key, seg.key)] = y

    idx = next(iter(yields.values())).index
    cols = {}
    for d in dists:
        acc = pd.Series(0.0, index=idx)
        for skey, share in d.segment_mix.items():
            y = yields.get((d.governorate, skey))
            if y is not None:
                acc = acc.add(y.reindex(idx).fillna(0.0) * share, fill_value=0.0)
        cols[d.key] = acc * d.capacity_mw

    D = pd.DataFrame(cols)
    G = D.T.groupby({d.key: d.governorate for d in dists}).sum().T
    return ForecastBundle(
        generated_at=pd.Timestamp.now(tz=TZ),
        weather_stamp=_weather_cache_stamp(),
        districts=D, governorates=G, national=G.sum(axis=1),
        notes=(
            "Simulated fleet: Tunisia publishes no installation registry.",
            "L0 physics only -- no Tunisian site has measured production to "
            "fit a learned correction to.",
            "Districts share their governorate's weather; the NWP grid is "
            "coarser than a delegation.",
        ),
    )


def get_forecast(force: bool = False) -> tuple[ForecastBundle, bool]:
    """Cached forecast, rebuilt when new weather has landed or it has aged out.

    Returns (bundle, rebuilt). Thread-safe: the API may be handling concurrent
    requests, and building twice in parallel would double the API calls to
    Open-Meteo for no benefit.
    """
    global _cached
    with _lock:
        stale = (
            force
            or _cached is None
            or _cached.age_seconds() > MAX_AGE_SECONDS
            or _weather_cache_stamp() > _cached.weather_stamp
        )
        if stale:
            _cached = build_forecast(refresh=force)
            return _cached, True
        return _cached, False


def evaluated_performance() -> pd.DataFrame:
    """Measured error by horizon, from the frozen evaluation (step 17).

    Served next to the forecast so a consumer can see what the numbers are worth
    rather than having to trust them. Empty if step 17 has not been run.
    """
    p = ARTIFACTS / "step17_horizons.csv"
    return pd.read_csv(p, index_col=0) if p.exists() else pd.DataFrame()


def selection_history() -> pd.DataFrame:
    """Which model the learning loop picked each month (step 16)."""
    p = ARTIFACTS / "step16_replay.csv"
    return pd.read_csv(p, index_col=0) if p.exists() else pd.DataFrame()
