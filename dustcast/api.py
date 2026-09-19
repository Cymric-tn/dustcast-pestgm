"""HTTP API for exchange with grid-operation and load-forecasting tools.

The concept note asks for automatic exchange with the operator's existing tools.
A real STEG connection is out of scope, so this is the interface a downstream
tool would consume, documented and working, with a sample consumer in
`scripts/step19_consumer.py`.

Two design choices worth defending:

  PROVENANCE TRAVELS WITH THE NUMBERS. Every forecast payload carries the model
  that produced it, whether the fleet is simulated, and whether the horizon has
  a measured error. A downstream tool that silently treats a simulated national
  total as a metered one is a failure mode this interface is built to prevent.

  FRESHNESS IS EXPLICIT. Responses report when the forecast was built and
  whether that request rebuilt it, so a consumer can tell a fresh number from a
  cached one without guessing.

    .venv/bin/uvicorn dustcast.api:app --port 8000
"""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from . import fleet, service, tunisia

app = FastAPI(
    title="DustCast",
    summary="Rooftop PV production forecasts for Tunisia at district, "
            "governorate and national scale.",
    version="0.1.0",
)

GOV_KEYS = {g.key: g for g in tunisia.GOVERNORATES}


def _series_payload(s: pd.Series, start: str | None, end: str | None,
                    rng: pd.Series | None = None) -> list[dict]:
    """Time series, optionally with the illustrative range around each point.

    The range is emitted as explicit low/high fields named `*_illustrative` so a
    consumer cannot mistake them for calibrated interval bounds. The caveat
    itself travels in the envelope.
    """
    if start:
        s = s[s.index >= pd.Timestamp(start, tz=service.TZ)]
    if end:
        s = s[s.index <= pd.Timestamp(end, tz=service.TZ)]
    out = []
    for t, v in s.items():
        row = {"time": t.isoformat(), "mw": round(float(v), 4)}
        if rng is not None and t in rng.index:
            half = float(rng.loc[t]) / 2.0
            row["low_illustrative"] = round(max(float(v) - half, 0.0), 4)
            row["high_illustrative"] = round(float(v) + half, 4)
        out.append(row)
    return out


def _envelope(bundle: service.ForecastBundle, rebuilt: bool, **extra) -> dict:
    ev = service.evaluated_performance()
    return {
        "generated_at": bundle.generated_at.isoformat(),
        "rebuilt_on_this_request": rebuilt,
        "age_seconds": round(bundle.age_seconds(), 1),
        "timezone": service.TZ,
        "units": "MW (AC)",
        "model": bundle.model,
        "fleet": "SIMULATED -- no installation registry available to this project",
        "evaluated_horizons": list(ev.index) if not ev.empty else [],
        "uncertainty": {
            "status": "ILLUSTRATIVE -- not a validated interval",
            "caveat": service.UNCERTAINTY_CAVEAT,
            "spatial_correlation_rho": (None if bundle.spatial_rho is None
                                        else round(bundle.spatial_rho, 4)),
            "rho_source": "mean pairwise correlation of FORECAST CLEAR-SKY "
                          "INDEX across governorates, used as a proxy for "
                          "forecast-error correlation",
        },
        "caveats": list(bundle.notes),
        **extra,
    }


@app.get("/health")
def health() -> dict:
    bundle, rebuilt = service.get_forecast()
    return {"status": "ok", "generated_at": bundle.generated_at.isoformat(),
            "rebuilt_on_this_request": rebuilt,
            "horizon_hours": bundle.horizon_hours}


@app.get("/meta")
def meta() -> dict:
    """Scope, scales and fleet composition -- what this service does and does not cover."""
    dists = tunisia.districts()
    return {
        "scales": {
            "district": len(dists),
            "governorate": len(tunisia.GOVERNORATES),
            "national": 1,
        },
        "segments": [
            {"key": s.key, "name": s.name, "voltage": s.voltage,
             "capacity_mw": s.national_mw,
             "unit_capacity_kw_range": list(s.cap_range)}
            for s in fleet.SEGMENTS
        ],
        "total_rooftop_mw": tunisia.rooftop_total_mw(),
        "district_weather": "shared within a governorate -- a simplification we "
                            "chose to keep the service to 24 grid requests, "
                            "not a limit of the ~11 km forecast grid",
        "district_names": "not modelled -- only delegation counts are",
        "district_status": "PROTOTYPE aggregation units. Correspondence to "
                           "STEG operational units (substations, MV feeders) "
                           "is NOT established",
        "provenance": "capacity split and fleet composition are simulated "
                      "assumptions; no installation-level registry was "
                      "available to this project",
    }


@app.get("/forecast/national")
def national(start: str | None = None, end: str | None = None,
             refresh: bool = Query(False, description="force a rebuild")) -> dict:
    bundle, rebuilt = service.get_forecast(force=refresh)
    return _envelope(bundle, rebuilt, scale="national",
                     series=_series_payload(bundle.national, start, end,
                                            bundle.national_range))


@app.get("/forecast/governorate/{key}")
def governorate(key: str, start: str | None = None, end: str | None = None) -> dict:
    bundle, rebuilt = service.get_forecast()
    if key not in bundle.governorates.columns:
        raise HTTPException(404, f"unknown governorate '{key}'; "
                                 f"see /meta and /forecast/governorates")
    g = GOV_KEYS[key]
    return _envelope(bundle, rebuilt, scale="governorate", key=key, name=g.name,
                     region=g.region,
                     series=_series_payload(bundle.governorates[key], start, end))


@app.get("/forecast/governorates")
def governorates() -> dict:
    """Every governorate at once -- the shape a map or a load-forecasting tool wants."""
    bundle, rebuilt = service.get_forecast()
    peak = bundle.national.idxmax()
    return _envelope(
        bundle, rebuilt, scale="governorate",
        national_peak_time=peak.isoformat(),
        regions=[{"key": k, "name": GOV_KEYS[k].name,
                  "mw_at_national_peak": round(float(bundle.governorates.loc[peak, k]), 3)}
                 for k in bundle.governorates.columns])


@app.get("/forecast/district/{key}")
def district(key: str, start: str | None = None, end: str | None = None) -> dict:
    bundle, rebuilt = service.get_forecast()
    if key not in bundle.districts.columns:
        raise HTTPException(404, f"unknown district '{key}'")
    d = next(x for x in tunisia.districts() if x.key == key)
    return _envelope(bundle, rebuilt, scale="district", key=key, name=d.name,
                     governorate=d.governorate,
                     capacity_mw=round(d.capacity_mw, 4),
                     segment_mix={k: round(v, 4) for k, v in d.segment_mix.items()},
                     series=_series_payload(bundle.districts[key], start, end))


@app.get("/models/performance")
def performance() -> dict:
    """Measured error by horizon, and what it does and does not cover."""
    ev = service.evaluated_performance()
    return {
        "measured_on": "DKASC Alice Springs -- an Australian desert site, not Tunisia",
        "note": "Horizons absent here are PRODUCED but NOT validated. The "
                "national layer runs L0 physics, so L0's row is the relevant one.",
        "nmae_pct_by_horizon": ({} if ev.empty
                                else ev.round(2).to_dict(orient="index")),
    }


@app.get("/models/selection")
def selection() -> dict:
    """What the learning loop actually chose, month by month."""
    sel = service.selection_history()
    if sel.empty:
        return {"available": False, "reason": "run scripts/step16_replay.py"}
    return {
        "available": True,
        "protocol": "every candidate refitted on past data each month; the "
                    "winner on a held-back 42-day window forecasts the next "
                    "month blind",
        "months": len(sel),
        "selected_counts": sel["selected"].value_counts().to_dict(),
        "history": [{"period": str(i), "selected": r["selected"]}
                    for i, r in sel.iterrows()],
    }


@app.post("/refresh")
def refresh() -> dict:
    """Force a rebuild from a fresh weather pull."""
    bundle, rebuilt = service.get_forecast(force=True)
    return {"rebuilt": rebuilt, "generated_at": bundle.generated_at.isoformat(),
            "horizon_hours": bundle.horizon_hours}
