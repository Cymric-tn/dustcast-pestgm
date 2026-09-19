"""Step 23: build the dashboard payload from the CURRENT service.

The dashboard had been left on the step-5 pipeline: 500 MW residential only, a
superseded model, and interval labels the evidence no longer supports. It
therefore showed a different system from the one the report describes.

This rebuilds the payload from the same snapshot the report and the API example
come from, so capacity, model, issue time and uncertainty wording agree across
all four deliverables.

Two deliberate changes from the old payload:

  * The band is the ILLUSTRATIVE range (dustcast.uncertainty), carried with its
    caveat. It is not called a 90% interval.
  * The "dust model off" ablation is GONE. The national layer runs L0 physics on
    Open-Meteo irradiance, which already embeds aerosol effects; there is no
    dust knob to switch off, so offering the toggle would claim a capability the
    pipeline does not have. Dust and AOD remain on the map as observed context.

    python scripts/step23_dashboard_export.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import (decisions, features, fleet, live, physics,  # noqa: E402
                      service, tunisia)
from dustcast.config import ARTIFACTS  # noqa: E402

TZ = service.TZ
HISTORY_DAYS = 2


def main() -> int:
    snap = json.loads((ARTIFACTS / "step22_snapshot.json").read_text())
    issued = pd.Timestamp(snap["issued_at"])
    bundle, _ = service.get_forecast()

    nat, G, rng = bundle.national, bundle.governorates, bundle.national_range
    idx = nat.index
    half = (rng / 2.0) if rng is not None else pd.Series(0.0, index=idx)

    dusts = live.fetch_dust_multi(tunisia.latitudes(), tunisia.longitudes(),
                                  TZ, past_days=60, forecast_days=5, tag="tn")
    weathers = live.fetch_weather_multi(tunisia.latitudes(), tunisia.longitudes(),
                                        TZ, past_days=2, forecast_days=7, tag="tn")
    dists = tunisia.districts()
    cap_by_gov = {g.key: sum(d.capacity_mw for d in dists if d.governorate == g.key)
                  for g in tunisia.GOVERNORATES}

    def times(ix):
        return [t.strftime("%Y-%m-%dT%H:%M") for t in ix]

    fut = nat[idx >= issued.floor("1h")]
    now_i = int(max(0, idx.searchsorted(issued.floor("1h"))))
    reserve = float(half.loc[fut.index].max()) if len(fut) else 0.0
    ramp = decisions.ramp_risk(fut, window="3h") if len(fut) else {
        "max_down_kw": 0.0, "max_up_kw": 0.0}

    payload = {
        "meta": {
            "generated": issued.strftime("%Y-%m-%d %H:%M %Z"),
            "issued_at": issued.strftime("%Y-%m-%dT%H:%M"),
            "horizon_hours": int(len(fut)),
            "now": idx[min(now_i, len(idx) - 1)].strftime("%Y-%m-%dT%H:%M"),
            "now_index": now_i,
            "history_days": HISTORY_DAYS,
            "installed_mw": float(snap["config"]["fleet_mw"]),
            "segments": snap["config"]["segments"],
            "units": snap["config"]["districts"],
            "confidence": 0.90,
            "band_label": "illustrative range",
            "band_caveat": service.UNCERTAINTY_CAVEAT,
            "model": bundle.model,
            "spatial_rho": snap["config"]["spatial_rho"],
            "trained_on": "L0 physics; no Tunisian production data exists to "
                          "fit a local correction to",
        },
        "national": {
            "t": times(idx),
            "point": [round(v, 2) for v in nat],
            "lower": [round(max(v - h, 0.0), 2) for v, h in zip(nat, half)],
            "upper": [round(v + h, 2) for v, h in zip(nat, half)],
            # The dust ablation no longer exists (see module docstring). The
            # field is kept equal to the point forecast so the chart code has an
            # array to slice, and the control that would reveal it is hidden.
            "dust_off": [round(v, 2) for v in nat],
            "peak_mw": round(float(fut.max()), 1) if len(fut) else 0.0,
            "peak_at": fut.idxmax().strftime("%a %H:%M") if len(fut) else "",
            "energy_mwh": round(float(fut.sum()), 0) if len(fut) else 0.0,
        },
        "weather": {},
        "decisions": {
            "reserve_mw": round(reserve, 1),
            "ramp_down_mw": round(float(ramp["max_down_kw"]), 0),
            "ramp_up_mw": round(float(ramp["max_up_kw"]), 0),
        },
        "governorates": [],
    }

    wsum, total_cap = {k: None for k in ("ghi", "clearsky_ghi", "temp_air", "aod")}, 0.0
    for i, gov in enumerate(tunisia.GOVERNORATES):
        # The clear-sky envelope is what makes a dip in the power curve legible
        # rather than mysterious, and it comes from the physics layer, not from
        # the weather API. Computed on a single probe archetype per governorate.
        probe = fleet.sample_segment_archetypes(
            gov.key, fleet.SEGMENTS[0], gov.latitude, gov.longitude, 50.0, TZ,
            n=1, seed=i)[0].site
        w = physics.complete_weather(probe, weathers[i]).reindex(idx)
        d = dusts[i].reindex(idx)
        cap = cap_by_gov[gov.key]
        total_cap += cap
        cols = {"ghi": w.get("ghi"), "clearsky_ghi": w.get("clearsky_ghi"),
                "temp_air": w.get("temp_air"), "aod": d.get("aerosol_optical_depth")}
        for k, col in cols.items():
            if col is None:
                continue
            v = col.fillna(0).to_numpy() * cap
            wsum[k] = v if wsum[k] is None else wsum[k] + v

        share = (G[gov.key] / nat.replace(0, pd.NA)).fillna(0.0)
        g_half = half * share
        aod = d.get("aerosol_optical_depth", pd.Series(0.0, index=idx)).fillna(0)
        dust = d.get("dust", pd.Series(0.0, index=idx)).fillna(0)
        soil = features.soiling_accumulation(
            dust, w.get("precipitation", pd.Series(0.0, index=idx)).fillna(0))
        advice = decisions.cleaning_recommendation(
            soil.loc[:issued], w.get("precipitation",
                                     pd.Series(0.0, index=idx)).loc[issued:].fillna(0))

        payload["governorates"].append({
            "key": gov.key, "name": gov.name, "region": gov.region,
            "lat": gov.latitude, "lon": gov.longitude,
            "capacity_mw": round(cap, 2),
            "peak_mw": round(float(G[gov.key].loc[fut.index].max()) if len(fut) else 0.0, 2),
            "aod": round(float(aod.loc[fut.index].mean()) if len(fut) else 0.0, 3),
            "dust": round(float(dust.loc[fut.index].mean()) if len(fut) else 0.0, 1),
            "aod_series": [round(v, 3) for v in aod],
            "ghi": [round(float(v), 0) for v in w.get("ghi", pd.Series(0.0, index=idx)).fillna(0)],
            "clearsky_ghi": [round(float(v), 0) for v in
                             w.get("clearsky_ghi", pd.Series(0.0, index=idx)).fillna(0)],
            "temp_air": [round(float(v), 1) for v in
                         w.get("temp_air", pd.Series(0.0, index=idx)).fillna(0)],
            "soiling_pct": round(float(soil.iloc[-1] * 100), 2),
            "clean_now": bool(advice.recommend),
            "clean_reason": advice.reason,
            "t": times(idx),
            "dust_off": [round(v, 3) for v in G[gov.key]],
            "point": [round(v, 3) for v in G[gov.key]],
            "lower": [round(max(v - h, 0.0), 3) for v, h in zip(G[gov.key], g_half)],
            "upper": [round(v + h, 3) for v, h in zip(G[gov.key], g_half)],
        })

    if total_cap:
        payload["weather"] = {k: [round(float(v / total_cap), 2 if k == "aod" else 1)
                                  for v in arr]
                              for k, arr in wsum.items() if arr is not None}

    out = ARTIFACTS / "step6_dashboard.json"
    out.write_text(json.dumps(payload, separators=(",", ":")))
    m = payload["meta"]
    print(f"  wrote {out}  ({out.stat().st_size / 1024:.0f} kB)")
    print(f"  installed      {m['installed_mw']:.0f} MW across "
          f"{len(m['segments'])} segments, {m['units']} units")
    print(f"  model          {m['model']}")
    print(f"  issued         {m['issued_at']}")
    print(f"  band           {m['band_label']} (rho {m['spatial_rho']:.3f})")
    print(f"  national peak  {payload['national']['peak_mw']} MW "
          f"{payload['national']['peak_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
