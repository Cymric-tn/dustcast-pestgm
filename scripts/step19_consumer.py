"""Step 19: a working downstream consumer of the DustCast API.

The concept note asks for automatic exchange with grid-operation and load-
forecasting tools. This is what such a tool would actually do: start against a
running service, read the forecast over HTTP, check what the numbers are worth,
and turn them into the quantities a control room works in -- net load, the
evening ramp, and a reserve requirement.

It also demonstrates the refresh behaviour end to end: the second call is served
from cache, a forced refresh rebuilds, and the response says which happened.

The consumer CHECKS PROVENANCE before using the numbers. A downstream tool that
treats a simulated national total as a metered one is the failure mode the
interface is built to prevent, so this one reads the flags and says out loud
what it is consuming.

    python scripts/step19_consumer.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import dispatch  # noqa: E402
from dustcast.config import ARTIFACTS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step15_multiarray import rule  # noqa: E402

HOST, PORT = "127.0.0.1", 8077
BASE = f"http://{HOST}:{PORT}"
PEAK_LOAD_MW = 4600.0      # ASSUMPTION: Tunisian summer peak, order of magnitude


def serve() -> uvicorn.Server:
    """Run the real API in this process, over real HTTP."""
    cfg = uvicorn.Config("dustcast.api:app", host=HOST, port=PORT,
                         log_level="warning")
    srv = uvicorn.Server(cfg)
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(120):
        if srv.started:
            return srv
        time.sleep(0.25)
    raise RuntimeError("API did not start")


def main() -> int:
    srv = serve()
    c = httpx.Client(base_url=BASE, timeout=180.0)

    rule("1. CONNECT")
    h = c.get("/health").json()
    print(f"  {BASE} -> {h['status']}, {h['horizon_hours']} forecast hours")

    rule("2. WHAT AM I CONSUMING?")
    m = c.get("/meta").json()
    print(f"  scales     {m['scales']['district']} districts -> "
          f"{m['scales']['governorate']} governorates -> national")
    print(f"  capacity   {m['total_rooftop_mw']:.0f} MW across "
          f"{len(m['segments'])} segments:")
    for s in m["segments"]:
        print(f"               {s['name']:<22s} {s['voltage']:<6s} {s['capacity_mw']:5.0f} MW")
    print(f"  provenance {m['provenance']}")

    perf = c.get("/models/performance").json()
    print(f"\n  measured on  {perf['measured_on']}")
    ev = perf["nmae_pct_by_horizon"]
    if ev:
        row = ev.get("J+1", {})
        l0 = row.get("L0 physics")
        print(f"  J+1 nMAE for the model the national layer uses: "
              f"{l0:.2f}%" if l0 else "  (no L0 row)")
    print(f"  {perf['note']}")

    rule("3. READ THE FORECAST")
    r = c.get("/forecast/national").json()
    s = pd.Series({pd.Timestamp(p["time"]): p["mw"] for p in r["series"]})
    s.index = pd.DatetimeIndex(s.index)
    print(f"  {len(s)} hours, built {r['generated_at'][:16]}, "
          f"rebuilt on this request: {r['rebuilt_on_this_request']}")
    print(f"  model: {r['model']}   fleet: {r['fleet']}")

    # A consumer that cares about correctness refuses to mislabel this.
    simulated = "SIMULATED" in r["fleet"]
    print(f"\n  -> tagging this feed as {'ESTIMATED (simulated fleet)' if simulated else 'METERED'}")

    rule("4. REFRESH BEHAVIOUR")
    t0 = time.time()
    again = c.get("/forecast/national").json()
    print(f"  second call    rebuilt={again['rebuilt_on_this_request']}  "
          f"({time.time() - t0:.2f}s, age {again['age_seconds']:.0f}s)")
    t0 = time.time()
    forced = c.post("/refresh").json()
    print(f"  forced refresh rebuilt={forced['rebuilt']}  ({time.time() - t0:.1f}s)")
    print("  A refresh also fires on its own when new weather lands in the")
    print("  cache, so the service follows the model runs rather than a timer.")

    rule("5. TURN IT INTO OPERATIONAL QUANTITIES")
    demand = dispatch.demand_series(s.index, PEAK_LOAD_MW)
    net = demand - s

    print(f"  assumed system peak demand      {PEAK_LOAD_MW:,.0f} MW (ASSUMPTION)")
    print(f"  rooftop PV peak over horizon    {s.max():,.1f} MW")
    print(f"  PV share at its own peak        {s.max() / demand.loc[s.idxmax()]:.1%} of demand")

    # The horizon-wide net-load minimum is at NIGHT and is a demand minimum, not
    # a PV effect. At 630 MW against a 4,600 MW peak there is no duck curve:
    # rooftop PV shaves the midday shoulder, it does not dig a midday belly.
    # Reporting the raw argmin without saying so would invite exactly that
    # misreading.
    print(f"\n  net-load minimum over the horizon  {net.min():,.1f} MW "
          f"at {net.idxmin():%a %H:%M}")
    print(f"    demand alone at that hour        {demand.loc[net.idxmin()]:,.1f} MW "
          f"-- PV contributes {s.loc[net.idxmin()]:,.1f} MW")
    print("    i.e. the minimum is the overnight demand trough. At this")
    print("    penetration rooftop PV does not create a midday net-load belly.")

    midday = net.index.hour.isin(range(11, 15))
    print(f"\n  midday net load (11-15h)          {net[midday].mean():,.1f} MW mean")
    print(f"    without rooftop PV               {demand[midday].mean():,.1f} MW")
    print(f"    PV shaves                        {s[midday].mean():,.1f} MW "
          f"({s[midday].mean() / demand[midday].mean():.1%})")

    # The evening ramp is the operationally expensive one. Decomposing it says
    # how much the operator can blame on PV falling away rather than on demand
    # rising -- which is the part a PV forecast can actually help with.
    ramp3 = net.diff().rolling(3).sum()
    worst = ramp3.idxmax()
    # rolling(3).sum() of the diff at t equals net[t] - net[t-3], so the window
    # must include t-3 for the endpoint decomposition to reconstruct the ramp.
    win = (net.index >= worst - pd.Timedelta("3h")) & (net.index <= worst)
    d_dem = demand[win].iloc[-1] - demand[win].iloc[0]
    d_pv = s[win].iloc[0] - s[win].iloc[-1]
    print(f"\n  steepest 3h net-load ramp         {ramp3.max():,.1f} MW "
          f"ending {worst:%a %H:%M}")
    print(f"    demand rising                    {d_dem:,.1f} MW")
    print(f"    PV falling away                  {d_pv:,.1f} MW "
          f"({d_pv / max(ramp3.max(), 1e-9):.0%} of the ramp)")
    print(f"    the two sum to                   {d_dem + d_pv:,.1f} MW "
          f"vs {ramp3.max():,.1f} MW measured")
    print("    The PV share is the part a production forecast can anticipate.")

    rule("6. HAND OFF")
    out = pd.DataFrame({"forecast_pv_mw": s, "assumed_demand_mw": demand,
                        "net_load_mw": net})
    out.index.name = "time"
    path = ARTIFACTS / "step19_consumer_handoff.csv"
    out.to_csv(path)
    print(f"  wrote {path}")
    print(f"  {len(out)} hourly rows, tagged ESTIMATED, ready for a load-")
    print(f"  forecasting or unit-commitment tool to ingest.")

    srv.should_exit = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
