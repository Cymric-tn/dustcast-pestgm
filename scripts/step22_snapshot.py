"""Step 22: freeze ONE forecast snapshot that every deliverable quotes.

The report, the chart and the API example had drifted apart -- different runs,
different weather windows, different numbers, and at one point a hand-written
"example" response whose values did not match anything the service returns.

This writes a single snapshot with its issue time and configuration. The report
figures, the quoted national statistics and the verbatim API example are all
generated from it, so they cannot disagree. Regenerate it, then regenerate the
report.

    python scripts/step22_snapshot.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import fleet, service, tunisia  # noqa: E402
from dustcast.config import ARTIFACTS  # noqa: E402

SNAPSHOT = ARTIFACTS / "step22_snapshot.json"


def main() -> int:
    bundle, _ = service.get_forecast(force=True)
    issued = bundle.generated_at
    nat, rng = bundle.national, bundle.national_range

    # The issue time splits the series. Hours before it are CONTEXT recomputed
    # from archived weather -- a fresh estimate for an elapsed hour is not an
    # operational forecast of that hour, and labelling it one would be wrong.
    future = nat.index >= issued.floor("1h")
    fut = nat[future]
    peak_t = fut.idxmax()
    half = (rng.loc[peak_t] / 2.0) if rng is not None else None

    snap = {
        "issued_at": issued.isoformat(),
        "timezone": service.TZ,
        "model": bundle.model,
        "config": {
            "fleet_mw": tunisia.rooftop_total_mw(),
            "segments": {s.key: s.national_mw for s in fleet.SEGMENTS},
            "districts": len(tunisia.districts()),
            "governorates": len(tunisia.GOVERNORATES),
            "archetypes_per_segment": service.ARCHETYPES_PER_SEGMENT,
            "spatial_rho": bundle.spatial_rho,
        },
        "context_hours": int((~future).sum(),),
        "forecast_hours": int(future.sum()),
        "peak": {
            "time": peak_t.isoformat(),
            "mw": round(float(fut.loc[peak_t]), 1),
            "range_mw": None if half is None else round(float(half * 2), 1),
            "half_width_mw": None if half is None else round(float(half), 1),
        },
        "series": [
            {"time": t.isoformat(),
             "mw": round(float(v), 4),
             "is_forecast": bool(t >= issued.floor("1h")),
             **({} if rng is None or t not in rng.index else {
                 "low_illustrative": round(max(float(v) - rng.loc[t] / 2, 0.0), 4),
                 "high_illustrative": round(float(v) + rng.loc[t] / 2, 4)})}
            for t, v in nat.items()
        ],
    }
    SNAPSHOT.write_text(json.dumps(snap, indent=1))

    print(f"  issued at        {issued:%Y-%m-%d %H:%M %Z}")
    print(f"  context hours    {snap['context_hours']} (before issue time, "
          f"recomputed from archived weather -- NOT forecasts)")
    print(f"  forecast hours   {snap['forecast_hours']}")
    print(f"  fleet            {snap['config']['fleet_mw']:.0f} MW across "
          f"{len(snap['config']['segments'])} segments, "
          f"{snap['config']['districts']} units")
    print(f"  rho              {snap['config']['spatial_rho']:.3f}")
    p = snap["peak"]
    print(f"  forecast peak    {p['mw']} MW at {p['time'][:16]}, "
          f"range {p['range_mw']} MW (+/-{p['half_width_mw']})")
    print(f"\n  wrote {SNAPSHOT}")

    # A real response, captured verbatim -- never hand-written.
    example = ARTIFACTS / "step22_api_example.txt"
    try:
        from fastapi.testclient import TestClient

        from dustcast.api import app
        j = TestClient(app).get("/forecast/national").json()
        peak_row = max((r for r in j["series"] if r.get("high_illustrative")),
                       key=lambda r: r["mw"])
        # Wrapped to fit the report's text block; a verbatim block does not
        # wrap itself and silently runs off the page.
        text = (
            "$ curl -s localhost:8000/forecast/national\n"
            "{ \"generated_at\": \"%s\",\n"
            "  \"units\": \"%s\", \"model\": \"%s\",\n"
            "  \"fleet\": \"%s\",\n"
            "  \"evaluated_horizons\": %s,\n"
            "  \"uncertainty\": {\n"
            "    \"status\": \"%s\",\n"
            "    \"caveat\": \"%s...\",\n"
            "    \"spatial_correlation_rho\": %s },\n"
            "  \"series\": [ { \"time\": \"%s\",\n"
            "               \"mw\": %s,\n"
            "               \"low_illustrative\": %s,\n"
            "               \"high_illustrative\": %s }, ... ] }\n"
        ) % (j["generated_at"][:19] + "+01:00", j["units"], j["model"],
             j["fleet"], json.dumps(j["evaluated_horizons"]),
             j["uncertainty"]["status"], j["uncertainty"]["caveat"][:46],
             j["uncertainty"]["spatial_correlation_rho"],
             peak_row["time"], peak_row["mw"],
             peak_row["low_illustrative"], peak_row["high_illustrative"])
        example.write_text(text)
        # Also beside the report, where LaTeX \verbatiminput can reach it.
        (Path(__file__).resolve().parent.parent / "report" / "api_example.txt"
         ).write_text(text)
        print(f"  wrote {example} (captured from a live response)")
        print("  wrote report/api_example.txt")
    except Exception as exc:                       # pragma: no cover
        print(f"  could not capture API example: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
