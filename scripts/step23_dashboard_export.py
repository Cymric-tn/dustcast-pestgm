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

from dustcast import dashboard  # noqa: E402
from dustcast.config import ARTIFACTS  # noqa: E402



def main() -> int:
    payload = dashboard.build_payload()
    # The baked-in copy is the fallback the page uses when no API is reachable.
    payload["meta"] = {**payload["meta"], "source": "snapshot"}

    out = ARTIFACTS / "step6_dashboard.json"
    out.write_text(json.dumps(payload, separators=(",", ":")))
    m = payload["meta"]
    print(f"  wrote {out}  ({out.stat().st_size / 1024:.0f} kB)")
    print(f"  installed      {m['installed_mw']:.0f} MW, {m['units']} units")
    print(f"  model          {m['model']}")
    print(f"  issued         {m['issued_at']}")
    print(f"  band           {m['band_label']} (rho {m['spatial_rho']:.3f})")
    print(f"  national peak  {payload['national']['peak_mw']} MW "
          f"{payload['national']['peak_at']}")
    print("  this copy is the page's OFFLINE FALLBACK; when an API is reachable")
    print("  the page replaces it with a live fetch from /dashboard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
