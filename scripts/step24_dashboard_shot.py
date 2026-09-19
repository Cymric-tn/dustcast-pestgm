"""Step 24: capture the dashboard for the report.

Rendered from the file the demo will serve, at the moment the snapshot was
issued, so the screenshot in the report shows the same system, capacity, model
and issue time as every other deliverable.

    python scripts/step24_dashboard_shot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dustcast.config import ARTIFACTS  # noqa: E402

PAGE = ROOT / "dashboard" / "dustcast.html"
OUT = ROOT / "report" / "fig_dashboard.png"


def main() -> int:
    from playwright.sync_api import sync_playwright

    snap = json.loads((ARTIFACTS / "step22_snapshot.json").read_text())
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 1500, "height": 980},
                        device_scale_factor=2)
        pg.goto(PAGE.as_uri())
        pg.wait_for_timeout(2500)
        errors = pg.evaluate("window.__err || null")
        pg.screenshot(path=str(OUT), clip={"x": 0, "y": 0,
                                           "width": 1500, "height": 900})
        shown = pg.evaluate(
            "JSON.stringify({installed: document.body.innerText.match"
            "(/([0-9]+) MW installed/)?.[1], issued: document.body.innerText"
            ".match(/issued ([0-9-]+ [0-9:]+)/)?.[1]})")
        b.close()

    print(f"  wrote {OUT} ({OUT.stat().st_size / 1024:.0f} kB)")
    print(f"  page reports: {shown}")
    print(f"  snapshot says: installed {snap['config']['fleet_mw']:.0f} MW, "
          f"issued {snap['issued_at'][:16]}")
    if errors:
        print(f"  page errors: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
