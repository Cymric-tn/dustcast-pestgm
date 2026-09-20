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


HOST, PORT = "127.0.0.1", 8123


def serve():
    """Run the real API so the screenshot shows the page in its LIVE state.

    Opened as a file the page has no API to talk to and falls back to its baked-in
    copy, which would put "snapshot" in the masthead of a figure whose whole point
    is that the dashboard is a live client.
    """
    import threading
    import time

    import uvicorn

    cfg = uvicorn.Config("dustcast.api:app", host=HOST, port=PORT,
                         log_level="warning")
    srv = uvicorn.Server(cfg)
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(160):
        if srv.started:
            return srv
        time.sleep(0.25)
    return None


def main() -> int:
    from playwright.sync_api import sync_playwright

    snap = json.loads((ARTIFACTS / "step22_snapshot.json").read_text())
    srv = serve()
    url = f"http://{HOST}:{PORT}/" if srv else PAGE.as_uri()
    print(f"  rendering {url}")
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 1500, "height": 980},
                        device_scale_factor=2)
        pg.goto(url)
        # The first /dashboard call builds the payload; wait for the live fetch
        # to land rather than photographing the fallback.
        try:
            pg.wait_for_function("typeof M !== 'undefined' && M.source === 'api'",
                                 timeout=120_000)
        except Exception:
            print("  WARNING: live fetch did not land; capturing fallback state")
        pg.wait_for_timeout(1500)
        errors = pg.evaluate("window.__err || null")
        pg.screenshot(path=str(OUT), clip={"x": 0, "y": 0,
                                           "width": 1500, "height": 900})
        shown = pg.evaluate(
            "JSON.stringify({installed: document.body.innerText.match"
            "(/([0-9]+) MW installed/)?.[1], issued: document.body.innerText"
            ".match(/issued ([0-9-]+ [0-9:]+)/)?.[1], feed: (typeof M !== "
            "'undefined' ? M.source : null)})")
        b.close()
    if srv:
        srv.should_exit = True

    # Record what the RENDERED page showed, so the build can verify the figure
    # in the report agrees with the snapshot rather than trusting that it does.
    (ARTIFACTS / "step24_rendered.json").write_text(shown)
    print(f"  wrote {OUT} ({OUT.stat().st_size / 1024:.0f} kB)")
    print(f"  page reports: {shown}")
    print(f"  snapshot says: installed {snap['config']['fleet_mw']:.0f} MW, "
          f"issued {snap['issued_at'][:16]}")
    if errors:
        print(f"  page errors: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
