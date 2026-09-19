"""Regenerate every deliverable from ONE snapshot, in the only correct order.

The snapshot, the dashboard, the screenshot, the report figures and the report
text all quote the same forecast. Running these steps out of order leaves them
disagreeing -- which happened repeatedly, and is exactly the inconsistency the
reviews kept finding. The order is a dependency chain, so it lives in code:

    step22  take the snapshot            (the single source of truth)
    step23  dashboard payload            <- snapshot
    step6b  inline payload into the page <- payload
    step24  screenshot the page          <- page
    step20  figures + LaTeX macros       <- snapshot
    tectonic                             <- figures, macros, screenshot

    python scripts/build_report.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")

STEPS = [
    ("snapshot", [PY, "scripts/step22_snapshot.py"]),
    ("dashboard payload", [PY, "scripts/step23_dashboard_export.py"]),
    ("dashboard page", [PY, "scripts/step6_build_dashboard.py"]),
    ("dashboard screenshot", [PY, "scripts/step24_dashboard_shot.py"]),
    ("figures + macros", [PY, "scripts/step20_report_figures.py"]),
    ("report", ["tectonic", "-X", "compile", "DustCast.tex", "--outdir", "."]),
]


def main() -> int:
    for name, cmd in STEPS:
        cwd = ROOT / "report" if name == "report" else ROOT
        print(f"==> {name}")
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            return r.returncode
    print("\n==> consistency check")
    return check()


def check() -> int:
    """Fail loudly if the deliverables disagree about the snapshot."""
    import json
    import re

    snap = json.loads((ROOT / "data/artifacts/step22_snapshot.json").read_text())
    issued = snap["issued_at"][:16].replace("T", " ")
    fleet = f"{snap['config']['fleet_mw']:.0f}"
    half = f"{snap['peak']['half_width_mw']:,.1f}"

    macros = (ROOT / "report/figures.tex").read_text()
    page = (ROOT / "dashboard/dustcast.html").read_text()
    payload = json.loads((ROOT / "data/artifacts/step6_dashboard.json").read_text())

    checks = [
        ("macros issue time", f"{{{issued}}}" in macros),
        ("macros half-width", f"{{{half}}}" in macros),
        ("dashboard issue time", issued in page),
        ("dashboard fleet MW", f'"installed_mw":{fleet}' in page
         or f"{fleet} MW" in page),
        ("payload fleet MW", f"{payload['meta']['installed_mw']:.0f}" == fleet),
        ("payload issue time", payload["meta"]["issued_at"] == issued.replace(" ", "T")),
    ]
    bad = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'}  {n}")
    if bad:
        print(f"\n  {len(bad)} deliverable(s) disagree with the snapshot.")
        return 1
    print(f"\n  all deliverables quote the snapshot issued {issued}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
