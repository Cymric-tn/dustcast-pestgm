"""Step 7: the field node, and the loop it closes.

CLAUDE.md puts the ESP32 first on the cut list, and it is genuinely optional
for a forecast. It is not optional for a *calibrated* forecast.

Step 2 measured the problem: a one-off conformal calibration fell to 76%
coverage against a 90% target once the error distribution drifted, and only
recalibrating from a trailing window of OBSERVED error recovered 90.3%. That
needs measured output back within about a day, and STEG cannot see
behind-the-meter rooftop generation -- so the feedback has to come from a small
monitored sample. This node is one of that sample.

Runs without hardware: a simulator drives the real ingest path so the whole
chain is exercised. Point it at a broker with --mqtt to consume a real node.

    python scripts/step7_node.py
    python scripts/step7_node.py --mqtt 192.168.1.10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import telemetry  # noqa: E402
from dustcast.config import PROCESSED, SFAX_ROOFTOP  # noqa: E402

TZ = "Africa/Tunis"
ARRAY_DC_KW = SFAX_ROOFTOP.dc_capacity_w / 1000.0


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def load_forecast() -> pd.DataFrame:
    """The Sfax band, as specific yield so a node of any size can be compared."""
    df = pd.read_parquet(PROCESSED / "step3_live_forecast.parquet")
    out = pd.DataFrame({
        "point": df["point_kw"] / ARRAY_DC_KW,
        "lower": df["lower_kw"] / ARRAY_DC_KW,
        "upper": df["upper_kw"] / ARRAY_DC_KW,
        "elevation": df["solar_elevation"],
    })
    return out[out["elevation"] > 0]


def simulate(forecast: pd.DataFrame, hours: int, soiling_loss: float,
             seed: int = 7) -> list[str]:
    """Emit JSON payloads exactly as the firmware would.

    SYNTHETIC. The measurement is the forecast plus a plausible error, not a
    real panel, and it can only ever demonstrate that the plumbing works --
    never that the forecast is right. Two components:

    a multiplicative noise term standing in for the ordinary forecast error the
    band is meant to contain, and an optional constant `soiling_loss` standing
    in for a real physical drift the band was NOT told about. The second is the
    interesting one: it is what a dirty array looks like from the control room,
    and what should drag realized coverage below nominal and trigger a refit.
    """
    rng = np.random.default_rng(seed)
    recent = forecast.tail(hours)

    payloads = []
    for t, row in recent.iterrows():
        # Scale the noise with the band's own width, so the simulation is
        # consistent with the uncertainty the model actually claims.
        half_width = max((row["upper"] - row["lower"]) / 2.0, 1e-4)
        noise = rng.normal(0.0, half_width / 1.645)   # 90% inside by construction
        y = max(0.0, row["point"] * (1.0 - soiling_loss) + noise)

        payloads.append((t, json.dumps({
            "node": "sfax-01",
            "seq": len(payloads),
            "uptime_s": len(payloads) * 15,
            "array_dc_w": ARRAY_DC_KW * 1000.0,
            "voltage_v": round(float(380 + rng.normal(0, 6)), 2),
            "current_a": round(float(y * ARRAY_DC_KW * 1000.0 / 380.0), 3),
            "power_w": round(float(y * ARRAY_DC_KW * 1000.0), 1),
            "panel_c": round(float(28 + 34 * row["point"] + rng.normal(0, 1.5)), 2),
            "ghi_wm2": None,
            "rssi": int(rng.integers(-72, -52)),
        })))
    return payloads


def run(forecast: pd.DataFrame, payloads, label: str) -> dict:
    results = []
    for received, payload in payloads:
        reading = telemetry.parse_payload(payload, tz=TZ, received=received)
        results.append(telemetry.verify(reading, forecast))
    cov = telemetry.realized_coverage(results)
    print(f"  {label:<28s} n={cov['n']:3d}  coverage {cov['coverage']:6.1%}  "
          f"bias {cov['bias_kw']:+.3f} kW  |dev| {cov['mean_abs_dev_kw']:.3f} kW")
    return cov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mqtt", help="broker host for a real node")
    ap.add_argument("--hours", type=int, default=120)
    args = ap.parse_args()

    forecast = load_forecast()

    rule("1. FORECAST THE NODE IS CHECKED AGAINST")
    print(f"Sfax rooftop, {ARRAY_DC_KW:.1f} kW DC")
    print(f"{len(forecast):,} daylight hours  {forecast.index.min():%Y-%m-%d} .. "
          f"{forecast.index.max():%Y-%m-%d}")
    print(f"band width, mean over daylight: "
          f"{(forecast['upper'] - forecast['lower']).mean():.3f} kW/kW")

    if args.mqtt:
        rule("2. LIVE NODE")
        print(f"subscribing to {args.mqtt}:1883 dustcast/telemetry -- Ctrl-C to stop")
        results: list[dict] = []

        def handle(reading: telemetry.Reading) -> None:
            r = telemetry.verify(reading, forecast)
            results.append(r)
            if r["status"] != "ok":
                print(f"  {reading.received:%H:%M:%S}  {r['status']}")
                return
            mark = "IN " if r["inside"] else "OUT"
            cov = telemetry.realized_coverage(results, window=200)
            print(f"  {reading.received:%H:%M:%S}  {r['measured_kw']:.3f} kW  "
                  f"[{r['lower_kw']:.3f}..{r['upper_kw']:.3f}]  {mark}  "
                  f"rolling coverage {cov['coverage']:.0%} (n={cov['n']})")

        telemetry.subscribe(args.mqtt, on_reading=handle, tz=TZ)
        return 0

    rule("2. SIMULATED NODE THROUGH THE REAL INGEST PATH")
    print("SYNTHETIC measurements -- this proves the plumbing, never the forecast.\n")
    clean = run(forecast, simulate(forecast, args.hours, 0.00), "clean array")
    light = run(forecast, simulate(forecast, args.hours, 0.06), "6% soiling loss")
    heavy = run(forecast, simulate(forecast, args.hours, 0.15), "15% soiling loss")

    rule("3. WHY THIS IS THE POINT OF THE HARDWARE")
    print(f"  a clean array sits inside the band {clean['coverage']:.0%} of the time,")
    print(f"  which is what a 90% band should do.")
    print(f"\n  at 15% soiling the same band holds only {heavy['coverage']:.0%}, "
          f"with a\n  persistent {heavy['bias_kw']:+.3f} kW bias -- the forecast is "
          f"not wrong at\n  random, it is wrong in one direction.")
    print(f"\n  That signature is what triggers recalibration, and it is invisible")
    print(f"  without measured output. It is also the cleaning trigger arriving")
    print(f"  from the array itself rather than from a modelled soiling index")
    print(f"  that has no ground truth anywhere in this project.")

    rule("GATE: does the node close the loop?")
    ok = clean["n"] > 0 and clean["coverage"] > heavy["coverage"]
    print(f"  firmware            hardware/dustcast_node/dustcast_node.ino "
          f"(NOT yet run on hardware)")
    print(f"  ingest + verify     dustcast/telemetry.py")
    print(f"  degradation is detectable: {clean['coverage']:.0%} -> "
          f"{heavy['coverage']:.0%} coverage")
    print(f"\n  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
