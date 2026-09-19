"""Field-node telemetry: ingest, and verification against the forecast band.

The node's readings are only useful once they are compared with what was
predicted for that hour, so this module does both jobs: parse what the ESP32
publishes, and score it against the band.

Two outputs matter downstream.

`inside` per reading is the demo moment -- a physically measured dot landing
inside an interval that was computed before sunrise.

`realized_coverage` over a trailing window is the operational one. It is the
input the rolling conformal recalibration consumes, and the reason a hardware
node is in this project at all: the guarantee in step 2 needs observed error
back within about a day, and behind-the-meter rooftops give the operator none.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Below this the array is effectively off and the reading says nothing about
#: forecast quality; scoring it would pad coverage with free correct answers,
#: exactly as night-time hours do in the offline metrics.
MIN_INFORMATIVE_KW = 0.02


@dataclass(frozen=True)
class Reading:
    node: str
    received: pd.Timestamp
    power_kw: float | None
    array_dc_kw: float
    voltage_v: float | None = None
    current_a: float | None = None
    panel_c: float | None = None
    ghi_wm2: float | None = None
    rssi: int | None = None

    @property
    def specific_yield(self) -> float | None:
        """kW produced per kW installed -- the only comparable form.

        The forecast is upscaled from specific yield, so a raw wattage cannot
        be compared with it directly without knowing the array's rating. The
        node publishes its own rating for exactly this reason.
        """
        if self.power_kw is None or self.array_dc_kw <= 0:
            return None
        return self.power_kw / self.array_dc_kw


def parse_payload(payload: str | bytes, tz: str = "Africa/Tunis",
                  received: pd.Timestamp | None = None) -> Reading:
    """Parse one MQTT/HTTP JSON message from a node.

    The node's own clock is deliberately not trusted for the timestamp: an
    ESP32 without NTP starts at the epoch, and a reading filed under 1970 would
    silently never match a forecast hour. `uptime_s` is kept for diagnostics
    and the server's arrival time is used instead.
    """
    d = json.loads(payload)
    return Reading(
        node=str(d.get("node", "unknown")),
        received=received or pd.Timestamp.now(tz=tz),
        power_kw=None if d.get("power_w") is None else float(d["power_w"]) / 1000.0,
        array_dc_kw=float(d.get("array_dc_w", 0.0)) / 1000.0,
        voltage_v=d.get("voltage_v"),
        current_a=d.get("current_a"),
        panel_c=d.get("panel_c"),
        ghi_wm2=d.get("ghi_wm2"),
        rssi=d.get("rssi"),
    )


def verify(reading: Reading, forecast: pd.DataFrame) -> dict:
    """Score one reading against the forecast interval for its hour.

    `forecast` is indexed by time and carries point/lower/upper as SPECIFIC
    YIELD, matching what `Reading.specific_yield` returns.
    """
    y = reading.specific_yield
    if y is None or forecast.empty:
        return {"status": "no_data"}

    # Nearest forecast hour, but only if one is genuinely near. Reindexing with
    # method="nearest" and no tolerance will happily match a reading to a
    # forecast six hours away and report a confident miss.
    idx = forecast.index.get_indexer([reading.received], method="nearest",
                                     tolerance=pd.Timedelta("90min"))
    if idx[0] == -1:
        return {"status": "no_forecast"}

    row = forecast.iloc[idx[0]]
    informative = row["point"] * reading.array_dc_kw > MIN_INFORMATIVE_KW
    return {
        "status": "ok",
        "t": forecast.index[idx[0]],
        "measured_kw": reading.power_kw,
        "predicted_kw": float(row["point"] * reading.array_dc_kw),
        "lower_kw": float(row["lower"] * reading.array_dc_kw),
        "upper_kw": float(row["upper"] * reading.array_dc_kw),
        "inside": bool(row["lower"] <= y <= row["upper"]),
        "deviation_kw": float((y - row["point"]) * reading.array_dc_kw),
        "informative": bool(informative),
    }


def realized_coverage(results: list[dict], window: int | None = None) -> dict:
    """Coverage actually achieved on the ground, over informative hours.

    This is the quantity that tells an operator whether the band still means
    what it says. When it drifts away from nominal, the conformal correction
    needs refitting -- which is the whole reason for putting a node in a field.
    """
    scored = [r for r in results if r.get("status") == "ok" and r["informative"]]
    if window:
        scored = scored[-window:]
    if not scored:
        return {"n": 0, "coverage": float("nan"), "mean_abs_dev_kw": float("nan")}
    return {
        "n": len(scored),
        "coverage": float(np.mean([r["inside"] for r in scored])),
        "mean_abs_dev_kw": float(np.mean([abs(r["deviation_kw"]) for r in scored])),
        "bias_kw": float(np.mean([r["deviation_kw"] for r in scored])),
    }


def subscribe(host: str, port: int = 1883, topic: str = "dustcast/telemetry",
              on_reading=None, tz: str = "Africa/Tunis"):
    """Block on an MQTT topic, handing each parsed Reading to `on_reading`.

    Kept out of the import path: paho-mqtt is only needed when real hardware is
    attached, and step 7 has to run end-to-end without it.
    """
    import paho.mqtt.client as mqtt_client

    client = mqtt_client.Client()

    def _on_message(_client, _userdata, message):
        try:
            reading = parse_payload(message.payload, tz=tz)
        except (ValueError, KeyError) as exc:
            print(f"dropped malformed payload: {exc}")
            return
        if on_reading:
            on_reading(reading)

    client.on_message = _on_message
    client.connect(host, port, keepalive=60)
    client.subscribe(topic)
    client.loop_forever()
