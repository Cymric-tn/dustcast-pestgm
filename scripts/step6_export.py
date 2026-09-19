"""Step 6a: bundle the national forecast into JSON for the map UI.

Reads what step5_national.py wrote rather than recomputing: re-running 432
pvlib chains to draw a chart would make the page slow to rebuild and would let
the numbers on screen drift from the numbers in the gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import decisions, features, tunisia  # noqa: E402
from dustcast.config import ARTIFACTS, PROCESSED  # noqa: E402

TZ = "Africa/Tunis"
HORIZON_HOURS = 48          # the default window the dashboard opens on

#: How much history to ship alongside the forecast. The dashboard lets an
#: operator scrub backwards to a past dust event, which is the most convincing
#: thing the system can show -- but those hours are a HINDCAST, not a forecast:
#: Open-Meteo's past_days returns the best available analysis rather than the
#: forecast that was actually issued at the time. The payload marks the
#: boundary so the interface can say so rather than implying 21 days of
#: forecast skill.
HISTORY_DAYS = 14


def main() -> int:
    nat = pd.read_parquet(PROCESSED / "step5_national.parquet")
    govs = pd.read_csv(ARTIFACTS / "step5_governorates.csv", index_col=0)
    series = pd.read_parquet(PROCESSED / "step5_governorate_series.parquet")

    now = pd.Timestamp.now(tz=TZ).floor("1h")
    start = now - pd.Timedelta(days=HISTORY_DAYS)
    end = nat.index.max()
    nat_w = nat.loc[start:end]
    if nat_w.empty:                     # cached run entirely in the past
        nat_w = nat.tail(HISTORY_DAYS * 24)
        start, end = nat_w.index[0], nat_w.index[-1]
    horizon_end = min(end, now + pd.Timedelta(hours=HORIZON_HOURS))

    def times(index) -> list[str]:
        return [t.strftime("%Y-%m-%dT%H:%M") for t in index]

    # Headline decisions are computed over the DEFAULT window only. Quoting a
    # reserve figure across three weeks of mixed hindcast and forecast would be
    # meaningless; the dashboard recomputes them for whatever window is chosen.
    default = nat.loc[now:horizon_end]
    if default.empty:
        default = nat_w.tail(HORIZON_HOURS)
    reserve = decisions.reserve_requirement(default["point_mw"], default["lower_mw"])
    ramp = decisions.ramp_risk(default["point_mw"], window="3h")

    payload = {
        "meta": {
            "generated": pd.Timestamp.now(tz=TZ).strftime("%Y-%m-%d %H:%M %Z"),
            "horizon_hours": HORIZON_HOURS,
            # Snapped to the series grid (values sit on the half hour, being
            # interval midpoints) so the interface can mark the boundary by
            # index instead of fuzzy-matching a timestamp.
            "now": nat_w.index[max(0, nat_w.index.searchsorted(now) - 1)]
                   .strftime("%Y-%m-%dT%H:%M"),
            "now_index": int(max(0, nat_w.index.searchsorted(now) - 1)),
            "history_days": HISTORY_DAYS,
            "installed_mw": float(govs["capacity_mw"].sum()),
            "confidence": 0.90,
            "trained_on": "DKASC Alice Springs array 3, 2008-2025",
        },
        "national": {
            "t": times(nat_w.index),
            "point": [round(v, 2) for v in nat_w["point_mw"]],
            "lower": [round(v, 2) for v in nat_w["lower_mw"]],
            "upper": [round(v, 2) for v in nat_w["upper_mw"]],
            "dust_off": [round(v, 2) for v in nat_w["dust_off_mw"]],
            "peak_mw": round(float(default["point_mw"].max()), 1),
            "peak_at": default["point_mw"].idxmax().strftime("%a %H:%M"),
            "energy_mwh": round(float(default["point_mw"].sum()), 0),
            "energy_dust_off_mwh": round(float(default["dust_off_mw"].sum()), 0),
            "band_vs_naive": round(
                float(default["band_mw"].mean() / default["naive_band_mw"].mean()), 3),
        },
        "weather": {},
        "decisions": {
            "reserve_mw": round(reserve, 1),
            "ramp_down_mw": round(ramp["max_down_kw"], 0),
            "ramp_up_mw": round(ramp["max_up_kw"], 0),
        },
        "governorates": [],
    }

    # National weather is capacity-weighted. An unweighted mean would give
    # Tozeur the same say as Tunis, which has eight times the installed PV.
    wsum = {k: None for k in ("ghi", "clearsky_ghi", "temp_air", "aod")}
    total_cap = 0.0
    for gov in tunisia.GOVERNORATES:
        g = series[series["governorate"] == gov.key].loc[start:end]
        if g.empty:
            continue
        cap = float(govs.loc[gov.key, "capacity_mw"])
        total_cap += cap
        for k in wsum:
            col = g[k].fillna(0).to_numpy() * cap
            wsum[k] = col if wsum[k] is None else wsum[k] + col
    if total_cap:
        payload["weather"] = {
            k: [round(float(v / total_cap), 2 if k == "aod" else 1)
                for v in wsum[k]] for k in wsum if wsum[k] is not None
        }

    for gov in tunisia.GOVERNORATES:
        row = govs.loc[gov.key]
        g = series[series["governorate"] == gov.key].loc[start:end]
        if g.empty:
            g = series[series["governorate"] == gov.key].tail(HISTORY_DAYS * 24)

        # Soiling over the FULL available history (deposition integrates over
        # weeks), then the cleaning decision against the forward rain forecast.
        full = series[series["governorate"] == gov.key]
        soiling = features.soiling_accumulation(full["dust"],
                                                full["precipitation"])
        rain_ahead = full.loc[now:horizon_end, "precipitation"]
        advice = decisions.cleaning_recommendation(
            soiling.loc[:now], rain_ahead)

        payload["governorates"].append({
            "key": gov.key,
            "name": gov.name,
            "region": gov.region,
            "lat": gov.latitude,
            "lon": gov.longitude,
            "capacity_mw": round(float(row["capacity_mw"]), 2),
            "peak_mw": round(float(g.loc[now:horizon_end, "point_mw"].max()
                                   if not g.loc[now:horizon_end].empty
                                   else g["point_mw"].max()), 2),
            "aod": round(float(g.loc[now:horizon_end, "aod"].mean()
                               if not g.loc[now:horizon_end].empty
                               else g["aod"].mean()), 3),
            "dust": round(float(g.loc[now:horizon_end, "dust"].mean()
                                if not g.loc[now:horizon_end].empty
                                else g["dust"].mean()), 1),
            "aod_series": [round(v, 3) for v in g["aod"]],
            "ghi": [round(v, 0) for v in g["ghi"].fillna(0)],
            "clearsky_ghi": [round(v, 0) for v in g["clearsky_ghi"].fillna(0)],
            "temp_air": [round(v, 1) for v in g["temp_air"].fillna(0)],
            "soiling_pct": round(float(soiling.iloc[-1] * 100), 2),
            "clean_now": bool(advice.recommend),
            "clean_reason": advice.reason,
            "t": times(g.index),
            "point": [round(v, 3) for v in g["point_mw"]],
            "lower": [round(v, 3) for v in g["lower_mw"]],
            "upper": [round(v, 3) for v in g["upper_mw"]],
            "dust_off": [round(v, 3) for v in g["dust_off_mw"]],
        })

    out = ARTIFACTS / "step6_dashboard.json"
    out.write_text(json.dumps(payload, separators=(",", ":")))
    size_kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({size_kb:.0f} kB)")
    print(f"national peak {payload['national']['peak_mw']} MW at "
          f"{payload['national']['peak_at']}, reserve "
          f"{payload['decisions']['reserve_mw']} MW")
    print(f"{len(payload['governorates'])} governorates, "
          f"{len(payload['national']['t'])} hours each")
    flagged = [g["name"] for g in payload["governorates"] if g["clean_now"]]
    print(f"cleaning recommended: {flagged or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
