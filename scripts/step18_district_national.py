"""Step 18: district -> governorate -> national, across LV and MV segments.

The concept note asks for forecasts at district, governorate and national scale,
covering rooftop PV on both low and medium voltage networks. This produces all
three scales for all three segments.

THE MODEL USED HERE IS L0 PHYSICS, AND THAT IS NOT A SHORTCUT.

The competing models that beat physics on DKASC -- the hour-of-day correction
and the boosted residuals -- all learn from a site's OWN observed production.
No Tunisian rooftop in this fleet has any. There is no production history to fit
a correction to, so the only model that can run is the one that needs no local
data. The frozen tests on unseen arrays do not describe this case either: every
baseline in them adapted using the target array's past output.

What DOES speak to it is L0's own measured error on DKASC (step 17): 5.57% nMAE
intra-day, 6.67% at J+1. That is the honest accuracy statement for a Tunisian
rooftop with no history, and it is a transfer claim from an Australian desert
site, not a Tunisian measurement.

The fleet itself is simulated throughout -- Tunisia publishes no installation
registry at any voltage.

    python scripts/step18_district_national.py [--refresh]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import fleet, learning, live, model, physics, tunisia  # noqa: E402
from dustcast.config import ARTIFACTS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step15_multiarray import rule  # noqa: E402

TZ = "Africa/Tunis"
ARCHETYPES_PER_SEGMENT = 5
HORIZONS = [0, 1, 2, 3]


def segment_yield(gov, segment, weather, seed) -> pd.Series | None:
    """kW produced per kW installed, for one segment in one governorate."""
    archs = fleet.sample_segment_archetypes(
        gov.key, segment, gov.latitude, gov.longitude, 50.0, TZ,
        n=ARCHETYPES_PER_SEGMENT, seed=seed)

    l0 = learning.PhysicsOnly()
    parts, cap_kw = [], 0.0
    for a in archs:
        f = physics.build_physics_frame(a.site, weather, clearsky_scale=None)
        f = f[f["expected_kw"].notna()]
        if f.empty:
            continue
        parts.append(l0.predict(f, a.site))
        cap_kw += a.site.dc_capacity_w / 1000.0
    if not parts:
        return None
    return sum(parts) / cap_kw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    rule("1. THE FLEET")
    dists = tunisia.districts()
    shares = fleet.segment_shares()
    print(f"  {len(dists)} districts across {len(tunisia.GOVERNORATES)} governorates")
    print(f"  {tunisia.rooftop_total_mw():.0f} MW rooftop, by segment:")
    for s in fleet.SEGMENTS:
        print(f"    {s.name:<24s} {s.voltage:<6s} {s.national_mw:6.0f} MW "
              f"({shares[s.key]:5.1%})  unit size ~{pd.Series([s.cap_range]).iloc[0]}")
    print("\n  SIMULATED. No installation-level registry or generation series")
    print("  was available to this project at any voltage level. Districts are")
    print("  PROTOTYPE aggregation units -- correspondence to STEG operational")
    print("  units is not established; names are not modelled, only counts.")

    rule("2. WEATHER")
    lats, lons = tunisia.latitudes(), tunisia.longitudes()
    weathers = live.fetch_weather_multi(lats, lons, TZ, past_days=2,
                                        forecast_days=7, refresh=args.refresh,
                                        tag="tn")
    print(f"  {len(weathers)} governorate grid points x {len(weathers[0]):,} h")
    print("  Districts within a governorate SHARE this weather: the NWP grid is")
    print("  ~11 km and CAMS dust ~40 km, both coarser than most delegations.")

    rule("3. SEGMENT YIELDS PER GOVERNORATE")
    yields: dict[tuple[str, str], pd.Series] = {}
    for i, gov in enumerate(tunisia.GOVERNORATES):
        for j, seg in enumerate(fleet.SEGMENTS):
            y = segment_yield(gov, seg, weathers[i], seed=i * 10 + j)
            if y is not None:
                yields[(gov.key, seg.key)] = y
    idx = next(iter(yields.values())).index
    print(f"  {len(yields)} governorate x segment combinations, {len(idx):,} hours")

    rule("4. DISTRICT -> GOVERNORATE -> NATIONAL")
    dist_mw: dict[str, pd.Series] = {}
    for d in dists:
        acc = pd.Series(0.0, index=idx)
        for skey, share in d.segment_mix.items():
            y = yields.get((d.governorate, skey))
            if y is not None:
                acc = acc.add(y.reindex(idx).fillna(0.0) * share, fill_value=0.0)
        dist_mw[d.key] = acc * d.capacity_mw       # yield x MW installed = MW

    D = pd.DataFrame(dist_mw)
    gov_of = {d.key: d.governorate for d in dists}
    G = D.T.groupby(gov_of).sum().T
    national = G.sum(axis=1)

    cap_by_gov = G.columns.map(
        lambda k: sum(d.capacity_mw for d in dists if d.governorate == k))
    now = pd.Timestamp.now(tz=TZ)
    peak_t = national.idxmax()

    print(f"  national peak {national.max():.1f} MW at {peak_t:%Y-%m-%d %H:%M} "
          f"(capacity factor {national.max() / tunisia.rooftop_total_mw():.1%})")
    print(f"  scales: {len(D.columns)} districts -> {len(G.columns)} governorates -> 1 national\n")

    top = G.loc[peak_t].sort_values(ascending=False).head(6)
    print("  largest governorates at national peak:")
    for k, v in top.items():
        name = next(g.name for g in tunisia.GOVERNORATES if g.key == k)
        cap = sum(d.capacity_mw for d in dists if d.governorate == k)
        print(f"    {name:<14s} {v:6.1f} MW of {cap:6.1f} MW installed")

    dpeak = D.loc[peak_t].sort_values(ascending=False).head(5)
    print("\n  largest districts at national peak:")
    for k, v in dpeak.items():
        d = next(x for x in dists if x.key == k)
        print(f"    {d.name:<16s} {v:6.2f} MW of {d.capacity_mw:6.2f} MW installed")

    rule("5. HORIZONS PRODUCED")
    day0 = now.normalize()
    for h in HORIZONS:
        lo, hi = day0 + pd.Timedelta(days=h), day0 + pd.Timedelta(days=h + 1)
        win = national.loc[(national.index >= lo) & (national.index < hi)]
        if win.empty:
            print(f"  J+{h}  no forecast hours in window")
            continue
        print(f"  J+{h}  {lo:%Y-%m-%d}  peak {win.max():6.1f} MW  "
              f"energy {win.sum():7.0f} MWh")
    print("\n  Produced, not validated. Measured error by horizon is in step 17,")
    print("  on DKASC, for L0 physics: 5.57% nMAE intra-day, 6.67% at J+1.")

    D.to_csv(ARTIFACTS / "step18_districts.csv")
    G.to_csv(ARTIFACTS / "step18_governorates.csv")
    national.rename("national_mw").to_csv(ARTIFACTS / "step18_national.csv")
    print(f"\n  wrote district / governorate / national series to {ARTIFACTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
