"""Step 21: an interval for the model the national layer actually runs.

Two separate questions, kept apart because conflating them is the error the
report was criticised for:

  1. Does an L0 band calibrated at DKASC hold up out of sample AT A SITE?
     Answerable, and answered here.
  2. Does the aggregated national band have 90% coverage over Tunisia?
     NOT answerable. There is no measured Tunisian national series to score it
     against. What is reported is how the aggregate is CONSTRUCTED, and the
     spatial correlation it depends on -- not a coverage figure.

    python scripts/step21_national_uncertainty.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dustcast import (fleet, live, metrics, physics, service,  # noqa: E402
                      tunisia, uncertainty)
from dustcast.config import (ARTIFACTS, DKASC_SITE_03, DKASC_SITE_16C,  # noqa: E402
                             DKASC_SITE_16D)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step15_multiarray import build, nwp_archive, rule  # noqa: E402

SPLIT = "2025-03-30"
LEVEL = 0.90
TZ = "Africa/Tunis"


def main() -> int:
    nwp = nwp_archive()
    sites = (DKASC_SITE_03, DKASC_SITE_16C, DKASC_SITE_16D)
    frames = {s.key: build(s, nwp) for s in sites}

    rule("1. SITE-LEVEL COVERAGE OF THE L0 BAND (out of sample)")
    cal = [(f[f.index < pd.Timestamp(SPLIT, tz=f.index.tz)], s)
           for s, f in ((s, frames[s.key]) for s in sites)]
    band = uncertainty.fit_l0_band(cal, LEVEL)
    print(f"  calibrated on {sum(len(f) for f, _ in cal):,} daylight hours "
          f"before {SPLIT}, across {len(cal)} arrays")
    print(f"  overall half-width {band.attrs['overall']:.4f} of AC capacity, "
          f"{len(band)} elevation x clear-sky bins\n")

    print(f"  {'array':<16s} {'n':>6s} {'PICP':>7s} {'PINAW':>7s} {'mean kW':>8s}")
    for s in sites:
        f = frames[s.key]
        te = f[f.index >= pd.Timestamp(SPLIT, tz=f.index.tz)]
        te = te[metrics.daylight_mask(te)]
        lo, hi = uncertainty.apply_l0_band(te, s, band)
        act, cap = te["ac_power_kw"], s.ac_capacity_w / 1000.0
        print(f"  {s.key:<16s} {len(te):>6,} {metrics.picp(act, lo, hi):>6.1%} "
              f"{metrics.pinaw(lo, hi, cap):>7.3f} {np.mean(hi - lo):>8.3f}")
    print(f"\n  Nominal is {LEVEL:.0%}. These are SITE coverages, measured in "
          f"Australia.")

    rule("2. HOW THE NATIONAL BAND IS CONSTRUCTED")
    weathers = live.fetch_weather_multi(tunisia.latitudes(), tunisia.longitudes(),
                                        TZ, past_days=2, forecast_days=7, tag="tn")
    dists = tunisia.districts()
    cap_by_gov = {g.key: sum(d.capacity_mw for d in dists if d.governorate == g.key)
                  for g in tunisia.GOVERNORATES}

    widths, kt_cols, points = {}, {}, {}
    for i, gov in enumerate(tunisia.GOVERNORATES):
        w_parts, p_parts, cap_kw = [], [], 0.0
        for j, seg in enumerate(fleet.SEGMENTS):
            for a in fleet.sample_segment_archetypes(
                    gov.key, seg, gov.latitude, gov.longitude, 50.0, TZ,
                    n=service.ARCHETYPES_PER_SEGMENT, seed=i * 10 + j):
                f = physics.build_physics_frame(a.site, weathers[i], clearsky_scale=None)
                f = f[f["expected_kw"].notna()]
                if f.empty:
                    continue
                lo, hi = uncertainty.apply_l0_band(f, a.site, band)
                w_parts.append(hi - lo)
                p_parts.append(f["expected_kw"].clip(lower=0.0))
                cap_kw += a.site.dc_capacity_w / 1000.0
                kt_cols.setdefault(gov.key, f["clearsky_index"].where(
                    f["solar_elevation"] > 10.0))
        # specific width (per kW installed) x governorate capacity -> MW
        widths[gov.key] = sum(w_parts) / cap_kw * cap_by_gov[gov.key]
        points[gov.key] = sum(p_parts) / cap_kw * cap_by_gov[gov.key]

    W = pd.DataFrame(widths)
    P = pd.DataFrame(points)
    rho = fleet.estimate_spatial_correlation(pd.DataFrame(kt_cols).dropna(how="all"))
    nat_w = fleet.aggregate_interval_width(W, rho)
    summed = W.sum(axis=1)

    peak = P.sum(axis=1).idxmax()
    print(f"  spatial correlation of the clear-sky index across 24 "
          f"governorates: rho = {rho:.3f}")
    print(f"  aggregation factor sqrt((1+(n-1)rho)/n) with n=24: "
          f"{np.sqrt((1 + 23 * rho) / 24):.3f}")
    print(f"\n  at the national peak ({peak:%Y-%m-%d %H:%M}):")
    print(f"    point forecast              {P.sum(axis=1).loc[peak]:7.1f} MW")
    print(f"    summed regional width       {summed.loc[peak]:7.1f} MW  "
          f"(assumes perfect correlation)")
    print(f"    correlation-aware width     {nat_w.loc[peak]:7.1f} MW  "
          f"({nat_w.loc[peak] / summed.loc[peak]:.0%} of the sum)")
    print(f"    independent-error width     "
          f"{summed.loc[peak] / np.sqrt(24):7.1f} MW  (assumes rho = 0)")

    rule("3. WHAT IS AND IS NOT ESTABLISHED")
    print("  ESTABLISHED   the L0 band's coverage at three Australian sites,")
    print("                out of sample, reported above.")
    print("  NOT ESTABLISHED")
    print("                national coverage over Tunisia. Aggregating")
    print("                calibrated site bands does not calibrate the")
    print("                aggregate: that depends on rho, which is estimated")
    print("                from forecast clear-sky index and never checked")
    print("                against a measured national series, because none")
    print("                was available to this project.")
    print("                Tunisian site coverage is likewise unvalidated.")

    out = pd.DataFrame({"point_mw": P.sum(axis=1), "width_mw": nat_w,
                        "width_summed_mw": summed})
    out.to_csv(ARTIFACTS / "step21_national_band.csv")
    band.to_csv(ARTIFACTS / "step21_l0_band.csv", index=False)
    print(f"\n  wrote {ARTIFACTS / 'step21_national_band.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
