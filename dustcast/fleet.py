"""A virtual rooftop fleet, and the upscaling from a sample to a nation.

The method is the standard one: forecast a *sample* of sites, divide by the
sample's installed capacity to get a specific yield in kW per kW installed,
then multiply by the region's installed capacity. AEMO does this for Australian
rooftop PV and Sheffield Solar / Open Climate Fix do it for the UK. The
difference here is that their sample is real and metered, and ours is
simulated, because Tunisia publishes no registry.

What the simulation buys is not realism about any individual roof -- it is the
*fleet* effect. Real rooftops are not all south-facing at optimal tilt, and a
mixture of orientations produces a visibly different aggregate curve from a
single optimal array: a lower, flatter peak with fatter morning and afternoon
shoulders, because east-facing arrays peak early and west-facing ones peak
late. Modelling the fleet as one big optimal panel overstates midday output and
understates the evening ramp, which is exactly the quantity an operator cares
about.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .config import Site

#: Fleet composition. Every one of these is an ASSUMPTION -- there is no
#: Tunisian installation registry to fit them to. They are chosen to be
#: plausible and are stated here so they can be argued with.
TILT_MEAN, TILT_SD = 28.0, 8.0          # flat-roof mounts cluster near optimal
TILT_RANGE = (5.0, 40.0)
AZIMUTH_MEAN, AZIMUTH_SD = 180.0, 30.0  # mostly south, with real spread
AZIMUTH_RANGE = (90.0, 270.0)           # east through west; nobody faces north
CAPACITY_LOG_MEAN, CAPACITY_LOG_SD = np.log(3.0), 0.35   # kW, residential
CAPACITY_RANGE = (1.5, 8.0)
AGE_RANGE_YEARS = (0.5, 8.0)            # PROSOL has been running for years

#: Archetypes per governorate. Enough to span the orientation distribution
#: without making the run slow; the aggregate is insensitive beyond ~10.
ARCHETYPES_PER_GOVERNORATE = 9


@dataclass(frozen=True)
class Archetype:
    site: Site
    weight: float   # share of the governorate's capacity this archetype carries


def sample_archetypes(name: str, latitude: float, longitude: float,
                      altitude: float, tz: str, n: int = ARCHETYPES_PER_GOVERNORATE,
                      seed: int = 0) -> list[Archetype]:
    """Draw a small set of plausible rooftops for one governorate.

    Seeded per governorate so the fleet is reproducible: a national number that
    changes between runs because of resampling noise is not a forecast anybody
    can act on.
    """
    rng = np.random.default_rng(seed)

    tilts = np.clip(rng.normal(TILT_MEAN, TILT_SD, n), *TILT_RANGE)
    azimuths = np.clip(rng.normal(AZIMUTH_MEAN, AZIMUTH_SD, n), *AZIMUTH_RANGE)
    capacities = np.clip(
        rng.lognormal(CAPACITY_LOG_MEAN, CAPACITY_LOG_SD, n), *CAPACITY_RANGE)
    ages = rng.uniform(*AGE_RANGE_YEARS, n)

    total = capacities.sum()
    out = []
    for i in range(n):
        commissioned = (pd.Timestamp.now(tz=tz).normalize()
                        - pd.Timedelta(days=float(ages[i]) * 365.25))
        site = Site(
            key=f"{name}_a{i}",
            name=f"{name} archetype {i}",
            latitude=latitude, longitude=longitude, altitude=altitude, tz=tz,
            tilt=float(tilts[i]), azimuth=float(azimuths[i]),
            dc_capacity_w=float(capacities[i]) * 1000.0,
            # Residential inverters are commonly sized slightly under the array.
            ac_capacity_w=float(capacities[i]) * 1000.0 * 0.95,
            gamma_pdc=-0.0035, eta_inv_nom=0.97,
            commissioned=commissioned.strftime("%Y-%m-%d"),
            temperature_model="close_mount_glass_glass",
        )
        out.append(Archetype(site=site, weight=float(capacities[i] / total)))
    return out


def specific_yield(power_by_archetype: pd.DataFrame,
                   capacities_kw: pd.Series) -> pd.Series:
    """Sample output -> kW produced per kW installed.

    Capacity-weighted by construction: the sum of the sample's power divided by
    the sum of its capacity. Averaging per-archetype yields instead would give
    every roof equal say regardless of size.
    """
    return power_by_archetype.sum(axis=1) / capacities_kw.sum()


def upscale(specific: pd.Series, region_capacity_mw: float) -> pd.Series:
    """Specific yield -> regional MW."""
    return specific * region_capacity_mw


def aggregate_interval_width(widths_mw: pd.DataFrame,
                             correlation: float) -> pd.Series:
    """Combine regional interval widths into a national one.

    Summing the regional bounds -- the obvious thing, and what CLAUDE.md
    section 6.6 does -- assumes the forecast errors are perfectly correlated
    across the country, which **over-states** the national band. Treating them
    as independent would divide the width by sqrt(n) and **under-state** it.
    Neither is right: weather is spatially correlated but not identical, and
    Tunisia is ~800 km long.

    For n regions with roughly equal error scale and average pairwise
    correlation rho,

        Var(sum) = sigma^2 * n * (1 + (n-1) * rho)

    so the national width is the summed width scaled by
    sqrt((1 + (n-1)*rho) / n). That recovers the sum exactly at rho = 1 and the
    sqrt(n) reduction at rho = 0, and takes a defensible position in between
    when rho is estimated from the forecasts themselves.
    """
    n = widths_mw.shape[1]
    if n <= 1:
        return widths_mw.sum(axis=1)
    factor = np.sqrt((1.0 + (n - 1) * correlation) / n)
    return widths_mw.sum(axis=1) * factor


def estimate_spatial_correlation(driver: pd.DataFrame) -> float:
    """Average pairwise correlation of a forecast driver across regions.

    Estimated from the clear-sky index rather than assumed, because it is the
    variable that actually carries the correlated forecast error: if it is
    cloudy over the whole country the errors move together, and if the weather
    is patchy they do not.
    """
    corr = driver.corr().to_numpy()
    n = corr.shape[0]
    if n <= 1:
        return 1.0
    off_diagonal = corr[~np.eye(n, dtype=bool)]
    return float(np.nanmean(off_diagonal))


# --------------------------------------------------------------------------
# Connection segments: LV residential, LV/MV commercial, MV industrial
# --------------------------------------------------------------------------
#
# The concept note's scope is rooftop PV on LOW AND MEDIUM voltage networks, so
# a residential-only fleet covers only part of it. These three segments differ
# in ways that change the aggregate curve, not just its size: commercial and
# industrial arrays sit on flat roofs at shallow tilt, which flattens and
# broadens the midday peak, and their far larger unit capacity means a handful
# of sites can dominate a district.
#
# EVERY number below is an ASSUMPTION. Tunisia publishes no installation
# registry at any voltage level, so none of this is fitted -- it is stated so it
# can be argued with. The national split follows STEG's end-June 2026 statement
# of ~500 MW residential rooftop against ~130 MW on MV/HV networks; the division
# of that 130 MW between commercial and industrial is our own guess.

@dataclass(frozen=True)
class Segment:
    key: str
    name: str
    voltage: str
    national_mw: float
    cap_log_mean: float          # log kW
    cap_log_sd: float
    cap_range: tuple[float, float]
    tilt_mean: float
    tilt_sd: float
    azimuth_sd: float
    temperature_model: str
    gamma_pdc: float
    eta_inv_nom: float


SEGMENTS: tuple[Segment, ...] = (
    Segment("residential", "Residential", "LV", 500.0,
            float(np.log(3.0)), 0.35, (1.5, 8.0),
            28.0, 8.0, 30.0, "close_mount_glass_glass", -0.0035, 0.97),
    # Shops, offices, hotels, schools. Flat roofs, shallower tilt, installer-
    # standardised orientation, professional maintenance (so slightly better
    # inverters and a cleaner array than the residential stock).
    Segment("commercial", "Commercial / services", "LV-MV", 78.0,
            float(np.log(45.0)), 0.80, (10.0, 250.0),
            12.0, 5.0, 20.0, "open_rack_glass_glass", -0.0035, 0.975),
    # Factories and warehouses. Large flat roofs, near-horizontal mounting,
    # east-west layouts are common, so the azimuth spread is WIDER than
    # commercial despite the more professional install.
    Segment("industrial", "Industrial", "MV", 52.0,
            float(np.log(600.0)), 0.70, (250.0, 2000.0),
            10.0, 4.0, 45.0, "open_rack_glass_glass", -0.0034, 0.98),
)

SEGMENT_BY_KEY = {s.key: s for s in SEGMENTS}


def segment_shares() -> dict[str, float]:
    """Each segment's share of national rooftop capacity."""
    total = sum(s.national_mw for s in SEGMENTS)
    return {s.key: s.national_mw / total for s in SEGMENTS}


def sample_segment_archetypes(name: str, segment: Segment,
                              latitude: float, longitude: float,
                              altitude: float, tz: str,
                              n: int = 5, seed: int = 0) -> list[Archetype]:
    """Draw plausible installations for one segment in one place.

    Same construction as `sample_archetypes`, but with the segment's own size,
    tilt and orientation distributions, so a district dominated by industrial
    roofs produces a visibly flatter, broader curve than a residential one.
    """
    rng = np.random.default_rng(seed)
    tilts = np.clip(rng.normal(segment.tilt_mean, segment.tilt_sd, n), 0.0, 40.0)
    azimuths = np.clip(rng.normal(AZIMUTH_MEAN, segment.azimuth_sd, n), *AZIMUTH_RANGE)
    capacities = np.clip(
        rng.lognormal(segment.cap_log_mean, segment.cap_log_sd, n), *segment.cap_range)
    ages = rng.uniform(*AGE_RANGE_YEARS, n)

    total = capacities.sum()
    out = []
    for i in range(n):
        commissioned = (pd.Timestamp.now(tz=tz).normalize()
                        - pd.Timedelta(days=float(ages[i]) * 365.25))
        out.append(Archetype(
            site=Site(
                key=f"{name}_{segment.key}_a{i}",
                name=f"{name} {segment.name} archetype {i}",
                latitude=latitude, longitude=longitude, altitude=altitude, tz=tz,
                tilt=float(tilts[i]), azimuth=float(azimuths[i]),
                dc_capacity_w=float(capacities[i]) * 1000.0,
                ac_capacity_w=float(capacities[i]) * 1000.0 * 0.95,
                gamma_pdc=segment.gamma_pdc, eta_inv_nom=segment.eta_inv_nom,
                commissioned=commissioned.strftime("%Y-%m-%d"),
                temperature_model=segment.temperature_model,
            ),
            weight=float(capacities[i] / total)))
    return out
