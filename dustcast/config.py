"""Site metadata, paths and physical constants.

Everything here is either measured (published by the data provider) or an
explicit modelling assumption. Assumptions are marked ASSUMPTION so they can be
defended or challenged in the pitch, not silently inherited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
ARTIFACTS = DATA / "artifacts"

for _d in (RAW, PROCESSED, ARTIFACTS):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Site definition
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Site:
    """A single PV array with everything pvlib needs to model it."""

    key: str
    name: str

    # location
    latitude: float
    longitude: float
    altitude: float
    tz: str

    # array geometry
    tilt: float
    azimuth: float  # pvlib convention: degrees clockwise from true north

    # electrical
    dc_capacity_w: float
    ac_capacity_w: float
    gamma_pdc: float  # DC power temperature coefficient, fraction per degC
    eta_inv_nom: float

    commissioned: str  # ISO date

    # thermal model
    temperature_model: str = "open_rack_glass_glass"

    raw_file: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)


# DKASC Alice Springs, array 3.
#
# Chosen as the training site because it is the closest structural analogue in
# the DKASC fleet to a Tunisian residential rooftop: fixed tilt, roof mounted,
# ~5 kW, poly-Si. Alice Springs is hot, arid and dusty -- climatically close to
# southern Tunisia. See CLAUDE.md section 10 for the transfer argument and its
# limits.
#
# Specs published at
# https://dkasolarcentre.com.au/source/alice-springs/dka-m5-a-phase
DKASC_SITE_03 = Site(
    key="dkasc_site03",
    name="DKASC Alice Springs #3 - BP Solar 4.95kW poly-Si fixed, roof mounted",
    # DKASC published site coordinates (Desert Knowledge Precinct).
    latitude=-23.7621,
    longitude=133.8744,
    altitude=576.0,
    # Northern Territory does not observe daylight saving, so this is a fixed
    # UTC+09:30 offset all year. Getting this wrong shifts the whole diurnal
    # cycle and silently destroys the physics baseline.
    tz="Australia/Darwin",
    tilt=20.0,
    # DKASC publishes "Azi = 0 (Solar North)". The array is in the southern
    # hemisphere, so it faces north -- 0 deg in pvlib's clockwise-from-true-
    # north convention. (A Tunisian array would be 180: due south.)
    azimuth=0.0,
    dc_capacity_w=4950.0,  # 30 x BP 3165, 165 W each
    ac_capacity_w=6000.0,  # SMA SMC 6000A
    # ASSUMPTION: -0.45 %/degC. BP 3165 datasheet Pmax coefficient is
    # -0.5 %/degC; PVWatts default for the technology is -0.47 %/degC. The
    # residual model learns whatever this gets wrong, which is precisely the
    # point of the residual design.
    gamma_pdc=-0.0045,
    # ASSUMPTION: SMC 6000A Euro efficiency is ~95.2%; nominal peak ~96.1%.
    eta_inv_nom=0.96,
    commissioned="2008-11-11",
    # ASSUMPTION: DKASC's own metadata is self-contradictory -- the array
    # listing says "Fixed: Ground Mount" while the detail page says "Roof
    # Mounted". Open rack runs cooler than a close roof mount. We follow the
    # open-rack choice; a close mount would raise cell temperature by a few
    # degC, and the residual model absorbs that bias.
    temperature_model="open_rack_glass_glass",
    raw_file="dkasc_site03_5min.csv",
    notes=(
        "Provider metadata conflicts on mounting type (roof vs ground).",
        # MEASURED, not assumed. actual/expected on high-sun hours falls
        # monotonically from 0.955 (2009) to 0.713 (2025): -1.55 %/yr
        # log-linear over 2009-2021, -25% total. That is roughly 3x the
        # 0.5 %/yr rule of thumb in CLAUDE.md section 4, and it is fitted per
        # site -- it must NOT be carried over to Tunisian rooftops, which are a
        # different vintage, technology and cleaning regime.
        "Degradation measured at -1.55 %/yr, ~3x the usual rule of thumb.",
        "Anemometer stops reporting after 2016; wind is imputed from a "
        "month-by-hour climatology (costs 0.54% of mean output).",
        "One row near 2025-08-22 is malformed in the provider's export.",
    ),
)

# A representative Tunisian residential rooftop, at Sfax.
#
# EVERY electrical and geometric value here is an ASSUMPTION. There is no open
# Tunisian PV registry -- no per-site installation records, no generation time
# series (CLAUDE.md section 10). This is a plausible archetype, not a real
# installation, and the national layer in step 5 will be an explicitly
# simulated fleet built from distributions of exactly these parameters.
SFAX_ROOFTOP = Site(
    key="tn_sfax",
    name="Sfax representative residential rooftop (SIMULATED archetype)",
    latitude=34.7406,
    longitude=10.7603,
    altitude=11.0,  # Open-Meteo's grid elevation for this cell
    tz="Africa/Tunis",
    # ASSUMPTION: near-optimal fixed tilt for latitude ~35 N, facing due south.
    # Real Tunisian rooftops vary widely with roof pitch; step 5 samples this.
    tilt=30.0,
    azimuth=180.0,
    # ASSUMPTION: a typical residential system under the PROSOL / net-metering
    # scheme. Decree 2016-1123 caps surplus sales to STEG at 30% of annual
    # production, which biases households toward self-consumption sizing.
    dc_capacity_w=3000.0,
    ac_capacity_w=3000.0,
    # ASSUMPTION: modern mono-Si, better than the 2008 poly-Si at DKASC.
    gamma_pdc=-0.0035,
    eta_inv_nom=0.97,
    # ASSUMPTION: commissioned five years ago.
    commissioned="2021-01-01",
    # Roof-mounted residential arrays sit close to the tiles with poor airflow,
    # so they run hotter than DKASC's rack. This is the physically right choice
    # here even though DKASC uses open_rack.
    temperature_model="close_mount_glass_glass",
    notes=(
        "No measured output exists for this site -- it is an archetype.",
        "Ageing and clear-sky calibration fitted at DKASC must NOT transfer.",
    ),
)

#: DKASC array 16C -- the FROZEN HOLDOUT.
#:
#: Downloaded early and then deliberately left untouched: never trained on,
#: never tuned on, never inspected. After several rounds of revising models,
#: baselines, calibration and dispatch design while looking at array 3's
#: 2025-03-31..08-20 window, that window has become development data. This array
#: is the only clean test left, and it is a hard one: east-facing rather than
#: north, 1.98 kW rather than 4.95, a different inverter, and its record runs to
#: 2026-02 where array 3 stopped reporting in 2025-08.
#:
#: It confounds an unseen period with an unseen site. That is stated rather than
#: hidden -- and it is closer to deployment than a pure temporal holdout, since
#: a Tunisian rooftop is also an array the model has never seen.
#:
#: Specs: https://dkasolarcentre.com.au/source/alice-springs/dka-m1-c-phase
DKASC_SITE_16C = Site(
    key="dkasc_site16c",
    name="DKASC Alice Springs #16C - BP Solar 1.98kW poly-Si fixed, EAST facing",
    latitude=-23.7621,
    longitude=133.8744,
    altitude=576.0,
    tz="Australia/Darwin",
    tilt=20.0,
    # 90 deg = due east. Array 3 faces 0 deg (north), so this directly exercises
    # the azimuth_offset feature introduced for the hemisphere transfer.
    azimuth=90.0,
    dc_capacity_w=1980.0,           # 12 x BP 3165J
    ac_capacity_w=2500.0,           # SMA SB 2500
    gamma_pdc=-0.0045,
    eta_inv_nom=0.96,
    commissioned="2008-11-11",
    temperature_model="open_rack_glass_glass",
    raw_file="fleet/dka_16c_east.csv",
    notes=("Frozen holdout. Never used for training, tuning or inspection.",),
)

#: DKASC array 16D -- the least-contaminated evaluation array available.
#:
#: NOT a verified installation. Provenance is weak and is recorded here rather
#: than implied:
#:
#:   IDENTITY    unknown. The local filename `dka_16d_west.csv` is a label from
#:               an earlier session with no download script recorded, and DKASC's
#:               Alice Springs listing has NO M1 D-phase. The provider page for
#:               the assumed code returns 404. The array is therefore identified
#:               only by its own behaviour.
#:   GEOMETRY    FITTED, not published. Tilt and azimuth were recovered by
#:               maximising correlation between measured output and modelled
#:               plane-of-array irradiance over 14,012 clear high-sun hours,
#:               restricted to 2015-2020 -- years before any evaluation window,
#:               so this fit does not touch evaluation data. Best fit azimuth
#:               280 deg (r=0.987); the diurnal profile is a near mirror image of
#:               16C's, peaking at 13-14h where 16C peaks at 10-11h. The same fit
#:               returned tilt 25 deg for BOTH arrays against a published 20 deg
#:               for 16C; on a 5 deg grid that is consistent with either, and the
#:               published value is kept.
#:   CAPACITY    ASSUMED EQUAL TO 16C, from a max/p99.9 power comparison over the
#:               full record (2.184 vs 2.226 kW max). That comparison DID read
#:               the evaluation period, as an aggregate over all years. A
#:               west-facing profile does not establish either identity or
#:               capacity, and capacity feeds the residual normalisation
#:               directly, so any transfer result using this array inherits the
#:               assumption.
#:
#: Untouched status: never used for fitting or model selection, and not inspected
#: before its first evaluation beyond the two items above. Weaker than "frozen",
#: and stated that way.
DKASC_SITE_16D = Site(
    key="dkasc_site16d",
    name="DKASC Alice Springs #16D - BP Solar 1.98kW poly-Si fixed, WEST facing",
    latitude=-23.7621,
    longitude=133.8744,
    altitude=576.0,
    tz="Australia/Darwin",
    tilt=20.0,
    azimuth=280.0,                  # fitted; training arrays are 0 and 90
    dc_capacity_w=1980.0,
    ac_capacity_w=2500.0,
    gamma_pdc=-0.0045,
    eta_inv_nom=0.96,
    commissioned="2008-11-11",
    temperature_model="open_rack_glass_glass",
    raw_file="fleet/dka_16d_west.csv",
    notes=(
        "Identity unverified; geometry fitted; capacity ASSUMED equal to 16C.",
        "Geometry fitted on 2015-2020, outside every evaluation window.",
        "Capacity inferred from a full-record power comparison that did read "
        "the evaluation period as an aggregate.",
        "Record ends 2025-09-04, earlier than 16C's 2026-02-19.",
    ),
)

SITES: dict[str, Site] = {
    DKASC_SITE_03.key: DKASC_SITE_03,
    DKASC_SITE_16C.key: DKASC_SITE_16C,
    DKASC_SITE_16D.key: DKASC_SITE_16D,
    SFAX_ROOFTOP.key: SFAX_ROOFTOP,
}


# --------------------------------------------------------------------------
# Modelling constants
# --------------------------------------------------------------------------

#: Resample the native 5-minute record to this frequency. Hourly matches the
#: resolution of the Open-Meteo forecasts the live pipeline will consume, so we
#: train at the resolution we deploy at.
TARGET_FREQ = "1h"

#: Solar elevation at or below this is treated as night: power is forced to
#: zero and the sample is excluded from every accuracy metric. Without this,
#: thousands of trivially-correct zeros inflate the apparent skill of every
#: model and make the intervals look far better calibrated than they are.
NIGHT_ELEVATION_DEG = 5.0

#: Chronological split. Never shuffle a time series: a random split lets the
#: model interpolate between neighbouring 5-minute samples and reports a score
#: it could never achieve on a real forecast.
SPLIT_TRAIN_END = "2021-12-31"
SPLIT_CAL_END = "2023-12-31"

#: Nominal coverage for the prediction intervals (step 2).
CONFIDENCE_LEVEL = 0.90
