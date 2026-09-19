"""Tunisia's governorates, and how rooftop PV capacity is spread across them.

READ THIS BEFORE QUOTING ANY NUMBER FROM HERE.

There is no open Tunisian PV registry. STEG and ANME publish nothing equivalent
to Australia's CER postcode capacity data or the UK's Sheffield Solar PVLive:
no per-site installation records, no generation time series, no per-governorate
capacity split. Every capacity figure below is therefore an **allocation of a
national total by proxy**, not a measurement. The national total itself is
sourced; its distribution is not.

That is a limitation to say out loud in the pitch, not to hide. What is being
demonstrated is the *upscaling method* -- the same one AEMO and Sheffield Solar
use on real registries -- applied to a transparently simulated fleet.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Governorate:
    key: str
    name: str
    latitude: float
    longitude: float
    population: int      # 2014 census, the last full enumeration
    region: str          # north / centre / south


#: Coordinates are the governorate capital. Populations are the 2014 census
#: (Institut National de la Statistique), the most recent full enumeration;
#: they are used only as relative weights, so their absolute age matters less
#: than their relative pattern.
GOVERNORATES: tuple[Governorate, ...] = (
    Governorate("tunis",       "Tunis",       36.8065, 10.1815, 1_056_247, "north"),
    Governorate("ariana",      "Ariana",      36.8625, 10.1956,   576_088, "north"),
    Governorate("ben_arous",   "Ben Arous",   36.7545, 10.2278,   631_842, "north"),
    Governorate("manouba",     "Manouba",     36.8081, 10.0972,   379_518, "north"),
    Governorate("nabeul",      "Nabeul",      36.4561, 10.7376,   787_920, "north"),
    Governorate("zaghouan",    "Zaghouan",    36.4029, 10.1429,   176_945, "north"),
    Governorate("bizerte",     "Bizerte",     37.2744,  9.8739,   568_219, "north"),
    Governorate("beja",        "Beja",        36.7256,  9.1817,   303_032, "north"),
    Governorate("jendouba",    "Jendouba",    36.5011,  8.7802,   401_477, "north"),
    Governorate("kef",         "Kef",         36.1742,  8.7049,   243_156, "north"),
    Governorate("siliana",     "Siliana",     36.0850,  9.3708,   223_087, "north"),
    Governorate("kairouan",    "Kairouan",    35.6781, 10.0963,   570_559, "centre"),
    Governorate("kasserine",   "Kasserine",   35.1676,  8.8365,   439_243, "centre"),
    Governorate("sidi_bouzid", "Sidi Bouzid", 35.0382,  9.4849,   429_912, "centre"),
    Governorate("sousse",      "Sousse",      35.8256, 10.6084,   674_971, "centre"),
    Governorate("monastir",    "Monastir",    35.7643, 10.8113,   548_828, "centre"),
    Governorate("mahdia",      "Mahdia",      35.5047, 11.0622,   410_812, "centre"),
    Governorate("sfax",        "Sfax",        34.7406, 10.7603,   955_421, "centre"),
    Governorate("gafsa",       "Gafsa",       34.4250,  8.7842,   337_331, "south"),
    Governorate("tozeur",      "Tozeur",      33.9197,  8.1335,   107_912, "south"),
    Governorate("kebili",      "Kebili",      33.7044,  8.9690,   156_961, "south"),
    Governorate("gabes",       "Gabes",       33.8815, 10.0982,   374_300, "south"),
    Governorate("medenine",    "Medenine",    33.3549, 10.5055,   479_520, "south"),
    Governorate("tataouine",   "Tataouine",   32.9297, 10.4518,   149_453, "south"),
)

#: Residential rooftop PV on the low-voltage network, STEG statement of
#: end-June 2026. CLAUDE.md section 10 records the conflict worth flagging: the
#: Ministry has separately cited "~400 MW installed vs 70 MW operational" for
#: low-voltage PV (Dec 2025), which appears to conflate registered pipeline
#: with commissioned capacity. This is the more recent figure and the one used
#: here; the ambiguity should be named rather than silently resolved.
NATIONAL_ROOFTOP_MW = 500.0

#: MV/HV connected PV, same STEG statement.
#:
#: Earlier versions excluded this entirely on the grounds that it is metered and
#: therefore visible to the operator. That was too blunt. The concept note puts
#: rooftop PV on BOTH low and medium voltage networks in scope, and the ~130 MW
#: covers commercial and industrial ROOFTOP installations as well as ground
#: plant. The platform now forecasts all three segments (see fleet.SEGMENTS) and
#: keeps the observability distinction explicit instead of dropping the capacity:
#: LV residential is the genuinely invisible part, while MV-connected industrial
#: is more likely already metered, so the two carry different operational value
#: even though both are forecast.
NATIONAL_MV_HV_MW = 130.0


def capacity_allocation(national_mw: float = NATIONAL_ROOFTOP_MW) -> pd.DataFrame:
    """Split the national rooftop total across governorates by population.

    ASSUMPTION, and a crude one. Population is a transparent proxy, not a
    model of adoption. Real PROSOL uptake almost certainly skews towards
    higher-income coastal governorates and towards detached housing, and away
    from dense apartment stock in Tunis -- so this allocation probably
    over-weights the capital and under-weights the Sahel.

    The reason to use it anyway is that it is auditable: one line, one
    assumption, no hidden parameters. A judge can disagree with it in a
    sentence, which is better than a sophisticated allocation nobody can check.
    """
    df = pd.DataFrame([
        {"key": g.key, "name": g.name, "latitude": g.latitude,
         "longitude": g.longitude, "population": g.population, "region": g.region}
        for g in GOVERNORATES
    ]).set_index("key")

    df["population_share"] = df["population"] / df["population"].sum()
    df["capacity_mw"] = df["population_share"] * national_mw
    return df


def latitudes() -> list[float]:
    return [g.latitude for g in GOVERNORATES]


def longitudes() -> list[float]:
    return [g.longitude for g in GOVERNORATES]


# --------------------------------------------------------------------------
# Districts (delegations / mu'tamadiyat)
# --------------------------------------------------------------------------
#
# The concept note asks for forecasts at district, governorate and national
# scale. Tunisia's administrative tier below the governorate is the delegation.
#
# WHAT IS REAL AND WHAT IS NOT:
#   counts     approximate published delegation counts per governorate. They sum
#              to DELEGATION_TOTAL below against the ~264 usually cited
#              nationally, so treat them as the right order and structure rather
#              than an authoritative register; INS publishes the definitive list.
#   names      NOT modelled. Districts are identified as "<Governorate> D<n>".
#              Inventing delegation names would look authoritative and be false.
#   capacity   SIMULATED. Split within each governorate by a reproducible draw
#              concentrated on the lower-numbered districts, standing in for the
#              fact that PV uptake is not uniform across a governorate.
#   weather    SHARED WITHIN A GOVERNORATE, deliberately. Open-Meteo's grid is
#              ~11 km and CAMS dust is ~40 km, both coarser than most
#              delegations, so district forecasts here differ by fleet
#              composition and capacity, NOT by resolved local weather. Claiming
#              otherwise would be claiming resolution the inputs do not have.

DISTRICT_COUNTS: dict[str, int] = {
    "tunis": 21, "ariana": 7, "ben_arous": 12, "manouba": 8,
    "nabeul": 16, "zaghouan": 6, "bizerte": 14, "beja": 9,
    "jendouba": 9, "kef": 11, "siliana": 11, "kairouan": 13,
    "kasserine": 13, "sidi_bouzid": 12, "sousse": 16, "monastir": 13,
    "mahdia": 11, "sfax": 16, "gafsa": 11, "tozeur": 5,
    "kebili": 6, "gabes": 10, "medenine": 9, "tataouine": 7,
}
DELEGATION_TOTAL = sum(DISTRICT_COUNTS.values())

#: Urban districts carry more commercial load; the governorates containing the
#: big industrial zones carry more industrial. ASSUMPTION, stated so it can be
#: argued with.
URBAN = {"tunis", "ariana", "ben_arous", "manouba", "sousse", "monastir", "sfax"}
INDUSTRIAL_HEAVY = {"sfax", "gabes", "bizerte", "ben_arous"}


@dataclass(frozen=True)
class District:
    key: str
    name: str
    governorate: str
    capacity_mw: float
    segment_mix: dict           # segment key -> share of this district's capacity


def rooftop_total_mw() -> float:
    """All rooftop capacity the platform forecasts, across every segment."""
    from .fleet import SEGMENTS

    return float(sum(s.national_mw for s in SEGMENTS))


def districts(national_mw: float = None, seed: int = 7) -> list[District]:
    """Split every governorate into its delegations, with a segment mix.

    Capacity within a governorate follows a Dirichlet draw with concentration
    below 1, which produces a realistically uneven split -- a few districts
    carrying much of the capacity -- rather than an implausible flat division.
    """
    import numpy as np

    from .fleet import segment_shares

    # The full rooftop fleet across all segments, not the residential subset:
    # NATIONAL_ROOFTOP_MW alone would leave commercial and industrial capacity
    # in the segment mix with no geography to sit in.
    caps = capacity_allocation(rooftop_total_mw() if national_mw is None
                               else national_mw)
    base = segment_shares()
    rng = np.random.default_rng(seed)

    out: list[District] = []
    for gov in GOVERNORATES:
        n = DISTRICT_COUNTS[gov.key]
        gov_mw = float(caps.loc[gov.key, "capacity_mw"])
        share = rng.dirichlet(np.full(n, 0.8))
        share = np.sort(share)[::-1]           # D1 is the largest district

        mix = dict(base)
        if gov.key in URBAN:
            mix["commercial"] *= 1.6
        if gov.key in INDUSTRIAL_HEAVY:
            mix["industrial"] *= 1.8
        tot = sum(mix.values())
        mix = {k: v / tot for k, v in mix.items()}

        for i in range(n):
            out.append(District(
                key=f"{gov.key}_d{i + 1}",
                name=f"{gov.name} D{i + 1}",
                governorate=gov.key,
                capacity_mw=gov_mw * float(share[i]),
                segment_mix=mix,
            ))
    return out
