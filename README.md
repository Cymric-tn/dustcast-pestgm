# DustCast

A forecasting platform for Tunisia's rooftop solar production, at district,
governorate and national scale, with per-site calibrated uncertainty (the
national aggregation is illustrative — see §7) and an operator decision layer.

Submission for the **PESTGM 7.0 CDC Tech Challenge, Track 1**.

---

## 1. What the problem is

Tunisia has roughly 500 MW of residential rooftop PV plus ~130 MW on MV/HV
networks. STEG cannot see this generation minute to minute: it is behind the
meter, unmetered at the granularity that matters, and weather-driven. The
operator holds more reserve than necessary and gets surprised by ramps.

At present penetration there is **no duck curve** — rooftop PV shaves about 9.9%
off midday net load, and the overnight demand trough is still the system
minimum. But **PV falling away is 36% of the steepest three-hour evening ramp**
(281 MW of 787 MW in the modelled case). That share is the part a production
forecast can anticipate, and it is the argument for the platform.

## 2. What it does

```
Open-Meteo NWP  ──►  pvlib physics  ──►  competing models  ──►  selector
   + CAMS dust                              L0 … L4            (held-back
                                                                window)
                                                    │
                        ┌───────────────────────────┤
                        ▼                           ▼
              conformal intervals          district → governorate → national
                        │                           │
                        └──────────┬────────────────┘
                                   ▼
                    decision layer · HTTP API · dashboard
```

| Capability | Status |
|---|---|
| District / governorate / national scales | 266 prototype units → 24 governorates → national |
| LV and MV segments | residential 500 MW (LV), commercial 78 MW (LV-MV), industrial 52 MW (MV) |
| Horizons produced | J+0 … J+7 |
| Horizons **evaluated** | J+0 … J+3 only — see §5 |
| Learning from observed error | five models refit monthly, selected on held-back data |
| Automatic refresh | rebuilds when new weather lands in cache, not on a timer |
| Machine exchange | documented HTTP API + working sample consumer |
| Uncertainty | per-site conformal intervals (coverage measured); national aggregation **illustrative only** |

## 3. The core idea, and where it broke

**Physics first, ML on the residual.** pvlib computes what a panel *should*
produce; a model learns only the gap. The residual has no diurnal cycle, no
seasonal cycle, and — in principle — no dependence on installed capacity.

**In practice it did depend on capacity, and that was a bug.**
`residual_kw = actual − expected` is in kW. The physics removes the diurnal and
seasonal *shape*, not the *scale*. A model trained on a 6.0 kW array and applied
to a 2.5 kW one over-corrects by the capacity ratio — a positive bias at every
hour. Fixed in `model.normalise_residual`; it accounted for about half the
transfer failure below.

## 4. The competing models

| | model | fitted from |
|---|---|---|
| **L0** | physics | site geometry only — needs no local data |
| **L1** | + fitted derate | one scalar |
| **L2** | + diurnal bias | thirteen numbers: mean residual by hour |
| **L3** | + boosted | XGBoost on the capacity-normalised residual |
| **L4** | + diurnal, then boosted | L2 first, XGBoost on what it leaves |

They **compete**. This is not a progression in which more data eventually
promotes the ML.

### Chronological learning replay — `scripts/step16_replay.py`

Every candidate is refitted each month on past data only, the best on a
held-back 42-day window is selected, and that choice forecasts the next month
blind. Nineteen months, DKASC array 16C:

| | |
|---|---|
| L2 selected | 14 / 19 months |
| L4 selected | 5 / 19 months |
| Switches | 7 |
| Selector was the month's actual winner | 12 / 19 |
| **An ML model won the month outright** | **8 / 19** |
| Selector vs best fixed choice *in hindsight* | **−0.5%** |

Pooled over the replay: L2 **3.90%** nMAE, L4 4.16%, L3 4.83%, L0 5.15%,
L1 5.18%.

The ML is competitive, not decorative — but its losing months are the heavy
ones, which is why it trails once hours are pooled.

## 5. Horizons — produced vs evaluated

`scripts/step17_horizons.py`. nMAE %, array 3:

| horizon | L0 physics | L2 diurnal | best ML | evaluated? |
|---|---|---|---|---|
| J+0 | 5.57 | 3.90 | **2.93** (L4) | yes |
| J+1 | 6.67 | 4.42 | **4.23** (L3) | yes |
| J+2 | 6.70 | 4.52 | **3.72** (L3) | yes |
| J+3 | 6.74 | 4.67 | **3.78** (L3) | yes |
| J+4 … J+7 | — | — | — | **no — produced, not validated** |

Two warnings attached to this table:

1. **It is not independent.** Array 3 is the array the feature set and
   hyperparameters were chosen on. The ML advantage here (CIs exclude zero:
   +24.7%, +4.3%, +17.6%, +19.0%) is consistent with those choices having been
   fitted to this array.
2. **The J+1→J+2 ordering is non-monotonic** and untested. Do not read a horizon
   trend into these three numbers in either direction.

Intra-day (J+0) is the same-day model run, **not a nowcast** — there is no
sub-hourly or sky-imagery input, so no minute-scale skill is claimed.

## 6. The transfer result

Frozen evaluations on arrays never used for fitting. Skill vs `physics + diurnal
bias`, day-ahead, forecast weather:

| test | array | setup | DustCast | + site bias |
|---|---|---|---|---|
| A *(diagnostic, 2nd look)* | 16C east | kW target | −50.5% | −38.5% |
| A | 16C east | **normalized** | −23.1% | −15.6% |
| B *(least-contaminated)* | 16D west | normalized, 2 arrays | −66.2% | **−9.4%** |
| C *(development experiment)* | 16D west | target array in pool | −10.6% | **−7.1%** |

Labels are load-bearing. **A** is a second look at an array already used once.
**C** reuses B's window and was written after seeing B, so it carries no
independent weight. Only **B** is close to a clean test, and even it is
qualified: 16D's identity is unverified, its geometry fitted (on 2015–2020,
outside every evaluation window), its capacity assumed.

Winning on the development array and losing on unseen ones is the signature of
site overfitting, quantified on both sides.

## 7. Uncertainty

Conformal intervals, 90% nominal, scored on identical hours (frozen 16C
holdout). Coverage and sharpness are reported together — either alone is
gameable, since a band from 0 to ∞ has perfect coverage and zero value.

> **The national aggregation is illustrative, not a validated interval.** The
> per-site band is conformal-calibrated on three arrays at one Australian
> facility and its site coverage is measured. Aggregating those bands to a
> national figure uses a shrinkage factor that assumes equal regional error
> scales and one common correlation (neither holds for population-weighted
> regions), treats conformal half-widths as standard deviations (they are not),
> and estimates the correlation from forecast *clear-sky index* as a proxy for
> forecast-*error* correlation — a substitution we measured to differ
> substantially (0.444 vs 0.089 on the same data). The API returns these bounds
> as `low_illustrative` / `high_illustrative` with the caveat attached.

| band | PICP | PINAW | Winkler |
|---|---|---|---|
| physics + diurnal bias | 88.9% | **0.197** | **0.980** |
| DustCast | 89.9% | 0.299 | 1.153 |
| DustCast CQR | 94.6% | 0.351 | 1.006 |

The ML band is wider for the coverage it achieves.

> **Provenance note.** §6 and §7 are scored on the same holdout array and period
> but come from different scripts, and their daylight/feature filters leave
> slightly different row counts (2,007 vs 2,188 hours). The comparisons *within*
> each table are on identical hours, which is what the skill and coverage figures
> require; figures should not be carried between the two tables. The report
> quotes one frame throughout.

## 8. The Tunisian layer — and its binding constraint

Every model that beats physics learns from a site's **own observed production**.
No Tunisian rooftop in this fleet has any: no per-site registry or generation
series was available to this project to fit a correction to.

**So the national layer runs L0 physics. That is not a shortcut — it is what the
absence of Tunisian data forces.**

The honest accuracy statement for a Tunisian rooftop with no history is therefore
L0's own measured error: **5.57% nMAE intra-day, 6.67% at J+1** — explicitly a
transfer claim from an Australian desert site, not a Tunisian measurement.

The frozen tests in §6 do **not** describe this case: every baseline in them
adapts using the target array's past output.

The fleet is a **transparent simulation** throughout — 266 **prototype
aggregation units** built from published delegation counts, with capacity split
and segment mix as stated assumptions. We have *not* established that these
correspond to STEG's operational units, which are more likely defined by network
topology (HV/MV substations and MV feeders) than by administrative boundaries;
the intended mapping is a lookup from connection point to feeder, replacing the
administrative split without changing the aggregation code. Unit *names* are not
modelled.

All units in a governorate currently share one weather series. That is a
simplification we chose to keep the service to 24 grid requests, **not** a limit
imposed by the data: an ~11 km forecast grid resolves variation well inside a
Tunisian governorate. CAMS dust at ~40 km is the coarser constraint.

## 9. Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Production data is not in the repo (~880 MB across three arrays):

```bash
B=https://solarcentre.spinifexvalley.com.au/export
curl -C - -o data/raw/dkasc_site03_5min.csv  $B/70-Site_DKA-M5_A-Phase.csv
curl -C - -o data/raw/fleet/dka_16c_east.csv $B/105-Site_DKA-M1_C-Phase.csv
curl -C - -o data/raw/fleet/dka_16d_west.csv $B/81-Site_DKA-M2_A-Phase.csv
```

| array | provider source | bytes | SHA-256 (first 16) |
|---|---|---|---|
| 3 | 70, `Site_DKA-M5_A-Phase` | 313,051,557 | `0cb0746fe5abda4e` |
| 16C | 105, `Site_DKA-M1_C-Phase` | 285,663,460 | `2b155248911195b2` |
| 16D | 81, `Site_DKA-M2_A-Phase` | 284,948,839 | `b829d6ce08612026` |

All three remote `Content-Length` values match these files exactly. The provider
labels source 105 "BP Solar, 2.0kW, poly-Si, Fixed, 2008, **East**" and source 81
the same but **West**, confirming capacity and orientation for both.

Rebuild every deliverable from one forecast snapshot. This is a dependency
chain, and the script verifies afterwards that the report, dashboard, screenshot
and API example all quote the same snapshot:

```bash
.venv/bin/python scripts/build_report.py
```

| | |
|---|---|
| `scripts/step16_replay.py` | learning loop — monthly refit and selection |
| `scripts/step17_horizons.py` | evaluated performance J+0…J+3 |
| `scripts/step18_district_national.py` | district → governorate → national |
| `scripts/step19_consumer.py` | API + downstream consumer, end to end |
| `scripts/step14_frozen_holdout.py` | the frozen evaluation |
| `scripts/step15_multiarray.py` | capacity-normalised multi-array tests |

### The API

```bash
.venv/bin/uvicorn dustcast.api:app --port 8000
```

Then open <http://localhost:8000/>. The dashboard fetches `/dashboard` on load
and every five minutes, and its masthead says whether it is showing **live** data
or the copy baked in at build time — it falls back to that copy when no API
answers, so the same file also works as a standalone artifact.

| endpoint | returns |
|---|---|
| `GET /` | the operator dashboard, served from the same origin as its data |
| `GET /dashboard` | everything that page draws, in one request |
| `GET /health` | freshness and horizon length |
| `GET /meta` | scales, segments, capacity, provenance |
| `GET /forecast/national` | national MW time series |
| `GET /forecast/governorate/{key}` | one governorate |
| `GET /forecast/governorates` | all 24, for a map |
| `GET /forecast/district/{key}` | one district, with its segment mix |
| `GET /models/performance` | measured nMAE by horizon |
| `GET /models/selection` | what the learning loop chose, month by month |
| `POST /refresh` | force a rebuild |

Every payload carries the model that produced it, whether the fleet is
simulated, and which horizons are evaluated. A downstream tool that silently
treats a simulated national total as a metered one is the failure mode this
interface exists to prevent — the sample consumer reads those flags and tags its
output `ESTIMATED`.

## 10. Known weaknesses

- **Suitable installation-level Tunisian data were not available to this
  project.** We found no accessible per-site registry or generation series. This
  is a statement about what we could obtain — ANME guidance describes
  commissioning records and inverter production histories, which would change
  what is possible here if accessible. The national view is a simulation; only
  the *upscaling method* is validated, on real multi-site data.
- **Tunisian accuracy and national interval coverage are unvalidated.** The
  5.57% / 6.67% figures were measured at DKASC in Australia. Site coverage is
  measured; national coverage depends on an estimated spatial correlation that
  was never checked against a measured national series.
- **Training data is Australian.** Alice Springs is climatically close to
  southern Tunisia, less so to the coast.
- **Soiling has no ground truth.** It is a relative index and a maintenance
  trigger, never a measured percentage.
- **The ML does not currently improve accuracy** on unseen arrays. It ships
  because it is competitive month to month and because the selector makes its
  use conditional and visible — not because it wins.
- **J+4 to J+7 are produced but not evaluated.**
- **Capacity figures conflict.** ~500 MW residential (STEG, end-June 2026) is
  used here; the Ministry's "~400 MW installed vs 70 MW operational" appears to
  conflate pipeline with commissioned capacity.
- **Dust magnitudes are transferred from elsewhere** (Spain 2022, West Africa
  2021). They establish mechanism and order of magnitude, not Tunisian values.

## 11. Layout

```
dustcast/
  physics.py    pvlib clear-sky, transposition, ageing
  features.py   21 features, hemisphere-invariant
  model.py      residual model, capacity normalisation, baselines
  learning.py   competing models L0..L4 + replay engine
  conformal.py  calibrated intervals
  fleet.py      virtual fleet, LV/MV segments, upscaling
  tunisia.py    governorates, districts, capacity allocation
  service.py    forecast build + auto-refresh
  api.py        HTTP interface
  dispatch.py   reserve, ramp, decision costs
```
