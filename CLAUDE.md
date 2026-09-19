# CLAUDE.md — DustCast

> National Intelligent Platform for Forecasting Rooftop Solar Production (Tunisia)
> Hackathon Track 1 submission. This file is the working context for the project:
> what was decided, why, how it's built, and what's still unresolved.

---

## 1. The problem in one paragraph

Tunisia has ~500 MW of residential rooftop PV plus ~130 MW on MV/HV networks (STEG, end-June 2026), inside a national PV fleet of ~895 MW (IRENA 2026). STEG cannot see this generation minute-to-minute. It is behind-the-meter, unmetered at the granularity that matters, and weather-driven. The operator therefore holds more spinning reserve than necessary, and gets surprised by ramps it could have anticipated. The track brief asks for a platform that forecasts this production at multiple spatial and temporal scales, quantifies uncertainty, analyses grid impact, and produces decision support.

## 2. Track selection

**Track 1 (solar forecasting), not Track 2 (load shedding).**

Track 2 is an operations-research problem — combinatorial scheduling, fairness constraints, priority ordering. Track 1 is ML + time series + physical modelling, which matches the team's existing background (XGBoost, 1D-CNN, DSP pipelines, embedded/IoT). Track 1 also has a hardware hook that Track 2 does not.

## 3. Core concept

Three ideas stacked into one platform:

1. **Physics first, ML on the residual.** A clear-sky physical model computes what a panel *should* produce. ML learns only the gap between that and reality.
2. **Saharan dust as the differentiator.** Free CAMS dust forecasts feed both an irradiance-attenuation effect and a soiling-accumulation state. Almost no competing team will model this.
3. **Uncertainty as a first-class output.** Conformal prediction gives calibrated intervals with coverage guarantees, which is what converts a forecast into an operator decision.

Plus a spatial layer: upscale from a small monitored sample to governorate and national totals — the standard method used by AEMO (Australia) and Sheffield Solar / Open Climate Fix (UK).

### Why residual learning

```
sunlight forecast → [ losses ] → actual electricity
```

Physics knows the first arrow exactly (sun angle, panel tilt, azimuth, area, time of year). It cannot know the losses:

| Loss source | Magnitude | Learnable from |
|---|---|---|
| Heat derating | ~0.4%/°C above 25°C; 15–18% on a 45°C Tunisian afternoon | temperature, wind |
| Soiling | accumulates over weeks; dust storms can halve output | dust AOD history, days since rain |
| Inverter clipping/efficiency | a few % | irradiance level, time |
| Shading / real orientation | site-specific | learned per-site offset |
| Ageing | ~0.5%/year | site age |
| NWP forecast bias | systematic, conditional | all weather features |

So the model target is:

```
residual = actual_power − clearsky_physics_prediction
```

This target has no diurnal cycle, no seasonal cycle, no dependence on installed capacity — physics stripped all of that out. What remains is a small, roughly zero-centred signal driven by clouds, dust and heat. Much easier to learn, and it generalises across sites and climates far better than raw power. Critically, **it needs less training data**, which matters because Tunisian data does not exist.

---

## 4. Architecture

### 4.1 Offline pipeline (runs once, ships trained artefacts)

```
DKASC Alice Springs production data
        │
        ├─→ pvlib ModelChain (Ineichen–Perez clear-sky)  ──→ expected_power
        │
        └─→ actual_power
                    │
                    ▼
        residual = actual − expected          ← training target
                    │
                    ▼
              XGBoost regressor
                    │
                    ▼
        MAPIE conformal calibration (held-out split)
                    │
                    ▼
        model.pkl + calibrator.pkl
```

### 4.2 Live pipeline (runs per dashboard refresh)

```
Open-Meteo forecast API ──┐
                          ├──→ feature frame (per governorate, hourly, 48h)
Open-Meteo air-quality  ──┘
  (CAMS dust + AOD)
                          │
                          ▼
              pvlib on forecast weather  ──→ expected_power
                          │
                          ▼
              XGBoost predicts residual   ──→ point forecast
                          │
                          ▼
              MAPIE → [lower, upper] @ 90%
                          │
                          ▼
              upscale: sample yield × installed capacity
                          │
                          ├──→ governorate totals
                          └──→ national total
                          │
                          ▼
              decision layer (reserve / ramp / cleaning)
                          │
                          ▼
              map + curves + decision panel
```

### 4.3 Optional hardware loop

```
ESP32 (INA219 current/voltage + DS18B20 panel temp + pyranometer)
        │ MQTT
        ▼
   backend ingest → live dot overlaid on forecast band
```

One node. Not a fleet. Its purpose is to prove the pipeline touches physical reality, not to provide training data.

---

## 5. Data sources

Each source does one job. Do not confuse them.

| Source | Job | Direction | Resolution | Latency | Cost |
|---|---|---|---|---|---|
| **NASA POWER** | long weather history (2001–now) for training | past | 0.5° (~50 km) | 5–7 days for solar | free, no key |
| **Open-Meteo forecast** | live 1–7 day weather + irradiance | future | fine | real-time | free (non-commercial) |
| **Open-Meteo air-quality** | dust + AOD (CAMS-derived) | past 92d + future 5d | CAMS global | real-time | free, no key |
| **CAMS ADS** | primary dust source, full grid | future 5d, 2×/day | 0.4° | ~hours | free, registration |
| **PVGIS** | TMY, panel geometry, synthetic histories | climatology | Europe + Africa | n/a | free |
| **DKASC Alice Springs** | **training target — real desert PV production** | past 15+ yr | 5-min / 10-s | n/a | free |

### Endpoint reference

Open-Meteo weather forecast — <https://open-meteo.com/en/docs>
Open-Meteo air quality — <https://open-meteo.com/en/docs/air-quality-api>
Open-Meteo historical (ERA5) — <https://open-meteo.com/en/docs/historical-weather-api>
NASA POWER — <https://power.larc.nasa.gov/docs/services/api/>
PVGIS — <https://joint-research-centre.ec.europa.eu/pvgis-online-tool_en>
CAMS ADS dataset — <https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts>
CAMS request walkthrough (EUMETSAT) — <https://dust.trainhub.eumetsat.int/docs/cams_global.html>
CAMS forecast charts (for slides) — <https://atmosphere.copernicus.eu/global-forecast-plots>
DKASC — <https://dkasolarcentre.com.au/download>

### Dust variables — the distinction that matters

- `aerosol_optical_depth` — whole atmospheric column. Drives **irradiance attenuation** (sunlight blocked before it reaches the panel).
- `dust` — surface-level concentration. Drives **soiling accumulation** (particles settling on glass).

Use both. They are different physical mechanisms with different time constants: AOD affects today, soiling integrates over weeks.

### Libraries

| Library | Role |
|---|---|
| `pvlib` | clear-sky models, ModelChain, temperature models, solar position |
| `xgboost` | residual regressor |
| `mapie` | conformal prediction intervals |
| `pandas` / `numpy` | data handling |
| `requests` | API calls |
| `cdsapi` | CAMS ADS direct access (only if going past Open-Meteo) |
| `folium` / `plotly` / `deck.gl` | map layer |
| `fastapi` + `uvicorn` | backend |

Optional, only if scope allows: `neuralforecast`, `ngboost`, `darts`, `quartz-solar-forecast` (Open Climate Fix, pip-installable, GBT on 9 Open-Meteo NWP variables, trained on 25,000 sites, ~13% MAE excluding night — useful as an external baseline to benchmark against).

---

## 6. Code snippets

### 6.1 Open-Meteo dust + AOD (Sfax, 60 days back + 5 forward)

```
https://air-quality-api.open-meteo.com/v1/air-quality
  ?latitude=34.74&longitude=10.76
  &hourly=dust,aerosol_optical_depth
  &domains=cams_global
  &past_days=60&forecast_days=5
```

`domains=cams_global` is **required** — the default `auto` may resolve to the European CAMS domain, which does not properly cover Tunisia.

```python
import requests, pandas as pd

def fetch_dust(lat, lon, past_days=60, forecast_days=5):
    r = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": lat, "longitude": lon,
            "hourly": "dust,aerosol_optical_depth",
            "domains": "cams_global",
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": "Africa/Tunis",
        },
        timeout=30,
    )
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")
```

### 6.2 CAMS direct (only if Open-Meteo is insufficient)

```python
import cdsapi

c = cdsapi.Client()  # reads ~/.cdsapirc
c.retrieve(
    "cams-global-atmospheric-composition-forecasts",
    {
        "variable": "dust_aerosol_optical_depth_550nm",
        "date": "2026-09-08/2026-09-08",
        "time": "00:00",
        "leadtime_hour": [str(h) for h in range(0, 121, 3)],
        "type": "forecast",
        "area": [38, 7, 30, 12],   # N, W, S, E — Tunisia bounding box
        "format": "netcdf",
    },
    "dust_tunisia.nc",
)
```

### 6.3 Clear-sky physics baseline

```python
import pvlib
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

def build_chain(lat, lon, tz, tilt, azimuth, module, inverter):
    site = Location(lat, lon, tz=tz)
    system = PVSystem(
        surface_tilt=tilt,
        surface_azimuth=azimuth,          # 180 = due south
        module_parameters=module,
        inverter_parameters=inverter,
        temperature_model_parameters=(
            TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
        ),
    )
    return ModelChain(system, site, aoi_model="physical",
                      spectral_model="no_loss")

def expected_power(chain, weather):
    """weather: DataFrame with ghi, dni, dhi, temp_air, wind_speed."""
    chain.run_model(weather)
    return chain.results.ac
```

Clear-sky reference (for the normalised clear-sky index feature):

```python
clearsky = site.get_clearsky(times, model="ineichen")
kt = weather["ghi"] / clearsky["ghi"].clip(lower=1)   # clear-sky index
```

### 6.4 Feature engineering

```python
def make_features(weather, dust, site_meta):
    X = pd.DataFrame(index=weather.index)

    # weather
    X["temp_air"]   = weather["temp_air"]
    X["wind_speed"] = weather["wind_speed"]
    X["rh"]         = weather["relative_humidity"]
    X["ghi"]        = weather["ghi"]
    X["kt"]         = weather["clearsky_index"]

    # dust: instantaneous attenuation
    X["aod"]  = dust["aerosol_optical_depth"]
    X["dust"] = dust["dust"]

    # dust: accumulated soiling state
    X["dust_7d"]  = dust["dust"].rolling("7D").mean()
    X["dust_30d"] = dust["dust"].rolling("30D").mean()
    X["days_since_rain"] = days_since_rain(weather["precipitation"])
    X["soiling_index"]   = soiling_accumulation(
        dust["dust"], weather["precipitation"]
    )

    # temporal
    X["hour"]        = X.index.hour
    X["doy"]         = X.index.dayofyear
    X["hour_sin"]    = np.sin(2*np.pi*X["hour"]/24)
    X["hour_cos"]    = np.cos(2*np.pi*X["hour"]/24)
    X["solar_elev"]  = weather["solar_elevation"]

    # site
    X["tilt"]     = site_meta["tilt"]
    X["azimuth"]  = site_meta["azimuth"]
    X["age_yrs"]  = site_meta["age_years"]

    return X
```

Soiling accumulation — a simple deposition/cleaning balance. Reset on rain above a threshold, otherwise integrate deposition:

```python
def soiling_accumulation(dust, precip, rain_reset_mm=3.0, k=1e-4):
    s, out = 0.0, []
    for d, p in zip(dust.fillna(0), precip.fillna(0)):
        if p >= rain_reset_mm:
            s = 0.0
        else:
            s += k * d
        out.append(min(s, 0.30))      # cap loss at 30%
    return pd.Series(out, index=dust.index)
```

`k` and `rain_reset_mm` are **tunable, not measured**. See constraints.

### 6.5 Residual model + conformal wrapper

```python
from xgboost import XGBRegressor
from mapie.regression import MapieRegressor

# target
y = actual_power - expected_power

# chronological split — never shuffle time series
cut_train, cut_cal = "2022-12-31", "2023-12-31"
X_tr, y_tr = X[:cut_train], y[:cut_train]
X_cal, y_cal = X[cut_train:cut_cal], y[cut_train:cut_cal]
X_te, y_te = X[cut_cal:], y[cut_cal:]

base = XGBRegressor(
    n_estimators=600, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="reg:squarederror", n_jobs=-1,
)

mapie = MapieRegressor(base, method="plus", cv="prefit")
base.fit(X_tr, y_tr)
mapie.fit(X_cal, y_cal)

resid_pred, resid_int = mapie.predict(X_te, alpha=0.10)   # 90%

power_pred  = expected_power_test + resid_pred
power_lower = np.maximum(expected_power_test + resid_int[:, 0, 0], 0)
power_upper = expected_power_test + resid_int[:, 1, 0]
```

Night handling: force `power = 0` and collapse the interval to zero when solar elevation ≤ 0. Otherwise night hours contaminate every aggregate metric and inflate apparent coverage.

### 6.6 Spatial upscaling

```python
def upscale(sample_forecasts, sample_capacity_kw, region_capacity_kw):
    """Sample sites → regional total. AEMO / Sheffield Solar method."""
    yield_frac = sample_forecasts.sum(axis=1) / sample_capacity_kw
    return yield_frac * region_capacity_kw

national = sum(
    upscale(gov_sample[g], gov_sample_cap[g], gov_installed_cap[g])
    for g in GOVERNORATES
)
```

Intervals propagate the same way — scale the sample's lower/upper bounds. Note this assumes perfect correlation across the region, which **over-states** the national interval width. A defensible refinement is to scale interval width by `1/sqrt(n_effective)` with `n_effective` derived from inter-site correlation. Mention this in the pitch; implement only if time allows.

### 6.7 Decision layer

```python
def reserve_requirement(net_load_lower, net_load_point):
    """Reserve sized off the pessimistic edge, not the point forecast."""
    return (net_load_lower - net_load_point).abs().max()

def ramp_risk(forecast, window="3h", limit_mw_per_h=None):
    ramp = forecast.diff().rolling(window).sum()
    return {"max_ramp": ramp.max(),
            "breach": bool(limit_mw_per_h and ramp.max() > limit_mw_per_h)}

def cleaning_recommendation(soiling_idx, rain_forecast_mm,
                            threshold=0.08, cost_per_clean=None):
    """Recommend cleaning only if loss exceeds threshold AND no rain coming."""
    if soiling_idx.iloc[-1] < threshold:
        return None
    if rain_forecast_mm.sum() >= 3.0:
        return "defer — rain forecast will clean panels"
    return f"clean now — modelled loss {soiling_idx.iloc[-1]:.1%}"
```

---

## 7. Metrics

Report these. **Do not lead with RMSE.**

| Metric | What it shows |
|---|---|
| **CRPS** | overall probabilistic forecast quality |
| **Pinball loss** | per-quantile accuracy (asymmetric, matches operator cost) |
| **PICP** | interval coverage — does the 90% band actually contain 90%? |
| **PINAW** | interval sharpness — narrow bands are worth money |
| **Ramp detection skill** | did we catch the sunset/dust ramps? |
| **Reserve shortfall** | how often would the operator have been caught short? |

PICP and PINAW must be reported together. Either alone is gameable: a band from 0 to infinity has perfect coverage and zero value.

Also report **skill score vs. two baselines**: persistence, and clear-sky physics alone. If the ML doesn't beat clear-sky physics, the ML is not earning its place.

---

## 8. Build order

Freeze at whichever step you reach. Every stopping point from 3 onward is a coherent submission.

| # | Build | Gate |
|---|---|---|
| 1 | DKASC ingest + pvlib baseline + XGBoost residual, one site, no UI | residual model beats clear-sky-only |
| 2 | MAPIE conformal calibration | PICP within ±3% of nominal |
| 3 | Live pipeline for one Tunisian city (Open-Meteo → forecast → interval) | end-to-end runs on real forecast data |
| 4 | Dust features + soiling index + cleaning logic | intervals visibly widen on dust events |
| 5 | Virtual fleet + governorate/national upscaling + grid-impact layer | national number is plausible vs. installed capacity |
| 6 | Map UI + decision panel | demo-able |
| 7 | ESP32 node | only if time remains |

**Cut list, in order:** ESP32 first, then the national upscaling refinement, then map polish. **The dust layer is never cut** — it is the entire differentiation.

---

## 9. Rejected alternatives (and why — expect judges to ask)

**Transformers (PatchTST, iTransformer, Informer).** A 2025 Tunisia-sited study (Rjim Matouk) found a Prophet + mutual-information pipeline beat iTransformer, PatchTST, TimesFM, Chronos and SOFTS on MAPE with roughly 96% less training time, and flagged 6–13 h training and 156–507 ms inference as operational barriers. PatchTST is the strongest transformer for solar (its patching aligns with the diurnal cycle), but it does not reliably beat gradient boosting at site level and needs far more data than we have. **Frame this as an engineering decision, not a shortcut.**

**Time-series foundation models (Chronos, TimesFM, Moirai).** Genuinely interesting for *cold-start* — forecasting a brand-new rooftop with zero history by seeding a foundation model with a pvlib/PVGIS synthetic history. Real research direction, real fit with Tunisia's stream of new unmonitored installs. Cut for scope and inference cost. Keep as "future work" in the pitch.

**Spatio-temporal GNNs.** The scientifically strongest answer to the "multiple spatial scales" requirement — nearby PV sites act as virtual weather stations sensing cloud motion (GCLSTM/GCTrafo, DEST-GNN). Highest-risk build in a hackathon: graph construction plus training time. Cut.

**BTM disaggregation** (recovering hidden PV from feeder net load). Complementary and operator-relevant, but needs a plausible net-load signal we'd have to simulate anyway. Cut.

**Sky-image / satellite nowcasting** (1–30 min). Would extend temporal coverage at the sub-hour end and offers an IoT hook via a sky camera. Optical-flow version is feasible; U-Net (HelioNet-style) is not. Cut, with optical flow as a stretch.

---

## 10. Open constraints

These are real. Name them in the pitch before a judge finds them.

**No open Tunisian PV data.** STEG and ANME publish nothing equivalent to Australia's CER postcode capacity data or the UK's Sheffield Solar PVLive. No generation time series, no per-site installation registry. Consequence: the national view is a **transparent simulation** — a virtual fleet built from realistic tilt/azimuth/capacity distributions — with the *upscaling method* validated on real multi-site data (Ausgrid, UK-PV). Say this out loud.

**Training data is Australian.** DKASC Alice Springs is the training set: real desert PV, 15+ years, hot, dusty, sunny. Climatically close to southern Tunisia, less so to the coast. The residual-learning design is precisely what makes this transfer defensible — we learn *loss mechanisms*, which are physical and portable, rather than a site-specific power curve. But it is still a transfer assumption and it should be tested (compare residual distributions across DKASC arrays with different orientations as a proxy for transfer robustness).

**Soiling has no ground truth.** We model soiling; we cannot validate the absolute loss. Present it as a **relative index and a maintenance trigger**, never as a measured percentage. The `k` and `rain_reset_mm` constants above are tuned, not derived.

**Capacity figures conflict.** The Ministry has cited "~400 MW installed vs 70 MW operational" for low-voltage PV (Dec 2025), which appears to conflate registered pipeline with commissioned capacity. STEG's end-June 2026 statement of ~500 MW residential is the more recent figure and the one used here. Flag the ambiguity rather than picking silently.

**API constraints.** Open-Meteo free tier is non-commercial with rate limits. NASA POWER solar has 5–7 day latency — fine for training, useless live. CAMS reanalysis lags ~2 days; use the *forecast* endpoints for anything live. Cache aggressively; do not hammer APIs from the demo loop.

**Regulatory context.** Tunisian net metering (Decree 2016-1123) caps surplus sales to STEG at 30% of annual production. Relevant if the decision layer touches self-consumption or battery dispatch economics.

**Evidence transferred from elsewhere.** The dust-impact magnitudes cited in the pitch are non-Tunisian: a March 2022 Saharan dust storm halved the Spanish national PV fleet's capacity factor for over two weeks (80% peak-day drop; Micheli et al., *Sustainable Energy Technologies and Assessments* 62, 2024), and a 2021 West African dry-season plume cut surface solar irradiance by ~18% on average, with dust-aware modelling reducing GHI estimation error by ~75% (*Atmospheric Chemistry and Physics* 25:997, 2025). These establish the **mechanism and order of magnitude**, not Tunisian values. Present them that way.

---

## 11. Demo script

1. Map of Tunisia, governorates shaded by current predicted rooftop PV output. National total in the header.
2. Click a governorate → 48-hour forecast curve with a 90% confidence band.
3. Scrub forward to a dust event. Output drops. **The band visibly widens.** This is the moment that sells the project.
4. Decision panel updates: `Reserve required: 180 MW` · `Cleaning recommended: Gabès, Medenine`.
5. Toggle "dust model off" → show the forecast miss the drop entirely. This is the ablation that proves the differentiator earns its place.
6. If the ESP32 exists: a live dot on the map, real hardware, real production, sitting inside the predicted band.

---

## 12. Pitch framing

Most teams will present a scatter plot of predicted vs. actual with a good R². That answers "is my model accurate?"

This project answers **"what should STEG do on Thursday?"** — hold this much reserve, send a cleaning crew to these two governorates, expect this ramp at sunset. For a phenomenon (Saharan dust) that is specifically Tunisian and that generic solar forecasting ignores entirely.

The uncertainty band is not decoration. An operator sizes reserve off the pessimistic edge of the interval, not the point forecast. Narrower, calibrated bands are directly worth money in avoided reserve. That is the argument.
