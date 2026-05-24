# 🔥 Heatvision

**Real-time aerial firefighting decision support for Air Tactical Group Supervisors (ATGS / Air Attack Officers).**

A full-stack visualization and intelligence tool that puts fire spread predictions, terrain analysis, and coordinated airtanker drop planning on a live satellite map — at the moment the drop decision is being made.

> **Context:** Only 55% of airtanker drops aimed at halting fire spread succeed. The variables that predict success (vegetation, terrain, wind, fuel moisture) are known — but none of this intelligence reaches pilots and air attack officers in real time. This tool is built to close that gap.

---

## Overview

The tool simulates and visualizes a complete aerial firefighting operation:

1. **Fire spread** — ML-predicted direction and severity, animated across real satellite terrain
2. **Aircraft choreography** — Bird Dog recon pass, tanker approach, retardant drop, and departure
3. **Environmental overlays** — shrubland zones (high drop effectiveness), gentle-slope terrain zones
4. **ATU-driven status** — door open/close events drive the retardant line drawing and status panel
5. **AAO briefing** — AI-generated Air Attack Officer talk-in briefing from a map screenshot

Both synthetic fires (place anywhere) and two real 2023 Nova Scotia fires (CWFIS satellite data) are supported.

---

## Architecture

```
drop-it-like-its-hot/
├── backend/          # FastAPI Python API
│   ├── main.py               # All API routes
│   ├── model_inference.py    # RandomForest prediction wrapper
│   ├── fuel_lookup.py        # Canadian FBP fuel type lookup (nearest-neighbor)
│   ├── aao_briefing.py       # AWS Bedrock (Amazon Nova 2) AAO briefing generator
│   ├── helpers/
│   │   ├── weatherapi.py     # WeatherAPI.com current conditions (TTL cached)
│   │   ├── terrain.py        # Slope/aspect via Canadian CDEM elevation API
│   │   ├── mock_fire.py      # Synthetic fire perimeter generator
│   │   ├── cwfis_fire_data.py # Real CWFIS NBAC satellite perimeter data
│   │   ├── mock_environment.py # Fallback shrub/terrain zone data
│   │   ├── sentinel_terrain.py # Real gentle-slope zones via Sentinel Hub DEM
│   │   └── sentinel_shrub.py   # Real shrubland zones via Sentinel Hub WorldCover
│   ├── data/
│   │   ├── barrington_2023.json  # Barrington Lake Fire perimeter (20,265 ha)
│   │   └── tantallon_2023.json   # Tantallon Fire perimeter (817 ha)
│   ├── wildfire_model.pkl        # Trained RandomForest model
│   ├── fuel_label_encoder.pkl    # FBP fuel type encoder
│   └── hfi_norm_bounds.npy       # HFI normalization bounds for severity
├── frontend/         # Next.js 16 + TypeScript UI
│   └── app/
│       └── components/
│           ├── FireMap.tsx       # Main visualization (DeckGL + MapLibre)
│           └── FireMapLoader.tsx # Data-fetching wrapper
├── model/            # Model training artifacts
│   ├── train_wildfire_model.py   # Training script
│   └── MODEL_README.md           # Model documentation
└── training_features_ns_2023.csv # Nova Scotia 2023 training data (3,490 rows)
```

---

## Backend API

The FastAPI server runs on `http://localhost:8000`. All endpoints return JSON.

### Endpoints

| Method | Path                | Description                                                       |
| ------ | ------------------- | ----------------------------------------------------------------- |
| `GET`  | `/health`           | Health check                                                      |
| `GET`  | `/simulate`         | Animated fire simulation at arbitrary coordinates                 |
| `GET`  | `/simulate-real`    | Simulation using real CWFIS satellite fire perimeter              |
| `GET`  | `/fires`            | List available real fires with metadata                           |
| `GET`  | `/fire-perimeter`   | Single synthetic fire perimeter polygon                           |
| `GET`  | `/terrain`          | Slope, aspect, and elevation at a point                           |
| `GET`  | `/weather`          | Current weather at a point (WeatherAPI)                           |
| `GET`  | `/terrain-zones`    | Gentle-slope drop zones (Sentinel Hub DEM, or mock fallback)      |
| `GET`  | `/shrub-zones`      | Shrubland drop zones (Sentinel Hub WorldCover, or mock fallback)  |
| `GET`  | `/mock-environment` | Mock shrub + terrain zone overlays for a named fire               |
| `POST` | `/aao-briefing`     | AI-generated Air Attack Officer talk-in briefing from a map image |

### `/simulate` — Core Simulation

```
GET /simulate?lat=44.85&lon=-63.55&radius_km=2.0&steps=60&seed=42
```

Returns `steps` GeoJSON frames, each containing:

- Fire perimeter polygon (growing and drifting with wind)
- Per-vertex spread direction and severity arrows
- Bird Dog and Air Tanker positions/headings/visibility
- Retardant line geometry (draws during door-open window)
- ATU event type (`door_open`, `door_close`, `none`)
- Status string (e.g. `"Drop in progress"`)

Also returns top-level `weather`, `terrain`, `fuel`, and `prediction` objects.

### `/simulate-real` — Real Fire Data

```
GET /simulate-real?fire=barrington&steps=60
```

Available fires: `barrington` (Barrington Lake, 20,265 ha) and `tantallon` (Upper Tantallon, 817 ha) — both May 2023, Nova Scotia. Perimeters are sourced from CWFIS NBAC (VIIRS/MODIS satellite IR, validated by ground crews).

### `/aao-briefing` — AI Briefing

```
POST /aao-briefing
Content-Type: application/json

{ "image_b64": "<base64-encoded PNG>", "mime_type": "image/png" }
```

Sends a map screenshot to Amazon Nova 2 Lite via AWS Bedrock. The model generates a one-to-two sentence Air Attack Officer talk-in briefing using only visual landmarks the tanker pilot can see from the cockpit (roads, rivers, ridgelines, clearcuts, grasslands). The drop zone must be marked in purple on the image.

---

## ML Model

A **Random Forest** model trained on Nova Scotia spring 2023 fire weather data (3,490 observations, March–May).

### Inputs

| Category | Features                                                                   |
| -------- | -------------------------------------------------------------------------- |
| Location | lat, lon                                                                   |
| Weather  | wind speed, wind direction (sin/cos), humidity, temperature, precipitation |
| FWI      | FFMC, DMC, DC, BUI, ISI                                                    |
| Terrain  | slope, aspect (sin/cos), elevation                                         |
| Fuel     | Canadian FBP fuel type (label-encoded)                                     |

### Outputs

| Output           | Description                                           |
| ---------------- | ----------------------------------------------------- |
| `fire_direction` | Predicted spread bearing (0–360°), ~2.8° circular MAE |
| `fire_severity`  | Normalized fire intensity (0–1), R² = 0.83            |

Direction is derived from a physics-based vector formula (wind force + upslope force). Severity is log-normalized Head Fire Intensity (HFI in kW/m) from the Canadian FBP system.

**Note:** This model is scoped to Nova Scotia. It was trained on spring season data and does not generalize to other provinces or summer peak conditions without retraining. See [model/MODEL_README.md](model/MODEL_README.md) for full details.

---

## Aircraft Simulation

The simulation plays out in four acts, designed around real aerial firefighting protocol:

| Act       | Frames  | What happens                                                                                |
| --------- | ------- | ------------------------------------------------------------------------------------------- |
| **Act 1** | 0–30%   | Bird Dog performs solo clockwise recon circle; Tanker waits offscreen                       |
| **Act 2** | 30–35%  | Both aircraft join up and approach from the north                                           |
| **Act 3** | 35–65%  | Tanker executes drop run south over the fire; door opens, retardant line draws, door closes |
| **Act 4** | 65–100% | Both aircraft depart south and fade out                                                     |

ATU events (`door_open` / `door_close`) are the trigger that draws the retardant line — mirroring how real Airborne Tracking Unit data works.

---

## Frontend

Built with **Next.js 16**, **TypeScript**, **DeckGL 9**, and **MapLibre GL**. Base map uses ESRI World Imagery (satellite).

### Layers

| Layer          | Library            | What it shows                                               |
| -------------- | ------------------ | ----------------------------------------------------------- |
| Fire perimeter | `PolygonLayer`     | Animated orange fire boundary                               |
| Spread arrows  | `ScatterplotLayer` | Per-vertex direction and severity indicators                |
| Retardant line | `PathLayer`        | Pink/red deterrent line drawn during the drop               |
| Bird Dog       | `IconLayer`        | Small aircraft icon with heading                            |
| Air Tanker     | `IconLayer`        | Large aircraft icon with heading and opacity fade           |
| Shrub zones    | `HeatmapLayer`     | Green heat blobs highlighting high-effectiveness drop areas |
| Terrain zones  | `PolygonLayer`     | Blue polygons marking gentle-slope terrain                  |

---

## Setup

### Prerequisites

- Python 3.13+, `uv` package manager
- Node.js 20+, npm
- API keys (see Environment Variables)

### Backend

```bash
cd backend
uv sync                     # install dependencies
cp .env.example .env        # fill in API keys (see below)
uv run fastapi dev main.py  # starts on :8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev                 # starts on :3000
```

---

## Environment Variables

Create `backend/.env`:

```env
# WeatherAPI.com — current weather for fire centre
WEATHERKEY=your_weatherapi_key

# AWS Bedrock — AAO briefing generation (Amazon Nova 2 Lite)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Sentinel Hub — real terrain and shrub zones (optional; falls back to mock data)
SENTINELHUB_CLIENT_ID=your_client_id
SENTINELHUB_CLIENT_SECRET=your_client_secret
```

The Sentinel Hub credentials are optional. Without them, `/terrain-zones` and `/shrub-zones` return pre-computed mock zones so the UI still works.

---

## Real Fire Data

Two real 2023 Nova Scotia fires are included:

| Fire         | Date                 | Area      | Description                                                |
| ------------ | -------------------- | --------- | ---------------------------------------------------------- |
| `barrington` | May 27 – Jun 2, 2023 | 20,265 ha | Nova Scotia's largest 2023 wildfire, near Shelburne County |
| `tantallon`  | May 28–29, 2023      | 817 ha    | Halifax Regional Municipality — forced evacuations         |

Perimeters were sourced from the **Canadian Wildland Fire Information System (CWFIS) National Burned Area Composite (NBAC)** — VIIRS/MODIS infrared satellite detections validated by ground crews. The raw MultiPolygon geometries were convex-hulled and resampled to 40 evenly-spaced vertices. Data is cached locally in `backend/data/`.

---

## Retraining the Model

```bash
cd model
# 1. Update DATA_PATH in train_wildfire_model.py or drop new CSV files in the folder
# 2. Run training
python train_wildfire_model.py
# 3. Copy the three artifacts to backend/
cp wildfire_model.pkl fuel_label_encoder.pkl hfi_norm_bounds.npy ../backend/
```

The three artifact files (`wildfire_model.pkl`, `fuel_label_encoder.pkl`, `hfi_norm_bounds.npy`) are always versioned together. Replace all three when retraining.

---

## Background: Why This Exists

Large airtankers (LATs) and very large airtankers (VLATs) are the most expensive tools in wildfire suppression. A single VLAT drop costs **$65,000+**, and the US spends roughly **$200M+ annually** on airtanker operations alone.

Research (AFUE study, 2015–2018, 5,000+ drops across 272 fires) shows only **55% of halt-spread drops succeed**. The variables that predict success — vegetation type, terrain, wind, fuel moisture — are all measurable. The problem is they don't reach the decision-makers (ATGS, Bird Dog pilots, Incident Commanders) at the moment the drop is ordered.

This tool is built to raise that success rate by putting the right intelligence in front of the right people in real time.
