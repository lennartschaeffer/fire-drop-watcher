# Wildfire Spread Prediction Model

Nova Scotia–specific model that predicts fire spread direction and intensity from local weather, fuel, and terrain conditions. Built for the bird dog demo.

---

## Quick start

```python
import joblib
from train_wildfire_model import predict

model   = joblib.load("wildfire_model.pkl")
encoder = joblib.load("fuel_label_encoder.pkl")

result = predict(
    lat=45.57, lon=-64.27,
    wind_speed=14.5, wind_direction=220,
    fuel="C2", slope=1.87, aspect=286,
    humidity=39, temp=13.1,
    ffmc=87.9, dmc=8.7, dc=18.7, bui=8.6, isi=6.6,
    pcp=0.0, elev=50,
    model=model, encoder=encoder,
)
# {'fire_direction': 269.8, 'fire_severity': 0.3966}
```

---

## Outputs

| Field | Type | Description |
|---|---|---|
| `fire_direction` | float, 0–360° | Predicted compass bearing the fire will spread toward |
| `fire_severity` | float, 0–1 | Normalized intensity at the current observation point (0 = negligible, 1 = extreme) |

**Important framing on severity:** the severity score reflects fire intensity *at the current location* given present conditions — it is not projected forward in the spread direction. Think of it as "how dangerous is it here right now" rather than "how dangerous will it be where the fire is heading." Closing this gap requires fire border coordinate data (see Limitations below).

---

## Inputs

### Location
| Param | Unit | Notes |
|---|---|---|
| `lat` | decimal degrees | Current observation position |
| `lon` | decimal degrees | Current observation position |

### Weather
| Param | Unit | Notes |
|---|---|---|
| `wind_speed` | m/s or km/h | Match whatever units your data source uses — must be consistent with training data |
| `wind_direction` | degrees 0–360 | Meteorological convention: the direction the wind is coming **from** |
| `humidity` | % (0–100) | Relative humidity |
| `temp` | °C | Air temperature |
| `pcp` | mm | Precipitation |

### Fire Weather Index components
These carry memory of past conditions and matter a lot for severity predictions. Get them from your weather feed rather than computing them yourself if possible.

| Param | What it represents |
|---|---|
| `ffmc` | Fine fuel moisture — yesterday's dryness, range 0–101 |
| `dmc` | Duff moisture — dryness over the past few days |
| `dc` | Drought code — long-term soil dryness over weeks/months |
| `bui` | Buildup Index — total available fuel (derived from dmc + dc) |
| `isi` | Initial Spread Index — how fast a new ignition would spread right now |

### Terrain
| Param | Unit | Notes |
|---|---|---|
| `slope` | FBP units (roughly degrees) | Slope magnitude at observation point |
| `aspect` | degrees 0–360 | Direction the slope faces (upslope direction) |
| `elev` | metres | Elevation |

### Fuel type
| Param | Valid values |
|---|---|
| `fuel` | `C1`, `C2`, `C5`, `D1`, `D2`, `M1_25`, `M1_35`, `M1_50`, `M1_65`, `M2_35`, `M2_50`, `M2_65`, `O1a`, `O1b`, `farm`, `urban` |

Pass any string not in this list and it will be encoded as `unknown` without erroring — predictions will degrade silently, so validate upstream.

---

## How the model works (short version)

**Direction** is predicted from wind direction, terrain aspect, slope, and location. Internally the labels were derived using a physics-based vector formula (wind force + upslope force) rather than observed ground truth. The model learns this relationship with high accuracy (~2.8° circular MAE).

**Severity** is predicted from all inputs combined. Labels were derived by log-normalizing Head Fire Intensity (HFI in kW/m) from the Canadian FBP system. The model explains ~83% of severity variance (R² = 0.83) on the held-out test set.

Neither output is a simulation — they are fast ML predictions suitable for real-time decision support.

---

## Training data

- **Source:** `training_features_ns_2023.csv`
- **Coverage:** Nova Scotia, spring fire season 2023 (March–May)
- **Rows:** 3,490 (3,478 used after null drops)
- **This model is Nova Scotia–specific.** It was deliberately scoped this way. Do not expect it to generalize to other provinces or fire seasons without retraining.

---

## Limitations

1. **No fire border input.** The model does not currently receive fire perimeter/boundary coordinates as input. Severity is therefore a local conditions score, not a forward-projected one. When perimeter data is available, add it as engineered features (border centroid lat/lon, distance to nearest border point, bearing from current position to centroid) and retrain.

2. **Spring season only.** Training data covers March–May. Summer fire behavior (higher DC, larger crown fires) is not represented. Severity predictions may be underconfident for peak summer conditions.

3. **Nova Scotia geography only.** The model has only seen NS fuel types and terrain. Inference outside NS is out-of-distribution.

4. **Severity ≠ forward severity.** See output framing note above.

---

## Retraining

To retrain with new or updated data:

1. Drop new CSV(s) in the project folder using the same column schema as `training_features_ns_2023.csv`
2. Update `DATA_PATH` at the top of `train_wildfire_model.py` (or concatenate multiple files before passing in)
3. Run `python train_wildfire_model.py`
4. New `wildfire_model.pkl`, `fuel_label_encoder.pkl`, and `hfi_norm_bounds.npy` will be written — commit all three together, they are versioned as a set

If new fuel types appear in the data, the encoder will pick them up automatically on retrain. Any code calling `predict()` with the old fuel list will still work.

---

## Artifacts

| File | Purpose |
|---|---|
| `wildfire_model.pkl` | Trained RandomForest model |
| `fuel_label_encoder.pkl` | Fuel type label encoder — required for inference |
| `hfi_norm_bounds.npy` | log(HFI) min/max bounds used to normalize severity labels |
| `train_wildfire_model.py` | Training script + `predict()` function |

All four files must be kept in sync. If you retrain, replace all of them.
