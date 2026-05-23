"""
fuel_lookup.py
--------------
Given a (lat, lon) coordinate in Nova Scotia, returns the fuel type
(vegetation / ground cover class) and associated fire-danger metrics
from the nearest point in the training dataset.

Fuel types follow the Canadian Forest Fire Behaviour Prediction (FBP) System:
  C1, C2, C5        – Conifer stands (spruce-fir, pine, etc.)
  D1, D2            – Deciduous / leafless hardwood
  M1_xx, M2_xx      – Mixed wood (xx = % conifer, e.g. M2_50 = 50 % conifer)
  O1a, O1b          – Open / grass / matted grass
  farm              – Agricultural land
  urban             – Urban / built-up

DMC (Duff Moisture Code) in the returned data is the value recorded at
the nearest incident point — it is a weather-derived index (higher = drier
organic layer = more flammable), NOT a fixed property of the location.
"""

import csv
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

# ---------------------------------------------------------------------------
# Fuel type human-readable descriptions
# ---------------------------------------------------------------------------
FUEL_DESCRIPTIONS: dict[str, str] = {
    "C1":    "Spruce-lichen woodland (open conifer, low canopy)",
    "C2":    "Boreal spruce (dense conifer, high fire intensity)",
    "C5":    "Red and white pine (mature pine stand)",
    "D1":    "Leafless aspen / hardwood (low spread, low intensity)",
    "D2":    "Green aspen / hardwood (moderate spread when dry)",
    "M1_25": "Mixed wood – 25 % conifer / 75 % deciduous",
    "M1_35": "Mixed wood – 35 % conifer / 65 % deciduous",
    "M1_50": "Mixed wood – 50 % conifer / 50 % deciduous",
    "M1_65": "Mixed wood – 65 % conifer / 35 % deciduous",
    "M2_35": "Mixed wood (green) – 35 % conifer / 65 % deciduous",
    "M2_50": "Mixed wood (green) – 50 % conifer / 50 % deciduous",
    "M2_65": "Mixed wood (green) – 65 % conifer / 35 % deciduous",
    "O1a":   "Matted grass (low, compressed grass layer)",
    "O1b":   "Standing grass (upright, highly flammable when dry)",
    "farm":  "Agricultural / farmland (variable, often low risk)",
    "urban": "Urban / built-up area",
}

# Rough flammability tier for each fuel type (informational)
FLAMMABILITY: dict[str, str] = {
    "C1": "moderate",  "C2": "high",     "C5": "high",
    "D1": "low",       "D2": "low-moderate",
    "M1_25": "low-moderate", "M1_35": "moderate", "M1_50": "moderate",
    "M1_65": "moderate-high","M2_35": "moderate", "M2_50": "moderate",
    "M2_65": "moderate-high","O1a": "moderate",   "O1b": "high",
    "farm": "low",     "urban": "very low",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class FuelLookupResult:
    # Input
    query_lat: float
    query_lon: float

    # Nearest match
    fuel_type: str
    fuel_description: str
    flammability: str

    # Fire-danger metrics at the nearest point
    dmc: float          # Duff Moisture Code (0–100+; higher = drier)
    dc: float           # Drought Code
    ffmc: float         # Fine Fuel Moisture Code
    fwi: float          # Fire Weather Index
    bui: float          # Buildup Index
    isi: float          # Initial Spread Index

    # Nearest point details
    nearest_lat: float
    nearest_lon: float
    distance_km: float

    def summary(self) -> str:
        return (
            f"Fuel type : {self.fuel_type} — {self.fuel_description}\n"
            f"Flammability : {self.flammability}\n"
            f"DMC (duff dryness) : {self.dmc:.1f}  |  FWI : {self.fwi:.1f}  "
            f"|  DC : {self.dc:.1f}  |  FFMC : {self.ffmc:.1f}\n"
            f"Nearest data point : ({self.nearest_lat:.4f}, {self.nearest_lon:.4f})  "
            f"— {self.distance_km:.2f} km away"
        )


# ---------------------------------------------------------------------------
# KD-tree spatial index (built once, cached)
# ---------------------------------------------------------------------------
@dataclass
class _DataPoint:
    lat: float
    lon: float
    fuel: str
    dmc: float
    dc: float
    ffmc: float
    fwi: float
    bui: float
    isi: float


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


@lru_cache(maxsize=1)
def _load_index(csv_path: str) -> list[_DataPoint]:
    """
    Load the training CSV into a list of _DataPoint objects.
    Cached so the file is only read once per process.
    Pure Python — no numpy/scipy required.
    """
    points: list[_DataPoint] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points.append(_DataPoint(
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    fuel=row["fuel"].strip(),
                    dmc=float(row["dmc"]),
                    dc=float(row["dc"]),
                    ffmc=float(row["ffmc"]),
                    fwi=float(row["fwi"]),
                    bui=float(row["bui"]),
                    isi=float(row["isi"]),
                ))
            except (ValueError, KeyError):
                continue  # skip malformed rows

    if not points:
        raise ValueError(f"No valid data loaded from {csv_path}")

    return points


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------
def get_fuel_at(
    lat: float,
    lon: float,
    csv_path: Optional[str] = None,
) -> FuelLookupResult:
    """
    Return the fuel type and fire-danger metrics for the given coordinates.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees (Nova Scotia: ~43.5 – 47.0 N)
    lon : float
        Longitude in decimal degrees (Nova Scotia: ~-66.4 – -59.7 W)
    csv_path : str, optional
        Path to the training CSV. Defaults to the bundled
        training_features_ns_2023.csv one directory above this file.

    Returns
    -------
    FuelLookupResult
        Dataclass with fuel type, description, flammability tier,
        fire-danger indices, and distance to the nearest data point.
    """
    if csv_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(here, "..", "training_features_ns_2023.csv")

    csv_path = os.path.abspath(csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training CSV not found at: {csv_path}")

    points = _load_index(csv_path)

    # Linear scan over ~3,500 points — fast enough in pure Python (~10ms)
    nearest = min(points, key=lambda p: _haversine_km(lat, lon, p.lat, p.lon))
    dist_km = _haversine_km(lat, lon, nearest.lat, nearest.lon)

    return FuelLookupResult(
        query_lat=lat,
        query_lon=lon,
        fuel_type=nearest.fuel,
        fuel_description=FUEL_DESCRIPTIONS.get(nearest.fuel, "Unknown fuel type"),
        flammability=FLAMMABILITY.get(nearest.fuel, "unknown"),
        dmc=nearest.dmc,
        dc=nearest.dc,
        ffmc=nearest.ffmc,
        fwi=nearest.fwi,
        bui=nearest.bui,
        isi=nearest.isi,
        nearest_lat=nearest.lat,
        nearest_lon=nearest.lon,
        distance_km=dist_km,
    )


# ---------------------------------------------------------------------------
# Manual test — run with:  python fuel_lookup.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    TEST_LOCATIONS = [
        ("Halifax",          44.6488, -63.5752),
        ("Truro",            45.3647, -63.2800),
        ("Cape Breton",      46.1368, -60.1942),
        ("Annapolis Royal",  44.7440, -65.5108),
        ("Amherst",          45.8363, -64.2186),
    ]

    print("=" * 60)
    print("  Nova Scotia Fuel Type Lookup — Test Run")
    print("=" * 60)

    for name, lat, lon in TEST_LOCATIONS:
        print(f"\n📍 {name}  ({lat}, {lon})")
        print("-" * 50)
        try:
            result = get_fuel_at(lat, lon)
            print(result.summary())
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
