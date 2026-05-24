"""
cwfis_fire_data.py
------------------
Loads real Nova Scotia 2023 wildfire perimeter data from the
Canadian Wildland Fire Information System (CWFIS) National Burned
Area Composite (NBAC).

Data was fetched via the CWFIS WFS endpoint:
  https://cwfis.cfs.nrcan.gc.ca/geoserver/public/ows
  Layer: public:nbac  (National Burned Area Composite 1972-2024)
  Filter: admin_area='NS' AND year=2023

The raw MultiPolygon geometries (45 parts for Barrington, 14 for Tantallon)
were convex-hulled and resampled to 40 evenly-spaced vertices to match the
format produced by helpers/mock_fire.generate_fire_perimeter().

Source: Natural Resources Canada / CWFIS — no API key required.
Data reflects VIIRS/MODIS infrared satellite detections validated by ground crews.
"""

import json
import math
import os
import random
from typing import Literal

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, "..", "data")

# ---------------------------------------------------------------------------
# Fire catalogue
# ---------------------------------------------------------------------------
FIRE_CATALOGUE: dict[str, dict] = {
    "barrington": {
        "label":       "Barrington Lake Fire (May 2023)",
        "description": "Nova Scotia's largest 2023 wildfire — 20,265 ha near Shelburne County.",
        "file":        "barrington_2023.json",
        "days":        6,   # May 27 – Jun 2, 2023
    },
    "tantallon": {
        "label":       "Tantallon / Upper Tantallon Fire (May 2023)",
        "description": "817 ha fire in Halifax Regional Municipality — forced evacuations.",
        "file":        "tantallon_2023.json",
        "days":        2,   # May 28-29, 2023
    },
}


def list_fires() -> list[dict]:
    """Return catalogue metadata for all available fires (no geometry)."""
    return [
        {
            "fire_id":     fid,
            "label":       meta["label"],
            "description": meta["description"],
            "days":        meta["days"],
        }
        for fid, meta in FIRE_CATALOGUE.items()
    ]


def load_fire(fire_id: str) -> dict:
    """
    Load the processed fire record from the local JSON cache.

    Returns a dict with keys:
      fire_id, year, poly_ha, start_date, end_date,
      center_lat, center_lon, radius_km, perimeter_ring, source
    where perimeter_ring is a closed GeoJSON ring [[lon,lat], ..., [lon,lat]].
    """
    meta = FIRE_CATALOGUE.get(fire_id)
    if meta is None:
        raise ValueError(f"Unknown fire_id '{fire_id}'. "
                         f"Available: {list(FIRE_CATALOGUE.keys())}")

    path = os.path.join(_DATA_DIR, meta["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cached fire data not found at {path}. "
            "Re-run backend/helpers/fetch_fire_data.py to refresh."
        )

    with open(path, encoding="utf-8") as f:
        record = json.load(f)

    record["fire_id"] = fire_id
    return record


# ---------------------------------------------------------------------------
# Polygon helpers
# ---------------------------------------------------------------------------

def _ring_centroid(ring: list[list[float]]) -> tuple[float, float]:
    """Average centroid of a closed ring [[lon,lat],...,[lon,lat]]."""
    pts = ring[:-1] if ring[0] == ring[-1] else ring
    return (
        sum(c[0] for c in pts) / len(pts),
        sum(c[1] for c in pts) / len(pts),
    )


def scale_ring(
    ring: list[list[float]],
    center: tuple[float, float],
    scale: float,
    seed: int = 0,
    noise: float = 0.0,
) -> list[list[float]]:
    """
    Scale a polygon ring inward/outward from `center` by `scale`.

    Optional `noise` (0..1) adds ±noise * scale radial jitter per vertex,
    seeded deterministically so repeated calls at the same scale are stable.
    """
    rng = random.Random(seed)
    pts = ring[:-1] if ring[0] == ring[-1] else ring
    clon, clat = center
    result = []
    for i, (lon, lat) in enumerate(pts):
        s = scale
        if noise > 0:
            s *= 1.0 + (rng.random() * 2 - 1) * noise
        result.append([
            round(clon + (lon - clon) * s, 6),
            round(clat + (lat - clat) * s, 6),
        ])
    result.append(result[0])  # close
    return result


def build_growth_frames(
    fire_record: dict,
    steps: int = 60,
    start_scale: float = 0.08,
    noise_per_frame: float = 0.025,
) -> list[list[list[float]]]:
    """
    Build `steps` perimeter rings that grow from `start_scale` (8% of final
    size) to 1.0 (the real satellite-detected boundary).

    Each frame is a closed GeoJSON ring [[lon,lat], ..., [lon,lat]] in the
    same format as helpers/mock_fire.generate_fire_perimeter().

    Growth is eased with a sqrt curve so the early frames look fast (like a
    real fire igniting) and the later frames slow down (fire edges stabilising).
    """
    ring = fire_record["perimeter_ring"]
    center = (fire_record["center_lon"], fire_record["center_lat"])
    frames = []
    for i in range(steps):
        t = i / max(steps - 1, 1)
        # sqrt easing: fast start, slower finish
        scale = start_scale + (1.0 - start_scale) * math.sqrt(t)
        frame_ring = scale_ring(
            ring,
            center,
            scale,
            seed=i,
            noise=noise_per_frame * (1.0 - t * 0.5),  # noise fades as fire matures
        )
        frames.append(frame_ring)
    return frames
