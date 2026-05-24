"""
sentinel_shrub.py
-----------------
Fetch ESA WorldCover 2020 (10 m) from the Sentinel Hub Process API for a given
bounding box and extract shrubland (class 20) zones as heatmap point clouds.

Flow
----
1.  OAuth2 client_credentials → bearer token (shared cache with sentinel_terrain)
2.  Process API  →  256 × 256 UINT8 GeoTIFF of WorldCover classifications
3.  Threshold == 20  →  binary shrubland mask
4.  scipy.ndimage.label  →  connected components; discard noise (< min_zone_pixels)
5.  Sample pixel-centre lat/lon from each component → points list for HeatmapLayer
6.  Return top-N zones (largest first) as ShrubZone dicts

Required env vars
-----------------
  SENTINELHUB_CLIENT_ID
  SENTINELHUB_CLIENT_SECRET

WorldCover class values (relevant subset)
-----------------------------------------
  10  Tree cover
  20  Shrubland          ← target class
  30  Grassland
  40  Cropland
  50  Built-up
  60  Bare / sparse vegetation
  80  Permanent water bodies
"""

from __future__ import annotations

import io
import math
import os
import time
import logging
from functools import lru_cache

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Sentinel Hub endpoints (shared with sentinel_terrain) ─────────────────────
_SH_TOKEN_URL   = "https://services.sentinel-hub.com/oauth/token"
_SH_PROCESS_URL = "https://services.sentinel-hub.com/api/v1/process"

# ── ESA WorldCover 2020 BYOC collection ──────────────────────────────────────
_WORLDCOVER_COLLECTION_ID = "0b940c63-45dd-4e6b-8019-c3660b81b884"
_WORLDCOVER_TYPE          = f"byoc-{_WORLDCOVER_COLLECTION_ID}"

# WorldCover class values treated as "open fuel" for retardant effectiveness.
# Both shrubland (20) and grassland (30) outperform forest for aerial retardant
# per AFUE research — retardant holds position on open vegetation, canopy
# interception is minimal, and ground crews can follow up effectively.
_OPEN_FUEL_CLASSES = {20, 30}  # Shrubland + Grassland

# ── Evalscript — returns raw classification integer as UINT8 ─────────────────
_EVALSCRIPT = """
//VERSION=3
function setup() {
  return { input: ["Map"], output: { bands: 1, sampleType: "UINT8" } };
}
function evaluatePixel(s) { return [s.Map]; }
"""

# ── In-process OAuth2 token cache ─────────────────────────────────────────────
_token_cache: dict[str, object] = {"token": None, "expires_at": 0.0}


def _get_token() -> str:
    """Return a valid Sentinel Hub bearer token, refreshing if < 60 s remain."""
    now = time.monotonic()
    if _token_cache["token"] and now < float(_token_cache["expires_at"]):  # type: ignore[arg-type]
        return str(_token_cache["token"])

    client_id     = os.environ.get("SENTINELHUB_CLIENT_ID", "")
    client_secret = os.environ.get("SENTINELHUB_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        raise EnvironmentError(
            "SENTINELHUB_CLIENT_ID and SENTINELHUB_CLIENT_SECRET must be set "
            "in the environment to use real shrub zones."
        )

    resp = requests.post(
        _SH_TOKEN_URL,
        data={
            "grant_type":    "client_credentials",
            "client_id":     client_id,
            "client_secret": client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    _token_cache["token"]      = data["access_token"]
    _token_cache["expires_at"] = now + data.get("expires_in", 3600) - 60
    return data["access_token"]


# ── WorldCover tile fetch ─────────────────────────────────────────────────────

def _fetch_worldcover_tiff(
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
    image_size: int = 256,
) -> bytes:
    """
    Call the Sentinel Hub Process API and return raw GeoTIFF bytes for a
    single-band UINT8 WorldCover classification tile.
    """
    token = _get_token()

    payload = {
        "input": {
            "bounds": {
                "bbox": [lon_min, lat_min, lon_max, lat_max],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": _WORLDCOVER_TYPE,
                    "dataFilter": {
                        "timeRange": {
                            "from": "2021-01-01T00:00:00Z",
                            "to":   "2021-12-31T23:59:59Z",
                        }
                    },
                }
            ],
        },
        "output": {
            "width":  image_size,
            "height": image_size,
            "responses": [
                {"identifier": "default", "format": {"type": "image/tiff"}}
            ],
        },
        "evalscript": _EVALSCRIPT,
    }

    resp = requests.post(
        _SH_PROCESS_URL,
        json=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.content


# ── Shrub zone extraction ─────────────────────────────────────────────────────

def _extract_shrub_zones(
    data: np.ndarray,
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
    min_zone_pixels: int,
    max_zones: int,
    max_points_per_zone: int = 200,
) -> list[dict]:
    """
    Given a 2-D WorldCover classification array and its geographic bounds,
    return a list of shrubland zone dicts for the largest class-20 regions.

    Each zone contains sampled pixel-centre lat/lon points suitable for
    a DeckGL HeatmapLayer, matching the ShrubZone interface in FireMap.tsx.
    """
    from scipy.ndimage import label as ndlabel
    import rasterio.features
    from rasterio.transform import from_bounds

    height, width = data.shape

    # ── Open-fuel mask: shrubland (20) + grassland (30) ──────────────────────
    shrub_mask = np.isin(data, list(_OPEN_FUEL_CLASSES)).astype(np.uint8)

    if shrub_mask.sum() == 0:
        logger.info("No open-fuel pixels (classes %s) found in this tile.", _OPEN_FUEL_CLASSES)
        return []

    # ── Connected components ──────────────────────────────────────────────────
    labeled, n_components = ndlabel(shrub_mask)

    comp_sizes = sorted(
        [(i + 1, int(np.sum(labeled == (i + 1)))) for i in range(n_components)],
        key=lambda x: -x[1],
    )

    zones: list[dict] = []
    for comp_id, pixel_count in comp_sizes:
        if pixel_count < min_zone_pixels:
            break
        if len(zones) >= max_zones:
            break

        # Pixel centre coordinates for every pixel in this component
        rows, cols = np.where(labeled == comp_id)

        # Row 0 = lat_max (top of image); col 0 = lon_min (left)
        zone_lats = lat_max - (rows + 0.5) * (lat_max - lat_min) / height
        zone_lons = lon_min + (cols + 0.5) * (lon_max - lon_min) / width

        zone_lat = float(np.mean(zone_lats))
        zone_lon = float(np.mean(zone_lons))

        # Subsample pixel centres for the heatmap (keeps payload small)
        n_sample = min(max_points_per_zone, len(rows))
        rng = np.random.default_rng(seed=int(comp_id))
        idx = rng.choice(len(rows), size=n_sample, replace=False)
        points = [
            {"lat": round(float(zone_lats[i]), 6), "lon": round(float(zone_lons[i]), 6)}
            for i in idx
        ]

        # Classify zone by the dominant class within this component
        comp_classes = data[labeled == comp_id]
        dominant_class = int(np.bincount(comp_classes.astype(int)).argmax())
        if dominant_class == 20:
            label     = "Shrubland"
            halt_rate = "~70%"
            class_desc = "shrubland (class 20)"
        else:
            label     = "Open Grassland"
            halt_rate = "~65%"
            class_desc = "grassland (class 30)"

        zones.append(
            {
                "center":      {"lat": round(zone_lat, 5), "lon": round(zone_lon, 5)},
                "points":      points,
                "label":       label,
                "halt_rate":   halt_rate,
                "description": (
                    f"Open fuel zone — ESA WorldCover {class_desc}, 10 m resolution. "
                    f"Pixel count: {pixel_count}. "
                    "Open vegetation has higher retardant hold and effectiveness than forest canopy (AFUE research)."
                ),
                "pixel_count": pixel_count,
                "source":      "Sentinel Hub — ESA WorldCover 2020",
            }
        )

    return zones


# ── In-process result cache (WorldCover is static — cache aggressively) ───────
_zone_cache: dict[tuple[float, float], tuple[float, list[dict]]] = {}
_ZONE_TTL = 3600 * 4  # 4 hours


def get_shrub_zones(
    center_lat: float,
    center_lon: float,
    radius_km: float = 25.0,
    min_zone_pixels: int = 50,
    max_zones: int = 5,
    image_size: int = 256,
) -> list[dict]:
    """
    Return shrubland zones (ESA WorldCover class 20) around (center_lat, center_lon).

    The WorldCover tile is fetched once from Sentinel Hub and cached for 4 hours.

    Parameters
    ----------
    center_lat, center_lon : float
        Centre of the area of interest.
    radius_km : float
        Half-width of the tile (default 25 km → 50 × 50 km coverage).
    min_zone_pixels : int
        Minimum connected-component size to include (default 50 ≈ ~2 km² at 256px/50km tile).
    max_zones : int
        Maximum number of zones to return, largest first.
    image_size : int
        Width = height of the requested tile in pixels (default 256).

    Returns
    -------
    list[dict]
        Each dict has keys: center, points, label, halt_rate, description
        matching the ShrubZone interface expected by the frontend.

    Raises
    ------
    EnvironmentError
        If SENTINELHUB_CLIENT_ID / SENTINELHUB_CLIENT_SECRET are not set.
    requests.HTTPError
        If the Sentinel Hub API returns a non-2xx response.
    """
    cache_key = (round(center_lat, 2), round(center_lon, 2))
    cached = _zone_cache.get(cache_key)
    if cached and (time.monotonic() - cached[0]) < _ZONE_TTL:
        logger.debug("shrub-zones cache hit for %s", cache_key)
        return cached[1]

    # ── Bounding box ──────────────────────────────────────────────────────────
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * math.cos(math.radians(center_lat)))
    lat_min = center_lat - dlat
    lat_max = center_lat + dlat
    lon_min = center_lon - dlon
    lon_max = center_lon + dlon

    logger.info(
        "Fetching WorldCover tile  bbox=[%.4f,%.4f,%.4f,%.4f]  size=%dx%d",
        lon_min, lat_min, lon_max, lat_max, image_size, image_size,
    )

    tiff_bytes = _fetch_worldcover_tiff(lat_min, lon_min, lat_max, lon_max, image_size)

    # ── Parse GeoTIFF in memory ────────────────────────────────────────────────
    import rasterio
    from rasterio.io import MemoryFile

    with MemoryFile(tiff_bytes) as memfile, memfile.open() as dataset:
        wc_data = dataset.read(1).astype(np.uint8)

    zones = _extract_shrub_zones(
        wc_data,
        lat_min, lon_min, lat_max, lon_max,
        min_zone_pixels=min_zone_pixels,
        max_zones=max_zones,
    )

    _zone_cache[cache_key] = (time.monotonic(), zones)
    logger.info("Extracted %d shrub zones for %s", len(zones), cache_key)
    return zones
