"""
sentinel_terrain.py
-------------------
Fetch Copernicus DEM GLO-30 from the Sentinel Hub Process API for a given
bounding box and extract gentle-slope (< 15°) terrain zones as polygon rings.

Flow
----
1.  OAuth2 client_credentials → bearer token (cached for 55 min)
2.  Process API  →  256 × 256 float32 GeoTIFF of raw elevation (metres)
3.  numpy.gradient  →  slope in degrees at every pixel
4.  Threshold < 15°  →  binary mask of "gentle / favorable" terrain
5.  scipy.ndimage.label  →  connected components; discard noise (< 200 px)
6.  rasterio.features.shapes  →  proper marching-squares polygon outlines
7.  Return top-N zones (largest first) as [[lon, lat], …] closed rings

Required env vars
-----------------
  SENTINELHUB_CLIENT_ID
  SENTINELHUB_CLIENT_SECRET

New pip dependency
------------------
  rasterio >= 1.4   (provides MemoryFile, from_bounds, features.shapes)
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

# ── Sentinel Hub endpoints ────────────────────────────────────────────────────

_SH_TOKEN_URL   = "https://services.sentinel-hub.com/oauth/token"
_SH_PROCESS_URL = "https://services.sentinel-hub.com/api/v1/process"

# ── DEM evalscript — returns raw elevation as float32 ────────────────────────

_EVALSCRIPT = """
//VERSION=3
function setup() {
  return { input: ["DEM"], output: { bands: 1, sampleType: "FLOAT32" } };
}
function evaluatePixel(s) { return [s.DEM]; }
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
            "in the environment to use real terrain zones."
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


# ── DEM fetch ─────────────────────────────────────────────────────────────────

def _fetch_dem_tiff(
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
    image_size: int = 256,
) -> bytes:
    """
    Call the Sentinel Hub Process API and return the raw GeoTIFF bytes for a
    single-band float32 DEM (Copernicus GLO-30, ~30 m resolution).
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
                    "type": "dem",
                    "dataFilter": {"demInstance": "COPERNICUS_30"},
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


# ── Polygon extraction ────────────────────────────────────────────────────────

def _extract_zones(
    elev: np.ndarray,
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
    slope_threshold_deg: float,
    min_zone_pixels: int,
    max_zones: int,
) -> list[dict]:
    """
    Given a 2-D elevation array and its geographic bounds, return a list of
    terrain zone dicts (ring + metadata) for the largest gentle-slope regions.
    """
    from scipy.ndimage import label as ndlabel
    import rasterio.features
    from rasterio.transform import from_bounds

    height, width = elev.shape

    # ── Slope computation ─────────────────────────────────────────────────────
    # Convert pixel spacing from degrees to metres so the gradient is in m/m
    pixel_lat_m = (lat_max - lat_min) * 111_320 / height
    pixel_lon_m = (
        (lon_max - lon_min)
        * 111_320
        * math.cos(math.radians((lat_min + lat_max) / 2))
        / width
    )

    # np.gradient returns (row-direction, col-direction) derivatives
    dz_dy, dz_dx = np.gradient(elev, pixel_lat_m, pixel_lon_m)
    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

    # ── Connected components of the gentle-slope mask ─────────────────────────
    gentle_mask = (slope_deg < slope_threshold_deg).astype(np.uint8)
    labeled, n_components = ndlabel(gentle_mask) #type: ignore

    # Sort components by size, largest first
    comp_sizes = sorted(
        [(i + 1, int(np.sum(labeled == (i + 1)))) for i in range(n_components)],
        key=lambda x: -x[1],
    )

    # Geo-transform for rasterio.features.shapes: pixel (row=0, col=0) → top-left
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)

    zones: list[dict] = []
    for comp_id, pixel_count in comp_sizes:
        if pixel_count < min_zone_pixels:
            break
        if len(zones) >= max_zones:
            break

        # Binary mask for this component only
        comp_mask = (labeled == comp_id).astype(np.uint8)

        # rasterio.features.shapes returns (geojson_geom, value) tuples.
        # connectivity=8 merges diagonally adjacent pixels.
        polys = [
            geom
            for geom, val in rasterio.features.shapes(
                comp_mask, mask=comp_mask, connectivity=8, transform=transform
            )
            if val == 1.0 and geom["type"] == "Polygon"
        ]

        if not polys:
            continue

        # Pick the largest polygon by vertex count (outer shell only)
        best = max(polys, key=lambda g: len(g["coordinates"][0]))
        ring: list[list[float]] = best["coordinates"][0]  # [[lon, lat], …] — already GeoJSON

        # Compute the component's centroid and mean slope for metadata
        rows, cols = np.where(labeled == comp_id)
        # Row 0 = lat_max (top of image); col 0 = lon_min (left of image)
        zone_lats = lat_max - (rows + 0.5) * (lat_max - lat_min) / height
        zone_lons = lon_min + (cols + 0.5) * (lon_max - lon_min) / width
        zone_lat  = float(np.mean(zone_lats))
        zone_lon  = float(np.mean(zone_lons))
        mean_slope = float(np.mean(slope_deg[labeled == comp_id]))

        zones.append(
            {
                "ring":        ring,
                "center":      {"lat": round(zone_lat, 5), "lon": round(zone_lon, 5)},
                "label":       f"Gentle Slope (< {slope_threshold_deg:.0f}°)",
                "description": (
                    f"Low-slope terrain — retardant holds position. "
                    f"Mean slope {mean_slope:.1f}°. Source: Copernicus DEM GLO-30."
                ),
                "halt_rate_modifier": "+12% vs steep terrain",
                "mean_slope_deg":     round(mean_slope, 1),
                "pixel_count":        pixel_count,
                "source":             "Sentinel Hub — Copernicus DEM GLO-30",
            }
        )

    return zones


# ── In-process result cache (terrain is static — cache aggressively) ──────────
# Key: (rounded_lat, rounded_lon) → (timestamp, zones)
_zone_cache: dict[tuple[float, float, float], tuple[float, list[dict]]] = {}
_ZONE_TTL = 3600 * 4  # 4 hours — DEM data never changes


def get_terrain_zones(
    center_lat: float,
    center_lon: float,
    radius_km: float = 25.0,
    slope_threshold_deg: float = 1.0,
    min_zone_pixels: int = 200,
    max_zones: int = 3,
    image_size: int = 256,
) -> list[dict]:
    """
    Return gentle-slope terrain zones around (center_lat, center_lon).

    The DEM tile is fetched once from Sentinel Hub and cached for 4 hours;
    subsequent calls for the same location are free.

    Parameters
    ----------
    center_lat, center_lon : float
        Centre of the area of interest.
    radius_km : float
        Half-width of the DEM tile (default 25 km → 50 × 50 km coverage).
    slope_threshold_deg : float
        Maximum slope angle considered "gentle / favorable" (default 15°).
    min_zone_pixels : int
        Minimum connected-component size to include (noise filter).
    max_zones : int
        Maximum number of zones to return (largest first).
    image_size : int
        Width = height of the requested DEM tile in pixels (default 256).

    Returns
    -------
    list[dict]
        Each dict has keys: ring, center, label, description, halt_rate_modifier
        matching the TerrainZone interface expected by the frontend.

    Raises
    ------
    EnvironmentError
        If SENTINELHUB_CLIENT_ID / SENTINELHUB_CLIENT_SECRET are not set.
    requests.HTTPError
        If the Sentinel Hub API returns a non-2xx response.
    """
    cache_key = (round(center_lat, 2), round(center_lon, 2), round(slope_threshold_deg, 1))
    cached = _zone_cache.get(cache_key)
    if cached and (time.monotonic() - cached[0]) < _ZONE_TTL:
        logger.debug("terrain-zones cache hit for %s", cache_key)
        return cached[1]

    # ── Compute bounding box ───────────────────────────────────────────────────
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * math.cos(math.radians(center_lat)))
    lat_min = center_lat - dlat
    lat_max = center_lat + dlat
    lon_min = center_lon - dlon
    lon_max = center_lon + dlon

    logger.info(
        "Fetching Sentinel Hub DEM  bbox=[%.4f,%.4f,%.4f,%.4f]  size=%dx%d",
        lon_min, lat_min, lon_max, lat_max, image_size, image_size,
    )

    tiff_bytes = _fetch_dem_tiff(lat_min, lon_min, lat_max, lon_max, image_size)

    # ── Parse GeoTIFF in memory ────────────────────────────────────────────────
    import rasterio
    from rasterio.io import MemoryFile

    with MemoryFile(tiff_bytes) as memfile, memfile.open() as dataset:
        elev = dataset.read(1).astype(np.float32)

    zones = _extract_zones(
        elev,
        lat_min, lon_min, lat_max, lon_max,
        slope_threshold_deg=slope_threshold_deg,
        min_zone_pixels=min_zone_pixels,
        max_zones=max_zones,
    )

    _zone_cache[cache_key] = (time.monotonic(), zones)
    logger.info("Extracted %d terrain zones for %s", len(zones), cache_key)
    return zones
