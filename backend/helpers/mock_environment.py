"""
mock_environment.py
-------------------
Mock environment overlay data for the fire map.

shrub_zones  — list of shrub vegetation clusters (HeatmapLayer points).
               Shrub fuel has the highest retardant halt-drop success rate
               (~70% vs ~55% avg) per the AFUE research.

terrain_zones — list of favorable-terrain polygons (DeckGL PolygonLayer rings).
                Co-located with their paired shrub zone so both overlays
                reinforce each other as a clear optimal drop zone.
"""

import math
import random

_FIRE_CENTERS: dict[str, tuple[float, float]] = {
    "barrington": (43.639699, -65.469716),
    "tantallon":  (44.693,    -63.780),
}


def _geo_offset(lat: float, lon: float, bearing_deg: float, dist_km: float) -> tuple[float, float]:
    b = math.radians(bearing_deg)
    dlat = (dist_km / 111.32) * math.cos(b)
    dlon = (dist_km / (111.32 * math.cos(math.radians(lat)))) * math.sin(b)
    return round(lat + dlat, 6), round(lon + dlon, 6)


def _scatter_points(center_lat: float, center_lon: float, radius_km: float, n: int, seed: int) -> list[dict]:
    """Scatter n points in a gaussian-ish blob around (center_lat, center_lon)."""
    rng = random.Random(seed)
    points = []
    for _ in range(n):
        r = abs(rng.gauss(0, radius_km * 0.45))
        bearing = rng.uniform(0, 360)
        lat, lon = _geo_offset(center_lat, center_lon, bearing, r)
        points.append({"lat": lat, "lon": lon})
    return points


def _ellipse_ring(
    center_lat: float,
    center_lon: float,
    radius_a_km: float,
    radius_b_km: float,
    bearing_deg: float,
    n_points: int = 48,
    seed: int = 0,
) -> list[list[float]]:
    """
    Build an irregular-ish ellipse polygon ring as [[lon, lat], ...].
    ±8% seeded noise per vertex keeps the shape organic.
    """
    rng = random.Random(seed)
    ring: list[list[float]] = []
    for i in range(n_points):
        theta = (2 * math.pi * i) / n_points
        r_km = (radius_a_km * radius_b_km) / math.sqrt(
            (radius_b_km * math.cos(theta)) ** 2
            + (radius_a_km * math.sin(theta)) ** 2
        )
        r_km *= 1.0 + rng.uniform(-0.08, 0.08)
        actual_bearing = (bearing_deg + math.degrees(theta)) % 360
        lat, lon = _geo_offset(center_lat, center_lon, actual_bearing, r_km)
        ring.append([lon, lat])
    ring.append(ring[0])
    return ring


def get_mock_environment(fire_id: str) -> dict:
    """
    Return all environment overlay data for the given fire as parallel lists:

      shrub_zones    — list of shrub clusters (heatmap point blobs)
      terrain_zones  — list of favorable-terrain polygons (ellipse rings)

    Each shrub zone is paired with a terrain zone at the same location.
    Primary zone: SW, ~12 km — overlapping shrub + terrain for a clear optimal drop spot.
    Secondary zone: ESE, ~18 km — smaller, lower-confidence sample point.
    """
    center_lat, center_lon = _FIRE_CENTERS.get(fire_id, _FIRE_CENTERS["barrington"])

    # ── Zone 1: SW — primary optimal drop zone ────────────────────────────────
    z1_lat, z1_lon = _geo_offset(center_lat, center_lon, 225, 12.0)

    shrub_zone_1 = {
        "center": {"lat": z1_lat, "lon": z1_lon},
        "points": _scatter_points(z1_lat, z1_lon, radius_km=5.5, n=60, seed=101),
        "label": "Dense Shrub",
        "halt_rate": "~70%",
        "description": "Shrub vegetation — highest retardant effectiveness per AFUE research",
    }
    t1_lat, t1_lon = _geo_offset(center_lat, center_lon, 240, 11.5)
    terrain_zone_1 = {
        "ring": _ellipse_ring(t1_lat, t1_lon, radius_a_km=5.5, radius_b_km=4.4, bearing_deg=240, seed=202),
        "center": {"lat": t1_lat, "lon": t1_lon},
        "label": "Gentle Slope (< 15°)",
        "description": "Low slope — retardant holds position. Ideal deterrent line anchor.",
        "halt_rate_modifier": "+12% vs steep terrain",
    }

    # ── Zone 2: shrub W, terrain N — secondary sample points ─────────────────
    z2_shrub_lat, z2_shrub_lon = _geo_offset(center_lat, center_lon, 270, 10.0)
    z2_terrain_lat, z2_terrain_lon = _geo_offset(center_lat, center_lon, 0, 10.0)

    shrub_zone_2 = {
        "center": {"lat": z2_shrub_lat, "lon": z2_shrub_lon},
        "points": _scatter_points(z2_shrub_lat, z2_shrub_lon, radius_km=3.5, n=35, seed=404),
        "label": "Mixed Shrub",
        "halt_rate": "~65%",
        "description": "Shrub-dominant fuel mix — good retardant effectiveness",
    }
    terrain_zone_2 = {
        "ring": _ellipse_ring(z2_terrain_lat, z2_terrain_lon, radius_a_km=4.0, radius_b_km=3.2, bearing_deg=0, seed=505),
        "center": {"lat": z2_terrain_lat, "lon": z2_terrain_lon},
        "label": "Gentle Slope (< 15°)",
        "description": "Low slope — retardant holds position.",
        "halt_rate_modifier": "+12% vs steep terrain",
    }

    return {
        "shrub_zones": [shrub_zone_1, shrub_zone_2],
        "terrain_zones": [terrain_zone_1, terrain_zone_2],
    }
