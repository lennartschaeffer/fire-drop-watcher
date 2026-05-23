import math
import random
from typing import Literal

def generate_fire_perimeter(
    center_lat: float,
    center_lon: float,
    radius_km: float = 1.0,
    num_points: int = 24,
    seed: int | None = None,
    spread_direction_deg: float | None = None,
    elongation: float = 2.5,
) -> list[list[float]]:
    # Produces a GeoJSON-compatible closed ring: [[lon, lat], ..., [lon, lat]]
    # Shape is an irregular polygon approximating a real fire perimeter — each
    # vertex is placed at an evenly-spaced angle but with ±30% random radial noise.
    # When spread_direction_deg is given, points facing that bearing are pushed further
    # out by `elongation`, creating a teardrop shape in the spread direction.
    rng = random.Random(seed)
    spread_rad = math.radians(spread_direction_deg) if spread_direction_deg is not None else None
    points = []
    for i in range(num_points):
        angle = (2 * math.pi * i) / num_points
        r = radius_km * rng.uniform(0.95, 1.05)  # ±5% noise on radius
        if spread_rad is not None:
            # alignment: 1.0 at spread direction, -1.0 at opposite; clamp to front half
            alignment = math.cos(angle - spread_rad)
            stretch = 1.0 + (elongation - 1.0) * max(0.0, alignment)
            r *= stretch
        # 111.32 km per degree latitude; longitude degrees shrink with cos(lat)
        dlat = (r / 111.32) * math.cos(angle)
        dlon = (r / (111.32 * math.cos(math.radians(center_lat)))) * math.sin(angle)
        points.append([round(center_lon + dlon, 6), round(center_lat + dlat, 6)])
    points.append(points[0])  # close the ring
    return points

