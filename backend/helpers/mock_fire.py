import math
import random
from typing import Literal

def generate_fire_perimeter(
    center_lat: float,
    center_lon: float,
    radius_km: float = 1.0,
    num_points: int = 24,
    seed: int | None = None,
) -> list[list[float]]:
    # Produces a GeoJSON-compatible closed ring: [[lon, lat], ..., [lon, lat]]
    # Shape is an irregular polygon approximating a real fire perimeter — each
    # vertex is placed at an evenly-spaced angle but with ±30% random radial noise.
    rng = random.Random(seed)
    points = []
    for i in range(num_points):
        angle = (2 * math.pi * i) / num_points
        r = radius_km * rng.uniform(0.7, 1.3)  # ±30% noise on radius
        # 111.32 km per degree latitude; longitude degrees shrink with cos(lat)
        dlat = (r / 111.32) * math.cos(angle)
        dlon = (r / (111.32 * math.cos(math.radians(center_lat)))) * math.sin(angle)
        points.append([round(center_lon + dlon, 6), round(center_lat + dlat, 6)])
    points.append(points[0])  # close the ring
    return points

