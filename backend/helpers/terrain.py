import math
import asyncio
import requests
import httpx


def fetch_altitude(lat: float, lon: float) -> float | None:
    url = "https://geogratis.gc.ca/services/elevation/cdem/altitude"
    try:
        response = requests.get(url, params={"lat": lat, "lon": lon}, timeout=10)
        if response.status_code == 200:
            return response.json().get("altitude")
    except Exception as e:
        print(f"API Error: {e}")
    return None


async def async_fetch_altitude(client: httpx.AsyncClient, lat: float, lon: float) -> float | None:
    url = "https://geogratis.gc.ca/services/elevation/cdem/altitude"
    try:
        response = await client.get(url, params={"lat": lat, "lon": lon}, timeout=10)
        if response.status_code == 200:
            return response.json().get("altitude")
    except Exception as e:
        print(f"API Error: {e}")
    return None


async def async_compute_slope(client: httpx.AsyncClient, lat: float, lon: float, delta: float = 0.001) -> dict:
    north, south, east, west, center = await asyncio.gather(
        async_fetch_altitude(client, lat + delta, lon),
        async_fetch_altitude(client, lat - delta, lon),
        async_fetch_altitude(client, lat, lon + delta),
        async_fetch_altitude(client, lat, lon - delta),
        async_fetch_altitude(client, lat, lon),
    )

    if any(v is None for v in [north, south, east, west]):
        return {"error": "Could not fetch elevation", "slope_degrees": 0.0, "aspect_degrees": 0.0, "elevation_m": 0.0}

    meters_per_deg_lat = 111_320
    meters_per_deg_lon = 111_320 * math.cos(math.radians(lat))

    dz_dy = (north - south) / (2 * delta * meters_per_deg_lat)
    dz_dx = (east - west)   / (2 * delta * meters_per_deg_lon)

    slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = math.degrees(slope_rad)

    aspect_rad = math.atan2(dz_dx, dz_dy)
    aspect_deg = (math.degrees(aspect_rad) + 360) % 360

    return {
        "slope_degrees": round(slope_deg, 4),
        "aspect_degrees": round(aspect_deg, 4),
        "elevation_m": round(center, 1) if center is not None else 0.0,
    }


def compute_slope(lat: float, lon: float, delta: float = 0.001) -> dict:
    north = fetch_altitude(lat + delta, lon)
    south = fetch_altitude(lat - delta, lon)
    east  = fetch_altitude(lat, lon + delta)
    west  = fetch_altitude(lat, lon - delta)

    if any(v is None for v in [north, south, east, west]):
        return {"error": "Could not fetch elevation for one or more sample points"}

    assert north is not None and south is not None and east is not None and west is not None

    meters_per_deg_lat = 111_320
    meters_per_deg_lon = 111_320 * math.cos(math.radians(lat))

    dz_dy = (north - south) / (2 * delta * meters_per_deg_lat)
    dz_dx = (east - west)   / (2 * delta * meters_per_deg_lon)

    slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = math.degrees(slope_rad)

    aspect_rad = math.atan2(dz_dx, dz_dy)
    aspect_deg = (math.degrees(aspect_rad) + 360) % 360

    center = fetch_altitude(lat, lon)
    return {
        "slope_degrees": round(slope_deg, 4),
        "aspect_degrees": round(aspect_deg, 4),
        "elevation_m": round(center, 1) if center is not None else 0.0,
    }
