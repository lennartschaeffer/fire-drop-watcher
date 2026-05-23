import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("WEATHERKEY")
BASE_URL = "https://api.weatherapi.com/v1"

# Simple in-memory TTL cache: key=(lat_rounded, lon_rounded) → (timestamp, data)
_WEATHER_CACHE: dict[tuple[float, float], tuple[float, dict]] = {}
_WEATHER_TTL_SECONDS = 600  # 10 minutes


def _cache_key(lat: float, lon: float) -> tuple[float, float]:
    # Round to 2 decimal places (~1 km grid) so nearby requests share a cache entry
    return (round(lat, 2), round(lon, 2))


def _fetch_current(lat: float, lon: float) -> dict:
    """Internal: fetch raw current weather data from WeatherAPI (with TTL cache)."""
    key = _cache_key(lat, lon)
    cached = _WEATHER_CACHE.get(key)
    if cached and (time.monotonic() - cached[0]) < _WEATHER_TTL_SECONDS:
        return cached[1]

    response = httpx.get(
        f"{BASE_URL}/current.json",
        params={"key": API_KEY, "q": f"{lat},{lon}"},
    )
    response.raise_for_status()
    data = response.json()["current"]
    _WEATHER_CACHE[key] = (time.monotonic(), data)
    return data


def get_weather(lat: float, lon: float) -> dict:
    """
    Fetch all weather fields in a single API call.

    Returns:
        dict with wind_kph, wind_dir, temp_c, humidity, precip_mm
    """
    current = _fetch_current(lat, lon)
    return {
        "wind_kph": current["wind_kph"],
        "wind_dir": current["wind_dir"],
        "temp_c": current["temp_c"],
        "humidity": current["humidity"],
        "precip_mm": current.get("precip_mm", 0.0),
    }


# Keep individual helpers for backwards compatibility
def get_wind(lat: float, lon: float) -> dict:
    current = _fetch_current(lat, lon)
    return {"wind_kph": current["wind_kph"], "wind_dir": current["wind_dir"]}


def get_temperature(lat: float, lon: float) -> dict:
    current = _fetch_current(lat, lon)
    return {"temp_c": current["temp_c"]}


def get_precipitation(lat: float, lon: float) -> dict:
    current = _fetch_current(lat, lon)
    return {"precip_mm": current.get("precip_mm", 0.0)}


def get_humidity(lat: float, lon: float) -> dict:
    current = _fetch_current(lat, lon)
    return {"humidity": current["humidity"]}


if __name__ == "__main__":
    lat = os.environ.get("WEATHERLAT")
    lon = os.environ.get("WEATHERLON")

    if not lat or not lon:
        raise ValueError("WEATHERLAT and WEATHERLON must be set in your .env file")

    lat, lon = float(lat), float(lon)

    weather = get_weather(lat, lon)
    print(f"  Temperature : {weather['temp_c']}°C")
    print(f"  Humidity    : {weather['humidity']}%")
    print(f"  Wind        : {weather['wind_kph']} kph {weather['wind_dir']}")
    print(f"  Precip      : {weather['precip_mm']} mm")
