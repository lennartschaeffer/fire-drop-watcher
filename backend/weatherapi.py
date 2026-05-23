import os
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("WEATHERKEY")
BASE_URL = "https://api.weatherapi.com/v1"


def _fetch_current(lat: float, lon: float) -> dict:
    """Internal: fetch raw current weather data from WeatherAPI."""
    response = httpx.get(
        f"{BASE_URL}/current.json",
        params={"key": API_KEY, "q": f"{lat},{lon}"},
    )
    response.raise_for_status()
    return response.json()["current"]


def get_wind(lat: float, lon: float) -> dict:
    """
    Get wind speed and direction for a location.

    Args:
        lat: Latitude (e.g. 40.7128)
        lon: Longitude (e.g. -74.0060)

    Returns:
        dict with wind_kph (float) and wind_dir (str, e.g. "SW")
    """
    current = _fetch_current(lat, lon)
    return {
        "wind_kph": current["wind_kph"],
        "wind_dir": current["wind_dir"],
    }


def get_temperature(lat: float, lon: float) -> dict:
    """
    Get temperature for a location.

    Args:
        lat: Latitude (e.g. 40.7128)
        lon: Longitude (e.g. -74.0060)

    Returns:
        dict with temp_c (float)
    """
    current = _fetch_current(lat, lon)
    return {
        "temp_c": current["temp_c"],
    }


def get_humidity(lat: float, lon: float) -> dict:
    """
    Get humidity for a location.

    Args:
        lat: Latitude (e.g. 40.7128)
        lon: Longitude (e.g. -74.0060)

    Returns:
        dict with humidity (int, percentage)
    """
    current = _fetch_current(lat, lon)
    return {
        "humidity": current["humidity"],
    }


if __name__ == "__main__":
    lat = os.environ.get("WEATHERLAT")
    lon = os.environ.get("WEATHERLON")

    if not lat or not lon:
        raise ValueError("WEATHERLAT and WEATHERLON must be set in your .env file")

    lat, lon = float(lat), float(lon)

    wind = get_wind(lat, lon)
    temp = get_temperature(lat, lon)
    humidity = get_humidity(lat, lon)

    print(f"  Temperature : {temp['temp_c']}°C")
    print(f"  Humidity    : {humidity['humidity']}%")
    print(f"  Wind        : {wind['wind_kph']} kph {wind['wind_dir']}")
