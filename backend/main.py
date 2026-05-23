import math
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from helpers.terrain import compute_slope
from helpers.weatherapi import get_wind, get_temperature, get_humidity
from helpers.mock_fire import generate_fire_perimeter

app = FastAPI(title="Drop It Like It's Hot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/fire-perimeter")
def fire_perimeter(
    lat: float = Query(...),
    lon: float = Query(...),
    radius_km: float = Query(1.0),
    num_points: int = Query(24),
    seed: int = Query(42),
):
    coords = generate_fire_perimeter(lat, lon, radius_km, num_points, seed)
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
        "properties": {"center_lat": lat, "center_lon": lon, "radius_km": radius_km},
    }

@app.get("/terrain")
def terrain(lat: float = Query(...), lon: float = Query(...)):
    return compute_slope(lat, lon)

@app.get("/weather")
def weather(lat: float = Query(...), lon: float = Query(...)):
    wind = get_wind(lat, lon)
    temp = get_temperature(lat, lon)
    humidity = get_humidity(lat, lon)
    return {**wind, **temp, **humidity}

WIND_DIR_DEGREES = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}

@app.get("/simulate")
def simulate(
    lat: float = Query(...),
    lon: float = Query(...),
    radius_km: float = Query(2.0),
    steps: int = Query(15),
    seed: int = Query(42),
):
    wind = get_wind(lat, lon)
    temp = get_temperature(lat, lon)
    humidity = get_humidity(lat, lon)
    terrain_data = compute_slope(lat, lon)

    # Wind comes FROM this bearing; fire spreads in the opposite direction.
    wind_from_deg = WIND_DIR_DEGREES.get(wind["wind_dir"], 0)
    spread_bearing_rad = math.radians((wind_from_deg + 180) % 360)

    # Center drifts ~1% of wind speed (kph) per step in km; radius grows 0.15 km/step.
    center_shift_km = wind["wind_kph"] * 0.01
    radius_growth_km = 0.15

    frames = []
    cur_lat, cur_lon, cur_radius = lat, lon, radius_km

    for i in range(steps):
        cur_lat += (center_shift_km / 111.32) * math.cos(spread_bearing_rad)
        cur_lon += (center_shift_km / (111.32 * math.cos(math.radians(cur_lat)))) * math.sin(spread_bearing_rad)
        cur_radius += radius_growth_km

        coords = generate_fire_perimeter(cur_lat, cur_lon, cur_radius, 24, seed + i)
        frames.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "center_lat": round(cur_lat, 6),
                "center_lon": round(cur_lon, 6),
                "radius_km": round(cur_radius, 3),
                "step": i,
            },
        })

    return {
        "frames": frames,
        "weather": {**wind, **temp, **humidity},
        "terrain": terrain_data,
        "steps": steps,
    }

