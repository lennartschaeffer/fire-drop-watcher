import math
import asyncio
import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from helpers.terrain import compute_slope, async_compute_slope
from helpers.weatherapi import get_wind, get_temperature, get_humidity, get_precipitation
from helpers.mock_fire import generate_fire_perimeter
from fuel_lookup import get_fuel_at
from model_inference import predict as model_predict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aao_briefing import get_aao_briefing

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
    precip = get_precipitation(lat, lon)
    return {**wind, **temp, **humidity, **precip}

WIND_DIR_DEGREES = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}

@app.get("/simulate")
async def simulate(
    lat: float = Query(...),
    lon: float = Query(...),
    radius_km: float = Query(2.0),
    steps: int = Query(15),
    seed: int = Query(42),
):
    wind = get_wind(lat, lon)
    temp = get_temperature(lat, lon)
    humidity = get_humidity(lat, lon)
    precip = get_precipitation(lat, lon)
    terrain_data = compute_slope(lat, lon)
    fuel_data = get_fuel_at(lat, lon)

    wind_from_deg = WIND_DIR_DEGREES.get(wind["wind_dir"], 0)

    base_prediction = model_predict(
        lat=lat, lon=lon,
        wind_speed=wind["wind_kph"],
        wind_direction=wind_from_deg,
        fuel=fuel_data.fuel_type,
        slope=terrain_data.get("slope_degrees", 0),
        aspect=terrain_data.get("aspect_degrees", 0),
        humidity=humidity["humidity"],
        temp=temp["temp_c"],
        ffmc=fuel_data.ffmc,
        dmc=fuel_data.dmc,
        dc=fuel_data.dc,
        bui=fuel_data.bui,
        isi=fuel_data.isi,
        pcp=precip["precip_mm"],
        elev=terrain_data.get("elevation_m", 0),
    )

    fire_direction = base_prediction["fire_direction"]
    fire_severity = base_prediction["fire_severity"]

    spread_bearing_rad = math.radians(fire_direction)
    center_shift_km = wind["wind_kph"] * 0.01
    radius_growth_km = 0.15

    # Compute perimeter points for step 0 to derive per-point terrain once.
    # Terrain is static — we reuse these lookups across all animation steps.
    initial_coords = generate_fire_perimeter(lat, lon, radius_km, 24, seed, spread_direction_deg=fire_direction)
    perimeter_points = [(c[1], c[0]) for c in initial_coords[:-1]]  # (lat, lon), drop closing point

    async with httpx.AsyncClient() as client:
        terrain_results = await asyncio.gather(
            *[async_compute_slope(client, p[0], p[1]) for p in perimeter_points]
        )

    perimeter_predictions = [
        model_predict(
            lat=p[0], lon=p[1],
            wind_speed=wind["wind_kph"],
            wind_direction=wind_from_deg,
            fuel=fuel_data.fuel_type,
            slope=t.get("slope_degrees", 0),
            aspect=t.get("aspect_degrees", 0),
            humidity=humidity["humidity"],
            temp=temp["temp_c"],
            ffmc=fuel_data.ffmc,
            dmc=fuel_data.dmc,
            dc=fuel_data.dc,
            bui=fuel_data.bui,
            isi=fuel_data.isi,
            pcp=precip["precip_mm"],
            elev=t.get("elevation_m", 0),
        )
        for p, t in zip(perimeter_points, terrain_results)
    ]

    frames = []
    cur_lat, cur_lon, cur_radius = lat, lon, radius_km

    for i in range(steps):
        cur_lat += (center_shift_km / 111.32) * math.cos(spread_bearing_rad)
        cur_lon += (center_shift_km / (111.32 * math.cos(math.radians(cur_lat)))) * math.sin(spread_bearing_rad)
        cur_radius += radius_growth_km

        coords = generate_fire_perimeter(cur_lat, cur_lon, cur_radius, 24, seed + i, spread_direction_deg=fire_direction)
        # Attach per-point predictions to current step's perimeter coordinates
        step_coords = coords[:-1]  # drop closing point
        arrows = [
            {
                "lon": step_coords[j][0],
                "lat": step_coords[j][1],
                "direction": perimeter_predictions[j]["fire_direction"],
                "severity": perimeter_predictions[j]["fire_severity"],
            }
            for j in range(len(step_coords))
        ]

        frames.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "center_lat": round(cur_lat, 6),
                "center_lon": round(cur_lon, 6),
                "radius_km": round(cur_radius, 3),
                "step": i,
                "fire_direction": fire_direction,
                "fire_severity": fire_severity,
                "perimeter_arrows": arrows,
            },
        })

    return {
        "frames": frames,
        "weather": {**wind, **temp, **humidity, **precip},
        "terrain": terrain_data,
        "fuel": {
            "fuel_type": fuel_data.fuel_type,
            "fuel_description": fuel_data.fuel_description,
            "flammability": fuel_data.flammability,
        },
        "prediction": base_prediction,
        "steps": steps,
    }

class BriefingRequest(BaseModel):
    image_b64: str          # Base64-encoded map image (drop zone marked in purple)
    mime_type: str = "image/png"  # e.g. "image/png", "image/jpeg"


@app.post("/aao-briefing")
def aao_briefing(req: BriefingRequest):
    """
    Generate an AAO talk-in briefing from a map image.
    The drop zone must be marked in purple on the image.
    Send the image as a base64-encoded string in the request body.
    """
    try:
        briefing = get_aao_briefing(
            image_b64=req.image_b64,
            mime_type=req.mime_type,
        )
        return {"briefing": briefing.text, "model": briefing.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
