import math
import time
import asyncio
import httpx
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from helpers.terrain import compute_slope, async_compute_slope
from helpers.weatherapi import get_weather, get_wind, get_temperature, get_humidity, get_precipitation
from helpers.mock_fire import generate_fire_perimeter
from fuel_lookup import get_fuel_at
from model_inference import predict as model_predict
from aao_briefing import get_aao_briefing

# ---------------------------------------------------------------------------
# In-memory TTL cache for terrain results.
# Each perimeter point needs 5 elevation API calls to compute slope/aspect.
# With 24 points that's 120 calls — cached for 15 min to make re-runs free.
# ---------------------------------------------------------------------------
_TERRAIN_CACHE: dict[tuple[float, float], tuple[float, dict]] = {}
_TERRAIN_TTL_SECONDS = 900  # 15 minutes


def _terrain_cache_key(lat: float, lon: float) -> tuple[float, float]:
    # Round to 3dp (~110 m grid) so nearby points can still share entries
    return (round(lat, 3), round(lon, 3))


async def _cached_slope(client: httpx.AsyncClient, lat: float, lon: float) -> dict:
    key = _terrain_cache_key(lat, lon)
    cached = _TERRAIN_CACHE.get(key)
    if cached and (time.monotonic() - cached[0]) < _TERRAIN_TTL_SECONDS:
        return cached[1]
    result = await async_compute_slope(client, lat, lon)
    _TERRAIN_CACHE[key] = (time.monotonic(), result)
    return result

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
    return get_weather(lat, lon)

WIND_DIR_DEGREES = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}

def _geo_offset(lat: float, lon: float, bearing_deg: float, dist_km: float) -> tuple[float, float]:
    """Return (lat, lon) offset from a point by dist_km in bearing_deg direction."""
    b = math.radians(bearing_deg)
    dlat = (dist_km / 111.32) * math.cos(b)
    dlon = (dist_km / (111.32 * math.cos(math.radians(lat)))) * math.sin(b)
    return round(lat + dlat, 6), round(lon + dlon, 6)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _add_aircraft_to_frames(
    frames: list[dict],
    _fire_direction: float,   # unused — paths are deterministic, not wind-relative
    origin_lat: float,
    origin_lon: float,
    radius_km: float,
) -> None:
    """
    Mutate each frame's properties in-place to add:
      bird_dog, tanker, retardant_line, atu_event, sim_status

    Aircraft paths are DETERMINISTIC — fixed bearings regardless of wind/fire
    direction so the demo always looks the same and is easy to follow.

    Tanker always approaches from the north and drops heading south.
    Retardant line is laid east-west, south of the fire origin.
    Act boundaries scale proportionally to total frame count.

      Act 1  0 – 28%   Bird Dog solo recon pass NE→SW over the fire
      Act 2  28 – 55%  Tanker en route south, Bird Dog repositions to escort slot
      Act 3  55 – 85%  Drop run S (door_open ~67%, door_close ~78%)
      Act 4  85 – 100% Post-drop, tanker departs north, Bird Dog orbits drop zone
    """
    total = len(frames)

    # ── Deterministic fixed bearings ──────────────────────────────────────────
    # Tanker always approaches from the north and drops heading south.
    # Retardant line is laid east-west, south of the fire origin.
    APPROACH_BEARING = 0    # tanker comes from the north
    DROP_BEARING     = 180  # tanker flies south on the drop run → line goes south of fire
    LEFT_OF_DROP     = (DROP_BEARING + 270) % 360   # 90° = east (left when heading south)
    RIGHT_OF_DROP    = (DROP_BEARING + 90)  % 360   # 270° = west (right when heading south)

    # Recon: Bird Dog flies a clockwise arc around the fire (NE → SE → SW → W → N)
    BD_ARC_RADIUS     = 7.0    # km — orbit distance from fire centre
    BD_ARC_START_BRG  = 45.0   # start bearing: NE
    BD_ARC_END_BRG    = 355.0  # end bearing: just west of north (310° of clockwise sweep)
    # Heading for a clockwise orbit = bearing_from_centre + 90°
    BD_ARC_END_HDG    = round(BD_ARC_END_BRG + 90) % 360  # ≈ 85° (heading NE-ish)

    # ── Act boundaries (proportional — works for any step count) ─────────────
    ACT2_START  = round(total * 0.48)   # recon runs longer — slowed down
    ACT3_START  = round(total * 0.57)   # bird-dog joins tanker quickly (9% window)
    DOOR_OPEN   = round(total * 0.67)
    DOOR_CLOSE  = round(total * 0.78)
    ACT4_START  = round(total * 0.85)

    # ── Key waypoints ─────────────────────────────────────────────────────────
    # Bird Dog recon arc: clockwise from NE to just-past-north
    bd_recon_end_lat, bd_recon_end_lon = _geo_offset(origin_lat, origin_lon, BD_ARC_END_BRG, BD_ARC_RADIUS)

    # Tanker base: north, 14 km out (always the same spot)
    tk_base_lat, tk_base_lon = _geo_offset(origin_lat, origin_lon, APPROACH_BEARING, 14.0)

    # Drop zone: south of the fire, ahead of the leading edge
    drop_zone_lat, drop_zone_lon = _geo_offset(origin_lat, origin_lon, DROP_BEARING, radius_km * 3.5)

    # Tanker run: enters from 8 km north, exits 3 km south of the drop zone
    tk_run_start_lat, tk_run_start_lon = _geo_offset(origin_lat, origin_lon, APPROACH_BEARING, 8.0)
    tk_run_end_lat,   tk_run_end_lon   = _geo_offset(drop_zone_lat, drop_zone_lon, DROP_BEARING, 3.0)

    # Retardant line: east-west, centred on drop zone, 3 km wide
    ret_start_lat, ret_start_lon = _geo_offset(drop_zone_lat, drop_zone_lon, LEFT_OF_DROP,  1.5)
    ret_end_lat,   ret_end_lon   = _geo_offset(drop_zone_lat, drop_zone_lon, RIGHT_OF_DROP, 1.5)

    # Tanker departure: exits back toward north base after drop
    tk_depart_lat, tk_depart_lon = _geo_offset(drop_zone_lat, drop_zone_lon, APPROACH_BEARING, 14.0)

    # ── Bird Dog escort start position (precomputed for smooth Act 2 → 3 join) ─
    # At t=0 of Act 3 the tanker is at tk_run_start; the bird dog should already
    # be behind-left of it so there is no jump when the act boundary crosses.
    _bd_escort_behind_lat, _bd_escort_behind_lon = _geo_offset(
        tk_run_start_lat, tk_run_start_lon, APPROACH_BEARING, 1.5
    )
    bd_escort_start_lat, bd_escort_start_lon = _geo_offset(
        _bd_escort_behind_lat, _bd_escort_behind_lon, LEFT_OF_DROP, 0.8
    )

    completed_retardant_line: list | None = None

    for i, frame in enumerate(frames):
        props = frame["properties"]

        # ── Act 1: Bird Dog clockwise recon arc NE → N, tanker parked at base ──
        if i < ACT2_START:
            t = i / max(ACT2_START - 1, 1)
            arc_brg = _lerp(BD_ARC_START_BRG, BD_ARC_END_BRG, t)
            bd_lat, bd_lon = _geo_offset(origin_lat, origin_lon, arc_brg % 360, BD_ARC_RADIUS)
            bd_hdg = round(arc_brg + 90) % 360  # tangent heading for clockwise orbit

            props["bird_dog"] = {"lat": bd_lat, "lon": bd_lon, "heading_deg": bd_hdg, "visible": True}
            props["tanker"]   = {"lat": tk_base_lat, "lon": tk_base_lon, "heading_deg": DROP_BEARING, "visible": False}
            props["retardant_line"] = None
            props["atu_event"]  = "none"
            props["sim_status"] = "Bird Dog performing reconnaissance"

        # ── Act 2: Tanker en route north → south run-start ────────────────────
        #    Bird Dog flies directly from its recon end position to the Act 3
        #    escort position — one smooth repositioning flight, no extra loops.
        elif i < ACT3_START:
            t = (i - ACT2_START) / max(ACT3_START - ACT2_START - 1, 1)

            # Tanker flies from north base to run-start (straight south)
            tk_lat = _lerp(tk_base_lat,       tk_run_start_lat, t)
            tk_lon = _lerp(tk_base_lon,       tk_run_start_lon, t)

            # Bird Dog: lerp from end-of-recon position to escort start position.
            # This gives one smooth diagonal flight with no spinning or snapping.
            bd_lat = _lerp(bd_recon_end_lat, bd_escort_start_lat, t)
            bd_lon = _lerp(bd_recon_end_lon, bd_escort_start_lon, t)
            # Heading gradually turns from recon heading (SW=225) to drop heading (S=180)
            bd_hdg = round(_lerp(BD_ARC_END_HDG, DROP_BEARING, t)) % 360

            props["bird_dog"] = {"lat": bd_lat, "lon": bd_lon, "heading_deg": bd_hdg, "visible": True}
            props["tanker"]   = {"lat": tk_lat, "lon": tk_lon, "heading_deg": DROP_BEARING, "visible": True}
            props["retardant_line"] = None
            props["atu_event"]  = "none"
            props["sim_status"] = "Tanker en route" if t < 0.5 else "Tanker on approach"

        # ── Act 3: Drop run (north → south) ───────────────────────────────────
        elif i < ACT4_START:
            t = (i - ACT3_START) / max(ACT4_START - ACT3_START - 1, 1)

            # Tanker flies straight: run_start → run_end (south)
            tk_lat = _lerp(tk_run_start_lat, tk_run_end_lat, t)
            tk_lon = _lerp(tk_run_start_lon, tk_run_end_lon, t)

            # Bird Dog holds escort: behind (north) and slightly east of the tanker
            bd_behind_lat, bd_behind_lon = _geo_offset(tk_lat, tk_lon, APPROACH_BEARING, 1.5)
            bd_lat, bd_lon = _geo_offset(bd_behind_lat, bd_behind_lon, LEFT_OF_DROP, 0.8)

            # ATU door events based on proportional frame positions
            if i < DOOR_OPEN:
                atu_event = "none"
                sim_status = "Bird Dog observing drop"
                current_retardant = None
            elif i == DOOR_OPEN:
                atu_event = "door_open"
                sim_status = "Drop in progress"
                current_retardant = [[ret_start_lon, ret_start_lat], [ret_start_lon, ret_start_lat]]
            elif i < DOOR_CLOSE:
                atu_event = "none"
                sim_status = "Drop in progress"
                door_t = (i - DOOR_OPEN) / max(DOOR_CLOSE - DOOR_OPEN, 1)
                partial_end_lat = _lerp(ret_start_lat, ret_end_lat, door_t)
                partial_end_lon = _lerp(ret_start_lon, ret_end_lon, door_t)
                current_retardant = [
                    [ret_start_lon, ret_start_lat],
                    [partial_end_lon, partial_end_lat],
                ]
            elif i == DOOR_CLOSE:
                atu_event = "door_close"
                sim_status = "Drop complete — retardant line laid"
                completed_retardant_line = [
                    [ret_start_lon, ret_start_lat],
                    [ret_end_lon,   ret_end_lat],
                ]
                current_retardant = completed_retardant_line
            else:
                atu_event = "none"
                sim_status = "Drop complete — retardant line laid"
                current_retardant = completed_retardant_line

            props["bird_dog"] = {"lat": bd_lat, "lon": bd_lon, "heading_deg": DROP_BEARING, "visible": True}
            props["tanker"]   = {"lat": tk_lat, "lon": tk_lon, "heading_deg": DROP_BEARING, "visible": True}
            props["retardant_line"] = current_retardant
            props["atu_event"]  = atu_event
            props["sim_status"] = sim_status

        # ── Act 4: Post-drop — both aircraft depart south together ──────────────
        else:
            t = (i - ACT4_START) / max(total - ACT4_START - 1, 1)

            # Both planes continue south from the tanker run-end position
            # Exit point: 20 km further south of the run end
            exit_lat, exit_lon = _geo_offset(tk_run_end_lat, tk_run_end_lon, DROP_BEARING, 20.0)

            tk_lat = _lerp(tk_run_end_lat, exit_lat, t)
            tk_lon = _lerp(tk_run_end_lon, exit_lon, t)

            # Bird dog flies in formation: slightly north and east of tanker
            bd_behind_lat, bd_behind_lon = _geo_offset(tk_lat, tk_lon, APPROACH_BEARING, 1.5)
            bd_lat, bd_lon = _geo_offset(bd_behind_lat, bd_behind_lon, LEFT_OF_DROP, 0.8)

            # Both disappear together once they've flown far enough south
            both_visible = t < 0.85

            props["bird_dog"] = {"lat": bd_lat, "lon": bd_lon, "heading_deg": DROP_BEARING, "visible": both_visible}
            props["tanker"]   = {"lat": tk_lat, "lon": tk_lon, "heading_deg": DROP_BEARING, "visible": both_visible}
            props["retardant_line"] = completed_retardant_line
            props["atu_event"]  = "none"
            props["sim_status"] = "Returning to base"


@app.get("/simulate")
async def simulate(
    lat: float = Query(...),
    lon: float = Query(...),
    radius_km: float = Query(2.0),
    steps: int = Query(60),
    seed: int = Query(42),
):
    # ------------------------------------------------------------------
    # 1. Single weather call (all fields, cached 10 min in weatherapi.py)
    # ------------------------------------------------------------------
    weather_data = get_weather(lat, lon)
    terrain_data = compute_slope(lat, lon)
    fuel_data = get_fuel_at(lat, lon)

    wind_from_deg = WIND_DIR_DEGREES.get(weather_data["wind_dir"], 0)

    base_prediction = model_predict(
        lat=lat, lon=lon,
        wind_speed=weather_data["wind_kph"],
        wind_direction=wind_from_deg,
        fuel=fuel_data.fuel_type,
        slope=terrain_data.get("slope_degrees", 0),
        aspect=terrain_data.get("aspect_degrees", 0),
        humidity=weather_data["humidity"],
        temp=weather_data["temp_c"],
        ffmc=fuel_data.ffmc,
        dmc=fuel_data.dmc,
        dc=fuel_data.dc,
        bui=fuel_data.bui,
        isi=fuel_data.isi,
        pcp=weather_data["precip_mm"],
        elev=terrain_data.get("elevation_m", 0),
    )

    fire_direction = base_prediction["fire_direction"]
    fire_severity = base_prediction["fire_severity"]

    spread_bearing_rad = math.radians(fire_direction)
    center_shift_km = weather_data["wind_kph"] * 0.005   # halved — slower drift per frame
    radius_growth_km = 0.07                               # was 0.15 — gradual expansion

    # ------------------------------------------------------------------
    # 2. Sample every 3rd perimeter point for terrain (8 of 24 points).
    #    Terrain varies smoothly across a ~2 km fire; we map un-sampled
    #    points to their nearest sampled neighbour.
    #    Terrain calls: 8 points × 5 altitude lookups = 40 (vs 120).
    #    All results are also TTL-cached for 15 min (_cached_slope).
    # ------------------------------------------------------------------
    NUM_POINTS = 24
    TERRAIN_STRIDE = 3

    initial_coords = generate_fire_perimeter(
        lat, lon, radius_km, NUM_POINTS, seed, spread_direction_deg=fire_direction
    )
    all_perimeter_points = [(c[1], c[0]) for c in initial_coords[:-1]]  # (lat, lon)

    sample_indices = list(range(0, NUM_POINTS, TERRAIN_STRIDE))
    sample_points = [all_perimeter_points[i] for i in sample_indices]

    async with httpx.AsyncClient() as client:
        sample_terrain = await asyncio.gather(
            *[_cached_slope(client, p[0], p[1]) for p in sample_points]
        )

    def nearest_terrain(idx: int) -> dict:
        nearest = min(sample_indices, key=lambda s: abs(s - idx))
        return sample_terrain[sample_indices.index(nearest)]

    perimeter_predictions = [
        model_predict(
            lat=all_perimeter_points[j][0], lon=all_perimeter_points[j][1],
            wind_speed=weather_data["wind_kph"],
            wind_direction=wind_from_deg,
            fuel=fuel_data.fuel_type,
            slope=nearest_terrain(j).get("slope_degrees", 0),
            aspect=nearest_terrain(j).get("aspect_degrees", 0),
            humidity=weather_data["humidity"],
            temp=weather_data["temp_c"],
            ffmc=fuel_data.ffmc,
            dmc=fuel_data.dmc,
            dc=fuel_data.dc,
            bui=fuel_data.bui,
            isi=fuel_data.isi,
            pcp=weather_data["precip_mm"],
            elev=nearest_terrain(j).get("elevation_m", 0),
        )
        for j in range(NUM_POINTS)
    ]

    frames = []
    cur_lat, cur_lon, cur_radius = lat, lon, radius_km

    for i in range(steps):
        cur_lat += (center_shift_km / 111.32) * math.cos(spread_bearing_rad)
        cur_lon += (center_shift_km / (111.32 * math.cos(math.radians(cur_lat)))) * math.sin(spread_bearing_rad)
        cur_radius += radius_growth_km

        coords = generate_fire_perimeter(
            cur_lat, cur_lon, cur_radius, NUM_POINTS, seed + i,
            spread_direction_deg=fire_direction
        )
        step_coords = coords[:-1]
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

    # ------------------------------------------------------------------
    # 3. Overlay aircraft (Bird Dog + Tanker) and ATU events on each frame
    # ------------------------------------------------------------------
    _add_aircraft_to_frames(frames, fire_direction, lat, lon, radius_km)

    return {
        "frames": frames,
        "weather": weather_data,
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
