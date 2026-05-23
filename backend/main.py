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
from helpers.cwfis_fire_data import load_fire, build_growth_frames, list_fires, _ring_centroid
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
    fire_direction: float,    # bearing the fire is spreading toward (degrees)
    origin_lat: float,
    origin_lon: float,
    radius_km: float,
) -> None:
    """
    Mutate each frame's properties in-place to add:
      bird_dog, tanker, retardant_line, atu_event, sim_status

    Tanker runs along the LEFT FLANK of the fire, hugging the perimeter edge.
    It flies PARALLEL to the fire's spread direction (not across it), positioned
    at the left side of the fire so it lays retardant along the flank.
    Drop zone is anchored to the ACTUAL fire state at drop time.

      Act 1  0 – 48%   Bird Dog clockwise recon arc around the fire
      Act 2  48 – 57%  Tanker en route from right-flank staging; Bird Dog repositions
      Act 3  57 – 85%  Drop run hugging the left flank (door_open ~67%, close ~78%)
      Act 4  85 – 100% Both aircraft depart ahead of fire
    """
    total = len(frames)

    # ── Directions relative to fire spread ────────────────────────────────────
    RUN_BEARING  = fire_direction                        # tanker flies same direction as fire spreads
    RUN_REVERSE  = (fire_direction + 180) % 360          # behind (tanker enters from here)
    # LEFT/RIGHT here are map-visual: +90 from fire_direction = left side on a north-up map
    LEFT_FLANK   = (fire_direction + 90) % 360           # visually left on map — drop zone goes here
    RIGHT_FLANK  = (fire_direction - 90 + 360) % 360    # visually right on map — tanker stages here

    # ── Act boundaries (proportional — works for any step count) ─────────────
    ACT2_START  = round(total * 0.48)
    ACT3_START  = round(total * 0.57)
    DOOR_OPEN   = round(total * 0.67)
    DOOR_CLOSE  = round(total * 0.78)
    ACT4_START  = round(total * 0.85)

    # ── Sample actual fire state at key moments ───────────────────────────────
    # The fire grows and drifts, so position everything using the actual frame
    # data at the relevant time rather than the initial origin/radius.
    def _fire_state(idx: int) -> tuple[float, float, float]:
        p = frames[min(idx, total - 1)]["properties"]
        return (
            p.get("center_lat", origin_lat),
            p.get("center_lon", origin_lon),
            p.get("radius_km", radius_km),
        )

    act1_lat, act1_lon, act1_r = _fire_state(ACT2_START // 2)
    BD_ARC_RADIUS = act1_r + 4.0   # orbit just outside the perimeter

    tk2_lat, tk2_lon, tk2_r = _fire_state(ACT2_START)
    dz_lat, dz_lon, dz_r    = _fire_state(DOOR_OPEN)   # fire edge at drop time

    # ── Bird Dog arc params ────────────────────────────────────────────────────
    BD_ARC_START_OFF = 135.0
    BD_ARC_SWEEP     = 270.0
    BD_ARC_START_BRG = (fire_direction + BD_ARC_START_OFF) % 360
    BD_ARC_END_BRG   = (BD_ARC_START_BRG + BD_ARC_SWEEP) % 360
    BD_ARC_END_HDG   = round(BD_ARC_START_BRG + BD_ARC_SWEEP + 90) % 360

    # ── Key waypoints ─────────────────────────────────────────────────────────
    bd_recon_end_lat, bd_recon_end_lon = _geo_offset(
        act1_lat, act1_lon, BD_ARC_END_BRG, BD_ARC_RADIUS
    )

    # Tanker stages on the RIGHT flank (opposite side from the drop)
    tk_base_lat, tk_base_lon = _geo_offset(tk2_lat, tk2_lon, RIGHT_FLANK, tk2_r + 6.0)

    # Drop zone: ON the left flank, right at the fire edge — tanker hugs it
    drop_zone_lat, drop_zone_lon = _geo_offset(dz_lat, dz_lon, LEFT_FLANK, dz_r * 1.0)

    # Tanker run: enters from BEHIND (6 km back), exits AHEAD (4 km forward),
    # flying parallel to fire spread direction along the left flank.
    tk_run_start_lat, tk_run_start_lon = _geo_offset(drop_zone_lat, drop_zone_lon, RUN_REVERSE, 6.0)
    tk_run_end_lat,   tk_run_end_lon   = _geo_offset(drop_zone_lat, drop_zone_lon, RUN_BEARING, 4.0)

    # Retardant line along the tanker path (parallel to fire spread, on the left flank)
    ret_start_lat, ret_start_lon = _geo_offset(drop_zone_lat, drop_zone_lon, RUN_REVERSE, 1.5)
    ret_end_lat,   ret_end_lon   = _geo_offset(drop_zone_lat, drop_zone_lon, RUN_BEARING, 1.5)

    # Bird Dog escort start: behind the tanker, slightly toward fire (right side)
    _bd_behind_lat, _bd_behind_lon = _geo_offset(
        tk_run_start_lat, tk_run_start_lon, RUN_REVERSE, 1.5
    )
    bd_escort_start_lat, bd_escort_start_lon = _geo_offset(
        _bd_behind_lat, _bd_behind_lon, RIGHT_FLANK, 0.8
    )

    completed_retardant_line: list | None = None

    for i, frame in enumerate(frames):
        props = frame["properties"]

        # ── Act 1: Bird Dog clockwise recon arc, tanker parked on right flank ──
        if i < ACT2_START:
            t = i / max(ACT2_START - 1, 1)
            arc_angle = BD_ARC_START_BRG + _lerp(0, BD_ARC_SWEEP, t)
            arc_brg   = arc_angle % 360
            bd_lat, bd_lon = _geo_offset(act1_lat, act1_lon, arc_brg, BD_ARC_RADIUS)
            bd_hdg = round(arc_angle + 90) % 360

            props["bird_dog"] = {"lat": bd_lat, "lon": bd_lon, "heading_deg": bd_hdg, "visible": True}
            props["tanker"]   = {"lat": tk_base_lat, "lon": tk_base_lon, "heading_deg": RUN_BEARING, "visible": False}
            props["retardant_line"] = None
            props["atu_event"]  = "none"
            props["sim_status"] = "Bird Dog performing reconnaissance"

        # ── Act 2: Tanker en route right flank → left flank run-start ────────
        elif i < ACT3_START:
            t = (i - ACT2_START) / max(ACT3_START - ACT2_START - 1, 1)

            tk_lat = _lerp(tk_base_lat,       tk_run_start_lat, t)
            tk_lon = _lerp(tk_base_lon,       tk_run_start_lon, t)

            bd_lat = _lerp(bd_recon_end_lat, bd_escort_start_lat, t)
            bd_lon = _lerp(bd_recon_end_lon, bd_escort_start_lon, t)
            bd_hdg = round(_lerp(BD_ARC_END_HDG, RUN_BEARING, t)) % 360

            props["bird_dog"] = {"lat": bd_lat, "lon": bd_lon, "heading_deg": bd_hdg, "visible": True}
            props["tanker"]   = {"lat": tk_lat, "lon": tk_lon, "heading_deg": RUN_BEARING, "visible": True}
            props["retardant_line"] = None
            props["atu_event"]  = "none"
            props["sim_status"] = "Tanker en route" if t < 0.5 else "Tanker on approach"

        # ── Act 3: Drop run — hugging the left flank ──────────────────────────
        elif i < ACT4_START:
            t = (i - ACT3_START) / max(ACT4_START - ACT3_START - 1, 1)

            tk_lat = _lerp(tk_run_start_lat, tk_run_end_lat, t)
            tk_lon = _lerp(tk_run_start_lon, tk_run_end_lon, t)

            # Bird dog: slightly AHEAD and toward the fire (right of tanker)
            bd_fwd_lat, bd_fwd_lon = _geo_offset(tk_lat, tk_lon, RUN_BEARING, 1.0)
            bd_lat, bd_lon = _geo_offset(bd_fwd_lat, bd_fwd_lon, RIGHT_FLANK, 0.6)

            if i < DOOR_OPEN:
                atu_event = "none"
                sim_status = "Approaching left flank"
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

            props["bird_dog"] = {"lat": bd_lat, "lon": bd_lon, "heading_deg": RUN_BEARING, "visible": True}
            props["tanker"]   = {"lat": tk_lat, "lon": tk_lon, "heading_deg": RUN_BEARING, "visible": True}
            props["retardant_line"] = current_retardant
            props["atu_event"]  = atu_event
            props["sim_status"] = sim_status

        # ── Act 4: Both aircraft depart ahead of fire ─────────────────────────
        else:
            t = (i - ACT4_START) / max(total - ACT4_START - 1, 1)

            exit_lat, exit_lon = _geo_offset(tk_run_end_lat, tk_run_end_lon, RUN_BEARING, 20.0)
            tk_lat = _lerp(tk_run_end_lat, exit_lat, t)
            tk_lon = _lerp(tk_run_end_lon, exit_lon, t)

            bd_fwd_lat, bd_fwd_lon = _geo_offset(tk_lat, tk_lon, RUN_BEARING, 1.0)
            bd_lat, bd_lon = _geo_offset(bd_fwd_lat, bd_fwd_lon, RIGHT_FLANK, 0.6)

            both_visible = t < 0.85

            props["bird_dog"] = {"lat": bd_lat, "lon": bd_lon, "heading_deg": RUN_BEARING, "visible": both_visible}
            props["tanker"]   = {"lat": tk_lat, "lon": tk_lon, "heading_deg": RUN_BEARING, "visible": both_visible}
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

# ===========================================================================
# Real satellite-derived fire endpoints
# Data: CWFIS NBAC (National Burned Area Composite) — VIIRS/MODIS IR detections
# ===========================================================================

@app.get("/fires")
def fires_catalogue():
    """List available real historical fires with metadata."""
    return {"fires": list_fires()}


@app.get("/simulate-real")
async def simulate_real(
    fire: str = Query("barrington", description="fire_id: 'barrington' or 'tantallon'"),
    steps: int = Query(60, ge=10, le=120),
):
    """
    Simulate fire spread from a real CWFIS satellite-derived perimeter.

    The fire grows from ~8% of its final documented size (frame 0) to the
    exact boundary recorded by satellite infrared bands (final frame),
    using a sqrt-eased growth curve to match the fast-ignition / slow-
    stabilisation pattern of the real fires.

    Weather and spread-model predictions use live data at the fire's centre.
    Aircraft overlay is identical to /simulate.
    """
    # ── 1. Load real perimeter data ─────────────────────────────────────────
    try:
        fire_record = load_fire(fire)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    lat = fire_record["center_lat"]
    lon = fire_record["center_lon"]
    final_ring = fire_record["perimeter_ring"]   # real satellite boundary
    radius_km  = fire_record["radius_km"]

    # ── 2. Weather / terrain / fuel at fire centre ──────────────────────────
    weather_data = get_weather(lat, lon)
    terrain_data = compute_slope(lat, lon)
    fuel_data    = get_fuel_at(lat, lon)

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
    fire_severity  = base_prediction["fire_severity"]

    # ── 3. Build frame rings: grow from 8% → real final boundary ────────────
    growth_rings = build_growth_frames(fire_record, steps=steps)

    # ── 4. Terrain sample on perimeter of the final ring (cached) ───────────
    NUM_POINTS    = len(final_ring) - 1   # exclude closing duplicate
    TERRAIN_STRIDE = 4
    sample_indices = list(range(0, NUM_POINTS, TERRAIN_STRIDE))

    final_pts = [(final_ring[i][1], final_ring[i][0]) for i in range(NUM_POINTS)]
    sample_pts = [final_pts[i] for i in sample_indices]

    async with httpx.AsyncClient() as client:
        sample_terrain = await asyncio.gather(
            *[_cached_slope(client, p[0], p[1]) for p in sample_pts]
        )

    def nearest_terrain(idx: int) -> dict:
        nearest = min(sample_indices, key=lambda s: abs(s - idx))
        return sample_terrain[sample_indices.index(nearest)]

    perimeter_predictions = [
        model_predict(
            lat=final_pts[j][0], lon=final_pts[j][1],
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

    # ── 5. Assemble frames ───────────────────────────────────────────────────
    frames = []
    center = _ring_centroid(final_ring)

    for i, ring in enumerate(growth_rings):
        step_coords = ring[:-1]   # exclude closing point for arrows
        arrows = [
            {
                "lon":       step_coords[j % len(step_coords)][0],
                "lat":       step_coords[j % len(step_coords)][1],
                "direction": perimeter_predictions[j % NUM_POINTS]["fire_direction"],
                "severity":  perimeter_predictions[j % NUM_POINTS]["fire_severity"],
            }
            for j in range(NUM_POINTS)
        ]

        # Scale radius linearly with growth factor so panel stats look sensible
        t     = i / max(steps - 1, 1)
        scale = 0.08 + 0.92 * (t ** 0.5)
        frames.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "center_lat":      round(center[1], 6),
                "center_lon":      round(center[0], 6),
                "radius_km":       round(radius_km * scale, 3),
                "step":            i,
                "fire_direction":  fire_direction,
                "fire_severity":   fire_severity,
                "perimeter_arrows": arrows,
                "source":          "CWFIS NBAC — satellite IR perimeter",
                "fire_id":         fire,
                "fire_label":      fire_record.get("label", fire),
            },
        })

    # ── 6. Overlay aircraft (same logic as /simulate) ───────────────────────
    _add_aircraft_to_frames(frames, fire_direction, lat, lon, radius_km)

    return {
        "frames":     frames,
        "weather":    weather_data,
        "terrain":    terrain_data,
        "fuel": {
            "fuel_type":        fuel_data.fuel_type,
            "fuel_description": fuel_data.fuel_description,
            "flammability":     fuel_data.flammability,
        },
        "prediction": base_prediction,
        "steps":      steps,
        "fire_meta": {
            "fire_id":    fire,
            "label":      fire_record.get("label", fire),
            "poly_ha":    fire_record.get("poly_ha"),
            "start_date": fire_record.get("start_date"),
            "end_date":   fire_record.get("end_date"),
            "source":     fire_record.get("source"),
        },
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
