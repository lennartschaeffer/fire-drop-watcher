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
from helpers.mock_environment import get_mock_environment
from helpers.sentinel_terrain import get_terrain_zones as _get_sentinel_terrain_zones
from helpers.sentinel_shrub import get_shrub_zones as _get_sentinel_shrub_zones
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
    _fire_direction: float,   # unused for aircraft paths — trajectories are deterministic
    origin_lat: float,
    origin_lon: float,
    radius_km: float,
) -> None:
    """
    Mutate each frame's properties in-place to add:
      bird_dog, tanker, deterrent_line, atu_event, sim_status

    All aircraft paths are DETERMINISTIC — fixed bearings independent of fire direction.

      Act 1  0 – 48%   Bird Dog full clockwise recon circle; Tanker parked at NE start
      Act 2  48 – 57%  Both fly from NE start → north approach position (joining up)
      Act 3  57 – 85%  Both fly south over the fire (door_open ~67%, close ~78%)
      Act 4  85 – 100% Both depart south and fade out
    """
    total = len(frames)

    # ── Fixed bearings (always south drop run, north approach) ───────────────
    DROP_BEARING     = 180   # both aircraft fly south during the drop
    APPROACH_BEARING = 0     # approach from north

    # ── Act boundaries (proportional — works for any step count) ─────────────
    ACT2_START  = round(total * 0.30)
    ACT3_START  = round(total * 0.35)
    DOOR_OPEN   = round(total * 0.52)   # was 0.67 — scaled to sit ~64% into compressed act 3
    DOOR_CLOSE  = round(total * 0.59)   # was 0.78 — scaled to sit ~86% into compressed act 3
    ACT4_START  = round(total * 0.65)   # was 0.85 — act 3 compressed from 50% → 30% of frames

    # ── Bird Dog arc: full 360° clockwise, starts NE (higher up, to the right) ─
    BD_ARC_RADIUS    = radius_km + 4.0   # fixed orbit distance from initial fire edge
    BD_ARC_START_BRG = 345.0            # NNW — higher up, slightly east of NW
    BD_ARC_SWEEP     = 360.0            # full clockwise circle
    BD_ARC_END_HDG   = round(BD_ARC_START_BRG + BD_ARC_SWEEP + 90) % 360

    # ── Shared start position: NW of fire — both aircraft begin here ──────────
    start_lat, start_lon = _geo_offset(origin_lat, origin_lon, BD_ARC_START_BRG, BD_ARC_RADIUS)

    # Tanker waits here during Act 1 (same position as bird dog arc start)
    tk_base_lat, tk_base_lon = start_lat, start_lon

    # ── Drop run: all waypoints locked to start_lon so there is zero lateral
    #    movement from Act 2 onwards — planes fly straight south only.
    #    tk_run_start must be SOUTH of start so Act 2 moves southward. ──────────
    tk_run_start_lat, _ = _geo_offset(start_lat, start_lon, DROP_BEARING, 2.0)
    tk_run_start_lon = start_lon   # same longitude as NW start — no lateral drift

    tk_run_end_lat   = _geo_offset(origin_lat, origin_lon, DROP_BEARING, radius_km + 4.0)[0]
    tk_run_end_lon   = start_lon   # same longitude throughout

    # deterrent line: east-west, centred on the drop longitude (start_lon)
    ret_start_lat, ret_start_lon = _geo_offset(origin_lat, start_lon, 270, 1.5)  # west end
    ret_end_lat,   ret_end_lon   = _geo_offset(origin_lat, start_lon,  90, 1.5)  # east end

    completed_deterrent_line: list | None = None

    for i, frame in enumerate(frames):
        props = frame["properties"]

        # ── Act 1: Bird Dog circles the fire; Tanker waits at NE start ────────
        if i < ACT2_START:
            t = i / max(ACT2_START - 1, 1)
            arc_angle = BD_ARC_START_BRG + t * BD_ARC_SWEEP
            arc_brg   = arc_angle % 360
            bd_lat, bd_lon = _geo_offset(origin_lat, origin_lon, arc_brg, BD_ARC_RADIUS)
            bd_hdg = round(arc_angle + 90) % 360

            props["bird_dog"] = {"lat": bd_lat, "lon": bd_lon, "heading_deg": bd_hdg, "visible": True}
            props["tanker"]   = {"lat": tk_base_lat, "lon": tk_base_lon, "heading_deg": DROP_BEARING, "visible": False, "opacity": 0.0}
            props["deterrent_line"] = None
            props["atu_event"]  = "none"
            props["sim_status"] = "Bird Dog performing reconnaissance"

        # ── Act 2: Both fly from NW start → north approach position ───────────
        #    No lateral drift — bird dog just slots behind-left of the tanker.
        elif i < ACT3_START:
            t = (i - ACT2_START) / max(ACT3_START - ACT2_START - 1, 1)

            tk_lat = _lerp(tk_base_lat, tk_run_start_lat, t)
            tk_lon = _lerp(tk_base_lon, tk_run_start_lon, t)

            # Bird dog: behind (north) and to the left (east, when heading south).
            # Offset lerps in from 0 so there's no snap at the act boundary.
            bd_behind_lat, bd_behind_lon = _geo_offset(tk_lat, tk_lon, APPROACH_BEARING, 2.5 * t)
            bd_lat, bd_lon               = _geo_offset(bd_behind_lat, bd_behind_lon, 90, 1.5 * t)
            bd_hdg = round(_lerp(BD_ARC_END_HDG, DROP_BEARING, t)) % 360

            # Ease opacity in quadratically so the tanker fades in smoothly
            tk_opacity = min(1.0, t * t * 3)
            props["bird_dog"] = {"lat": bd_lat, "lon": bd_lon, "heading_deg": bd_hdg, "visible": True}
            props["tanker"]   = {"lat": tk_lat, "lon": tk_lon, "heading_deg": DROP_BEARING, "visible": True, "opacity": round(tk_opacity, 3)}
            props["deterrent_line"] = None
            props["atu_event"]  = "none"
            props["sim_status"] = "Joining up" if t < 0.5 else "On approach"

        # ── Act 3: Both fly south over the fire (drop run) ────────────────────
        elif i < ACT4_START:
            t = (i - ACT3_START) / max(ACT4_START - ACT3_START - 1, 1)

            tk_lat = _lerp(tk_run_start_lat, tk_run_end_lat, t)
            tk_lon = _lerp(tk_run_start_lon, tk_run_end_lon, t)

            # Bird dog stays behind (north) and to the right (east) of the tanker —
            # same offset it reached at the end of the Act 2 transition.
            bd_behind_lat, bd_behind_lon = _geo_offset(tk_lat, tk_lon, APPROACH_BEARING, 2.5)
            bd_lat, bd_lon               = _geo_offset(bd_behind_lat, bd_behind_lon, 90, 1.5)

            if i < DOOR_OPEN:
                atu_event = "none"
                sim_status = "Approaching drop zone"
                current_deterrent = None
            elif i == DOOR_OPEN:
                atu_event = "door_open"
                sim_status = "Drop in progress"
                current_deterrent = [[ret_start_lon, ret_start_lat], [ret_start_lon, ret_start_lat]]
            elif i < DOOR_CLOSE:
                atu_event = "none"
                sim_status = "Drop in progress"
                door_t = (i - DOOR_OPEN) / max(DOOR_CLOSE - DOOR_OPEN, 1)
                partial_end_lat = _lerp(ret_start_lat, ret_end_lat, door_t)
                partial_end_lon = _lerp(ret_start_lon, ret_end_lon, door_t)
                current_deterrent = [
                    [ret_start_lon, ret_start_lat],
                    [partial_end_lon, partial_end_lat],
                ]
            elif i == DOOR_CLOSE:
                atu_event = "door_close"
                sim_status = "Drop complete — deterrent line laid"
                completed_deterrent_line = [
                    [ret_start_lon, ret_start_lat],
                    [ret_end_lon,   ret_end_lat],
                ]
                current_deterrent = completed_deterrent_line
            else:
                atu_event = "none"
                sim_status = "Drop complete — deterrent line laid"
                current_deterrent = completed_deterrent_line

            props["bird_dog"] = {"lat": bd_lat, "lon": bd_lon, "heading_deg": DROP_BEARING, "visible": True}
            props["tanker"]   = {"lat": tk_lat, "lon": tk_lon, "heading_deg": DROP_BEARING, "visible": True, "opacity": 1.0}
            props["deterrent_line"] = current_deterrent
            props["atu_event"]  = atu_event
            props["sim_status"] = sim_status

        # ── Act 4: Both depart south and fade out ─────────────────────────────
        else:
            t = (i - ACT4_START) / max(total - ACT4_START - 1, 1)

            exit_lat, exit_lon = _geo_offset(tk_run_end_lat, tk_run_end_lon, DROP_BEARING, 20.0)
            tk_lat = _lerp(tk_run_end_lat, exit_lat, t)
            tk_lon = _lerp(tk_run_end_lon, exit_lon, t)

            # Bird dog stays behind (north) and to the right (east) of tanker through departure —
            # same relative offset maintained from the drop run.
            bd_behind_lat, bd_behind_lon = _geo_offset(tk_lat, tk_lon, APPROACH_BEARING, 2.5)
            bd_lat, bd_lon               = _geo_offset(bd_behind_lat, bd_behind_lon, 90, 1.5)

            both_visible = t < 0.85

            props["bird_dog"] = {"lat": bd_lat, "lon": bd_lon, "heading_deg": DROP_BEARING, "visible": both_visible}
            props["tanker"]   = {"lat": tk_lat, "lon": tk_lon, "heading_deg": DROP_BEARING, "visible": both_visible, "opacity": 1.0}
            props["deterrent_line"] = completed_deterrent_line
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


@app.get("/mock-environment")
def mock_environment(
    fire: str = Query("barrington", description="fire_id: 'barrington' or 'tantallon'"),
):
    """
    Return mock environment overlay data for the given fire:
      - shrub_zones: list of shrub heatmap blobs
      - terrain_zones: list of favorable-terrain polygon rings
    """
    return get_mock_environment(fire)


@app.get("/terrain-zones")
async def terrain_zones(
    lat: float = Query(..., description="Fire centre latitude"),
    lon: float = Query(..., description="Fire centre longitude"),
    radius_km: float = Query(25.0, description="Half-width of the DEM tile to analyse"),
    slope_threshold: float = Query(1.0, description="Max slope (degrees) for 'gentle terrain'"),
    max_zones: int = Query(3, ge=1, le=6),
):
    """
    Return real gentle-slope terrain zones derived from the Copernicus DEM GLO-30
    via the Sentinel Hub Process API.

    Requires SENTINELHUB_CLIENT_ID + SENTINELHUB_CLIENT_SECRET in the environment.
    Falls back to mock terrain zones if those credentials are absent.

    Response shape mirrors /mock-environment's terrain_zones list so the frontend
    can swap them in without any interface changes.
    """
    try:
        zones = _get_sentinel_terrain_zones(
            center_lat=lat,
            center_lon=lon,
            radius_km=radius_km,
            slope_threshold_deg=slope_threshold,
            max_zones=max_zones,
        )
        return {"terrain_zones": zones, "source": "sentinel-hub"}
    except EnvironmentError:
        # No Sentinel Hub credentials — return mock zones so the UI still works
        env_data = get_mock_environment("barrington")
        return {"terrain_zones": env_data["terrain_zones"], "source": "mock-fallback"}


@app.get("/shrub-zones")
async def shrub_zones(
    lat: float = Query(..., description="Fire centre latitude"),
    lon: float = Query(..., description="Fire centre longitude"),
    radius_km: float = Query(25.0, description="Half-width of the WorldCover tile to analyse"),
    max_zones: int = Query(3, ge=1, le=6),
):
    """
    Return real shrubland zones derived from ESA WorldCover 2020 (class 20)
    via the Sentinel Hub Process API.

    Requires SENTINELHUB_CLIENT_ID + SENTINELHUB_CLIENT_SECRET in the environment.
    Falls back to mock shrub zones if those credentials are absent.

    Response shape mirrors /mock-environment's shrub_zones list so the frontend
    can swap them in without any interface changes.
    """
    try:
        zones = _get_sentinel_shrub_zones(
            center_lat=lat,
            center_lon=lon,
            radius_km=radius_km,
            max_zones=max_zones,
        )
        return {"shrub_zones": zones, "source": "sentinel-hub"}
    except EnvironmentError:
        # No Sentinel Hub credentials — return mock zones so the UI still works
        env_data = get_mock_environment("barrington")
        return {"shrub_zones": env_data["shrub_zones"], "source": "mock-fallback"}


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
