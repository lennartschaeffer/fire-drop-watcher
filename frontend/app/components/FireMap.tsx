"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Map, { MapRef } from "react-map-gl/maplibre";
import DeckGL from "@deck.gl/react";
import {
  PolygonLayer,
  IconLayer,
  PathLayer,
  ScatterplotLayer,
} from "@deck.gl/layers";
import { HeatmapLayer } from "@deck.gl/aggregation-layers";
import "maplibre-gl/dist/maplibre-gl.css";

const INITIAL_VIEW = {
  longitude: -65.469716,
  latitude: 43.639699,
  zoom: 10,
  pitch: 0,
  bearing: 0,
};

const ESRI_SATELLITE =
  "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";

const MAP_STYLE = {
  version: 8 as const,
  sources: {
    esri: {
      type: "raster" as const,
      tiles: [ESRI_SATELLITE],
      tileSize: 256,
      attribution: "ESRI World Imagery",
    },
  },
  layers: [{ id: "esri-satellite", type: "raster" as const, source: "esri" }],
};

const STEPS = 60;
const FIRE_INTERVAL_MS = 1500; // fire perimeter advances every 1.5 s (slow, realistic)
const PLANE_TICK_MS = 80; // plane interpolation tick — ~12 fps smooth movement

// ─── Types ────────────────────────────────────────────────────────────────────

interface WeatherData {
  wind_kph: number;
  wind_dir: string;
  temp_c: number;
  humidity: number;
  precip_mm: number;
}

interface TerrainData {
  slope_degrees: number;
  aspect_degrees: number;
  elevation_m?: number;
  error?: string;
}

interface FuelData {
  fuel_type: string;
  fuel_description: string;
  flammability: string;
}

interface ShrubPoint {
  lat: number;
  lon: number;
}

interface ShrubZone {
  center: { lat: number; lon: number };
  points: ShrubPoint[];
  label: string;
  halt_rate: string;
  description: string;
}

interface TerrainZone {
  ring: [number, number][]; // [[lon, lat], ...] closed ring
  center: { lat: number; lon: number };
  label: string;
  description: string;
  halt_rate_modifier: string;
}

interface AircraftData {
  lat: number;
  lon: number;
  heading_deg: number;
  visible: boolean;
  opacity?: number; // 0–1 fade-in (used by tanker during Act 2)
}

interface FireFeature {
  type: string;
  geometry: { type: string; coordinates: number[][][] };
  properties: {
    center_lat: number;
    center_lon: number;
    radius_km: number;
    step: number;
    fire_direction: number;
    fire_severity: number;
    perimeter_arrows: {
      lon: number;
      lat: number;
      direction: number;
      severity: number;
    }[];
    bird_dog?: AircraftData;
    tanker?: AircraftData;
    retardant_line?: [number, number][] | null;
    atu_event?: "none" | "door_open" | "door_close";
    sim_status?: string;
  };
}

interface FireMeta {
  fire_id: string;
  label: string;
  poly_ha: number;
  start_date: string;
  end_date: string;
  source: string;
}

interface SimulateResponse {
  frames: FireFeature[];
  weather: WeatherData;
  terrain: TerrainData;
  fuel: FuelData;
  prediction: { fire_direction: number; fire_severity: number };
  steps: number;
  fire_meta?: FireMeta; // only present on /simulate-real responses
}

const API = "http://localhost:8000";

type SimStatus = "idle" | "loading" | "playing" | "done";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function severityColor(s: number): [number, number, number, number] {
  return [255, Math.round(180 * (1 - s)), 0, Math.round(40 + 80 * s)];
}

function severityLineColor(s: number): [number, number, number, number] {
  return [255, Math.round(160 * (1 - s)), 0, 220];
}

// Plane SVGs — both point north (up) at 0°; DeckGL rotates by heading.
// getAngle in DeckGL IconLayer: 0 = up, clockwise positive.
// We pass -(heading_deg) because DeckGL rotates counter-clockwise from north.

const BIRD_DOG_SVG = encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="64" height="64">
  <!-- fuselage -->
  <ellipse cx="16" cy="16" rx="2.5" ry="11" fill="white"/>
  <!-- main wings -->
  <ellipse cx="16" cy="19" rx="13" ry="2.2" fill="white"/>
  <!-- tail fin -->
  <ellipse cx="16" cy="8"  rx="5"  ry="1.4" fill="white"/>
  <!-- nose -->
  <ellipse cx="16" cy="27" rx="2"  ry="1.5" fill="#d1d5db"/>
</svg>
`);

const TANKER_SVG = encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="64" height="64">
  <!-- fuselage (thicker, boxy belly) -->
  <rect x="13" y="6" width="6" height="20" rx="2.5" fill="#fbbf24"/>
  <!-- wide wings -->
  <rect x="2" y="18" width="28" height="3.5" rx="1.5" fill="#fbbf24"/>
  <!-- tail fins -->
  <rect x="12" y="6" width="8" height="4" rx="1" fill="#f59e0b"/>
  <!-- retardant belly tank -->
  <rect x="13.5" y="16" width="5" height="7" rx="1" fill="#dc2626"/>
</svg>
`);

function statusIcon(status: string): string {
  if (status.includes("reconnaissance")) return "🛩";
  if (status.includes("en route")) return "✈️";
  if (status.includes("on approach")) return "✈️";
  if (status.includes("observing")) return "👁";
  if (status.includes("in progress")) return "🔴";
  if (status.includes("complete")) return "✅";
  if (status.includes("returning")) return "🏠";
  return "📡";
}

/** Circular mean of an array of degree angles (handles 0°/360° wraparound). */
function circularMean(angles: number[]): number {
  if (angles.length === 0) return 0;
  const sinSum = angles.reduce((s, a) => s + Math.sin((a * Math.PI) / 180), 0);
  const cosSum = angles.reduce((s, a) => s + Math.cos((a * Math.PI) / 180), 0);
  return ((Math.atan2(sinSum, cosSum) * 180) / Math.PI + 360) % 360;
}

/** Scale a GeoJSON ring inward/outward around a center point. */
function scalePolygon(
  coords: number[][],
  center: [number, number],
  scale: number,
): number[][] {
  return coords.map(([lon, lat]) => [
    center[0] + (lon - center[0]) * scale,
    center[1] + (lat - center[1]) * scale,
  ]);
}

/** Deterministic pseudo-random in [0, 1) from an integer seed. */
function seededRand(n: number): number {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

/** Ray-casting point-in-polygon for a GeoJSON ring (array of [lon, lat] pairs). */
function pointInPolygon(lon: number, lat: number, ring: number[][]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    if (
      yi > lat !== yj > lat &&
      lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi
    ) {
      inside = !inside;
    }
  }
  return inside;
}

// ─── Header sub-component ─────────────────────────────────────────────────────

interface HeaderProps {
  status: SimStatus;
  simStatus: string;
  currentFrame: number;
  totalFrames: number;
  weather: WeatherData | null;
  severity: number;
  dataSource: "mock" | "satellite";
  fireMeta: FireMeta | null;
}

function CommandHeader({
  status,
  simStatus,
  currentFrame,
  totalFrames,
  weather,
  severity,
  dataSource,
  fireMeta,
}: HeaderProps) {
  const progress =
    totalFrames > 0 ? ((currentFrame + 1) / totalFrames) * 100 : 0;

  // Severity-based accent color for the live indicator
  const severityHue = Math.round(30 - severity * 30); // 30 (orange) → 0 (red)
  const severityRgb = `hsl(${severityHue}, 100%, 55%)`;

  return (
    <header className="shrink-0 flex flex-col bg-[#07080a] border-b border-zinc-800/80">
      {/* Main row — three equal columns: brand | status | data */}
      <div className="grid grid-cols-3 items-center h-20 px-0">
        {/* ── Col 1: Brand ── */}
        <div className="flex items-center gap-0">
          {/* Flame accent bar */}
          <div
            className="w-1.5 self-stretch shrink-0"
            style={{
              background:
                status === "playing"
                  ? `linear-gradient(to bottom, #ff6a00, #ff2200)`
                  : "linear-gradient(to bottom, #ff6a0044, #ff220033)",
              transition: "background 0.6s ease",
            }}
          />
          <div className="flex items-center gap-4 pl-5">
            <span
              className="text-[48px] tracking-[0.1em] leading-none select-none"
              style={{
                fontFamily: "var(--font-bebas)",
                color: "#ff4500",
                textShadow:
                  status === "playing" ? "0 0 28px #ff450077" : "none",
                transition: "text-shadow 0.6s ease",
              }}
            >
              HEATVISION
            </span>
            <div className="flex flex-col gap-1.5 pt-0.5"></div>
          </div>
        </div>

        {/* ── Col 2: Status (centered) ── */}
        <div className="flex items-center justify-center">
          {simStatus && (status === "playing" || status === "done") ? (
            <div
              className="flex items-center gap-3 px-6 py-2.5 rounded-lg"
              style={{
                background: `${severityRgb}12`,
                border: `1px solid ${severityRgb}40`,
              }}
            >
              <span
                className="w-3 h-3 rounded-full shrink-0"
                style={{
                  backgroundColor: severityRgb,
                  boxShadow: `0 0 10px ${severityRgb}`,
                  animation:
                    status === "playing"
                      ? "pulse 1.4s ease-in-out infinite"
                      : "none",
                }}
              />
              <span
                className="text-base font-semibold tracking-widest"
                style={{
                  fontFamily: "var(--font-geist-mono)",
                  color: statusTextColor(simStatus),
                }}
              >
                {statusIcon(simStatus)}&nbsp;{simStatus.toUpperCase()}
              </span>
            </div>
          ) : status === "loading" ? (
            <div className="flex items-center gap-3 px-6 py-2.5 rounded-lg bg-amber-950/30 border border-amber-800/40">
              <span
                className="w-3 h-3 rounded-full bg-amber-400 shrink-0"
                style={{ animation: "pulse 0.8s ease-in-out infinite" }}
              />
              <span className="text-base font-mono tracking-widest text-amber-400">
                LOADING DATA…
              </span>
            </div>
          ) : (
            <span className="text-base font-mono tracking-[0.2em] text-zinc-700 uppercase">
              — STANDBY —
            </span>
          )}
        </div>

        {/* ── Col 3: Weather + Frame counter (right-aligned) ── */}
        <div className="flex items-center justify-end gap-0 pr-0">
          {/* Weather readout */}
          {weather && status !== "idle" && (
            <>
              <div className="flex items-center gap-5 px-5">
                <WeatherChip icon="🌡" value={`${weather.temp_c}°C`} />
                <WeatherChip icon="💧" value={`${weather.humidity}%`} />
                <WeatherChip
                  icon="💨"
                  value={`${weather.wind_kph} kph ${weather.wind_dir}`}
                />
              </div>
              <div className="w-px self-stretch bg-zinc-800 shrink-0" />
            </>
          )}
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-[2px] bg-zinc-900 w-full">
        <div
          className="h-full"
          style={{
            width: `${progress}%`,
            background:
              progress > 0
                ? `linear-gradient(to right, #ff4500, ${severityRgb})`
                : "transparent",
            transition: "width 0.4s ease, background 0.6s ease",
            boxShadow: progress > 0 ? `0 0 6px ${severityRgb}88` : "none",
          }}
        />
      </div>
    </header>
  );
}

function WeatherChip({ icon, value }: { icon: string; value: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-base leading-none">{icon}</span>
      <span
        className="text-sm text-zinc-400 tracking-wide"
        style={{ fontFamily: "var(--font-geist-mono)" }}
      >
        {value}
      </span>
    </div>
  );
}

/** Colour for the status text in the header (lighter, text-only palette) */
function statusTextColor(status: string): string {
  if (status.includes("reconnaissance")) return "#7dd3fc"; // sky-300
  if (status.includes("en route")) return "#fde68a"; // amber-200
  if (status.includes("on approach")) return "#fcd34d"; // amber-300
  if (status.includes("observing")) return "#fdba74"; // orange-300
  if (status.includes("in progress")) return "#fca5a5"; // red-300
  if (status.includes("complete")) return "#86efac"; // green-300
  if (status.includes("returning")) return "#d4d4d8"; // zinc-300
  return "#a1a1aa";
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function FireMap() {
  const [frames, setFrames] = useState<FireFeature[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [weather, setWeather] = useState<WeatherData | null>(null);
  const [terrain, setTerrain] = useState<TerrainData | null>(null);
  const [fuel, setFuel] = useState<FuelData | null>(null);

  const [status, setStatus] = useState<SimStatus>("idle");
  // 0..1 interpolation factor between currentFrame and currentFrame+1 for planes
  const [planeT, setPlaneT] = useState(0);

  // Real satellite-derived fire mode
  const [selectedFire, setSelectedFire] = useState<string>("barrington");
  const [fireMeta, setFireMeta] = useState<FireMeta | null>(null);
  const [dataSource, setDataSource] = useState<"mock" | "satellite">("mock");

  // Shrub zone overlays
  const [shrubZones, setShrubZones] = useState<ShrubZone[]>([]);
  const [showShrubs, setShowShrubs] = useState(true);

  // Terrain zone overlays
  const [terrainZones, setTerrainZones] = useState<TerrainZone[]>([]);
  const [showFavorableTerrain, setShowFavorableTerrain] = useState(true);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const planeIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const frameCountRef = useRef(0);

  // Draw mode
  const [drawMode, setDrawMode] = useState(false);
  const [dropPoints, setDropPoints] = useState<[number, number][]>([]);
  const [briefingLoading, setBriefingLoading] = useState(false);
  const [messages, setMessages] = useState<
    { id: number; text: string; ts: string }[]
  >([]);
  const [showComms, setShowComms] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const deckRef = useRef<any>(null);
  const mapRef = useRef<MapRef>(null);

  const lat = INITIAL_VIEW.latitude;
  const lon = INITIAL_VIEW.longitude;

  function handleSimulate() {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (planeIntervalRef.current) clearInterval(planeIntervalRef.current);
    setFrames([]);
    setCurrentFrame(0);
    setPlaneT(0);
    setStatus("loading");
    setDataSource("mock");
    setFireMeta(null);

    fetch(
      `${API}/simulate?lat=${lat}&lon=${lon}&radius_km=2&steps=${STEPS}&seed=42`,
    )
      .then((r) => r.json())
      .then((data: SimulateResponse) => startPlayback(data))
      .catch(() => setStatus("idle"));
  }

  // ── Real satellite-derived fire simulation ─────────────────────────────────

  function startPlayback(data: SimulateResponse) {
    setWeather(data.weather);
    setTerrain(data.terrain);
    setFuel(data.fuel);
    setFireMeta(data.fire_meta ?? null);
    setFrames(data.frames);
    setStatus("playing");

    frameCountRef.current = 0;
    intervalRef.current = setInterval(() => {
      frameCountRef.current += 1;
      if (frameCountRef.current >= data.frames.length) {
        clearInterval(intervalRef.current!);
        clearInterval(planeIntervalRef.current!);
        setStatus("done");
      } else {
        const f = frameCountRef.current;
        setCurrentFrame(f);
        setPlaneT(0);
      }
    }, FIRE_INTERVAL_MS);

    planeIntervalRef.current = setInterval(() => {
      setPlaneT((t) => Math.min(t + PLANE_TICK_MS / FIRE_INTERVAL_MS, 1));
    }, PLANE_TICK_MS);
  }

  function handleSimulateReal() {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (planeIntervalRef.current) clearInterval(planeIntervalRef.current);
    setFrames([]);
    setCurrentFrame(0);
    setPlaneT(0);
    setStatus("loading");
    setDataSource("satellite");
    setShrubZones([]);
    setTerrainZones([]);

    // Kick off all three fetches in parallel:
    //   1. Real satellite fire perimeter frames
    //   2. Real shrub zones from Sentinel Hub WorldCover (falls back to mock if no API key)
    //   3. Real terrain zones from Sentinel Hub DEM (falls back to mock if no API key)
    Promise.all([
      fetch(`${API}/simulate-real?fire=${selectedFire}&steps=${STEPS}`).then(
        (r) => r.json(),
      ),
      fetch(
        `${API}/shrub-zones?lat=${INITIAL_VIEW.latitude}&lon=${INITIAL_VIEW.longitude}`,
      ).then((r) => r.json()),
      fetch(
        `${API}/terrain-zones?lat=${INITIAL_VIEW.latitude}&lon=${INITIAL_VIEW.longitude}`,
      ).then((r) => r.json()),
    ])
      .then(
        ([simData, shrubData, terrainData]: [
          SimulateResponse,
          { shrub_zones: ShrubZone[]; source: string },
          { terrain_zones: TerrainZone[]; source: string },
        ]) => {
          setShrubZones(shrubData.shrub_zones ?? []);
          setTerrainZones(terrainData.terrain_zones ?? []);
          startPlayback(simData);
        },
      )
      .catch(() => setStatus("idle"));
  }

  useEffect(
    () => () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (planeIntervalRef.current) clearInterval(planeIntervalRef.current);
    },
    [],
  );

  // ── Draw mode ──────────────────────────────────────────────────────────────

  // Auto-scroll chat to bottom whenever a new message arrives
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function toggleDrawMode() {
    setDrawMode((d) => !d);
    setDropPoints([]);
  }

  function handleDeckClick(info: { coordinate?: number[] }) {
    if (!drawMode || !info.coordinate) return;
    const pt = info.coordinate as [number, number];
    setDropPoints((prev) => (prev.length >= 2 ? [pt] : [...prev, pt]));
  }

  async function handleGetBriefing() {
    if (dropPoints.length < 2) return;
    setBriefingLoading(true);
    setShowComms(true);

    // Capture the map + drop zone into a single PNG.
    // WebGL canvases clear their buffer after each frame (preserveDrawingBuffer=false),
    // so we must capture INSIDE a render event — not after the fact.
    const captureImage = (): Promise<string> =>
      new Promise((resolve) => {
        const map = mapRef.current?.getMap();
        if (!map) {
          resolve("");
          return;
        }

        const doCapture = () => {
          const mapCanvas = map.getCanvas();
          const dpr = window.devicePixelRatio || 1;
          const w = mapCanvas.width; // already in device pixels
          const h = mapCanvas.height;

          const composite = document.createElement("canvas");
          composite.width = w;
          composite.height = h;
          const ctx = composite.getContext("2d")!;

          // 1. Satellite tiles (captured synchronously during render — buffer is live)
          ctx.drawImage(mapCanvas, 0, 0);

          // 2. DeckGL overlay (may be populated; best-effort)
          const deckCanvas = deckRef.current?.deck
            ?.canvas as HTMLCanvasElement | null;
          if (deckCanvas) {
            try {
              ctx.drawImage(deckCanvas, 0, 0);
            } catch (_) {
              /* tainted? skip */
            }
          }

          // 3. Draw drop zone line ourselves using coordinate projection
          //    — guaranteed to appear regardless of DeckGL buffer state
          if (dropPoints.length === 2) {
            const [p1, p2] = dropPoints;
            // map.project() returns CSS pixels; multiply by dpr for device-pixel canvas
            const pt1 = map.project(p1 as [number, number]);
            const pt2 = map.project(p2 as [number, number]);

            ctx.save();
            ctx.strokeStyle = "rgba(160, 32, 240, 1)";
            ctx.lineWidth = 4 * dpr;
            ctx.lineCap = "round";
            ctx.beginPath();
            ctx.moveTo(pt1.x * dpr, pt1.y * dpr);
            ctx.lineTo(pt2.x * dpr, pt2.y * dpr);
            ctx.stroke();

            // endpoint dots
            ctx.fillStyle = "rgba(160, 32, 240, 1)";
            for (const pt of [pt1, pt2]) {
              ctx.beginPath();
              ctx.arc(pt.x * dpr, pt.y * dpr, 8 * dpr, 0, Math.PI * 2);
              ctx.fill();
            }
            ctx.restore();
          }

          resolve(composite.toDataURL("image/png").split(",")[1]);
        };

        // Fire the capture inside the next render callback so the WebGL buffer is live
        map.once("render", doCapture);
        map.triggerRepaint();
      });

    try {
      const b64 = await captureImage();

      const res = await fetch(`${API}/aao-briefing`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: b64, mime_type: "image/png" }),
      });
      const data = await res.json();
      const text = data.briefing ?? "No briefing returned.";
      const ts = new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });
      setMessages((prev) => [...prev, { id: Date.now(), text, ts }]);
    } catch (e) {
      const ts = new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });
      setMessages((prev) => [
        ...prev,
        { id: Date.now(), text: "Error generating briefing.", ts },
      ]);
      console.error(e);
    } finally {
      setBriefingLoading(false);
    }
  }

  // ── Interpolation helpers ────────────────────────────────────────────────
  function lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t;
  }
  function lerpAngle(a: number, b: number, t: number): number {
    // JS % preserves sign of dividend, so normalise with +540 before the second %
    // e.g. 350° → 10°: raw diff = -340; (-340 % 360 + 540) % 360 - 180 = 20 ✓
    const diff = ((((b - a) % 360) + 540) % 360) - 180;
    return a + diff * t;
  }

  const activeFeature = frames[currentFrame] ?? null;
  const props = activeFeature?.properties;
  const severity = props?.fire_severity ?? 0;
  const birdDog = props?.bird_dog ?? null;
  const tanker = props?.tanker ?? null;
  const simStatus = props?.sim_status ?? "";

  // Interpolate plane positions between currentFrame and currentFrame+1
  const nextProps =
    frames[Math.min(currentFrame + 1, frames.length - 1)]?.properties;

  const interpBirdDog: AircraftData | null = birdDog
    ? {
        visible: birdDog.visible,
        lat: nextProps?.bird_dog
          ? lerp(birdDog.lat, nextProps.bird_dog.lat, planeT)
          : birdDog.lat,
        lon: nextProps?.bird_dog
          ? lerp(birdDog.lon, nextProps.bird_dog.lon, planeT)
          : birdDog.lon,
        heading_deg: nextProps?.bird_dog
          ? lerpAngle(
              birdDog.heading_deg,
              nextProps.bird_dog.heading_deg,
              planeT,
            )
          : birdDog.heading_deg,
      }
    : null;

  const interpTanker: AircraftData | null = tanker
    ? {
        visible: tanker.visible,
        lat: nextProps?.tanker
          ? lerp(tanker.lat, nextProps.tanker.lat, planeT)
          : tanker.lat,
        lon: nextProps?.tanker
          ? lerp(tanker.lon, nextProps.tanker.lon, planeT)
          : tanker.lon,
        heading_deg: nextProps?.tanker
          ? lerpAngle(tanker.heading_deg, nextProps.tanker.heading_deg, planeT)
          : tanker.heading_deg,
        opacity: nextProps?.tanker
          ? lerp(tanker.opacity ?? 1, nextProps.tanker.opacity ?? 1, planeT)
          : (tanker.opacity ?? 1),
      }
    : null;

  const centerLon = activeFeature?.properties.center_lon ?? 0;
  const centerLat = activeFeature?.properties.center_lat ?? 0;
  const radiusKm = activeFeature?.properties.radius_km ?? 1;
  const center: [number, number] = [centerLon, centerLat];

  // ── Ember particles — flicker at plane-tick rate (~12 fps) ─────────────────
  interface Ember {
    position: [number, number];
    color: [number, number, number, number];
    size: number;
  }
  const embers = useMemo<Ember[]>(() => {
    if (!activeFeature) return [];
    const ring = activeFeature.geometry.coordinates[0];
    // Bounding box for fast candidate generation
    const lons = ring.map((p) => p[0]);
    const lats = ring.map((p) => p[1]);
    const minLon = Math.min(...lons),
      maxLon = Math.max(...lons);
    const minLat = Math.min(...lats),
      maxLat = Math.max(...lats);
    const dLon = maxLon - minLon,
      dLat = maxLat - minLat;
    const tickSeed = Math.round(planeT * 1000);
    const pts: Ember[] = [];
    let attempt = 0;
    while (pts.length < 300 && attempt < 1400) {
      const s = currentFrame * 2311 + tickSeed * 97 + attempt;
      const lon = minLon + seededRand(s) * dLon;
      const lat = minLat + seededRand(s + 4999) * dLat;
      if (pointInPolygon(lon, lat, ring)) {
        const t = seededRand(s + 200);
        const alpha = Math.round(40 + seededRand(s + 300) * 80); // half opacity
        let color: [number, number, number, number];
        if (t < 0.1)
          color = [255, 255, 225, alpha]; // white-hot
        else if (t < 0.28)
          color = [255, 220, 50, alpha]; // bright yellow
        else if (t < 0.58)
          color = [255, 115, 10, alpha]; // deep orange
        else if (t < 0.8)
          color = [210, 42, 5, alpha]; // red-orange
        else color = [135, 12, 0, alpha]; // dark ember
        const size = 0.5 + seededRand(s + 400) * 1.8; // smaller sparks
        pts.push({ position: [lon, lat], color, size });
      }
      attempt++;
    }
    return pts;
  }, [currentFrame, planeT]); // eslint-disable-line react-hooks/exhaustive-deps

  const fireLayers = activeFeature
    ? [
        // 1. Outermost atmosphere haze — very wide, near-transparent orange bloom
        new ScatterplotLayer({
          id: "fire-haze",
          data: [{ position: center }],
          getPosition: (d) => d.position,
          getFillColor: [255, 70, 0, 18],
          getRadius: radiusKm * 1400,
          updateTriggers: { getPosition: [currentFrame] },
        }),
        // 2. Outer perimeter polygon — dark red-orange, defines the fire boundary
        new PolygonLayer({
          id: "fire-outer",
          data: [activeFeature.geometry.coordinates[0]],
          getPolygon: (d) => d,
          getFillColor: [200, 30, 0, 90],
          getLineColor: severityLineColor(severity),
          getLineWidth: 3,
          lineWidthUnits: "pixels",
          filled: true,
          stroked: true,
          pickable: false,
          updateTriggers: {
            getFillColor: [severity],
            getLineColor: [severity],
          },
        }),
        // 3. Mid zone — brighter orange, 70% of perimeter
        new PolygonLayer({
          id: "fire-mid",
          data: [
            scalePolygon(activeFeature.geometry.coordinates[0], center, 0.7),
          ],
          getPolygon: (d) => d,
          getFillColor: [255, 95, 0, 130],
          filled: true,
          stroked: false,
          pickable: false,
          updateTriggers: { getFillColor: [severity] },
        }),
        // 4. Inner zone — yellow-orange, 44% of perimeter
        new PolygonLayer({
          id: "fire-inner",
          data: [
            scalePolygon(activeFeature.geometry.coordinates[0], center, 0.44),
          ],
          getPolygon: (d) => d,
          getFillColor: [255, 180, 20, 165],
          filled: true,
          stroked: false,
          pickable: false,
          updateTriggers: { getFillColor: [severity] },
        }),
        // 5. White-hot core, 18% of perimeter
        new PolygonLayer({
          id: "fire-core",
          data: [
            scalePolygon(activeFeature.geometry.coordinates[0], center, 0.18),
          ],
          getPolygon: (d) => d,
          getFillColor: [255, 245, 195, 210],
          filled: true,
          stroked: false,
          pickable: false,
        }),
        // 6. Ember / spark particles — tight inside the polygon, flicker at ~12 fps
        new ScatterplotLayer<Ember>({
          id: "fire-embers",
          data: embers,
          getPosition: (d) => d.position,
          getFillColor: (d) => d.color,
          getRadius: (d) => d.size,
          radiusUnits: "pixels",
          updateTriggers: {
            getPosition: [currentFrame, planeT],
            getFillColor: [currentFrame, planeT],
            getRadius: [currentFrame, planeT],
          },
        }),
      ]
    : [];

  // Purple drop zone line + endpoint dots
  const dropLineLayers = [
    ...(dropPoints.length === 2
      ? [
          new PathLayer({
            id: "drop-line",
            data: [{ path: dropPoints }],
            getPath: (d) => d.path,
            getColor: [160, 32, 240, 255],
            getWidth: 4,
            widthUnits: "pixels",
            capRounded: true,
          }),
        ]
      : []),
    ...(dropPoints.length > 0
      ? [
          new ScatterplotLayer({
            id: "drop-points",
            data: dropPoints.map((p) => ({ position: p })),
            getPosition: (d) => d.position,
            getFillColor: [160, 32, 240, 255],
            getRadius: 8,
            radiusUnits: "pixels",
          }),
        ]
      : []),
  ];

  // Bird Dog icon layers — plane body + Snoop Dogg head overlay
  const birdDogLayer = interpBirdDog?.visible
    ? [
        // Plane body — SVG nose is at bottom (south at 0°), so add 180° to face heading
        new IconLayer<AircraftData>({
          id: "bird-dog-plane",
          data: [interpBirdDog],
          getPosition: (d) => [d.lon, d.lat],
          getIcon: () => ({
            url: `data:image/svg+xml;charset=utf-8,${BIRD_DOG_SVG}`,
            width: 64,
            height: 64,
            anchorX: 32,
            anchorY: 32,
          }),
          getSize: 80,
          getAngle: (d) => -d.heading_deg + 180,
          billboard: false,
          sizeUnits: "pixels",
          updateTriggers: {
            getAngle: [currentFrame, planeT],
            getPosition: [currentFrame, planeT],
          },
        }),
        // Snoop Dogg head — same rotation so face points in direction of travel
        new IconLayer<AircraftData>({
          id: "bird-dog-head", // Snoop Dogg
          data: [interpBirdDog],
          getPosition: (d) => [d.lon, d.lat],
          getIcon: () => ({
            url: "/snoop-dogg-head.png",
            width: 512,
            height: 512,
            anchorX: 256,
            anchorY: 256,
          }),
          getSize: 44,
          getAngle: (d) => -d.heading_deg + 180,
          billboard: false,
          sizeUnits: "pixels",
          updateTriggers: {
            getAngle: [currentFrame, planeT],
            getPosition: [currentFrame, planeT],
          },
        }),
      ]
    : [];

  // Air Tanker icon layers — plane body + Drake head overlay
  const tankerOpacity = interpTanker?.opacity ?? 1;
  const tankerLayer = interpTanker?.visible
    ? [
        // Plane body — SVG nose is at bottom (south at 0°), so add 180° to face heading
        new IconLayer<AircraftData>({
          id: "tanker-plane",
          data: [interpTanker],
          getPosition: (d) => [d.lon, d.lat],
          getIcon: () => ({
            url: `data:image/svg+xml;charset=utf-8,${TANKER_SVG}`,
            width: 64,
            height: 64,
            anchorX: 32,
            anchorY: 32,
          }),
          getSize: 100,
          getAngle: (d) => -d.heading_deg + 180,
          opacity: tankerOpacity,
          billboard: false,
          sizeUnits: "pixels",
          updateTriggers: {
            getAngle: [currentFrame, planeT],
            getPosition: [currentFrame, planeT],
            opacity: [currentFrame, planeT],
          },
        }),
        // Drake head — same rotation so face points in direction of travel
        new IconLayer<AircraftData>({
          id: "tanker-head",
          data: [interpTanker],
          getPosition: (d) => [d.lon, d.lat],
          getIcon: () => ({
            url: "/drake-head.png",
            width: 512,
            height: 512,
            anchorX: 256,
            anchorY: 256,
          }),
          getSize: 38,
          getAngle: (d) => -d.heading_deg + 180,
          opacity: tankerOpacity,
          billboard: false,
          sizeUnits: "pixels",
          updateTriggers: {
            getAngle: [currentFrame, planeT],
            getPosition: [currentFrame, planeT],
            opacity: [currentFrame, planeT],
          },
        }),
      ]
    : [];

  // ── Shrub zone heatmap layers (one per zone) ──────────────────────────────
  const shrubLayer = showShrubs
    ? shrubZones.map(
        (zone, i) =>
          new HeatmapLayer<ShrubPoint>({
            id: `shrub-heatmap-${i}`,
            data: zone.points,
            getPosition: (d) => [d.lon, d.lat],
            getWeight: 1,
            radiusPixels: 60,
            intensity: 1,
            threshold: 0.05,
            colorRange: [
              [20, 83, 45, 0],
              [21, 128, 61, 80],
              [22, 163, 74, 140],
              [34, 197, 94, 180],
              [74, 222, 128, 210],
              [187, 247, 208, 230],
            ],
          }),
      )
    : [];

  // ── Terrain zone polygon layers (one per zone) ────────────────────────────
  // Soft purple outline + barely-there fill — reads as "zone boundary" not "solid area"
  const TERRAIN_PURPLE_FILL: [number, number, number, number] = [
    160, 32, 240, 18,
  ];
  const TERRAIN_PURPLE_STROKE: [number, number, number, number] = [
    160, 32, 240, 90,
  ];

  const favorableTerrainLayer = showFavorableTerrain
    ? terrainZones.map(
        (zone, i) =>
          new PolygonLayer({
            id: `favorable-terrain-${i}`,
            data: [zone.ring],
            getPolygon: (d) => d,
            getFillColor: TERRAIN_PURPLE_FILL,
            getLineColor: TERRAIN_PURPLE_STROKE,
            getLineWidth: 2,
            lineWidthUnits: "pixels",
            filled: true,
            stroked: true,
            pickable: false,
          }),
      )
    : [];

  const layers = [
    ...shrubLayer,
    ...favorableTerrainLayer,
    ...fireLayers,
    ...dropLineLayers,
    ...birdDogLayer,
    ...tankerLayer,
  ];

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col w-full h-full">
      <CommandHeader
        status={status}
        simStatus={simStatus}
        currentFrame={currentFrame}
        totalFrames={frames.length}
        weather={weather}
        severity={severity}
        dataSource={dataSource}
        fireMeta={fireMeta}
      />

      <div className="relative flex-1">
        <DeckGL
          ref={deckRef}
          initialViewState={INITIAL_VIEW}
          controller={true}
          layers={layers}
          onClick={handleDeckClick}
          getCursor={({ isDragging }) =>
            drawMode ? "crosshair" : isDragging ? "grabbing" : "grab"
          }
        >
          <Map ref={mapRef} mapStyle={MAP_STYLE} />
        </DeckGL>

        {/* Right panel */}
        <div className="absolute top-4 right-4 w-68 rounded-xl bg-black/70 text-white text-sm p-4 space-y-3 backdrop-blur-sm">
          {/* ── Title / data-source indicator ──────────────────────────── */}
          <div className="space-y-0.5">
            <h2 className="font-semibold text-base text-orange-400">
              Wildfire Zone (Nova Scotia)
            </h2>
            {dataSource === "satellite" && fireMeta ? (
              <p className="text-blue-300 text-xs flex items-center gap-1">
                <span>🛰️</span>
                <span className="truncate">{fireMeta.label}</span>
              </p>
            ) : (
              <p className="text-zinc-400 text-xs">
                {lat.toFixed(4)}°N, {Math.abs(lon).toFixed(4)}°W
              </p>
            )}
          </div>

          {/* ── Mock simulation ─────────────────────────────────────────── */}
          {/* <div className="space-y-1.5">
            <p className="text-zinc-500 text-[10px] uppercase tracking-wide">
              Synthetic simulation
            </p>
            <button
              onClick={handleSimulate}
              disabled={status === "loading" || status === "playing"}
              className="w-full py-1.5 rounded-lg bg-orange-500 hover:bg-orange-400 disabled:bg-zinc-700 disabled:text-zinc-500 text-white text-xs font-semibold transition-colors"
            >
              {status === "loading" && dataSource === "mock"
                ? "Loading…"
                : status === "playing" && dataSource === "mock"
                  ? "Simulating…"
                  : "Simulate (mock)"}
            </button>
          </div> */}

          {/* ── Real satellite fire simulation ──────────────────────────── */}
          <div className="space-y-1.5 border-t border-zinc-700 pt-3">
            <p className="text-zinc-500 text-[10px] uppercase tracking-wide flex items-center gap-1">
              <span>🛰️</span> Real satellite IR perimeter
            </p>

            {/* Fire selector */}
            <select
              value={selectedFire}
              onChange={(e) => setSelectedFire(e.target.value)}
              disabled={status === "loading" || status === "playing"}
              className="w-full py-1 px-2 rounded-lg bg-zinc-800 border border-zinc-600 text-zinc-200 text-xs disabled:opacity-50 focus:outline-none focus:border-blue-400"
            >
              <option value="barrington">🔥 Barrington Lake — 20,265 ha</option>
              <option value="tantallon">🔥 Tantallon (HRM) — 817 ha</option>
            </select>

            <button
              onClick={handleSimulateReal}
              disabled={status === "loading" || status === "playing"}
              className="w-full py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white text-xs font-semibold transition-colors"
            >
              {status === "loading" && dataSource === "satellite"
                ? "Fetching satellite data…"
                : status === "playing" && dataSource === "satellite"
                  ? "Playing…"
                  : "Simulate Real Fire"}
            </button>

            {/* Fire metadata card — shown after a real-fire run */}
            {fireMeta && dataSource === "satellite" && status !== "idle" && (
              <div className="rounded-lg bg-blue-950/50 border border-blue-800 px-2.5 py-2 space-y-1">
                <p className="text-blue-300 text-[10px] uppercase tracking-wide">
                  CWFIS NBAC — Satellite IR
                </p>
                <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-[11px]">
                  <span className="text-zinc-400">Size</span>
                  <span className="text-zinc-200">
                    {fireMeta.poly_ha.toLocaleString(undefined, {
                      maximumFractionDigits: 0,
                    })}{" "}
                    ha
                  </span>
                  <span className="text-zinc-400">Start</span>
                  <span className="text-zinc-200">
                    {fireMeta.start_date?.slice(0, 10)}
                  </span>
                  <span className="text-zinc-400">End</span>
                  <span className="text-zinc-200">
                    {fireMeta.end_date?.slice(0, 10)}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* ── Shrub zone toggle ────────────────────────────────────────── */}
          {dataSource === "satellite" && shrubZones.length > 0 && (
            <div className="border-t border-zinc-700 pt-3">
              <button
                onClick={() => setShowShrubs((v) => !v)}
                className={`w-full py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                  showShrubs
                    ? "bg-green-800 hover:bg-green-700 text-green-200 ring-1 ring-green-600"
                    : "bg-zinc-700 hover:bg-zinc-600 text-zinc-300"
                }`}
              >
                🌿 Shrub Zones
              </button>
            </div>
          )}

          {/* ── Terrain zone toggle ───────────────────────────────────────── */}
          {dataSource === "satellite" && terrainZones.length > 0 && (
            <div className="border-t border-zinc-700 pt-3 space-y-2">
              <button
                onClick={() => setShowFavorableTerrain((v) => !v)}
                className={`w-full py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                  showFavorableTerrain
                    ? "bg-purple-900/60 hover:bg-purple-900/80 text-purple-200 ring-1 ring-purple-600"
                    : "bg-zinc-700 hover:bg-zinc-600 text-zinc-300"
                }`}
              >
                🏔 Favorable Terrain
              </button>
            </div>
          )}

          {/* ── Draw drop zone ───────────────────────────────────────────── */}
          <div className="space-y-2 border-t border-zinc-700 pt-3">
            <button
              onClick={toggleDrawMode}
              className={`w-full py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                drawMode
                  ? "bg-purple-600 hover:bg-purple-500 text-white ring-2 ring-purple-400"
                  : "bg-zinc-700 hover:bg-zinc-600 text-zinc-200"
              }`}
            >
              {drawMode ? "✏️ Drawing… (click 2 points)" : "Draw Drop Zone"}
            </button>

            {drawMode && (
              <p className="text-zinc-400 text-xs text-center">
                {dropPoints.length === 0 && "Click start point"}
                {dropPoints.length === 1 && "Click end point"}
                {dropPoints.length === 2 && "Line set — get briefing or redraw"}
              </p>
            )}

            {dropPoints.length === 2 && (
              <button
                onClick={handleGetBriefing}
                disabled={briefingLoading}
                className="w-full py-1.5 rounded-lg bg-purple-700 hover:bg-purple-600 disabled:bg-zinc-700 disabled:text-zinc-500 text-white text-xs font-semibold transition-colors"
              >
                {briefingLoading ? "Generating…" : "Get AAO Briefing"}
              </button>
            )}
          </div>

          {weather && status !== "idle" && (
            <div className="space-y-1">
              <p className="text-zinc-400 text-xs uppercase tracking-wide">
                Current Weather
              </p>
              <div className="grid grid-cols-2 gap-x-2 gap-y-1">
                <span className="text-zinc-300">Temp</span>
                <span>{weather.temp_c} °C</span>
                <span className="text-zinc-300">Humidity</span>
                <span>{weather.humidity}%</span>
                <span className="text-zinc-300">Wind</span>
                <span>
                  {weather.wind_kph} kph {weather.wind_dir}
                </span>
              </div>
            </div>
          )}

          {terrain?.error && (
            <p className="text-red-400 text-xs">{terrain.error}</p>
          )}
        </div>

        {/* Satellite data badge — bottom-left when in real-fire mode */}
        {dataSource === "satellite" && status !== "idle" && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 pointer-events-none">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-950/90 border border-blue-600 text-blue-200 text-[10px] font-medium backdrop-blur-sm shadow-lg">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
              🛰️ CWFIS NBAC · VIIRS/MODIS IR · Nova Scotia 2023
            </div>
          </div>
        )}

        {/* AAO Comms — left side chat panel */}
        {showComms && (
          <div className="absolute top-4 left-4 w-80 h-[calc(100%-2rem)] max-h-150 flex flex-col rounded-xl bg-black/80 backdrop-blur-sm text-white shadow-2xl overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 bg-zinc-900/80 border-b border-zinc-700 shrink-0">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-purple-400 animate-pulse" />
                <span className="font-semibold text-sm text-purple-300">
                  AAO Comms
                </span>
              </div>
              <button
                onClick={() => setShowComms(false)}
                className="text-zinc-400 hover:text-white text-xs px-1.5 py-0.5 rounded hover:bg-zinc-700 transition-colors"
              >
                ✕
              </button>
            </div>

            {/* Message list */}
            <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">
              {messages.length === 0 && (
                <p className="text-zinc-500 text-xs text-center mt-8">
                  No transmissions yet.
                </p>
              )}
              {messages.map((msg) => (
                <div key={msg.id} className="flex flex-col gap-1">
                  {/* Avatar + timestamp row */}
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 rounded-full bg-purple-700 flex items-center justify-center text-[10px] font-bold shrink-0">
                      AAO
                    </div>
                    <span className="text-zinc-500 text-[10px]">{msg.ts}</span>
                  </div>
                  {/* Bubble */}
                  <div className="ml-8 bg-zinc-800 rounded-xl rounded-tl-sm px-3 py-2">
                    <p className="text-[12px] text-zinc-200 leading-relaxed">
                      {msg.text}
                    </p>
                  </div>
                </div>
              ))}
              {briefingLoading && (
                <div className="flex flex-col gap-1">
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 rounded-full bg-purple-700 flex items-center justify-center text-[10px] font-bold shrink-0">
                      AAO
                    </div>
                  </div>
                  <div className="ml-8 bg-zinc-800 rounded-xl rounded-tl-sm px-3 py-2">
                    <span className="text-zinc-400 text-xs italic">
                      Generating briefing…
                    </span>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Footer */}
            <div className="px-3 py-2 border-t border-zinc-700 shrink-0">
              <p className="text-zinc-500 text-[10px] text-center">
                Draw a drop zone on the map and press{" "}
                <span className="text-purple-400">Get AAO Briefing</span>
              </p>
            </div>
          </div>
        )}
      </div>
      {/* end relative flex-1 map area */}
    </div>
  );
}
