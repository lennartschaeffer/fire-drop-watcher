"use client";

import { useEffect, useRef, useState } from "react";
import Map, { MapRef } from "react-map-gl/maplibre";
import DeckGL from "@deck.gl/react";
import { PolygonLayer, IconLayer, PathLayer, ScatterplotLayer } from "@deck.gl/layers";
import "maplibre-gl/dist/maplibre-gl.css";

const INITIAL_VIEW = {
  longitude: -64.27,
  latitude: 45.57,
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

const STEPS = 15;
const FRAME_INTERVAL_MS = 1200;

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

interface PerimeterArrow {
  lon: number;
  lat: number;
  direction: number;
  severity: number;
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
    perimeter_arrows: PerimeterArrow[];
  };
}

interface SimulateResponse {
  frames: FireFeature[];
  weather: WeatherData;
  terrain: TerrainData;
  fuel: FuelData;
  prediction: { fire_direction: number; fire_severity: number };
  steps: number;
}

const API = "http://localhost:8000";

type SimStatus = "idle" | "loading" | "playing" | "done";

// Map severity 0–1 to RGBA: low=yellow, mid=orange, high=deep red
function severityColor(s: number): [number, number, number, number] {
  const r = 255;
  const g = Math.round(180 * (1 - s));
  const b = 0;
  return [r, g, b, Math.round(40 + 80 * s)];
}

function severityLineColor(s: number): [number, number, number, number] {
  const r = 255;
  const g = Math.round(160 * (1 - s));
  const b = 0;
  return [r, g, b, 220];
}

function severityLabel(s: number): string {
  if (s < 0.25) return "Low";
  if (s < 0.5)  return "Moderate";
  if (s < 0.75) return "High";
  return "Extreme";
}

function severityTextColor(s: number): string {
  if (s < 0.25) return "text-yellow-400";
  if (s < 0.5)  return "text-orange-400";
  if (s < 0.75) return "text-orange-500";
  return "text-red-500";
}

// Build a simple SVG arrow pointing up (north), rotated to fire_direction
function ArrowSVG({ direction, size = 32 }: { direction: number; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      style={{ transform: `rotate(${direction}deg)`, transition: "transform 0.4s ease" }}
    >
      <polygon
        points="16,2 22,22 16,18 10,22"
        fill="#f97316"
        stroke="#fff"
        strokeWidth="1.5"
      />
    </svg>
  );
}

export default function FireMap() {
  const [frames, setFrames] = useState<FireFeature[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [weather, setWeather] = useState<WeatherData | null>(null);
  const [terrain, setTerrain] = useState<TerrainData | null>(null);
  const [fuel, setFuel] = useState<FuelData | null>(null);
  const [prediction, setPrediction] = useState<{ fire_direction: number; fire_severity: number } | null>(null);
  const [status, setStatus] = useState<SimStatus>("idle");
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Draw mode
  const [drawMode, setDrawMode] = useState(false);
  const [dropPoints, setDropPoints] = useState<[number, number][]>([]);
  const [briefingLoading, setBriefingLoading] = useState(false);
  const [messages, setMessages] = useState<{ id: number; text: string; ts: string }[]>([]);
  const [showComms, setShowComms] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const deckRef = useRef<any>(null);
  const mapRef = useRef<MapRef>(null);

  const lat = INITIAL_VIEW.latitude;
  const lon = INITIAL_VIEW.longitude;

  function handleSimulate() {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setFrames([]);
    setCurrentFrame(0);
    setStatus("loading");

    fetch(`${API}/simulate?lat=${lat}&lon=${lon}&radius_km=2&steps=${STEPS}`)
      .then((r) => r.json())
      .then((data: SimulateResponse) => {
        setWeather(data.weather);
        setTerrain(data.terrain);
        setFuel(data.fuel);
        setPrediction(data.prediction);
        setFrames(data.frames);
        setStatus("playing");

        let frame = 0;
        intervalRef.current = setInterval(() => {
          frame += 1;
          if (frame >= data.frames.length) {
            clearInterval(intervalRef.current!);
            setStatus("done");
          } else {
            setCurrentFrame(frame);
          }
        }, FRAME_INTERVAL_MS);
      })
      .catch(() => setStatus("idle"));
  }

  useEffect(() => () => { if (intervalRef.current) clearInterval(intervalRef.current); }, []);

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
        if (!map) { resolve(""); return; }

        const doCapture = () => {
          const mapCanvas = map.getCanvas();
          const dpr = window.devicePixelRatio || 1;
          const w = mapCanvas.width;   // already in device pixels
          const h = mapCanvas.height;

          const composite = document.createElement("canvas");
          composite.width = w;
          composite.height = h;
          const ctx = composite.getContext("2d")!;

          // 1. Satellite tiles (captured synchronously during render — buffer is live)
          ctx.drawImage(mapCanvas, 0, 0);

          // 2. DeckGL overlay (may be populated; best-effort)
          const deckCanvas = deckRef.current?.deck?.canvas as HTMLCanvasElement | null;
          if (deckCanvas) {
            try { ctx.drawImage(deckCanvas, 0, 0); } catch (_) { /* tainted? skip */ }
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
      const ts = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      setMessages((prev) => [...prev, { id: Date.now(), text, ts }]);
    } catch (e) {
      const ts = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      setMessages((prev) => [...prev, { id: Date.now(), text: "Error generating briefing.", ts }]);
      console.error(e);
    } finally {
      setBriefingLoading(false);
    }
  }

  const activeFeature = frames[currentFrame] ?? null;
  const props = activeFeature?.properties;
  const severity = props?.fire_severity ?? 0;

  const arrowData: PerimeterArrow[] = activeFeature?.properties.perimeter_arrows ?? [];

  const fireLayers = activeFeature
    ? [
        new PolygonLayer({
          id: "fire-perimeter",
          data: [activeFeature.geometry.coordinates[0]],
          getPolygon: (d) => d,
          getFillColor: severityColor(severity),
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
        new IconLayer<PerimeterArrow>({
          id: "fire-direction-arrows",
          data: arrowData,
          getPosition: (d) => [d.lon, d.lat],
          getIcon: () => ({
            url: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(
              `<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 32 32">
                <polygon points="16,2 22,22 16,18 10,22" fill="#f97316" stroke="white" stroke-width="1.5"/>
              </svg>`
            )}`,
            width: 64,
            height: 64,
            anchorX: 32,
            anchorY: 32,
          }),
          getSize: 32,
          getAngle: (d) => -d.direction,
          billboard: false,
          sizeUnits: "pixels",
          updateTriggers: { getAngle: [currentFrame] },
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

  const layers = [...fireLayers, ...dropLineLayers];

  return (
    <div className="relative w-full h-full">
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
        <Map
          ref={mapRef}
          mapStyle={MAP_STYLE}
        />
      </DeckGL>

      <div className="absolute top-4 right-4 w-64 rounded-xl bg-black/70 text-white text-sm p-4 space-y-3 backdrop-blur-sm">
        <h2 className="font-semibold text-base text-orange-400">Fire Zone — Nova Scotia</h2>
        <p className="text-zinc-400 text-xs">
          {lat.toFixed(4)}°N, {Math.abs(lon).toFixed(4)}°W
        </p>

        <button
          onClick={handleSimulate}
          disabled={status === "loading" || status === "playing"}
          className="w-full py-1.5 rounded-lg bg-orange-500 hover:bg-orange-400 disabled:bg-zinc-700 disabled:text-zinc-500 text-white text-xs font-semibold transition-colors"
        >
          {status === "loading" ? "Loading…" : status === "playing" ? "Simulating…" : "Simulate"}
        </button>

        {/* Draw drop zone */}
        <div className="space-y-2">
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

        {props && (status === "playing" || status === "done") && (
          <div className="space-y-1">
            <p className="text-zinc-400 text-xs uppercase tracking-wide">Simulation</p>
            <div className="grid grid-cols-2 gap-x-2 gap-y-1">
              <span className="text-zinc-300">Step</span>
              <span>{props.step + 1} / {STEPS}</span>
              <span className="text-zinc-300">Radius</span>
              <span>{props.radius_km.toFixed(2)} km</span>
            </div>
            <div className="mt-1 h-1 rounded bg-zinc-700">
              <div
                className="h-1 rounded bg-orange-500 transition-all duration-300"
                style={{ width: `${((props.step + 1) / STEPS) * 100}%` }}
              />
            </div>
          </div>
        )}

        {prediction && status !== "idle" && (
          <div className="space-y-1">
            <p className="text-zinc-400 text-xs uppercase tracking-wide">Model Prediction</p>
            <div className="grid grid-cols-2 gap-x-2 gap-y-1 items-center">
              <span className="text-zinc-300">Severity</span>
              <span className={`font-semibold ${severityTextColor(prediction.fire_severity)}`}>
                {severityLabel(prediction.fire_severity)} ({(prediction.fire_severity * 100).toFixed(0)}%)
              </span>
              <span className="text-zinc-300">Direction</span>
              <span className="flex items-center gap-1.5">
                <ArrowSVG direction={prediction.fire_direction} size={18} />
                {prediction.fire_direction.toFixed(0)}°
              </span>
            </div>
            <div className="mt-1 h-1.5 rounded bg-zinc-700">
              <div
                className="h-1.5 rounded transition-all duration-300"
                style={{
                  width: `${prediction.fire_severity * 100}%`,
                  backgroundColor: `rgb(255, ${Math.round(180 * (1 - prediction.fire_severity))}, 0)`,
                }}
              />
            </div>
          </div>
        )}

        {weather && status !== "idle" && (
          <div className="space-y-1">
            <p className="text-zinc-400 text-xs uppercase tracking-wide">Weather</p>
            <div className="grid grid-cols-2 gap-x-2 gap-y-1">
              <span className="text-zinc-300">Temp</span>
              <span>{weather.temp_c} °C</span>
              <span className="text-zinc-300">Humidity</span>
              <span>{weather.humidity}%</span>
              <span className="text-zinc-300">Wind</span>
              <span>{weather.wind_kph} kph {weather.wind_dir}</span>
            </div>
          </div>
        )}

        {terrain && !terrain.error && status !== "idle" && (
          <div className="space-y-1">
            <p className="text-zinc-400 text-xs uppercase tracking-wide">Terrain</p>
            <div className="grid grid-cols-2 gap-x-2 gap-y-1">
              <span className="text-zinc-300">Slope</span>
              <span>{terrain.slope_degrees}°</span>
              <span className="text-zinc-300">Aspect</span>
              <span>{terrain.aspect_degrees}°</span>
              {terrain.elevation_m !== undefined && (
                <>
                  <span className="text-zinc-300">Elevation</span>
                  <span>{terrain.elevation_m} m</span>
                </>
              )}
            </div>
          </div>
        )}

        {fuel && status !== "idle" && (
          <div className="space-y-1">
            <p className="text-zinc-400 text-xs uppercase tracking-wide">Fuel</p>
            <div className="grid grid-cols-2 gap-x-2 gap-y-1">
              <span className="text-zinc-300">Type</span>
              <span>{fuel.fuel_type}</span>
              <span className="text-zinc-300">Risk</span>
              <span className="capitalize">{fuel.flammability}</span>
            </div>
            <p className="text-zinc-500 text-xs leading-tight">{fuel.fuel_description}</p>
          </div>
        )}

        {terrain?.error && <p className="text-red-400 text-xs">{terrain.error}</p>}
      </div>

      {/* AAO Comms — left side chat panel */}
      {showComms && (
        <div className="absolute top-4 left-4 w-80 h-[calc(100%-2rem)] max-h-150 flex flex-col rounded-xl bg-black/80 backdrop-blur-sm text-white shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 bg-zinc-900/80 border-b border-zinc-700 shrink-0">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-purple-400 animate-pulse" />
              <span className="font-semibold text-sm text-purple-300">AAO Comms</span>
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
              <p className="text-zinc-500 text-xs text-center mt-8">No transmissions yet.</p>
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
                  <span className="text-zinc-400 text-xs italic">Generating briefing…</span>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Footer */}
          <div className="px-3 py-2 border-t border-zinc-700 shrink-0">
            <p className="text-zinc-500 text-[10px] text-center">
              Draw a drop zone on the map and press <span className="text-purple-400">Get AAO Briefing</span>
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
