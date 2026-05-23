"use client";

import { useEffect, useRef, useState } from "react";
import Map from "react-map-gl/maplibre";
import DeckGL from "@deck.gl/react";
import { PolygonLayer } from "@deck.gl/layers";
import "maplibre-gl/dist/maplibre-gl.css";

const INITIAL_VIEW = {
  longitude: -119.4,
  latitude: 50.6,
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
}

interface TerrainData {
  slope_degrees: number;
  aspect_degrees: number;
  error?: string;
}

interface FireFeature {
  type: string;
  geometry: { type: string; coordinates: number[][][] };
  properties: { center_lat: number; center_lon: number; radius_km: number; step: number };
}

interface SimulateResponse {
  frames: FireFeature[];
  weather: WeatherData;
  terrain: TerrainData;
  steps: number;
}

const API = "http://localhost:8000";

type SimStatus = "idle" | "loading" | "playing" | "done";

export default function FireMap() {
  const [frames, setFrames] = useState<FireFeature[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [weather, setWeather] = useState<WeatherData | null>(null);
  const [terrain, setTerrain] = useState<TerrainData | null>(null);
  const [status, setStatus] = useState<SimStatus>("idle");
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

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

  const activeFeature = frames[currentFrame] ?? null;

  const layers = activeFeature
    ? [
        new PolygonLayer({
          id: "fire-perimeter",
          data: [activeFeature.geometry.coordinates[0]],
          getPolygon: (d) => d,
          getFillColor: [255, 80, 0, 60],
          getLineColor: [255, 140, 0, 220],
          getLineWidth: 3,
          lineWidthUnits: "pixels",
          filled: true,
          stroked: true,
          pickable: false,
        }),
      ]
    : [];

  const props = activeFeature?.properties;

  return (
    <div className="relative w-full h-full">
      <DeckGL initialViewState={INITIAL_VIEW} controller={true} layers={layers}>
        <Map mapStyle={MAP_STYLE} />
      </DeckGL>

      <div className="absolute top-4 right-4 w-64 rounded-xl bg-black/70 text-white text-sm p-4 space-y-3 backdrop-blur-sm">
        <h2 className="font-semibold text-base text-orange-400">Fire Zone — Okanagan</h2>
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
            </div>
          </div>
        )}

        {terrain?.error && <p className="text-red-400 text-xs">{terrain.error}</p>}
      </div>
    </div>
  );
}
