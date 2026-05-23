import FireMapLoader from "./components/FireMapLoader";

export default function Home() {
  return (
    <main className="flex flex-col flex-1 h-screen">
      <header className="shrink-0 px-4 py-3 bg-zinc-950 border-b border-zinc-800 flex items-center gap-3">
        <span className="text-orange-400 text-lg">🔥</span>
        <h1 className="text-white font-semibold text-sm tracking-wide">
          Drop It Like It&apos;s Hot
        </h1>
        <span className="text-zinc-500 text-xs ml-2">Wildfire Spread Visualizer</span>
      </header>
      <div className="flex-1 relative">
        <FireMapLoader />
      </div>
    </main>
  );
}
