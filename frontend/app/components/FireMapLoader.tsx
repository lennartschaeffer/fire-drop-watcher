"use client";

import dynamic from "next/dynamic";

const FireMap = dynamic(() => import("./FireMap"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center bg-zinc-950 text-zinc-500 text-sm">
      Loading map…
    </div>
  ),
});

export default function FireMapLoader() {
  return <FireMap />;
}
