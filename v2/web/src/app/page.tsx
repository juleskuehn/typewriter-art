"use client";

import { SidebarHistory } from "@/components/sidebar-history";
import { SidebarControls } from "@/components/sidebar-controls";
import { CenterPane } from "@/components/center-pane";
import { useOptimizerStore } from "@/store/optimizer";

export default function Home() {
  const hydrated = useOptimizerStore((s) => s.hydrated);

  // History loads from IndexedDB asynchronously. Rendering the empty state
  // first would flash "Drop an image here" over entries that are already there.
  if (!hydrated) {
    return (
      <div className="h-full flex items-center justify-center">
        <p className="text-sm text-gray-400">Loading history…</p>
      </div>
    );
  }

  return (
    <div className="h-full flex">
      {/* Left sidebar — history */}
      <SidebarHistory />

      {/* Center — image / tabs */}
      <CenterPane />

      {/* Right sidebar — controls */}
      <SidebarControls />
    </div>
  );
}
