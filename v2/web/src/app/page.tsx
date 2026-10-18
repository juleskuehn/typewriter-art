"use client";

import { SidebarHistory } from "@/components/sidebar-history";
import { SidebarControls } from "@/components/sidebar-controls";
import { CenterPane } from "@/components/center-pane";

export default function Home() {
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
