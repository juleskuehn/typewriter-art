"use client";

import { Plus } from "lucide-react";
import { SidebarHistory } from "@/components/sidebar-history";
import { SidebarControls } from "@/components/sidebar-controls";
import { CenterPane } from "@/components/center-pane";
import { useOptimizerStore } from "@/store/optimizer";

export default function Home() {
  const newImage = useOptimizerStore((s) => s.newImage);

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
