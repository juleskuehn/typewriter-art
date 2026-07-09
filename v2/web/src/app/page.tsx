"use client";

import { ImageUpload } from "@/components/image-upload";
import { ControlPanel } from "@/components/control-panel";
import { LivePreview } from "@/components/live-preview";
import { DownloadPanel } from "@/components/download-panel";
import { LayerGallery } from "@/components/layer-gallery";
import { useOptimizerStore, useIsIdle, useIsConfiguring } from "@/store/optimizer";
import { ArrowLeft, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function Home() {
  const isIdle = useIsIdle();
  const isConfiguring = useIsConfiguring();
  const reset = useOptimizerStore((s) => s.reset);
  const state = useOptimizerStore((s) => s.state);
  const isComplete = state.phase === "complete";

  return (
    <main className="flex-1 flex flex-col items-center">
      {/* Header */}
      <header className="w-full max-w-5xl px-6 py-4 flex items-center justify-between border-b border-gray-800">
        <div className="flex items-center gap-2">
          {!isIdle && !isConfiguring && (
            <Button variant="ghost" size="icon" onClick={reset}>
              <ArrowLeft className="w-4 h-4" />
            </Button>
          )}
          <h1 className="text-lg font-semibold tracking-tight">Typewriter Art</h1>
        </div>
        {isComplete && (
          <Button variant="ghost" size="sm" onClick={reset} className="gap-1">
            <RefreshCw className="w-3 h-3" />
            New Image
          </Button>
        )}
      </header>

      {/* Content */}
      <div className="w-full max-w-5xl flex-1 p-6">
        {isIdle && <LandingView />}
        {isConfiguring && <ConfigureView />}
        {!isIdle && !isConfiguring && <ResultView />}
      </div>
    </main>
  );
}

function LandingView() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] gap-6">
      <div className="text-center space-y-2">
        <h2 className="text-2xl font-bold tracking-tight">
          Convert photographs into typewriter art
        </h2>
        <p className="text-gray-400 max-w-md mx-auto">
          Upload a grayscale photo and watch as characters are placed across
          multiple overlapping layers to recreate continuous tone through ink accumulation.
        </p>
      </div>
      <ImageUpload />
    </div>
  );
}

function ConfigureView() {
  return (
    <div className="flex flex-col lg:flex-row gap-6 items-start justify-center">
      <div className="flex flex-col gap-4 items-center">
        <ImageUpload />
      </div>
      <ControlPanel />
    </div>
  );
}

function ResultView() {
  const state = useOptimizerStore((s) => s.state);
  const reset = useOptimizerStore((s) => s.reset);
  const isComplete = state.phase === "complete";
  const isError = state.phase === "error";

  if (isError) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] gap-4">
        <div className="text-red-400 text-lg font-medium">Optimization failed</div>
        <div className="text-gray-400 text-sm max-w-md text-center">
          {(state as { phase: "error"; message: string }).message}
        </div>
        <Button onClick={reset} variant="outline">
          Try again
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col lg:flex-row gap-6 items-start justify-center">
      <div className="flex flex-col gap-6 items-center w-full max-w-xl">
        <LivePreview />
        <LayerGallery />
      </div>
      <DownloadPanel />
    </div>
  );
}
