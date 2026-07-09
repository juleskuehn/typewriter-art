"use client";

import { useRef, useEffect } from "react";
import { useOptimizerStore } from "@/store/optimizer";
import type { OptimizeResult } from "@/lib/types";

/**
 * Renders the optimizer mockup onto a canvas.
 * Loads from base64 PNG data when optimization completes.
 */
export function LivePreview() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const state = useOptimizerStore((s) => s.state);

  // When we have a result or configuring, show the image
  const imageDataUrl =
    state.phase === "configuring" || state.phase === "optimizing" || state.phase === "complete"
      ? state.imageDataUrl
      : null;

  const result = state.phase === "complete" ? state.result : null;

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (result) {
      // Show the optimized mockup from base64
      const img = new Image();
      img.onload = () => {
        // Scale canvas to fit the image while maintaining aspect ratio
        const maxW = canvas.parentElement?.clientWidth ?? 600;
        const scale = Math.min(1, maxW / img.width);
        canvas.width = img.width * scale;
        canvas.height = img.height * scale;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      };
      img.src = `data:image/png;base64,${result.mockup_base64}`;
    } else if (imageDataUrl) {
      // Show the uploaded original image
      const img = new Image();
      img.onload = () => {
        const maxW = canvas.parentElement?.clientWidth ?? 600;
        const scale = Math.min(1, maxW / img.width);
        canvas.width = img.width * scale;
        canvas.height = img.height * scale;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      };
      img.src = imageDataUrl;
    }
  }, [result, imageDataUrl]);

  if (!imageDataUrl && !result) {
    return (
      <div className="w-full max-w-xl aspect-[3/4] bg-gray-100 dark:bg-gray-900 rounded-lg flex items-center justify-center text-gray-400 text-sm">
        Upload an image to see the preview
      </div>
    );
  }

  return (
    <div className="w-full max-w-xl">
      <div className="relative">
        <canvas
          ref={canvasRef}
          className="rounded-lg border border-gray-200 dark:border-gray-800 w-full"
          style={{ imageRendering: "pixelated" }}
        />
        {state.phase === "optimizing" && (
          <div className="absolute inset-0 bg-black/20 rounded-lg flex items-center justify-center">
            <div className="bg-white dark:bg-gray-800 rounded-lg px-4 py-2 shadow-lg text-sm font-medium">
              Optimizing...
            </div>
          </div>
        )}
      </div>
      {result && (
        <div className="mt-2 text-center text-sm text-gray-500">
          Error: {result.stats.final_error.toFixed(6)}
          {" · "}
          Time: {(result.stats.total_time_ns / 1e6).toFixed(1)} ms
          {" · "}
          Grid: {result.num_cols}×{result.num_rows}
        </div>
      )}
    </div>
  );
}
