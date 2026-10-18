"use client";

import { useState } from "react";
import type { OptimizeResult } from "@/lib/types";

export function LayerView({ result }: { result: OptimizeResult }) {
  const [selected, setSelected] = useState(0);

  const totalLayers = result.layers_base64.length;
  const cols = Math.min(4, totalLayers);
  const selectedLayer = Math.min(selected, totalLayers - 1);

  if (totalLayers === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-sm text-gray-400">
        Layer images aren&apos;t available for this entry — re-optimize to generate
        them.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 w-full h-full overflow-y-auto">
      {/* Grid of layer thumbnails */}
      <div
        className="grid gap-2"
        style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}
      >
        {result.layers_base64.map((b64, i) => {
          const [y, x] = result.fractional_offsets[i] ?? [0, 0];
          return (
            <button
              key={i}
              className={`relative border-2 rounded-lg overflow-hidden transition-all
                ${selectedLayer === i
                  ? "border-blue-500 ring-1 ring-blue-300"
                  : "border-gray-200 hover:border-gray-400"
                }`}
              onClick={() => setSelected(i)}
            >
              <img
                src={`data:image/png;base64,${b64}`}
                alt={`Layer ${i}`}
                className="w-full"
                style={{ imageRendering: "pixelated" }}
              />
              <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-white text-[10px] py-0.5 px-1 text-center">
                L{i} ({y},{x})
              </div>
            </button>
          );
        })}
      </div>

      {/* Enlarged view of selected layer */}
      <div className="flex-1 flex items-center justify-center border border-gray-200 rounded-lg p-4 bg-gray-50 min-h-[200px]">
        <img
          src={`data:image/png;base64,${result.layers_base64[selectedLayer]}`}
          alt={`Layer ${selectedLayer}`}
          className="max-w-full max-h-full"
          style={{ imageRendering: "pixelated" }}
        />
      </div>
    </div>
  );
}
