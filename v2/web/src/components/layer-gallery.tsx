"use client";

import { useState } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { useOptimizerStore } from "@/store/optimizer";

export function LayerGallery() {
  const state = useOptimizerStore((s) => s.state);
  const result = state.phase === "complete" ? state.result : null;
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  if (!result) return null;

  return (
    <>
      <div className="w-full max-w-xl">
        <h3 className="text-sm font-medium text-gray-500 mb-2">Layers</h3>
        <Tabs defaultValue="0">
          <TabsList className="w-full overflow-x-auto flex-wrap h-auto gap-1">
            <TabsTrigger value="all" className="text-xs">
              All
            </TabsTrigger>
            {result.layers_base64.map((_, i) => {
              const [y, x] = result.fractional_offsets[i];
              return (
                <TabsTrigger key={i} value={String(i)} className="text-xs">
                  L{i} ({y},{x})
                </TabsTrigger>
              );
            })}
          </TabsList>

          {/* All layers grid */}
          <TabsContent value="all" className="mt-3">
            <div className="grid grid-cols-4 gap-1">
              {result.layers_base64.map((b64, i) => (
                <button
                  key={i}
                  className="relative group cursor-zoom-in"
                  onClick={() => setLightboxIndex(i)}
                >
                  <img
                    src={`data:image/png;base64,${b64}`}
                    alt={`Layer ${i}`}
                    className="w-full rounded border border-gray-200 dark:border-gray-800"
                    style={{ imageRendering: "pixelated" }}
                  />
                  <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors rounded flex items-center justify-center">
                    <span className="text-[10px] text-white opacity-0 group-hover:opacity-100 bg-black/50 px-1 rounded">
                      L{i}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </TabsContent>

          {/* Individual layers */}
          {result.layers_base64.map((b64, i) => (
            <TabsContent key={i} value={String(i)} className="mt-3">
              <button
                className="cursor-zoom-in w-full"
                onClick={() => setLightboxIndex(i)}
              >
                <img
                  src={`data:image/png;base64,${b64}`}
                  alt={`Layer ${i}`}
                  className="w-full rounded border border-gray-200 dark:border-gray-800"
                  style={{ imageRendering: "pixelated" }}
                />
              </button>
            </TabsContent>
          ))}
        </Tabs>
      </div>

      {/* Lightbox */}
      <Dialog
        open={lightboxIndex !== null}
        onOpenChange={(open) => {
          if (!open) setLightboxIndex(null);
        }}
      >
        <DialogContent className="max-w-4xl">
          {lightboxIndex !== null && (
            <div className="flex flex-col items-center gap-3">
              <img
                src={`data:image/png;base64,${result.layers_base64[lightboxIndex]}`}
                alt={`Layer ${lightboxIndex}`}
                className="max-h-[70vh] rounded"
                style={{ imageRendering: "pixelated" }}
              />
              <div className="flex items-center gap-2">
                <button
                  className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded disabled:opacity-30"
                  disabled={lightboxIndex === 0}
                  onClick={() => setLightboxIndex(lightboxIndex - 1)}
                >
                  ←
                </button>
                <span className="text-sm text-gray-500">
                  Layer {lightboxIndex} of {result.layers_base64.length - 1}
                  {(() => {
                    const [y, x] = result.fractional_offsets[lightboxIndex];
                    return ` (offset ${y}, ${x})`;
                  })()}
                </span>
                <button
                  className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded disabled:opacity-30"
                  disabled={lightboxIndex === result.layers_base64.length - 1}
                  onClick={() => setLightboxIndex(lightboxIndex + 1)}
                >
                  →
                </button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
