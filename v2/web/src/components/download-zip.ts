"use client";

import JSZip from "jszip";
import type { OptimizeResult } from "@/lib/types";

/**
 * Bundle all results into a ZIP file and trigger download.
 */
export async function downloadZip(result: OptimizeResult) {
  const zip = new JSZip();

  // Final mockup
  zip.file("final.png", base64ToBlob(result.mockup_base64), { base64: true });

  // Individual layers
  const layersFolder = zip.folder("layers")!;
  result.layers_base64.forEach((b64, i) => {
    const [y, x] = result.fractional_offsets[i];
    layersFolder.file(`layer_${i}_${y}_${x}.png`, base64ToBlob(b64), {
      base64: true,
    });
  });

  // Choices JSON
  zip.file("choices.json", JSON.stringify(result.choices, null, 2));

  // Stats text
  const statsText = [
    `Total time: ${(result.stats.total_time_ns / 1e6).toFixed(1)} ms`,
    `Total comparisons: ${result.stats.total_comparisons}`,
    `Final error: ${result.stats.final_error}`,
    `Grid: ${result.num_cols}×${result.num_rows}`,
  ].join("\n");
  zip.file("stats.txt", statsText);

  const blob = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "typewriter-art.zip";
  a.click();
  URL.revokeObjectURL(url);
}

function base64ToBlob(base64: string): string {
  // JSZip accepts base64 as string directly
  return base64;
}
