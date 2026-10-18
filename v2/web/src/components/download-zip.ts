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

  // Auto-alignment scores (same format as the CLI's alignment_scores.json)
  if (result.alignment) {
    zip.file(
      "alignment_scores.json",
      JSON.stringify(result.alignment, null, 2)
    );
  }

  // Stats text — same format as the CLI's stats.txt
  const usPerComparison =
    result.stats.total_comparisons > 0
      ? `Time per comparison: ${(result.stats.total_time_ns / result.stats.total_comparisons / 1000).toFixed(3)} µs\n`
      : "";
  const statsText = [
    `Total time: ${(result.stats.total_time_ns / 1e9).toFixed(3)} s`,
    `Total comparisons: ${result.stats.total_comparisons}`,
    usPerComparison.trimEnd(),
    `Final error: ${result.stats.final_error.toFixed(6)}`,
  ]
    .filter(Boolean)
    .join("\n");
  zip.file("stats.txt", statsText);

  // Error over time — same format as the CLI's error_over_time.csv
  const csv = [
    "iteration,error",
    ...result.error_over_time.map((err, i) => `${i},${err}`),
  ].join("\n");
  zip.file("error_over_time.csv", csv);

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
