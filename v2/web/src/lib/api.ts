import type { OptimizeConfig, OptimizeResult } from "@/lib/types";

/**
 * Send an image + config to the server and get back the optimization result.
 *
 * Converts the image data URL to a Blob, sends as multipart form data
 * along with the optimization config, and the server returns the JSON result.
 */
export async function runOptimization(
  imageDataUrl: string,
  config: OptimizeConfig
): Promise<OptimizeResult> {
  // Decode data URL to Blob (more reliable than fetch(dataUrl) for large files)
  const [header, base64] = imageDataUrl.split(",");
  const mimeType = header.match(/data:([^;]+)/)?.[1] || "image/png";
  const ext = mimeType.split("/")[1] || "png";

  const binaryStr = atob(base64);
  const bytes = new Uint8Array(binaryStr.length);
  for (let i = 0; i < binaryStr.length; i++) {
    bytes[i] = binaryStr.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: mimeType });

  const formData = new FormData();
  formData.append("image", blob, `input.${ext}`);
  formData.append("config", JSON.stringify(config));

  const res = await fetch("/api/optimize", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Optimization failed (${res.status}): ${text}`);
  }

  return res.json();
}
