import type { OptimizeConfig, OptimizeResult } from "@/lib/types";

/**
 * Progress event from the SSE stream (before completion).
 */
export type ProgressEvent = AlignProgressEvent | LoopProgressEvent;

/** One auto-alignment candidate finished evaluating. */
export interface AlignProgressEvent {
  type: "align";
  /** Candidates evaluated so far. */
  candidate: number;
  /** Total evaluations in the sweep (candidates + refinements). */
  total: number;
}

/** One layer optimization pass finished. */
export interface LoopProgressEvent {
  type: "loop";
  loop_num: number;
  num_loops: number;
  layer_num: number;
  num_layers: number;
  error: number;
  /** Base64 PNG of the current mockup — sent on the layer-0 pass of each loop. */
  mockup_b64?: string;
}

export type OptimizeProgressCallback = (event: ProgressEvent) => void;

/**
 * Send an image + config to the server and receive SSE progress + final result.
 *
 * Calls `onProgress` for each progress event (with optional mockup base64),
 * and resolves with the final result on completion.
 */
export async function runOptimization(
  imageDataUrl: string,
  config: OptimizeConfig,
  onProgress?: OptimizeProgressCallback
): Promise<OptimizeResult> {
  // Decode data URL to Blob
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

  // Read SSE stream
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  return new Promise((resolve, reject) => {
    let settled = false;
    const settleResolve = (result: OptimizeResult) => {
      if (!settled) {
        settled = true;
        resolve(result);
      }
    };
    const settleReject = (err: Error) => {
      if (!settled) {
        settled = true;
        reject(err);
      }
    };

    function pump() {
      reader!.read().then(({ done, value }) => {
        if (value) {
          buffer += decoder.decode(value, { stream: !done });
          buffer = processSSEBuffer(buffer, (eventType, data) => {
            if (eventType === "progress") {
              try {
                const progress: ProgressEvent = JSON.parse(data);
                onProgress?.(progress);
              } catch { /* ignore parse errors */ }
            } else if (eventType === "complete") {
              try {
                const result: OptimizeResult = JSON.parse(data);
                settleResolve(result);
              } catch (e) {
                settleReject(e instanceof Error ? e : new Error(String(e)));
              }
            } else if (eventType === "error") {
              try {
                const err = JSON.parse(data);
                settleReject(new Error(err.message || "Unknown error"));
              } catch { settleReject(new Error("Unknown error")); }
            }
          });
        }
        if (done) {
          // Stream ended without a terminal event — don't hang forever
          settleReject(new Error("Optimization stream ended without a result"));
          return;
        }
        pump();
      }).catch((e) =>
        settleReject(e instanceof Error ? e : new Error(String(e)))
      );
    }
    pump();
  });
}

/**
 * Process SSE buffer, calling handler for each complete event.
 * Returns the remaining (incomplete) portion of the buffer.
 *
 * SSE events are delimited by double newlines (\n\n). We accumulate
 * the buffer until we have at least one complete event, process it,
 * and keep the remainder.
 */
function processSSEBuffer(
  buffer: string,
  handle: (eventType: string, data: string) => void
): string {
  let idx = 0;
  while (true) {
    const eventEnd = buffer.indexOf("\n\n", idx);
    if (eventEnd === -1) break;

    const eventBlock = buffer.slice(idx, eventEnd);
    idx = eventEnd + 2;

    // Parse event block lines
    let eventType = "";
    let data = "";
    for (const line of eventBlock.split("\n")) {
      if (line.startsWith("event: ")) {
        eventType = line.slice(7);
      } else if (line.startsWith("data: ")) {
        data += (data ? "\n" : "") + line.slice(6);
      }
    }

    if (eventType && data) {
      handle(eventType, data);
    }
  }

  return buffer.slice(idx);
}
