"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type {
  OptimizeConfig,
  OptimizeResult,
  HistoryItem,
  CenterTab,
} from "@/lib/types";
import { DEFAULT_CONFIG } from "@/lib/types";
import { runOptimization } from "@/lib/api";

function uid(): string {
  return crypto.randomUUID().slice(0, 8);
}

function makeThumbnail(dataUrl: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const maxDim = 200;
      const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL("image/png"));
    };
    img.onerror = reject;
    img.src = dataUrl;
  });
}

/**
 * Persistence budgeting.
 *
 * localStorage is a ~5 MB synchronous string store, and zustand's persist
 * middleware calls `setItem()` after **every** `set()`. A history item can carry
 * a full source image plus a full-resolution mockup, and `optimize()` calls
 * `set()` once per progress event — so a single 480-event run used to serialize
 * the entire history 480 times and then throw QuotaExceededError.
 *
 * Three defences:
 *  1. `toPersisted` keeps the payload under a fixed byte budget, dropping the
 *     oldest items first (they survive in the live session, just not a reload).
 *  2. Persistence is suspended during a run so progress ticks cost O(1) and the
 *     result is written exactly once, when the run finishes.
 *  3. `persistStorage.setItem` never throws — on quota pressure the write is
 *     skipped rather than breaking the run.
 */
const STORAGE_KEY = "typewriter-art-history";

/** Stay comfortably under the typical 5 MB localStorage quota. */
const PERSIST_BUDGET_BYTES = 3_000_000;

/**
 * Per-item cap on the source image. Matches the original rule (`< 500_000`),
 * and guarantees one oversized upload cannot starve every other entry.
 * Larger uploads are simply not re-optimizable after a reload.
 */
const MAX_SOURCE_IMAGE_CHARS = 500_000;

/**
 * Rough persisted size of a string field. Base64 is ~1 byte per char and the
 * JSON escaping it needs is negligible, so `.length` is a good enough estimate;
 * the try/catch in `persistStorage` covers the tail risk.
 */
function estimateChars(value: string | undefined): number {
  return value ? value.length : 0;
}

/** True while an optimization run is in flight — see note above. */
let persistSuspended = false;

/** Last serialized snapshot, reused while persistence is suspended. */
let lastSnapshot: PersistedState | null = null;

/** Minimal persisted data — bounded thumbnails, config and stats. */
interface PersistedItem {
  id: string;
  thumbnail: string;
  /**
   * Full image, kept only while the budget allows. Absent means we only have
   * the thumbnail after a reload — see `sourceImageLost`.
   */
  imageDataUrl?: string;
  config: OptimizeConfig;
  /** Stats + mockup only — full layers are never persisted. */
  resultMeta: {
    stats: OptimizeResult["stats"];
    mockup_base64: string;
    num_rows: number;
    num_cols: number;
    error_over_time: number[];
  } | null;
  timestamp: number;
  /** Set when the full source image had to be dropped to stay in budget. */
  sourceImageLost?: boolean;
}

interface PersistedState {
  history: PersistedItem[];
}

/**
 * History items in persistence order (newest first), trimmed to the byte
 * budget.
 *
 * Allocation is two passes so one oversized field cannot starve the rest:
 *  - Pass 1 gives every entry its thumbnail and its result — that is the work
 *    the user actually waited for and what the UI shows.
 *  - Pass 2 spends what is left on source images (newest first), which are only
 *    needed to re-optimize and can be re-uploaded.
 */
function toPersisted(history: HistoryItem[]): PersistedItem[] {
  const out: PersistedItem[] = [];
  let budget = PERSIST_BUDGET_BYTES;

  // Pass 1 — thumbnails and results.
  for (const h of history) {
    const item: PersistedItem = {
      id: h.id,
      thumbnail: h.thumbnail,
      config: h.config,
      resultMeta: null,
      timestamp: h.timestamp,
      sourceImageLost: true,
    };

    let size = estimateChars(h.thumbnail) + 256; // 256 ≈ id/config/timestamp JSON

    if (h.result) {
      const mockup = estimateChars(h.result.mockup_base64);
      const curve = h.result.error_over_time.length * 24; // number + separators
      if (size + mockup + curve <= budget) {
        item.resultMeta = {
          stats: h.result.stats,
          mockup_base64: h.result.mockup_base64,
          num_rows: h.result.num_rows,
          num_cols: h.result.num_cols,
          error_over_time: h.result.error_over_time,
        };
        size += mockup + curve;
      }
    }

    // Not even room for the thumbnail — everything older is larger still.
    if (size > budget) break;

    budget -= size;
    out.push(item);
  }

  // Pass 2 — source images, newest first, each individually capped.
  for (let i = 0; i < out.length; i++) {
    // Already degraded: `imageDataUrl` is just the ~200px thumbnail at this
    // point. Never re-assert it as a real source image, or the next reload
    // would silently "restore" a preview and allow optimizing it.
    if (history[i].sourceImageLost) continue;
    const src = history[i].imageDataUrl;
    if (src === history[i].thumbnail) continue;
    const srcSize = estimateChars(src);
    if (!src || srcSize > MAX_SOURCE_IMAGE_CHARS || srcSize > budget) continue;
    out[i].imageDataUrl = src;
    out[i].sourceImageLost = false;
    budget -= srcSize;
  }

  return out;
}

/**
 * Direct `PersistStorage` rather than `createJSONStorage`, so the stringify
 * happens here and can be skipped entirely while a run is in flight.
 */
const persistStorage = {
  getItem: (name: string): { state: PersistedState; version?: number } | null => {
    try {
      const raw = localStorage.getItem(name);
      return raw ? (JSON.parse(raw) as { state: PersistedState; version?: number }) : null;
    } catch {
      return null;
    }
  },
  setItem: (name: string, value: { state: PersistedState; version?: number }): void => {
    if (persistSuspended) return;
    try {
      localStorage.setItem(name, JSON.stringify(value));
    } catch {
      // QuotaExceededError (or private-mode storage) — losing persisted history
      // is far better than failing the optimization the user just waited for.
    }
  },
  removeItem: (name: string): void => {
    try {
      localStorage.removeItem(name);
    } catch {
      // ignore
    }
  },
};

interface OptimizerStore {
  history: HistoryItem[];
  activeId: string | null;
  centerTab: CenterTab;
  /** Whether the user is in "new image" upload mode (empty center) */
  newImageMode: boolean;

  activeItem: () => HistoryItem | null;

  addImage: (dataUrl: string) => Promise<string>;
  selectImage: (id: string) => void;
  newImage: () => void;
  setConfig: <K extends keyof OptimizeConfig>(key: K, value: OptimizeConfig[K]) => void;
  optimize: () => Promise<void>;
  setTab: (tab: CenterTab) => void;
  removeImage: (id: string) => void;
}

export const useOptimizerStore = create<OptimizerStore>()(
  persist(
    (set, get) => ({
      history: [],
      activeId: null,
      centerTab: "input",
      newImageMode: true,

      activeItem: () => {
        const { history, activeId } = get();
        return history.find((h) => h.id === activeId) ?? null;
      },

      addImage: async (dataUrl: string) => {
        const existing = get().history.find(
          (h) => h.imageDataUrl === dataUrl
        );
        if (existing) {
          set({ activeId: existing.id, newImageMode: false, centerTab: existing.result ? "output" : "input" });
          return existing.id;
        }

        const id = uid();
        const thumbnail = await makeThumbnail(dataUrl);

        const item: HistoryItem = {
          id,
          thumbnail,
          imageDataUrl: dataUrl,
          sourceImageLost: false,
          config: { ...DEFAULT_CONFIG },
          result: null,
          timestamp: Date.now(),
        };

        set((s) => ({
          history: [item, ...s.history],
          activeId: id,
          newImageMode: false,
          centerTab: "input",
        }));

        return id;
      },

      selectImage: (id) => {
        const item = get().history.find((h) => h.id === id);
        set({
          activeId: id,
          newImageMode: false,
          centerTab: item?.result ? "output" : "input",
        });
      },

      newImage: () => {
        set({ activeId: null, newImageMode: true, centerTab: "input" });
      },

      setConfig: (key, value) => {
        const { activeId } = get();
        if (!activeId) return;
        set((s) => ({
          history: s.history.map((h) =>
            h.id === activeId
              ? { ...h, config: { ...h.config, [key]: value }, timestamp: Date.now() }
              : h
          ),
        }));
      },

      optimize: async () => {
        const { activeId } = get();
        if (!activeId) return;
        const item = get().history.find((h) => h.id === activeId);
        if (!item) return;

        // Refuse to optimize a degraded thumbnail. After a reload we may only
        // still have the ~200px preview, which would silently produce a much
        // worse result than the one on screen — ask for a re-upload instead.
        if (item.sourceImageLost) {
          throw new Error(
            "The full-size image for this entry was not kept after a page reload. Re-upload the image to optimize it."
          );
        }

        // Errors for the current run only — never append to a previous run's curve
        const runErrors: number[] = [];

        // Suspend persistence for the duration of the run: every progress event
        // calls set(), and each one would otherwise serialize the whole history.
        persistSuspended = true;
        try {
          const result: OptimizeResult = await runOptimization(
            item.imageDataUrl,
            item.config,
            // Progress callback: update the result incrementally
            (progress) => {
              runErrors.push(progress.error);
              set((s) => ({
                history: s.history.map((h) =>
                  h.id === activeId
                    ? {
                        ...h,
                        result: {
                          mockup_base64: progress.mockup_b64 ?? h.result?.mockup_base64 ?? "",
                          layers_base64: h.result?.layers_base64 ?? [],
                          num_rows: h.result?.num_rows ?? 0,
                          num_cols: h.result?.num_cols ?? 0,
                          fractional_offsets: h.result?.fractional_offsets ?? [],
                          stats: {
                            ...(h.result?.stats ?? { total_time_ns: 0, total_comparisons: 0, final_error: 0 }),
                            final_error: progress.error,
                          },
                          error_over_time: [...runErrors],
                          choices: h.result?.choices ?? {},
                        },
                        timestamp: Date.now(),
                      }
                    : h
                ),
              }));
            }
          );

          // Re-enable persistence before the final set() so the completed
          // result is written exactly once, here.
          persistSuspended = false;
          set((s) => ({
            history: s.history.map((h) =>
              h.id === activeId
                ? { ...h, result, timestamp: Date.now() }
                : h
            ),
            centerTab: "output",
          }));
        } catch (err) {
          console.error("Optimization failed:", err);
          throw err;
        } finally {
          persistSuspended = false;
        }
      },

      setTab: (tab) => set({ centerTab: tab }),

      removeImage: (id) => {
        set((s) => {
          const newHistory = s.history.filter((h) => h.id !== id);
          const newActiveId =
            s.activeId === id ? (newHistory[0]?.id ?? null) : s.activeId;
          const newImageMode = newHistory.length === 0 && s.activeId === id;
          return {
            history: newHistory,
            activeId: newImageMode ? null : newActiveId,
            newImageMode,
          };
        });
      },
    }),
    {
      name: STORAGE_KEY,
      storage: persistStorage,
      // Build a budgeted snapshot instead of persisting the raw history.
      partialize: (state) => {
        // O(1) during a run — see `persistSuspended`.
        if (persistSuspended) return lastSnapshot ?? { history: [] };
        lastSnapshot = { history: toPersisted(state.history) };
        return lastSnapshot;
      },
      // Rehydrate: reconstruct full HistoryItem from persisted data
      merge: (persisted: unknown, current) => {
        const p = persisted as {
          history?: PersistedItem[];
        };
        if (!p?.history) return current as OptimizerStore;

        const rehydrated: HistoryItem[] = p.history.map((pi) => {
          // A "source" that is byte-identical to the thumbnail is a degraded
          // fallback, not a real upload — treat it as lost even if an older
          // build recorded `sourceImageLost: false`. This self-heals records
          // that were persisted before the per-item cap existed.
          const degraded = !pi.imageDataUrl || pi.imageDataUrl === pi.thumbnail;
          return {
            id: pi.id,
            thumbnail: pi.thumbnail,
            // Used for display only; `sourceImageLost` blocks re-optimizing.
            imageDataUrl: pi.imageDataUrl ?? pi.thumbnail,
            sourceImageLost: degraded || pi.sourceImageLost === true,
            config: pi.config ?? { ...DEFAULT_CONFIG },
            result: pi.resultMeta
              ? {
                  mockup_base64: pi.resultMeta.mockup_base64,
                  layers_base64: [], // layers not persisted — will be empty
                  num_rows: pi.resultMeta.num_rows,
                  num_cols: pi.resultMeta.num_cols,
                  fractional_offsets: [],
                  stats: pi.resultMeta.stats,
                  error_over_time: pi.resultMeta.error_over_time,
                  choices: {},
                }
              : null,
            timestamp: pi.timestamp ?? 0,
          };
        });

        return {
          ...(current as OptimizerStore),
          history: rehydrated,
        };
      },
    }
  )
);

export function useActiveItem() {
  return useOptimizerStore((s) => s.activeItem());
}
