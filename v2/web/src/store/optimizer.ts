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

/** Minimal persisted data — thumbnails + config + stats only (no full images). */
interface PersistedItem {
  id: string;
  thumbnail: string;
  /** Full image stored in memory only; on reload, we use thumbnail. */
  imageDataUrl?: string;
  config: OptimizeConfig;
  /** Stats + mockup thumbnail only — full layers not persisted. */
  resultMeta: {
    stats: OptimizeResult["stats"];
    mockup_base64: string;
    num_rows: number;
    num_cols: number;
    error_over_time: number[];
  } | null;
  timestamp: number;
}

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

        try {
          const result: OptimizeResult = await runOptimization(
            item.imageDataUrl,
            item.config,
            // Progress callback: update the result incrementally
            (progress) => {
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
                          error_over_time: [...(h.result?.error_over_time ?? []), progress.error],
                          choices: h.result?.choices ?? {},
                        },
                        timestamp: Date.now(),
                      }
                    : h
                ),
              }));
            }
          );

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
      name: "typewriter-art-history",
      // Only persist metadata + thumbnails — no full images to stay under quota
      partialize: (state) => ({
        history: state.history.map((h) => ({
          id: h.id,
          thumbnail: h.thumbnail,
          config: h.config,
          resultMeta: h.result
            ? {
                stats: h.result.stats,
                mockup_base64: h.result.mockup_base64,
                num_rows: h.result.num_rows,
                num_cols: h.result.num_cols,
                error_over_time: h.result.error_over_time,
              }
            : null,
          timestamp: h.timestamp,
        })),
      }),
      // Rehydrate: reconstruct full HistoryItem from persisted data
      merge: (persisted: unknown, current) => {
        const p = persisted as {
          history?: PersistedItem[];
        };
        if (!p?.history) return current as OptimizerStore;

        const rehydrated: HistoryItem[] = p.history.map((pi) => ({
          id: pi.id,
          thumbnail: pi.thumbnail,
          imageDataUrl: pi.imageDataUrl ?? pi.thumbnail, // fall back to thumbnail
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
        }));

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
