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
import { idbGet, idbSet, idbRemove } from "@/lib/idb";

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
 * Persistence.
 *
 * History lives in IndexedDB, not localStorage. localStorage is a synchronous
 * ~5 MB store and one optimized entry is roughly 1 MB (full source image plus a
 * full-resolution mockup), so two or three entries overflowed it — surfacing as
 * QuotaExceededError — and any photo over ~375 KB could not be re-optimized
 * after a reload. IndexedDB is asynchronous with a far larger per-origin quota,
 * so full images and results are kept intact.
 *
 * Three things still matter even with a large quota:
 *  1. zustand's persist middleware runs `setItem()` after **every** `set()`, and
 *     `optimize()` calls `set()` once per SSE progress event. Persistence is
 *     suspended during a run so progress ticks cost O(1) and the result is
 *     written exactly once, when the run finishes.
 *  2. Storage access never throws into application code — a full or unavailable
 *     store must not fail the optimization the user just waited for.
 *  3. Writes must not race hydration. `setItem()` fires after every `set()`,
 *     including `set()`s that land while the asynchronous read in `getItem` is
 *     still in flight — persisting that pre-hydration state would overwrite
 *     stored history with an empty list. Writes are gated on hydration
 *     completion instead (see `storageReady`).
 */
const STORAGE_KEY = "typewriter-art-history";

/**
 * Safety valve rather than a design limit: if history ever grows past this the
 * oldest entries are dropped, instead of filling the origin's storage quota.
 */
const PERSIST_BUDGET_BYTES = 64 * 1024 * 1024;

/**
 * Rough persisted size of a string field. Base64 is ~1 byte per char and the
 * structured clone overhead is negligible, so `.length` is close enough.
 */
function estimateChars(value: string | undefined): number {
  return value ? value.length : 0;
}

/** True while an optimization run is in flight — see note above. */
let persistSuspended = false;

/**
 * False until hydration has finished (success or failure).
 *
 * zustand's persist middleware runs `setItem()` after **every** `set()` —
 * including `set()`s issued while the asynchronous read in `getItem` is still
 * in flight. Writing then persists the pre-hydration default state (an empty
 * history) over the real data: that is how history entries vanished after a
 * reload. Writes are dropped until `onRehydrateStorage`'s callback below has
 * fired, i.e. until the stored state has been merged into the store.
 */
let storageReady = false;

/**
 * Set false when stored history could not be *read* (as opposed to simply
 * being absent). An empty history that follows a read error is untrustworthy,
 * so writes stay off for the session rather than clobbering data that merely
 * failed to load.
 */
let storageWritable = true;

/** Last snapshot, reused while persistence is suspended. */
let lastSnapshot: PersistedState | null = null;

interface PersistedState {
  history: HistoryItem[];
}

/** Approximate persisted footprint of one entry. */
function estimateItem(h: HistoryItem): number {
  let size = estimateChars(h.thumbnail) + estimateChars(h.imageDataUrl) + 512;
  const r = h.result;
  if (r) {
    size += estimateChars(r.mockup_base64);
    for (const layer of r.layers_base64) size += estimateChars(layer);
    size += r.error_over_time.length * 24; // number + separators
    size += 2048; // choices grid + stats + fractional offsets
  }
  return size;
}

/**
 * Entries in persistence order (newest first), trimmed to the budget.
 *
 * Only whole entries are dropped and only as a last resort — with IndexedDB
 * this is a safety valve, not a routine occurrence. The live session is never
 * affected by trimming.
 */
function toPersisted(history: HistoryItem[]): PersistedState {
  const out: HistoryItem[] = [];
  let budget = PERSIST_BUDGET_BYTES;

  for (const h of history) {
    const size = estimateItem(h);
    if (size > budget) break;
    budget -= size;
    out.push(h);
  }

  return { history: out };
}

/**
 * Validate whatever was found in storage into a `HistoryItem`.
 *
 * The schema is strict: anything that is not a current entry is dropped rather
 * than repaired. There is no deployed data to be compatible with — the v2
 * store in `lib/idb.ts` wipes the old database on upgrade — so legacy shapes
 * (the localStorage `resultMeta` summaries, thumbnail-only "sources") are not
 * supported.
 *
 * The one normalization kept is merging `config` over the defaults: a config
 * must be structurally complete no matter which fields a stored entry carries,
 * because a `checked={undefined}` input flips from uncontrolled to controlled
 * on first click (React warning).
 */
function normalizeItem(raw: unknown): HistoryItem | null {
  const item = raw as Partial<HistoryItem> | null | undefined;
  if (!item || typeof item.id !== "string") return null;
  if (typeof item.thumbnail !== "string" || typeof item.imageDataUrl !== "string") return null;

  return {
    id: item.id,
    thumbnail: item.thumbnail,
    imageDataUrl: item.imageDataUrl,
    sourceImageLost: item.sourceImageLost === true,
    config: { ...DEFAULT_CONFIG, ...(item.config ?? {}) },
    result: item.result && typeof item.result === "object" ? item.result : null,
    timestamp: typeof item.timestamp === "number" ? item.timestamp : 0,
  };
}

function normalizeStored(input: unknown): PersistedState | null {
  const container = input as { state?: { history?: unknown[] } } | null | undefined;
  const history = container?.state?.history;
  if (!Array.isArray(history)) return null;
  return { history: history.map(normalizeItem).filter((h): h is HistoryItem => h !== null) };
}

/**
 * `PersistStorage` over IndexedDB.
 *
 * `getItem` always resolves rather than rejecting, so zustand's hydration
 * always completes and the UI can leave its loading state even when storage is
 * unavailable (e.g. private browsing).
 */
const persistStorage = {
  getItem: async (
    name: string
  ): Promise<{ state: PersistedState; version?: number } | null> => {
    // One-time cleanup from the localStorage → IndexedDB transition: older
    // builds kept history under this key in localStorage (the ~5 MB store that
    // kept overflowing). Nothing reads it any more.
    try {
      if (typeof localStorage !== "undefined") localStorage.removeItem(name);
    } catch {
      // ignore
    }
    try {
      const stored = normalizeStored(await idbGet<unknown>(name));
      return stored ? { state: stored } : null;
    } catch {
      // A failed read is not an empty store: keep writes disabled so the
      // empty history that follows can never overwrite unread data.
      storageWritable = false;
      return null;
    }
  },
  setItem: async (
    name: string,
    value: { state: PersistedState; version?: number }
  ): Promise<void> => {
    // The single choke point for writes — see the notes on `storageReady` and
    // `persistSuspended` above: zustand persists after *every* `set()`, so this
    // is what keeps pre-hydration state and progress ticks out of storage.
    if (!storageReady || !storageWritable || persistSuspended) return;
    try {
      await idbSet(name, value);
    } catch {
      // Losing persisted history is far better than failing the optimization
      // the user just waited for.
    }
  },
  removeItem: async (name: string): Promise<void> => {
    try {
      await idbRemove(name);
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
  /**
   * False until persisted history has loaded. Storage is asynchronous, so the
   * UI waits rather than flashing the empty state over real history.
   */
  hydrated: boolean;
  /**
   * Transient status line while a run is in flight ("Aligning 12/72…",
   * "Optimizing…"). Not persisted.
   */
  progressStatus: string | null;

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
      hydrated: false,
      progressStatus: null,

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
        set({ progressStatus: "Preparing…" });
        try {
          const result: OptimizeResult = await runOptimization(
            item.imageDataUrl,
            item.config,
            // Progress callback: update the result incrementally
            (progress) => {
              if (progress.type === "align") {
                set({
                  progressStatus: `Aligning ${progress.candidate}/${progress.total}…`,
                });
                return;
              }
              runErrors.push(progress.error);
              set((s) => ({
                progressStatus: "Optimizing…",
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
          set({ progressStatus: null });
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
        lastSnapshot = toPersisted(state.history);
        return lastSnapshot;
      },
      // `getItem` has already validated whatever was in storage.
      merge: (persisted: unknown, current) => {
        const p = persisted as PersistedState | undefined;
        const c = current as OptimizerStore;
        if (!p?.history) return c;
        // Hydration ends with a replace-style `set`, so anything added to
        // `history` while the async read was in flight would be dropped here.
        // Keep those entries (history is newest-first).
        const pending = c.history.filter((h) => !p.history.some((x) => x.id === h.id));
        return { ...c, history: [...pending, ...p.history] };
      },
      // Fires after the merge above on both the success and error paths — only
      // from here on may `setItem` touch storage.
      onRehydrateStorage: () => (_state, error) => {
        storageReady = true;
        if (error) storageWritable = false;
      },
    }
  )
);

//
// Storage is asynchronous, so history lands after the first render. The UI
// gates on `hydrated` rather than flashing the empty state over real history.
// `persistStorage.getItem` always resolves instead of rejecting, so this
// listener is guaranteed to fire even when storage is unavailable.
//
const finishHydration = () => useOptimizerStore.setState({ hydrated: true });
useOptimizerStore.persist.onFinishHydration(finishHydration);
if (useOptimizerStore.persist.hasHydrated()) finishHydration();

export function useActiveItem() {
  return useOptimizerStore((s) => s.activeItem());
}
