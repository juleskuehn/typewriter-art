"use client";

import { create } from "zustand";
import type { OptimizeConfig, OptimizeResult, OptimizerState } from "@/lib/types";
import { DEFAULT_CONFIG } from "@/lib/types";
import { runOptimization } from "@/lib/api";

interface OptimizerStore {
  state: OptimizerState;
  config: OptimizeConfig;

  /** Set the uploaded file and enter configuring phase. */
  setFile: (file: File) => void;

  /** Update a config parameter. */
  setConfig: <K extends keyof OptimizeConfig>(
    key: K,
    value: OptimizeConfig[K]
  ) => void;

  /** Start optimization. */
  optimize: () => Promise<void>;

  /** Reset to idle. */
  reset: () => void;
}

export const useOptimizerStore = create<OptimizerStore>((set, get) => ({
  state: { phase: "idle" },
  config: { ...DEFAULT_CONFIG },

  setFile: (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      set({
        state: {
          phase: "configuring",
          imageDataUrl: reader.result as string,
        },
      });
    };
    reader.readAsDataURL(file);
  },

  setConfig: (key, value) => {
    set((s) => ({ config: { ...s.config, [key]: value } }));
  },

  optimize: async () => {
    const current = get();
    if (current.state.phase !== "configuring") return;

    set({ state: { phase: "optimizing", imageDataUrl: current.state.imageDataUrl } });

    try {
      const result: OptimizeResult = await runOptimization(
        current.state.imageDataUrl,
        current.config
      );

      set({
        state: {
          phase: "complete",
          imageDataUrl: current.state.imageDataUrl,
          result,
        },
      });
    } catch (err) {
      set({
        state: { phase: "error", message: String(err) },
      });
    }
  },

  reset: () => {
    set({ state: { phase: "idle" }, config: { ...DEFAULT_CONFIG } });
  },
}));

/** Convenience hooks for specific state slices. */
export function useIsIdle() {
  return useOptimizerStore((s) => s.state.phase === "idle");
}
export function useIsConfiguring() {
  return useOptimizerStore((s) => s.state.phase === "configuring");
}
export function useIsOptimizing() {
  return useOptimizerStore((s) => s.state.phase === "optimizing");
}
export function useIsComplete() {
  return useOptimizerStore((s) => s.state.phase === "complete");
}
