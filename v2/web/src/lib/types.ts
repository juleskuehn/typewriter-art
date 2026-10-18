// TypeScript types matching the Rust optimizer types

/** Configuration passed to the optimizer. */
export interface OptimizeConfig {
  charset: string;
  row_length: number;
  num_loops: number;
  init_mode: "random" | "blank";
  asymmetry: number;
  search: "simAnneal" | "greedy";
  init_temp: number;
  layers: string;
  shuffle: boolean;
}

/** Statistics from the optimizer. */
export interface OptimizeStats {
  total_time_ns: number;
  total_comparisons: number;
  final_error: number;
}

/** Choices grid: dict of layer key → 2D array of character indices. */
export type ChoicesGrid = Record<string, number[][]>;

/** Full result from the optimizer. */
export interface OptimizeResult {
  mockup_base64: string;
  layers_base64: string[];
  num_rows: number;
  num_cols: number;
  fractional_offsets: [number, number][];
  stats: OptimizeStats;
  error_over_time: number[];
  choices: ChoicesGrid;
}

/** A single item in the image history (persisted to localStorage). */
export interface HistoryItem {
  id: string;
  /** Small thumbnail (resized, ~200px wide, base64 PNG) */
  thumbnail: string;
  /** Full-resolution image data URL */
  imageDataUrl: string;
  /** The config last used for this image */
  config: OptimizeConfig;
  /** The optimization result, if one exists */
  result: OptimizeResult | null;
  /** When this image was added/last modified */
  timestamp: number;
}

/** Center pane tab selection */
export type CenterTab = "input" | "output" | "layers";

/** Default configuration. */
export const DEFAULT_CONFIG: OptimizeConfig = {
  charset: "hermes-sm",
  row_length: 20,
  num_loops: 5,
  init_mode: "random",
  asymmetry: 0.1,
  search: "simAnneal",
  init_temp: 0.001,
  layers: "4x2",
  shuffle: true,
};

/** Available charsets. */
export const CHARSETS = [
  { value: "hermes-sm", label: "Hermes (small, 18×10)" },
  { value: "hermes-md", label: "Hermes (medium, 28×16)" },
  { value: "hermes", label: "Hermes (full)" },
  { value: "sc-1", label: "SC-1 (87 chars)" },
  { value: "sc-2", label: "SC-2 (175 chars, default)" },
  { value: "sc-3", label: "SC-3 (265 chars)" },
] as const;

/** Available layer configs. */
export const LAYER_CONFIGS = [
  { value: "1x1", label: "1×1 (single layer)" },
  { value: "4x1", label: "4×1 (4 layers)" },
  { value: "4x2", label: "4×2 (8 layers, default)" },
  { value: "16x1", label: "16×1 (16 layers)" },
] as const;
