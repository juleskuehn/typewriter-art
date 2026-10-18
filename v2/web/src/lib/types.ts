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
  /** Run the pre-optimization auto-alignment sweep (64-crop search). */
  align_enabled: boolean;
  /** Deluxe: deep-rerank the top crops with a real optimization run. */
  align_deluxe: boolean;
  /** Spatial weighting of the loss and alignment score ("center focus"). */
  center_focus: "off" | "soft" | "strong";
}

/** Statistics from the optimizer. */
export interface OptimizeStats {
  total_time_ns: number;
  total_comparisons: number;
  final_error: number;
}

/** Choices grid: dict of layer key → 2D array of character indices. */
export type ChoicesGrid = Record<string, number[][]>;

/** One crop candidate from the auto-alignment sweep (step units, 4 = 1 char). */
export interface CropTransform {
  zoom: number;
  shift_x: number;
  shift_y: number;
}

/** Scored result for one alignment candidate. */
export interface AlignmentScore {
  transform: CropTransform;
  psnr: number;
  ssim: number;
  score: number;
  /** True when rescored with the full layer stack (two-stage sweep). */
  refined: boolean;
}

/** Outcome of the auto-alignment sweep. */
export interface AlignmentResult {
  best: CropTransform;
  steps: number;
  /** Screen/refine candidates, sorted best-first by the quick score. */
  scores: AlignmentScore[];
  /**
   * Deluxe rerank scores: the top candidates re-evaluated with a deeper run.
   * Not comparable with `scores`; `best` follows this ranking when present.
   */
  rerank?: AlignmentScore[];
}

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
  /** Auto-alignment outcome, when the sweep ran. */
  alignment?: AlignmentResult;
}

/** A single item in the image history (persisted to localStorage). */
export interface HistoryItem {
  id: string;
  /** Small thumbnail (resized, ~200px wide, base64 PNG) */
  thumbnail: string;
  /**
   * Image data URL used for display and optimization.
   *
   * Full resolution in a live session. After a page reload this may be only the
   * ~200px thumbnail, because localStorage has a ~5 MB quota and we cannot keep
   * every source image. When that happens `sourceImageLost` is set.
   */
  imageDataUrl: string;
  /**
   * True when `imageDataUrl` is a degraded thumbnail rather than the original
   * upload. Re-optimizing would silently produce a much worse result, so the UI
   * blocks it and asks for a re-upload.
   */
  sourceImageLost?: boolean;
  /** The config last used for this image */
  config: OptimizeConfig;
  /** The optimization result, if one exists */
  result: OptimizeResult | null;
  /** When this image was added/last modified */
  timestamp: number;
}

/** Center pane tab selection */
export type CenterTab = "input" | "output" | "layers";

/** Default configuration (matches the Rust CLI defaults). */
export const DEFAULT_CONFIG: OptimizeConfig = {
  charset: "sc-2",
  row_length: 30,
  num_loops: 15,
  init_mode: "random",
  asymmetry: 0.1,
  search: "simAnneal",
  init_temp: 0.001,
  layers: "4x2",
  shuffle: true,
  align_enabled: true,
  align_deluxe: false,
  center_focus: "soft",
};

/**
 * Center focus presets (must match `CenterFocus` in the Rust library).
 * Labels describe the emphasis on the middle of the image.
 */
export const CENTER_FOCUSES = [
  { value: "off", label: "Off (uniform)" },
  { value: "soft", label: "Soft" },
  { value: "strong", label: "Strong (fuzzy borders)" },
] as const;

/**
 * Available charsets (must match directories under `v2/assets/charsets/`).
 * Labels show post-filter glyph count and glyph size (height×width),
 * as loaded by the optimizer.
 */
export const CHARSETS = [
  { value: "sc-2", label: "Smith-Corona 2-tone (173 chars, 52×26)" },
  { value: "sc-1", label: "Smith-Corona medium (87 chars, 50×26)" },
  { value: "sc-3", label: "Smith-Corona 3-tone (263 chars, 50×26)" },
  { value: "sc-round", label: "Smith-Corona round (7 chars, 50×26)" },
  { value: "hermes-sm", label: "Hermes small (88 chars, 18×10)" },
  { value: "hermes-md", label: "Hermes medium (88 chars, 28×16)" },
  { value: "hermes", label: "Hermes full (88 chars, 52×30)" },
  { value: "daisywriter-shrunken", label: "Daisywriter shrunken (95 chars, 32×16)" },
  { value: "daisywriter", label: "Daisywriter full (95 chars, 82×42)" },
  { value: "dw-no-chev", label: "Daisywriter no chevrons (91 chars, 32×16)" },
] as const;

/**
 * Available layer configs (must match keys in `v2/assets/layers.json`).
 * Labels show the number of layers.
 */
export const LAYER_CONFIGS = [
  { value: "1x1", label: "1 layer" },
  { value: "4x2", label: "4×2 (8 layers)" },
  { value: "4x1", label: "4×1 (4 layers)" },
  { value: "16x1", label: "16×1 (16 layers)" },
  { value: "2Hx1", label: "2H×1 (2 layers, horizontal)" },
  { value: "2Hx2", label: "2H×2 (4 layers, horizontal)" },
  { value: "2Vx1", label: "2V×1 (2 layers, vertical)" },
  { value: "2Vx2", label: "2V×2 (4 layers, vertical)" },
  { value: "1x2", label: "1×2 (2 layers, same offset)" },
  { value: "1x4", label: "1×4 (4 layers, same offset)" },
  { value: "daisy_full", label: "Daisy full (50 layers)" },
  { value: "daisy_x2", label: "Daisy ×2 (100 layers)" },
] as const;
