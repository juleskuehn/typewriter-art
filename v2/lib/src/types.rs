//! Core types for the typewriter-art optimization engine.
//!
//! These types represent the data model: charsets, targets, layer configurations,
//! optimization parameters, and results.

use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

/// A 3D array of character glyphs: (num_chars, height, width).
/// Values are in [0.0, 1.0] where 1.0 = white (no ink) and 0.0 = black (full ink).
pub type Chars = Array3<f32>;

/// A 2D grayscale image: (height, width). Values in [0.0, 1.0].
pub type Image = Array2<f32>;

/// A grid of character indices: (num_layers, num_positions).
/// Each position holds the index into the charset for that character.
pub type Choices = ndarray::Array2<u16>;

/// A 3D stack of layer images: (num_layers, height, width).
pub type Layers = Array3<f32>;

/// Character grid dimensions (height, width in pixels).
pub type CharShape = (usize, usize);

/// Layer offset in pixels: (y_offset, x_offset).
pub type LayerOffset = (usize, usize);

/// Fractional layer offset: (y_frac, x_frac) as fractions of char height/width.
pub type FractionalOffset = (f32, f32);

/// Padding applied to the target image, in pixels.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Padding {
    pub top: usize,
    pub right: usize,
    pub bottom: usize,
    pub left: usize,
}

impl Padding {
    /// The image content inside a padded target of `shape` (rows, cols):
    /// `(y0, y1, x0, x1)` with exclusive ends — i.e. the container minus the
    /// white border that `resize_target` adds (half a character on each side,
    /// plus any bottom rows padded to reach a character boundary).
    pub fn content_rect(&self, shape: (usize, usize)) -> (usize, usize, usize, usize) {
        (
            self.top,
            shape.0.saturating_sub(self.bottom),
            self.left,
            shape.1.saturating_sub(self.right),
        )
    }
}

/// Spatial weighting of the loss and the auto-alignment score ("center focus").
///
/// Emphasizes the middle of the image over its edges and margins. For the
/// alignment score this changes which crop wins; for the optimization loss it
/// mainly stops charging edge characters for ink in the padding border —
/// positions optimize their own tiles independently, so a slowly varying
/// profile cannot reallocate effort between regions. See [`crate::weights`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CenterFocus {
    /// Uniform weights (pixel parity with the paper's loss).
    Off,
    /// Gentle center emphasis; margins lightly de-emphasized.
    Soft,
    /// Strong center emphasis; margins nearly free (fuzzy borders).
    Strong,
}

/// Parameters of the radial weight profile (see [`crate::weights`]).
#[derive(Debug, Clone, Copy)]
pub struct FocusPreset {
    /// Weight at the corners of the content region.
    pub floor: f32,
    /// Gaussian sigma; the rect's edge midpoints sit at r = 1.
    pub sigma: f32,
    /// Weight outside the content region (padding border / margin).
    pub padding: f32,
}

impl CenterFocus {
    /// Profile parameters, or `None` for [`CenterFocus::Off`] (uniform).
    pub fn preset(self) -> Option<FocusPreset> {
        match self {
            CenterFocus::Off => None,
            CenterFocus::Soft => Some(FocusPreset {
                floor: 0.25,
                sigma: 0.65,
                padding: 0.1,
            }),
            CenterFocus::Strong => Some(FocusPreset {
                floor: 0.05,
                sigma: 0.45,
                padding: 0.0,
            }),
        }
    }
}

/// One crop candidate for the auto-alignment sweep (paper §3.4).
///
/// All fields are in fine steps: value `v` means `v/steps` of a character
/// (with `steps = 8` the default, 1 = ⅛ char). Unlike v1's one-sided space,
/// parameters are signed: shifts can move the content in any direction
/// (positive = left/up, matching v1) and zoom can shrink as well as grow
/// (positive = the container grows by that fraction of a character).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CropTransform {
    /// Scale: boundary growth in fine steps, signed (negative shrinks).
    pub zoom: i32,
    /// Horizontal translation in fine steps, signed (positive = left).
    pub shift_x: i32,
    /// Vertical translation in fine steps, signed (positive = up).
    pub shift_y: i32,
}

/// Scored result for one alignment candidate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AlignmentScore {
    pub transform: CropTransform,
    pub psnr: f32,
    pub ssim: f32,
    /// `ssim * ssim_weight + psnr` (the paper's `SSIM × 4 + PSNR`).
    pub score: f32,
    /// True when this candidate was rescored with the full layer stack
    /// (two-stage sweep); unrefined scores come from the single-pass screen.
    #[serde(default)]
    pub refined: bool,
}

/// Outcome of the auto-alignment sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentResult {
    /// The chosen crop transform (the applied one — with the deluxe rerank
    /// this can differ from `scores[0]`).
    pub best: CropTransform,
    /// Fine steps per parameter (`v` = `v/steps` of a character).
    pub steps: u32,
    /// Screen/refine candidates, sorted best-first by the quick score.
    pub scores: Vec<AlignmentScore>,
    /// Deluxe rerank scores: the top candidates re-evaluated with a deeper
    /// optimization run (scores are not comparable with `scores`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank: Option<Vec<AlignmentScore>>,
}

/// Full output of [`align::search()`], including optional candidate mockups.
///
/// [`align::search()`]: crate::align::search
#[derive(Debug, Clone)]
pub struct AlignmentSearch {
    pub result: AlignmentResult,
    /// Quick mockup per candidate (cropped to its score region), best-first.
    /// Only populated when requested.
    pub mockups: Vec<(CropTransform, Image)>,
}

/// Configuration for the pre-optimization auto-alignment sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignConfig {
    /// Run the sweep before optimizing.
    pub enabled: bool,
    /// Fine resolution: one parameter step = `1/steps` of a character
    /// (8 → ⅛-char fine grid; the coarse grid is ¼-char).
    pub steps: u32,
    /// Quick greedy passes per candidate (paper: a single cycle).
    pub passes: u32,
    /// Score = `ssim * ssim_weight + psnr` (paper: 4.0).
    pub ssim_weight: f32,
    /// Two-stage sweep: after screening every candidate with a single
    /// placement pass, rescore the top `refine_top` with the full layer stack
    /// and rank by that. 0 disables refinement.
    pub refine_top: u32,
    /// Deluxe rerank: run a deeper optimization on the top `rerank_top`
    /// candidates and pick the winner by that (approximate-index style exact
    /// reranking). 0 = off.
    pub rerank_top: u32,
    /// Optimization loops per reranked candidate; 0 = auto
    /// (`max(3, num_loops / 2)`). The winner continues to `num_loops` total.
    pub rerank_loops: u32,
}

impl Default for AlignConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            steps: 8,
            passes: 1,
            ssim_weight: 4.0,
            refine_top: 8,
            rerank_top: 0,
            rerank_loops: 0,
        }
    }
}

/// Configuration for a charset sprite sheet (matches config.json format).
#[derive(Debug, Clone, Deserialize)]
pub struct CharsetConfig {
    #[serde(rename = "charset_name")]
    pub charset_name: String,
    #[serde(rename = "image_path")]
    pub image_path: String,
    #[serde(rename = "slicesX")]
    pub slices_x: usize,
    #[serde(rename = "slicesY")]
    pub slices_y: usize,
    /// Indices of characters to exclude (v1 semantics: value `v` excludes the
    /// glyph at index `v - 1` in the blank-prepended list).
    #[serde(rename = "excludeChars", default)]
    pub exclude_chars: Vec<usize>,
    /// Indices of characters to include (takes precedence over `exclude_chars`
    /// when non-empty; the blank tile at index 0 is always kept).
    #[serde(rename = "includeChars", default)]
    pub include_chars: Vec<usize>,
    #[serde(rename = "whiteThreshold")]
    pub white_threshold: f32,
    #[serde(rename = "blankSpace")]
    pub blank_space: bool,
}

/// Search algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchAlgorithm {
    /// Always pick the best character (fastest convergence, no escape from local minima).
    Greedy,
    /// Simulated annealing: probabilistically accept worse characters to escape local minima.
    SimAnneal,
}

/// Initialization mode for character assignments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitMode {
    /// All positions start with the blank character (index 0).
    Blank,
    /// All positions are assigned random characters.
    Random,
}

/// Complete optimization configuration.
#[derive(Debug, Clone)]
pub struct OptimizeConfig {
    /// Path to charset directory (relative to assets/).
    pub charset: String,
    /// Path to target image.
    pub target: String,
    /// Layer configuration key in layers.json.
    pub layers: String,
    /// Number of characters per row.
    pub row_length: u32,
    /// Number of optimization passes per layer.
    pub num_loops: u32,
    /// Initial character assignment strategy.
    pub init_mode: InitMode,
    /// Asymmetry factor for MSE (penalizes too-light regions).
    pub asymmetry: f32,
    /// Spatial weighting of the loss and alignment score ("center focus").
    pub center_focus: CenterFocus,
    /// RNG seed for reproducible runs (random init, SA decisions, layer
    /// shuffle). `None` = nondeterministic.
    pub seed: Option<u64>,
    /// Search algorithm.
    pub search: SearchAlgorithm,
    /// Initial temperature for simulated annealing.
    pub init_temp: f32,
    /// Whether to shuffle layer order each loop.
    pub shuffle: bool,
    /// Output file path for the final mockup PNG.
    pub out_file: String,
    /// Pre-optimization auto-alignment sweep ("optimized cropping").
    pub align: AlignConfig,
    /// Skip the sweep and use this crop transform directly.
    pub align_override: Option<CropTransform>,
}

/// Statistics collected during optimization.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Stats {
    pub total_time_ns: u64,
    pub total_comparisons: u64,
    pub final_error: f32,
}

/// Complete optimization result.
#[derive(Debug, Clone)]
pub struct OptimizeResult {
    /// Final mockup image.
    pub mockup: Image,
    /// Individual layer images.
    pub layers: Layers,
    /// Character assignments per layer.
    pub choices: Choices,
    /// Error value after each layer pass.
    pub error_over_time: Vec<f32>,
    /// Optimization statistics.
    pub stats: Stats,
    /// Target image padding info.
    pub padding: Padding,
    /// Number of character rows in the output grid.
    pub num_rows: usize,
    /// Number of character columns in the output grid.
    pub num_cols: usize,
    /// Fractional layer offsets (for output formatting).
    pub fractional_offsets: Vec<FractionalOffset>,
    /// Layer offsets in pixels (for cropping/export).
    pub layer_offsets: Vec<LayerOffset>,
    /// Auto-alignment outcome, when the sweep (or a manual override) ran.
    pub alignment: Option<AlignmentResult>,
}
