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
    /// Search algorithm.
    pub search: SearchAlgorithm,
    /// Initial temperature for simulated annealing.
    pub init_temp: f32,
    /// Whether to shuffle layer order each loop.
    pub shuffle: bool,
    /// Output file path for the final mockup PNG.
    pub out_file: String,
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
}
