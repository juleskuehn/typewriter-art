//! Layer configuration.
//!
//! Loads `layers.json` and converts fractional offsets to pixel offsets.

use std::collections::HashMap;
use std::path::Path;

use crate::types::{FractionalOffset, LayerOffset};

/// Load layer offset configurations from a JSON file.
///
/// Returns a map from layer name (e.g., "4x2") to list of fractional offsets.
pub fn load_layer_configs(path: &Path) -> anyhow::Result<HashMap<String, Vec<FractionalOffset>>> {
    let content = std::fs::read_to_string(path)?;
    let configs: HashMap<String, Vec<[f32; 2]>> = serde_json::from_str(&content)?;

    Ok(configs
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().map(|[y, x]| (y, x)).collect()))
        .collect())
}

/// Convert fractional layer offsets to pixel offsets.
///
/// Fractional offsets are given as fractions of character height and width.
pub fn fractional_to_pixel(
    offsets: &[FractionalOffset],
    char_height: usize,
    char_width: usize,
) -> Vec<LayerOffset> {
    offsets
        .iter()
        .map(|&(y_frac, x_frac)| {
            (
                (char_height as f32 * y_frac).round() as usize,
                (char_width as f32 * x_frac).round() as usize,
            )
        })
        .collect()
}
