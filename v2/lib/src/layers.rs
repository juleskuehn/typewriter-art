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
/// Truncates (like v1's `int(charHeight * o)`), so e.g. 0.25 × 18 → 4, not 5.
pub fn fractional_to_pixel(
    offsets: &[FractionalOffset],
    char_height: usize,
    char_width: usize,
) -> Vec<LayerOffset> {
    offsets
        .iter()
        .map(|&(y_frac, x_frac)| {
            // Multiply in f64, truncate like v1's `int()`
            (
                (char_height as f64 * f64::from(y_frac)) as usize,
                (char_width as f64 * f64::from(x_frac)) as usize,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractional_to_pixel_truncates() {
        // v1 uses int(), which truncates: 0.25 × 18 → 4 (not 5),
        // 0.25 × 16 → 4 exactly.
        let offsets = [(0.0, 0.0), (0.5, 0.5), (0.25, 0.75), (0.125, 0.375)];
        let pixel = fractional_to_pixel(&offsets, 18, 20);
        assert_eq!(pixel[0], (0, 0));
        assert_eq!(pixel[1], (9, 10));
        assert_eq!(pixel[2], (4, 15));
        assert_eq!(pixel[3], (2, 7)); // 0.375 × 20 = 7.5 → 7
    }
}
