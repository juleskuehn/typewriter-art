//! Charset loading and preprocessing.
//!
//! Ports `chop_charset` and `prep_charset` from the Python codebase.
//! Loads a sprite sheet PNG, slices it into individual character glyphs,
//! and normalizes to [0.0, 1.0] float32 arrays.

use std::path::Path;

use ndarray::{Array3, Axis};

use crate::types::{Chars, CharsetConfig};

/// Result of loading and preprocessing a charset.
pub struct LoadedCharset {
    /// Character glyphs as (num_chars, height, width) with values in [0.0, 1.0].
    pub chars: Chars,
    /// Ratio of original slice width to adjusted (even) slice width.
    pub x_change: f32,
    /// Ratio of original slice height to adjusted (even) slice height.
    pub y_change: f32,
}

/// Load a charset from its configuration directory.
///
/// Reads `config.json` from the directory, loads the sprite sheet image,
/// slices it into individual characters, and returns the normalized glyph array.
pub fn prep_charset(config_dir: &Path) -> anyhow::Result<LoadedCharset> {
    let config_path = config_dir.join("config.json");
    let config: CharsetConfig =
        serde_json::from_reader(std::fs::File::open(&config_path)?)?;

    let image_path = config_dir.join(&config.image_path);
    chop_charset(&image_path, &config)
}

/// Chop a sprite sheet into individual character glyphs.
///
/// Port of `chop_charset` from utils.py.
/// Returns normalized character glyphs along with the aspect ratio change factors.
pub fn chop_charset(
    image_path: &Path,
    config: &CharsetConfig,
) -> anyhow::Result<LoadedCharset> {
    let img = image::open(image_path)?.into_luma8();

    let (img_w, img_h) = img.dimensions();
    let slices_x = config.slices_x;
    let slices_y = config.slices_y;

    // Step sizes in the original image
    let step_x = img_w as f32 / slices_x as f32;
    let step_y = img_h as f32 / slices_y as f32;

    // Round up to integers, force even
    let mut new_step_x = step_x.ceil() as usize;
    let mut new_step_y = step_y.ceil() as usize;
    if new_step_x % 2 != 0 {
        new_step_x += 1;
    }
    if new_step_y % 2 != 0 {
        new_step_y += 1;
    }

    let x_change = step_x / new_step_x as f32;
    let y_change = step_y / new_step_y as f32;

    // Resize the sprite sheet to the adjusted character dimensions
    let new_w = (new_step_x * slices_x) as u32;
    let new_h = (new_step_y * slices_y) as u32;
    let resized = crate::resize::resize_area(&img, new_w, new_h);

    // Convert to float32 [0.0, 1.0]
    let pixels: Vec<f32> = resized
        .pixels()
        .map(|p| p.0[0] as f32 / 255.0)
        .collect();

    // Create an ndarray view of the full resized sprite sheet
    let sheet =
        ndarray::Array2::from_shape_vec((new_h as usize, new_w as usize), pixels)?;

    // Slice individual characters
    let mut glyphs: Vec<ndarray::Array2<f32>> = Vec::new();

    for y in 0..slices_y {
        for x in 0..slices_x {
            let start_x = x * new_step_x;
            let start_y = y * new_step_y;
            let end_x = (x + 1) * new_step_x;
            let end_y = (y + 1) * new_step_y;

            let char_slice = sheet
                .slice(ndarray::s![start_y..end_y, start_x..end_x])
                .to_owned();

            // Skip characters that are mostly white (blank/empty)
            let mean = char_slice.mean().unwrap_or(1.0);
            if mean < config.white_threshold {
                glyphs.push(char_slice);
            }
        }
    }

    // Prepend blank tile (all white) as index 0
    if config.blank_space {
        glyphs.insert(
            0,
            ndarray::Array2::from_elem((new_step_y, new_step_x), 1.0f32),
        );
    }

    // Apply include/exclude filtering (v1 `chop_charset` semantics)
    glyphs = apply_include_exclude(
        glyphs,
        &config.include_chars,
        &config.exclude_chars,
    );

    // Stack into 3D array: (num_chars, height, width)
    let num_chars = glyphs.len();
    let h = glyphs[0].nrows();
    let w = glyphs[0].ncols();

    let mut chars = Array3::zeros((num_chars, h, w));
    for (i, glyph) in glyphs.into_iter().enumerate() {
        chars
            .index_axis_mut(Axis(0), i)
            .assign(&glyph);
    }

    log::info!(
        "Charset '{}': {} glyphs of {}×{} (from {}×{} sprite sheet)",
        config.charset_name,
        num_chars,
        h,
        w,
        img_w,
        img_h,
    );

    Ok(LoadedCharset {
        chars,
        x_change,
        y_change,
    })
}

/// Apply the v1 `chop_charset` include/exclude filter.
///
/// Call this *after* prepending the blank tile, so index 0 is the blank.
/// Port of v1:
/// - `includeChars`: `[c for i, c in enumerate(chars) if i in includeChars or i == 0]`
/// - `excludeChars`: `[c for i, c in enumerate(chars) if i + 1 not in excludeChars]`
///   (i.e. exclude value `v` drops the glyph at index `v - 1`)
fn apply_include_exclude(
    glyphs: Vec<ndarray::Array2<f32>>,
    include_chars: &[usize],
    exclude_chars: &[usize],
) -> Vec<ndarray::Array2<f32>> {
    if !include_chars.is_empty() {
        glyphs
            .into_iter()
            .enumerate()
            .filter(|(i, _)| *i == 0 || include_chars.contains(i))
            .map(|(_, g)| g)
            .collect()
    } else if !exclude_chars.is_empty() {
        glyphs
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !exclude_chars.contains(&(i + 1)))
            .map(|(_, g)| g)
            .collect()
    } else {
        glyphs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glyphs(n: usize) -> Vec<ndarray::Array2<f32>> {
        // Distinguish glyphs by fill value 0.1 * (i + 1)
        (0..n)
            .map(|i| ndarray::Array2::from_elem((2, 2), 0.1 * (i + 1) as f32))
            .collect()
    }

    fn values(gs: &[ndarray::Array2<f32>]) -> Vec<f32> {
        gs.iter().map(|g| g[[0, 0]]).collect()
    }

    #[test]
    fn test_exclude_chars_semantics() {
        // 5 glyphs (index 0 = blank). excludeChars value v drops index v-1,
        // so excluding 2 and 4 drops indices 1 and 3.
        let out = apply_include_exclude(glyphs(5), &[], &[2, 4]);
        assert_eq!(values(&out), vec![0.1, 0.3, 0.5]);
    }

    #[test]
    fn test_include_chars_semantics() {
        // includeChars keeps only the listed indices plus the blank at 0.
        let out = apply_include_exclude(glyphs(5), &[2, 4], &[9, 9, 9]);
        assert_eq!(values(&out), vec![0.1, 0.3, 0.5]);
    }

    #[test]
    fn test_no_filtering() {
        let out = apply_include_exclude(glyphs(4), &[], &[]);
        assert_eq!(values(&out), vec![0.1, 0.2, 0.3, 0.4]);
    }
}
