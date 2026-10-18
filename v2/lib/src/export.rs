//! Export optimization results to files.
//!
//! Handles saving mockup PNGs, individual layer PNGs, choices.json,
//! error-over-time data, and JSON-serializable result exports.

use std::io::Write;
use std::path::Path;

use serde::Serialize;

use crate::types::OptimizeResult;

/// Save the mockup image to a PNG file.
pub fn save_mockup(result: &OptimizeResult, path: &Path) -> anyhow::Result<()> {
    let img = float_image_to_gray(&result.mockup);
    img.save(path)?;
    Ok(())
}

/// Save individual layer images as PNG files.
///
/// Files are named `layer_{i}_{y_offset}_{x_offset}.png`.
pub fn save_layers(
    result: &OptimizeResult,
    dir: &Path,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;

    for l in 0..result.layers.shape()[0] {
        let layer_img = result.layers.index_axis(ndarray::Axis(0), l);
        let img = float_image_to_gray(&layer_img.to_owned());
        let (y_off, x_off) = result.fractional_offsets[l];
        let filename = format!("layer_{}_{}_{}.png", l, y_off, x_off);
        img.save(dir.join(&filename))?;
    }

    Ok(())
}

/// Save the character choices as a JSON file.
///
/// Format matches v1 Python output: a dict mapping layer keys to 2D arrays
/// of character indices.
pub fn save_choices(result: &OptimizeResult, path: &Path) -> anyhow::Result<()> {
    let mut choices_dict = serde_json::Map::new();

    for l in 0..result.choices.shape()[0] {
        let key = format!(
            "layer{}_{}_{}",
            l, result.fractional_offsets[l].0, result.fractional_offsets[l].1
        );
        let row = result.choices.index_axis(ndarray::Axis(0), l);
        let grid: Vec<Vec<u16>> = (0..result.num_rows)
            .map(|r| {
                row.slice(ndarray::s![r * result.num_cols..(r + 1) * result.num_cols])
                    .iter()
                    .copied()
                    .collect()
            })
            .collect();
        choices_dict.insert(key, serde_json::to_value(grid)?);
    }

    let json = serde_json::to_string_pretty(&choices_dict)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Save the statistics as a text file.
pub fn save_stats(result: &OptimizeResult, path: &Path) -> anyhow::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(
        f,
        "Total time: {:.3} s",
        result.stats.total_time_ns as f64 / 1e9
    )?;
    writeln!(f, "Total comparisons: {}", result.stats.total_comparisons)?;
    if result.stats.total_comparisons > 0 {
        writeln!(
            f,
            "Time per comparison: {:.3} µs",
            result.stats.total_time_ns as f64 / result.stats.total_comparisons as f64 / 1000.0
        )?;
    }
    writeln!(f, "Final error: {:.6}", result.stats.final_error)?;
    Ok(())
}

/// Save error-over-time data as CSV (for plotting).
pub fn save_error_over_time(result: &OptimizeResult, path: &Path) -> anyhow::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "iteration,error")?;
    for (i, err) in result.error_over_time.iter().enumerate() {
        writeln!(f, "{},{}", i, err)?;
    }
    Ok(())
}

/// Convert a float image [0.0, 1.0] to a grayscale image.
fn float_image_to_gray(img: &crate::types::Image) -> image::GrayImage {
    let (h, w) = (img.nrows(), img.ncols());
    let mut gray = image::GrayImage::new(w as u32, h as u32);

    for y in 0..h {
        for x in 0..w {
            let val = img[[y, x]];
            let clamped = (val.clamp(0.0, 1.0) * 255.0) as u8;
            gray.put_pixel(x as u32, y as u32, image::Luma([clamped]));
        }
    }

    gray
}

/// Crop the mockup to remove padding, matching v1 behavior.
///
/// In v1 (directory mode), the final mockup is cropped by the maximum layer
/// offset in each dimension: `mockup[:-max_offset_y, :-max_offset_x]`.
/// The white border added during target preprocessing is *not* removed,
/// because layer characters can straddle it.
pub fn crop_mockup(result: &OptimizeResult) -> crate::types::Image {
    let max_offset_y = result
        .layer_offsets
        .iter()
        .map(|o| o.0)
        .max()
        .unwrap_or(0);
    let max_offset_x = result
        .layer_offsets
        .iter()
        .map(|o| o.1)
        .max()
        .unwrap_or(0);

    let h = result.mockup.nrows();
    let w = result.mockup.ncols();

    // The crop matches how v1 saves the final image:
    // remove the padding at bottom/right (from layer offsets) only
    result
        .mockup
        .slice(ndarray::s![
            ..h.saturating_sub(max_offset_y),
            ..w.saturating_sub(max_offset_x)
        ])
        .to_owned()
}

/// Encode a float image as a base64 PNG string (for JSON output).
pub fn image_to_base64_png(img: &crate::types::Image) -> anyhow::Result<String> {
    use base64::Engine as _;
    let gray = float_image_to_gray(img);
    let mut buf = std::io::Cursor::new(Vec::new());
    gray.write_to(&mut buf, image::ImageFormat::Png)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(buf.into_inner()))
}

/// Serializable result for JSON output (used by `--json` CLI flag and web API).
#[derive(Serialize)]
pub struct JsonResult {
    pub mockup_base64: String,
    pub layers_base64: Vec<String>,
    pub num_rows: usize,
    pub num_cols: usize,
    pub fractional_offsets: Vec<(f32, f32)>,
    pub stats: crate::types::Stats,
    pub error_over_time: Vec<f32>,
    pub choices: serde_json::Value,
}

/// Build a JSON-serializable result from an optimization result.
pub fn build_json_result(result: &OptimizeResult) -> anyhow::Result<JsonResult> {
    // Encode mockup as base64 PNG
    let mockup_base64 = image_to_base64_png(&result.mockup)?;

    // Encode each layer as base64 PNG
    let mut layers_base64 = Vec::with_capacity(result.layers.shape()[0]);
    for l in 0..result.layers.shape()[0] {
        let layer_img = result.layers.index_axis(ndarray::Axis(0), l).to_owned();
        layers_base64.push(image_to_base64_png(&layer_img)?);
    }

    // Build choices dict (same format as save_choices)
    let mut choices_dict = serde_json::Map::new();
    for l in 0..result.choices.shape()[0] {
        let key = format!(
            "layer{}_{}_{}",
            l, result.fractional_offsets[l].0, result.fractional_offsets[l].1
        );
        let row = result.choices.index_axis(ndarray::Axis(0), l);
        let grid: Vec<Vec<u16>> = (0..result.num_rows)
            .map(|r| {
                row.slice(ndarray::s![r * result.num_cols..(r + 1) * result.num_cols])
                    .iter()
                    .copied()
                    .collect()
            })
            .collect();
        choices_dict.insert(key, serde_json::to_value(grid)?);
    }

    Ok(JsonResult {
        mockup_base64,
        layers_base64,
        num_rows: result.num_rows,
        num_cols: result.num_cols,
        fractional_offsets: result.fractional_offsets.clone(),
        stats: result.stats.clone(),
        error_over_time: result.error_over_time.clone(),
        choices: serde_json::Value::Object(choices_dict),
    })
}
