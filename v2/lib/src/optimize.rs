//! Optimization orchestration.
//!
//! Coordinates the full optimization pipeline:
//! charset loading → target preprocessing → layer optimization loops → final output.

use std::time::Instant;

use ndarray::Axis;
use rand::seq::SliceRandom;

use crate::charset::{prep_charset, LoadedCharset};
use crate::layers::{fractional_to_pixel, load_layer_configs};
use crate::render::{combine_layers, compute_background, render_all_layers, render_layer};
use crate::resize::resize_target;
use crate::search::optimize_layer_pass;
use crate::types::{
    CharShape, Image, InitMode, LayerOffset, Layers, OptimizeConfig,
    OptimizeResult, Stats,
};

/// Run the full optimization pipeline.
pub fn optimize(config: OptimizeConfig) -> anyhow::Result<OptimizeResult> {
    let start = Instant::now();

    // Load charset
    let base_path = std::env::current_dir()?;
    let assets_path = base_path.join("assets");
    let charset_path = if std::path::Path::new(&config.charset).is_dir() {
        std::path::PathBuf::from(&config.charset)
    } else {
        let under_assets = assets_path.join("charsets").join(&config.charset);
        if under_assets.is_dir() {
            under_assets
        } else {
            assets_path.join(&config.charset)
        }
    };

    // Resolve target path: try assets/ then raw path
    let target_path = {
        let p = std::path::Path::new(&config.target);
        if p.is_file() {
            p.to_path_buf()
        } else {
            let under_assets = assets_path.join(&config.target);
            if under_assets.is_file() {
                under_assets
            } else {
                p.to_path_buf()
            }
        }
    };

    let LoadedCharset {
        chars,
        x_change,
        y_change,
    } = prep_charset(&charset_path)?;

    let char_h = chars.shape()[1];
    let char_w = chars.shape()[2];
    let char_shape: CharShape = (char_h, char_w);

    // Load target image
    let target_img = image::open(&target_path)?.into_luma8();

    // Resize target to character grid
    let (target, padding) = resize_target(
        &target_img,
        config.row_length,
        char_shape,
        (x_change, y_change),
    );

    let num_rows = target.nrows() / char_h;
    let num_cols = target.ncols() / char_w;
    let num_positions = num_rows * num_cols;
    log::info!(
        "Grid: {} rows × {} cols = {} positions, char size: {}×{}",
        num_rows,
        num_cols,
        num_positions,
        char_h,
        char_w,
    );
    log::info!(
        "Target image: {}×{} (after resize + pad)",
        target.nrows(),
        target.ncols(),
    );

    // Load layer offsets
    let layers_json_path = assets_path.join("layers.json");
    let all_layer_configs = load_layer_configs(&layers_json_path)?;
    let fractional_offsets = all_layer_configs
        .get(&config.layers)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Layer config '{}' not found in layers.json. Available: {:?}",
                config.layers,
                all_layer_configs.keys().collect::<Vec<_>>()
            )
        })?
        .clone();
    let layer_offsets: Vec<LayerOffset> =
        fractional_to_pixel(&fractional_offsets, char_h, char_w);

    let num_layers = layer_offsets.len();
    log::info!(
        "Layers: {} ({})",
        num_layers,
        format_offsets(&fractional_offsets)
    );

    // Determine output shape with padding for layer offsets
    let max_offset_y = layer_offsets.iter().map(|o| o.0).max().unwrap_or(0);
    let max_offset_x = layer_offsets.iter().map(|o| o.1).max().unwrap_or(0);
    let out_h = target.nrows() + max_offset_y;
    let out_w = target.ncols() + max_offset_x;

    // Pad target to output size (fill with white = 1.0)
    let target_padded = {
        let mut padded = Image::from_elem((out_h, out_w), 1.0f32);
        let mut view = padded.slice_mut(ndarray::s![..target.nrows(), ..target.ncols()]);
        view.assign(&target);
        padded
    };

    // Initialize mockup (all white)
    let mut mockup = Image::from_elem((out_h, out_w), 1.0f32);

    // Initialize choices
    let mut choices = ndarray::Array2::<u16>::zeros((num_layers, num_positions));

    // Initialize layers based on init_mode
    let mut layers: Layers;
    if matches!(config.init_mode, InitMode::Random) {
        let mut rng = rand::thread_rng();
        for l in 0..num_layers {
            for p in 0..num_positions {
                choices[[l, p]] = rand::Rng::gen_range(&mut rng, 0..chars.shape()[0] as u16);
            }
        }
        layers = render_all_layers(
            &choices,
            &chars,
            num_cols,
            &layer_offsets,
            (out_h, out_w),
        );
        mockup = combine_layers(&layers);
    } else {
        // Blank — layers are all-white, choices are blank char (index 0)
        layers = Layers::from_elem((num_layers, out_h, out_w), 1.0f32);
    }

    let num_loops = config.num_loops as usize;
    // Error over time: one entry per layer pass
    let mut error_over_time: Vec<f32> = Vec::with_capacity(num_loops * num_layers);
    let mut total_comparisons: u64 = 0;

    // Create shuffled layer order list (re-shuffled each loop)
    let mut shuffled_layers: Vec<(usize, LayerOffset)> = layer_offsets
        .iter()
        .enumerate()
        .map(|(i, &o)| (i, o))
        .collect();

    let mut rng = rand::thread_rng();

    for loop_num in 0..num_loops {
        let temperature = if config.init_temp > 0.0 {
            config.init_temp / (loop_num + 1) as f32
        } else {
            0.0
        };

        if config.shuffle {
            shuffled_layers.shuffle(&mut rng);
        }

        for &(layer_num, layer_offset) in &shuffled_layers {
            // Compute background = product of all OTHER layers
            let bg = compute_background(&layers, layer_num);

            // Get this layer's choices as a mutable 1D slice
            let mut layer_choices = choices
                .index_axis(Axis(0), layer_num)
                .to_owned();

            // Run optimization pass
            let (comparisons, mean_err) = optimize_layer_pass(
                &bg,
                &mut mockup,
                &target_padded,
                &chars,
                &mut layer_choices,
                layer_offset,
                config.asymmetry,
                config.search,
                temperature,
            );

            // Update choices
            choices
                .index_axis_mut(Axis(0), layer_num)
                .assign(&layer_choices);

            // Re-render this layer with updated choices
            let updated_layer = render_layer(
                layer_choices.as_slice().unwrap(),
                &chars,
                num_cols,
                layer_offset,
                (out_h, out_w),
            );
            layers
                .index_axis_mut(Axis(0), layer_num)
                .assign(&updated_layer);

            // Update mockup
            mockup = combine_layers(&layers);

            total_comparisons += comparisons;
            error_over_time.push(mean_err);

            log::debug!(
                "Loop {}/{}, layer {}/{} (offset {:?}): err={:.6}, comps={}",
                loop_num + 1,
                num_loops,
                layer_num + 1,
                num_layers,
                layer_offset,
                mean_err,
                comparisons,
            );
        }
    }

    let elapsed = start.elapsed();
    let final_error = error_over_time.last().copied().unwrap_or(0.0);

    Ok(OptimizeResult {
        mockup,
        layers,
        choices,
        error_over_time,
        stats: Stats {
            total_time_ns: elapsed.as_nanos() as u64,
            total_comparisons,
            final_error,
        },
        padding,
        num_rows,
        num_cols,
        fractional_offsets,
    })
}

fn format_offsets(offsets: &[(f32, f32)]) -> String {
    offsets
        .iter()
        .map(|(y, x)| format!("({:.2},{:.2})", y, x))
        .collect::<Vec<_>>()
        .join(", ")
}
