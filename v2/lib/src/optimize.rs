//! Optimization orchestration.
//!
//! Coordinates the full optimization pipeline:
//! charset loading → target preprocessing → layer optimization loops → final output.

use std::sync::mpsc;
use std::time::Instant;

use ndarray::Axis;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::align;
use crate::charset::{prep_charset, LoadedCharset};
use crate::export::image_to_base64_png;
use crate::layers::{fractional_to_pixel, load_layer_configs};
use crate::render::{Ink, combine_layers, compute_background, render_all_layers, render_layer};
use crate::resize::resize_target;
use crate::search::{asymmetric_mse, optimize_layer_pass};
use crate::types::{
    AlignmentResult, AlignmentScore, Chars, CharShape, DitherMode, FractionalOffset, Image,
    InitMode, LayerOffset, Layers, OptimizeConfig, OptimizeResult, Padding, Stats,
};

/// Progress event emitted during optimization (for streaming).
///
/// Serialized with a `type` tag: `"type":"align"` while the auto-alignment
/// sweep evaluates candidates, `"type":"loop"` after each layer pass.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProgressEvent {
    /// One auto-alignment candidate finished evaluating.
    Align {
        /// Candidates evaluated so far.
        candidate: usize,
        /// Total candidates in the sweep.
        total: usize,
    },
    /// One layer optimization pass finished.
    Loop {
        loop_num: usize,
        num_loops: usize,
        layer_num: usize,
        num_layers: usize,
        error: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        mockup_b64: Option<String>,
    },
}

/// Run the full optimization pipeline.
pub fn optimize(
    config: OptimizeConfig,
) -> anyhow::Result<OptimizeResult> {
    optimize_inner(config, None)
}

/// Run optimization with optional progress streaming.
/// If `progress_tx` is provided, a ProgressEvent is sent after each layer pass.
pub fn optimize_with_progress(
    config: OptimizeConfig,
    progress_tx: mpsc::Sender<ProgressEvent>,
) -> anyhow::Result<OptimizeResult> {
    optimize_inner(config, Some(progress_tx))
}

fn optimize_inner(
    config: OptimizeConfig,
    progress_tx: Option<mpsc::Sender<ProgressEvent>>,
) -> anyhow::Result<OptimizeResult> {
    let start = Instant::now();

    let PreparedInputs {
        chars,
        ink,
        char_shape,
        target,
        padding,
        fractional_offsets,
        layer_offsets,
    } = load_inputs(&config)?;

    let (char_h, char_w) = char_shape;
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

    // Pre-optimization auto-alignment sweep ("optimized cropping", paper §3.4)
    // Scored over the image content only — not the padding border.
    let content = padding.content_rect((target.nrows(), target.ncols()));
    let progress_tx = progress_tx.map(std::sync::Mutex::new);
    let (best_transform, mut alignment) = match config.align_override {
        Some(t) => {
            log::info!("Using manual crop transform: {:?}", t);
            (
                Some(t),
                Some(AlignmentResult {
                    best: t,
                    steps: config.align.steps,
                    scores: Vec::new(),
                    rerank: None,
                }),
            )
        }
        None if config.align.enabled => {
            log::info!(
                "Auto-alignment: screening candidates (fine steps = 1/{} char)",
                config.align.steps
            );
            let progress = |completed: usize, total: usize| {
                log::debug!("Align: evaluated {}/{} candidates", completed, total);
                if let Some(tx) = progress_tx.as_ref() {
                    let _ = tx.lock().unwrap().send(ProgressEvent::Align {
                        candidate: completed,
                        total,
                    });
                }
            };
            let search = align::search(
                &target,
                &chars,
                &ink,
                &layer_offsets,
                &config.align,
                config.asymmetry,
                content,
                config.center_focus,
                false,
                Some(&progress),
            );
            log::info!(
                "Auto-alignment: best zoom {} shift ({}, {}), score {:.3}",
                search.result.best.zoom,
                search.result.best.shift_x,
                search.result.best.shift_y,
                search.result.scores.first().map(|s| s.score).unwrap_or(0.0),
            );
            let best = search.result.best;
            (Some(best), Some(search.result))
        }
        None => (None, None),
    };

    // Determine output shape with padding for layer offsets
    log::info!(
        "Layers: {} ({})",
        layer_offsets.len(),
        format_offsets(&fractional_offsets)
    );
    let max_offset_y = layer_offsets.iter().map(|o| o.0).max().unwrap_or(0);
    let max_offset_x = layer_offsets.iter().map(|o| o.1).max().unwrap_or(0);
    let out_shape = (target.nrows() + max_offset_y, target.ncols() + max_offset_x);
    let num_loops = config.num_loops;

    // Deluxe rerank: deep-evaluate the top candidates and keep the winner's
    // run state, which continues to `num_loops` instead of restarting.
    let rerank_top = if alignment
        .as_ref()
        .map_or(false, |a| !a.scores.is_empty() && config.align.rerank_top > 0)
    {
        (config.align.rerank_top as usize)
            .min(alignment.as_ref().unwrap().scores.len())
    } else {
        0
    };

    let mut reranked: Option<Vec<AlignmentScore>> = None;
    let mut winner: Option<(RunState, Image, (usize, usize, usize, usize))> = None;

    if rerank_top > 0 {
        let rerank_loops = if config.align.rerank_loops > 0 {
            config.align.rerank_loops
        } else {
            (num_loops / 2).max(3)
        }
        .min(num_loops);
        log::info!(
            "Deluxe rerank: top {} candidates × {} loops",
            rerank_top, rerank_loops
        );

        let mut scores = Vec::with_capacity(rerank_top);
        let mut best_score = f32::NEG_INFINITY;
        for (k, entry) in alignment.as_ref().unwrap().scores[..rerank_top]
            .iter()
            .enumerate()
        {
            let t = entry.transform;
            // Each candidate gets its own transform of the *original* target
            let (cand_target, cand_rect) =
                align::apply_crop(&target, &t, char_shape, content, config.align.steps);
            let cand_padded = pad_to(&cand_target, out_shape);
            let cand_weights =
                crate::weights::container_map(out_shape, cand_rect, config.center_focus);
            let mut cand_state = init_state(
                &cand_padded,
                &chars,
                &ink,
                &layer_offsets,
                num_cols,
                num_positions,
                &config,
            );
            run_loops(
                &mut cand_state,
                &cand_padded,
                &chars,
                &ink,
                &layer_offsets,
                cand_weights.as_ref(),
                num_cols,
                &config,
                rerank_loops,
                0,
                None,
            );
            // Score (and continue) the best state this candidate reached —
            // with SA the last pass need not be the best one.
            cand_state.restore_best(&ink, &chars, num_cols, &layer_offsets, out_shape);
            let (score, psnr, ssim) = align::score_mockup(
                &cand_target,
                &cand_state.mockup,
                cand_rect,
                config.center_focus,
                config.align.ssim_weight,
            );
            log::info!(
                "Deluxe rerank {}/{}: zoom {} shift ({}, {}) → score {:.3}",
                k + 1,
                rerank_top,
                t.zoom,
                t.shift_x,
                t.shift_y,
                score
            );
            if let Some(tx) = progress_tx.as_ref() {
                let _ = tx.lock().unwrap().send(ProgressEvent::Align {
                    candidate: k + 1,
                    total: rerank_top,
                });
            }
            scores.push(AlignmentScore {
                transform: t,
                psnr,
                ssim,
                score,
                refined: true,
            });
            if score > best_score {
                best_score = score;
                winner = Some((cand_state, cand_target, cand_rect));
            }
        }
        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        reranked = Some(scores);
    }

    // Resolve the target: the rerank winner's, or the quick best transform's
    let (target, content, mut state) = match winner {
        Some((state, target, rect)) => (target, rect, state),
        None => {
            let (target, content) = match best_transform {
                Some(t) => align::apply_crop(&target, &t, char_shape, content, config.align.steps),
                None => (target, content),
            };
            let target_padded = pad_to(&target, out_shape);
            let state = init_state(&target_padded, &chars, &ink, &layer_offsets, num_cols, num_positions, &config);
            (target, content, state)
        }
    };
    if let Some(ref mut a) = alignment {
        if let Some(scores) = reranked {
            a.best = scores[0].transform;
            a.rerank = Some(scores);
        }
    }

    let target_padded = pad_to(&target, out_shape);
    let loss_weights = crate::weights::container_map(out_shape, content, config.center_focus);
    let completed_loops = if rerank_top > 0 {
        state.error_over_time.len() / layer_offsets.len().max(1)
    } else {
        0
    };

    // Run the remaining optimization loops (all of them, unless the deluxe
    // rerank already advanced the winner)
    run_loops(
        &mut state,
        &target_padded,
        &chars,
        &ink,
        &layer_offsets,
        loss_weights.as_ref(),
        num_cols,
        &config,
        num_loops.saturating_sub(completed_loops as u32),
        completed_loops as u32,
        progress_tx.as_ref(),
    );

    // Simulated annealing accepts worse moves, so the last state is not
    // necessarily the best one: return the best state seen anywhere in the run
    // (including the rerank continuation). `error_over_time` keeps the true
    // trajectory; the returned art is its minimum point.
    state.restore_best(&ink, &chars, num_cols, &layer_offsets, out_shape);

    // Dither finishing stage (advanced): light-tone texture. It trades a
    // little pointwise error for low-frequency fidelity, so `final_error` is
    // recomputed from the returned art below.
    if config.dither.mode != DitherMode::None {
        let dither_stats = crate::dither::run(
            &mut state.mockup,
            &mut state.choices,
            &mut state.layers,
            &target_padded,
            &chars,
            &ink,
            &layer_offsets,
            num_cols,
            &config.dither,
        );
        log::info!(
            "Dither: seeded {} marks, accepted {} flips, LF error {:.6} → {:.6}",
            dither_stats.seeded,
            dither_stats.flipped,
            dither_stats.lf_before,
            dither_stats.lf_after
        );
    }

    let elapsed = start.elapsed();
    // Error of the returned artwork. Without dithering this is exactly the
    // minimum of `error_over_time`; with dithering it describes the
    // dither-refined art.
    let final_error = asymmetric_mse(
        &target_padded,
        &state.mockup,
        config.asymmetry,
        loss_weights.as_ref(),
    );

    Ok(OptimizeResult {
        mockup: state.mockup,
        layers: state.layers,
        choices: state.choices,
        error_over_time: state.error_over_time,
        stats: Stats {
            total_time_ns: elapsed.as_nanos() as u64,
            total_comparisons: state.total_comparisons,
            final_error,
        },
        padding,
        num_rows,
        num_cols,
        fractional_offsets,
        layer_offsets,
        alignment,
    })
}

/// Mutable state of an optimization run, so a run can be paused and continued
/// (the deluxe rerank keeps the winner's state and finishes it).
struct RunState {
    layers: Layers,
    choices: ndarray::Array2<u16>,
    mockup: Image,
    error_over_time: Vec<f32>,
    total_comparisons: u64,
    /// Best state seen anywhere in the run, measured by the error of the full
    /// state after each pass. Simulated annealing accepts worse moves, so the
    /// last state is not necessarily the best one — the result is restored
    /// from this snapshot at the end.
    best_choices: ndarray::Array2<u16>,
    best_error: f32,
}

impl RunState {
    /// Swap in the best state seen so far: re-render the layers and mockup
    /// from `best_choices` so the whole state is consistent again.
    fn restore_best(
        &mut self,
        ink: &Ink,
        chars: &Chars,
        num_cols: usize,
        layer_offsets: &[LayerOffset],
        out_shape: (usize, usize),
    ) {
        self.choices.assign(&self.best_choices);
        self.layers =
            render_all_layers(&self.best_choices, chars, num_cols, layer_offsets, out_shape);
        self.mockup = combine_layers(&self.layers, ink);
    }
}

/// Pad a target to the output shape (white = 1.0) for layer offsets.
fn pad_to(target: &Image, out_shape: (usize, usize)) -> Image {
    let mut padded = Image::from_elem(out_shape, 1.0f32);
    padded
        .slice_mut(ndarray::s![..target.nrows(), ..target.ncols()])
        .assign(target);
    padded
}

/// Initialize a run state (blank or random character assignments).
/// Seeded runs use a deterministic RNG.
fn init_state(
    target_padded: &Image,
    chars: &Chars,
    ink: &Ink,
    layer_offsets: &[LayerOffset],
    num_cols: usize,
    num_positions: usize,
    config: &OptimizeConfig,
) -> RunState {
    let num_layers = layer_offsets.len();
    let out_shape = (target_padded.nrows(), target_padded.ncols());
    let mut choices = ndarray::Array2::<u16>::zeros((num_layers, num_positions));

    let (layers, mockup) = if matches!(config.init_mode, InitMode::Random) {
        let mut rng = config
            .seed
            .map(rand::rngs::StdRng::seed_from_u64)
            .unwrap_or_else(rand::rngs::StdRng::from_entropy);
        for l in 0..num_layers {
            for p in 0..num_positions {
                choices[[l, p]] = rand::Rng::gen_range(&mut rng, 0..chars.shape()[0] as u16);
            }
        }
        let layers = render_all_layers(&choices, chars, num_cols, layer_offsets, out_shape);
        let mockup = combine_layers(&layers, ink);
        (layers, mockup)
    } else {
        // Blank — layers are all-white, choices are blank char (index 0)
        (
            Layers::from_elem((num_layers, out_shape.0, out_shape.1), 1.0),
            Image::from_elem(out_shape, 1.0),
        )
    };

    let best_choices = choices.clone();
    RunState {
        layers,
        choices,
        mockup,
        error_over_time: Vec::new(),
        total_comparisons: 0,
        best_choices,
        best_error: f32::INFINITY,
    }
}

/// Run `loops_to_run` optimization loops, continuing a run that already
/// completed `loops_done` loops (temperature schedule and progress numbering
/// continue where the previous run stopped).
#[allow(clippy::too_many_arguments)]
fn run_loops(
    state: &mut RunState,
    target_padded: &Image,
    chars: &Chars,
    ink: &Ink,
    layer_offsets: &[LayerOffset],
    loss_weights: Option<&Image>,
    num_cols: usize,
    config: &OptimizeConfig,
    loops_to_run: u32,
    loops_done: u32,
    progress_tx: Option<&std::sync::Mutex<mpsc::Sender<ProgressEvent>>>,
) {
    let num_layers = layer_offsets.len();
    let out_shape = (target_padded.nrows(), target_padded.ncols());

    // Layer order list (re-shuffled each loop)
    let mut shuffled_layers: Vec<(usize, LayerOffset)> = layer_offsets
        .iter()
        .enumerate()
        .map(|(i, &o)| (i, o))
        .collect();
    let mut shuffle_rng = config
        .seed
        .map(|s| rand::rngs::StdRng::seed_from_u64(s.wrapping_add(loops_done as u64)))
        .unwrap_or_else(rand::rngs::StdRng::from_entropy);

    // The starting state is also a candidate: with SA the very first pass can
    // accept a worse move, so "best seen" must include the initialization.
    if !state.best_error.is_finite() {
        state.best_error = asymmetric_mse(
            target_padded,
            &state.mockup,
            config.asymmetry,
            loss_weights,
        );
        state.best_choices.assign(&state.choices);
    }

    for i in 0..loops_to_run {
        let loop_num = loops_done + i;
        let temperature = if config.init_temp > 0.0 {
            config.init_temp / (loop_num + 1) as f32
        } else {
            0.0
        };

        if config.shuffle {
            shuffled_layers.shuffle(&mut shuffle_rng);
        }

        for &(layer_num, layer_offset) in &shuffled_layers {
            // Compute background = product of all OTHER layers
            let bg = compute_background(&state.layers, layer_num, ink);

            // Get this layer's choices as a mutable 1D slice
            let mut layer_choices = state.choices.index_axis(Axis(0), layer_num).to_owned();

            // Per-pass RNG seed (deterministic when the run is seeded)
            let pass_seed = config.seed.map_or_else(rand::random::<u64>, |s| {
                s.wrapping_add(((loop_num as u64) << 32) | layer_num as u64)
            });

            // Run optimization pass
            let (comparisons, cell_err) = optimize_layer_pass(
                &bg,
                &mut state.mockup,
                target_padded,
                ink,
                &mut layer_choices,
                layer_offset,
                config.asymmetry,
                config.search,
                temperature,
                loss_weights,
                pass_seed,
            );

            // Update choices
            state
                .choices
                .index_axis_mut(Axis(0), layer_num)
                .assign(&layer_choices);

            // Re-render this layer with updated choices
            let updated_layer = render_layer(
                layer_choices.as_slice().unwrap(),
                chars,
                num_cols,
                layer_offset,
                out_shape,
            );
            state
                .layers
                .index_axis_mut(Axis(0), layer_num)
                .assign(&updated_layer);

            // Update mockup
            state.mockup = combine_layers(&state.layers, ink);

            // Error of the full state after this pass. The trajectory
            // (`error_over_time`, the progress chart) and the best-seen
            // tracking share this one measure, so the returned artwork is
            // exactly the minimum of the plotted curve.
            let state_error = asymmetric_mse(
                target_padded,
                &state.mockup,
                config.asymmetry,
                loss_weights,
            );
            if state_error < state.best_error {
                state.best_error = state_error;
                state.best_choices.assign(&state.choices);
            }

            state.total_comparisons += comparisons;
            state.error_over_time.push(state_error);

            // Emit progress event if streaming (mockup only on first layer of each loop)
            if let Some(tx) = progress_tx {
                let mockup_b64 = if layer_num == 0 {
                    image_to_base64_png(&state.mockup).ok()
                } else {
                    None
                };
                let _ = tx.lock().unwrap().send(ProgressEvent::Loop {
                    loop_num: loop_num as usize + 1,
                    num_loops: config.num_loops as usize,
                    layer_num: layer_num + 1,
                    num_layers,
                    error: state_error,
                    mockup_b64,
                });
            }

            log::debug!(
                "Loop {}/{}, layer {}/{} (offset {:?}): err={:.6} (cell {:.6}), comps={}",
                loop_num + 1,
                config.num_loops,
                layer_num + 1,
                num_layers,
                layer_offset,
                state_error,
                cell_err,
                comparisons,
            );
        }
    }
}

/// Inputs shared by the optimization pipeline and the alignment sweep.
struct PreparedInputs {
    chars: Chars,
    /// Ink model resolved from `chars` and the configured floor.
    ink: Ink,
    char_shape: CharShape,
    /// Target image, resized and padded to the character grid.
    target: Image,
    padding: Padding,
    fractional_offsets: Vec<FractionalOffset>,
    layer_offsets: Vec<LayerOffset>,
}

/// Load the charset and target image, resize the target to the character grid,
/// and resolve the layer offsets. Shared by [`optimize`] and [`align_search`].
fn load_inputs(config: &OptimizeConfig) -> anyhow::Result<PreparedInputs> {
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

    // Load target image and resize it to the character grid
    let target_img = image::open(&target_path)?.into_luma8();
    let (target, padding) = resize_target(
        &target_img,
        config.row_length,
        char_shape,
        (x_change, y_change),
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

    let ink = Ink::new(config.ink_floor, &chars);
    Ok(PreparedInputs {
        chars,
        ink,
        char_shape,
        target,
        padding,
        fractional_offsets,
        layer_offsets,
    })
}

/// Run only the auto-alignment sweep and return its results.
///
/// Used by the CLI's `--align-only` mode. Unlike [`optimize`], this ignores
/// `config.align.enabled` — it always runs the sweep. When `keep_mockups` is
/// set, each candidate's quick mockup is returned alongside the scores.
pub fn align_search(
    config: &OptimizeConfig,
    keep_mockups: bool,
) -> anyhow::Result<crate::types::AlignmentSearch> {
    let inputs = load_inputs(config)?;
    let progress = |completed: usize, total: usize| {
        log::info!("Align: evaluated {}/{} candidates", completed, total);
    };
    let content = inputs
        .padding
        .content_rect((inputs.target.nrows(), inputs.target.ncols()));
    Ok(align::search(
        &inputs.target,
        &inputs.chars,
        &inputs.ink,
        &inputs.layer_offsets,
        &config.align,
        config.asymmetry,
        content,
        config.center_focus,
        keep_mockups,
        Some(&progress),
    ))
}

fn format_offsets(offsets: &[(f32, f32)]) -> String {
    offsets
        .iter()
        .map(|(y, x)| format!("({:.2},{:.2})", y, x))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::asymmetric_mse;
    use crate::types::SearchAlgorithm;
    use ndarray::arr3;

    fn test_chars() -> Chars {
        // 3 chars: blank (all white), half-dark columns, all-dark.
        arr3(&[
            [[1.0, 1.0], [1.0, 1.0]],
            [[1.0, 0.5], [1.0, 0.5]],
            [[0.5, 0.5], [0.5, 0.5]],
        ])
    }

    fn test_config() -> OptimizeConfig {
        OptimizeConfig {
            charset: String::new(),
            target: String::new(),
            layers: "1x1".to_string(),
            row_length: 2,
            num_loops: 1,
            init_mode: InitMode::Blank,
            asymmetry: 0.1,
            center_focus: crate::types::CenterFocus::Off,
            ink_floor: crate::types::InkFloor::Fixed(0.0),
            dither: crate::types::DitherConfig::default(),
            seed: Some(1),
            search: SearchAlgorithm::Greedy,
            init_temp: 0.0,
            shuffle: false,
            out_file: String::new(),
            align: crate::types::AlignConfig::default(),
            align_override: None,
        }
    }

    /// With SA at a huge temperature nearly any move is accepted, so late
    /// passes can easily be worse than earlier ones. The run must return the
    /// best state it saw, not the last one.
    #[test]
    fn test_returns_best_seen_state_not_last() {
        let chars = test_chars();
        let ink = Ink::new(crate::types::InkFloor::Fixed(0.0), &chars);
        // 2x2 grid of 2x2 cells.
        let target = ndarray::arr2(&[
            [0.2f32, 0.4, 0.6, 0.8],
            [0.9, 0.1, 0.3, 0.5],
            [0.7, 0.2, 0.4, 0.6],
            [0.1, 0.8, 0.3, 0.9],
        ]);
        let layer_offsets: Vec<LayerOffset> = vec![(0, 0)];
        let num_cols = 2;
        let num_positions = 4;
        let out_shape = (target.nrows(), target.ncols());
        let config = OptimizeConfig {
            num_loops: 6,
            init_mode: InitMode::Random,
            init_temp: 1.0e9,
            search: SearchAlgorithm::SimAnneal,
            seed: Some(11),
            ..test_config()
        };

        let mut state = init_state(
            &target,
            &chars,
            &ink,
            &layer_offsets,
            num_cols,
            num_positions,
            &config,
        );
        run_loops(
            &mut state,
            &target,
            &chars,
            &ink,
            &layer_offsets,
            None,
            num_cols,
            &config,
            config.num_loops,
            0,
            None,
        );
        state.restore_best(&ink, &chars, num_cols, &layer_offsets, out_shape);

        // The returned state is consistent with the tracked best…
        let restored_err = asymmetric_mse(&target, &state.mockup, config.asymmetry, None);
        assert!((restored_err - state.best_error).abs() < 1e-6);
        assert_eq!(&state.choices, &state.best_choices);
        // …it is at least as good as any pass on the plotted curve…
        let min_pass = state
            .error_over_time
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        assert!(state.best_error <= min_pass + 1e-7);
        // …and this seeded run really does end above its best (regression
        // guard: without best-state restore, final_error == last pass).
        assert!(
            state.best_error < *state.error_over_time.last().unwrap(),
            "expected the seeded SA run to end above its best pass"
        );
    }

    /// The starting state is a candidate too: if every pass is a regression,
    /// the initialization must win.
    #[test]
    fn test_best_can_be_the_initial_state() {
        let chars = test_chars();
        let ink = Ink::new(crate::types::InkFloor::Fixed(0.0), &chars);
        // A target the blank init matches exactly (all white).
        let target = Image::from_elem((4, 4), 1.0f32);
        let layer_offsets: Vec<LayerOffset> = vec![(0, 0)];
        let num_cols = 2;
        let config = OptimizeConfig {
            num_loops: 2,
            init_mode: InitMode::Blank,
            init_temp: 1.0e9,
            search: SearchAlgorithm::SimAnneal,
            seed: Some(3),
            ..test_config()
        };

        let mut state = init_state(&target, &chars, &ink, &layer_offsets, num_cols, 4, &config);
        let init_error =
            asymmetric_mse(&target, &state.mockup, config.asymmetry, None);
        assert_eq!(init_error, 0.0);

        run_loops(
            &mut state,
            &target,
            &chars,
            &ink,
            &layer_offsets,
            None,
            num_cols,
            &config,
            config.num_loops,
            0,
            None,
        );
        state.restore_best(&ink, &chars, num_cols, &layer_offsets, (4, 4));

        assert_eq!(state.best_error, 0.0);
        assert_eq!(&state.choices, &state.best_choices);
        // Everything the passes did was a regression from the init.
        assert!(
            state
                .error_over_time
                .iter()
                .all(|&e| e >= state.best_error)
        );
    }
}
