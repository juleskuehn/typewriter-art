//! Pre-optimization auto-alignment ("optimized cropping").
//!
//! Port of the `autoCrop` routine from v1 `kword.py` (removed from `src/` in
//! commit 85cba3d; described in the GI 2021 paper §3.4 "Optimized cropping").
//!
//! Characters can only be placed on a fixed character grid, so how the target
//! image happens to line up with that grid strongly affects shape matching —
//! this is "the primary factor when the typed image is small". Before the
//! iterative optimization starts, we therefore try every combination of small
//! zoom and shift values (`steps³` candidates; 64 at the default `steps = 4`),
//! run a single quick greedy optimization cycle for each cropped target, and
//! keep the crop that maximizes `ssim * ssim_weight + psnr`
//! (the paper's `SSIM × 4 + PSNR`).
//!
//! The crop is a scale + translate of the *grid-aligned* target within the
//! same fixed-size container (white fill), exactly as in v1:
//! - zoom: the container grows by `zoom/steps` of a character in each
//!   dimension — the paper's scale `s = (c + n) / n`.
//! - shift: content moves left/up by `shift/steps` of a character
//!   (the paper's `x = -a * charWidth`, `y = -b * charHeight`).
//!
//! Deliberate deviations from the v1 code:
//! - Shift steps follow the paper (¼ char per step at `steps = 4`); the v1
//!   code shifted by ⅛ char per step, contradicting the paper's parameter list.
//! - The score weights SSIM by 4 (paper); the v1 code scored `ssim + psnr`.
//! - The quick pass is an exact greedy pass over the whole charset; v1 used an
//!   ANN-assisted single placement visit per position.
//! - The winner is applied automatically and optimization continues; v1 stopped
//!   after the sweep and required re-running with manual parameters.
//! - Multi-layer configs use a two-stage sweep: every candidate is screened
//!   with a single placement pass (v1 parity), then the top [`AlignConfig::refine_top`]
//!   are rescored with the full layer stack and ranked by that. Screening alone
//!   ranks crops nearly as well (rank correlation ≈ 0.83 against the full
//!   stack on real images) and costs `1/num_layers` as much; the refinement
//!   keeps the very top of the ranking exact.
//! - Only the **image content** is compared — v1 scored the full padded
//!   container, including the half-character white border added by
//!   `resize_target`. That border is a big, crisp rectangle at a fixed
//!   position, and SSIM works much like an edge detector (as the paper itself
//!   notes), so "how sharp is the frame" could dominate "how well does the
//!   middle match". The score region here follows the content through the
//!   zoom/shift transform, so the padding never drives the ranking.

use ndarray::s;
use rayon::prelude::*;

use crate::metrics;
use crate::render::{combine_layers, compute_background, render_layer};
use crate::search::optimize_layer_pass;
use crate::types::{
    AlignConfig, AlignmentResult, AlignmentScore, AlignmentSearch, Chars, CropTransform, Image,
    LayerOffset, Layers, SearchAlgorithm,
};

/// Progress callback: `(completed_candidates, total_candidates)`.
///
/// Called after each candidate finishes (from parallel worker threads).
pub type AlignProgressFn<'a> = dyn Fn(usize, usize) + Sync + 'a;

/// All candidate crop transforms: every combination of
/// `(zoom, shift_y, shift_x)` with each parameter in `0..steps`.
pub fn candidate_transforms(steps: u32) -> Vec<CropTransform> {
    let steps = steps.max(1);
    let mut out = Vec::with_capacity((steps * steps * steps) as usize);
    for zoom in 0..steps {
        for shift_y in 0..steps {
            for shift_x in 0..steps {
                out.push(CropTransform {
                    zoom,
                    shift_x,
                    shift_y,
                });
            }
        }
    }
    out
}

/// Apply one crop transform to a grid-aligned target.
///
/// The target is scaled up by `zoom/steps` of a character in each dimension,
/// shifted left/up by `(shift_x, shift_y)/steps` of a character, then cropped
/// or white-padded back to the original container size.
///
/// Returns the transformed target and the score region `(y0, y1, x0, x1)`:
/// where the image `content` lands after the zoom + shift, clipped to the
/// container. Compare mockup and target only over this region — the padding
/// border must not drive the ranking (see module docs).
///
/// `content` is the image content inside the container — `(y0, y1, x0, x1)`
/// with exclusive ends, i.e. the container minus the padding added by
/// `resize_target` (see [`crate::types::Padding::content_rect`]).
pub fn apply_crop(
    target: &Image,
    t: &CropTransform,
    char_shape: (usize, usize),
    content: (usize, usize, usize, usize),
    steps: u32,
) -> (Image, (usize, usize, usize, usize)) {
    let (char_h, char_w) = char_shape;
    let steps = steps.max(1) as usize;
    let (h0, w0) = target.dim();

    // Zoom: grow the container by zoom/steps of a character per dimension
    let (h1, w1) = (
        h0 + (t.zoom as usize * char_h) / steps,
        w0 + (t.zoom as usize * char_w) / steps,
    );
    let scaled = crate::resize::resize_area_f32(target, w1, h1);

    // Shift: cut from the top-left, moving content left/up
    let shift_y = (t.shift_y as usize * char_h) / steps;
    let shift_x = (t.shift_x as usize * char_w) / steps;
    let (sh, sw) = (h1.saturating_sub(shift_y), w1.saturating_sub(shift_x));

    // Crop-or-pad back into the original container (white fill), v1 parity:
    // the whole container including the padding border is transformed.
    let (clip_h, clip_w) = (h0.min(sh), w0.min(sw));
    let mut out = Image::from_elem((h0, w0), 1.0);
    out.slice_mut(s![..clip_h, ..clip_w])
        .assign(&scaled.slice(s![shift_y..(shift_y + clip_h), shift_x..(shift_x + clip_w)]));

    // Where the image content lands after zoom + shift, clipped to the
    // container. Scaling maps source row y to dest row y * (h1/h0).
    let (cy0, cy1, cx0, cx1) = content;
    let (sy, sx) = (h1 as f64 / h0 as f64, w1 as f64 / w0 as f64);
    let map = |start: usize, end: usize, s: f64, shift: usize, max: usize| {
        let lo = ((start as f64 * s).round() as isize - shift as isize).max(0) as usize;
        let lo = lo.min(max);
        let hi = ((end as f64 * s).round() as isize - shift as isize).max(0) as usize;
        let hi = hi.min(max).max(lo);
        (lo, hi)
    };
    let (y0, y1) = map(cy0, cy1, sy, shift_y, h0);
    let (x0, x1) = map(cx0, cx1, sx, shift_x, w0);

    // Degenerate content mapping (pathological inputs): fall back to the
    // overlap of the shifted content with the container (v1's minShape).
    let score = if y1 <= y0 || x1 <= x0 {
        (0, clip_h, 0, clip_w)
    } else {
        (y0, y1, x0, x1)
    };

    (out, score)
}

/// Run the quick greedy evaluation for one candidate target.
///
/// One blank-init greedy coordinate-descent cycle over all layers — the v2
/// equivalent of v1's single `initOnly` placement pass. `passes` repeats the
/// cycle over all layers.
pub fn quick_mockup(
    target: &Image,
    chars: &Chars,
    layer_offsets: &[LayerOffset],
    passes: u32,
    asymmetry: f32,
) -> Image {
    let char_h = chars.shape()[1];
    let char_w = chars.shape()[2];
    let num_cols = target.ncols() / char_w;
    let num_rows = target.nrows() / char_h;
    let num_positions = num_rows * num_cols;

    let max_offset_y = layer_offsets.iter().map(|o| o.0).max().unwrap_or(0);
    let max_offset_x = layer_offsets.iter().map(|o| o.1).max().unwrap_or(0);
    let out_shape = (target.nrows() + max_offset_y, target.ncols() + max_offset_x);

    let mut target_padded = Image::from_elem(out_shape, 1.0);
    target_padded
        .slice_mut(s![..target.nrows(), ..target.ncols()])
        .assign(target);

    let num_layers = layer_offsets.len();
    let mut layers = Layers::from_elem((num_layers, out_shape.0, out_shape.1), 1.0);
    let mut choices = ndarray::Array2::<u16>::zeros((num_layers, num_positions));
    let mut mockup = Image::from_elem(out_shape, 1.0);

    for _ in 0..passes.max(1) {
        for (layer_num, &layer_offset) in layer_offsets.iter().enumerate() {
            let bg = compute_background(&layers, layer_num);
            let mut layer_choices = choices.index_axis(ndarray::Axis(0), layer_num).to_owned();
            optimize_layer_pass(
                &bg,
                &mut mockup,
                &target_padded,
                chars,
                &mut layer_choices,
                layer_offset,
                asymmetry,
                SearchAlgorithm::Greedy,
                0.0,
            );
            choices
                .index_axis_mut(ndarray::Axis(0), layer_num)
                .assign(&layer_choices);
            let updated = render_layer(
                layer_choices.as_slice().unwrap(),
                chars,
                num_cols,
                layer_offset,
                out_shape,
            );
            layers
                .index_axis_mut(ndarray::Axis(0), layer_num)
                .assign(&updated);
            mockup = combine_layers(&layers);
        }
    }

    combine_layers(&layers)
}

/// Score one candidate: crop the target, run the quick greedy pass, and
/// compare mockup and target over the transformed **content** region only
/// (the padding border is excluded — see module docs).
fn evaluate_candidate(
    target: &Image,
    chars: &Chars,
    layer_offsets: &[LayerOffset],
    config: &AlignConfig,
    asymmetry: f32,
    content: (usize, usize, usize, usize),
    t: &CropTransform,
    keep_mockup: bool,
) -> (AlignmentScore, Option<Image>) {
    let char_shape = (chars.shape()[1], chars.shape()[2]);
    let (candidate_target, (y0, y1, x0, x1)) =
        apply_crop(target, t, char_shape, content, config.steps);
    let mockup = quick_mockup(&candidate_target, chars, layer_offsets, config.passes, asymmetry);

    let mockup_scored = mockup.slice(s![y0..y1, x0..x1]).to_owned();
    let target_scored = candidate_target.slice(s![y0..y1, x0..x1]).to_owned();

    let psnr = metrics::psnr(&target_scored, &mockup_scored, 1.0);
    let ssim = metrics::ssim(&target_scored, &mockup_scored, 1.0);

    (
        AlignmentScore {
            transform: *t,
            psnr,
            ssim,
            score: ssim * config.ssim_weight + psnr,
            refined: false,
        },
        if keep_mockup { Some(mockup_scored) } else { None },
    )
}

/// Run the full alignment sweep over a grid-aligned target.
///
/// Every candidate is cropped, quickly optimized and scored; results are
/// sorted best-first. Candidates run in parallel in waves of at most one
/// candidate per worker thread (bounding peak memory).
///
/// With multi-layer configs the sweep runs in two stages: all candidates are
/// screened with a single placement pass (v1 parity), then the top
/// `config.refine_top` are rescored with the full layer stack and ranked by
/// that (their scores carry `refined = true`). Single-layer configs skip the
/// refinement — screening already is the full evaluation.
///
/// If `keep_mockups` is set, each candidate's quick mockup (cropped to its
/// score region) is returned best-first alongside the scores.
pub fn search(
    target: &Image,
    chars: &Chars,
    layer_offsets: &[LayerOffset],
    config: &AlignConfig,
    asymmetry: f32,
    content: (usize, usize, usize, usize),
    keep_mockups: bool,
    progress: Option<&AlignProgressFn<'_>>,
) -> AlignmentSearch {
    let steps = config.steps.max(1);
    let candidates = candidate_transforms(steps);
    let n = candidates.len();

    let refine_top = if layer_offsets.len() > 1 {
        (config.refine_top as usize).min(n)
    } else {
        0
    };
    let done = std::sync::atomic::AtomicUsize::new(0);
    let advance = || {
        let completed = done.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        if let Some(progress) = progress {
            progress(completed, n + refine_top);
        }
    };

    // Waves of at most one candidate per thread: full parallelism, bounded memory.
    let wave_size = rayon::current_num_threads().max(1);
    let mut scored: Vec<(AlignmentScore, Option<Image>)> = Vec::with_capacity(n);

    // Stage 1: screen every candidate with a single placement pass.
    let screen_offsets = &layer_offsets[..1.min(layer_offsets.len())];
    for wave in candidates.chunks(wave_size) {
        let results: Vec<_> = wave
            .par_iter()
            .map(|t| {
                let out = evaluate_candidate(
                    target,
                    chars,
                    screen_offsets,
                    config,
                    asymmetry,
                    content,
                    t,
                    keep_mockups,
                );
                advance();
                out
            })
            .collect();
        scored.extend(results);
    }
    scored.sort_by(|a, b| {
        b.0.score
            .partial_cmp(&a.0.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Stage 2: rescore the top candidates with the full layer stack.
    if refine_top > 0 {
        let top: Vec<CropTransform> = scored[..refine_top]
            .iter()
            .map(|(s, _)| s.transform)
            .collect();
        for wave in top.chunks(wave_size) {
            let results: Vec<_> = wave
                .par_iter()
                .map(|t| {
                    let (mut score, mockup) = evaluate_candidate(
                        target,
                        chars,
                        layer_offsets,
                        config,
                        asymmetry,
                        content,
                        t,
                        keep_mockups,
                    );
                    score.refined = true;
                    advance();
                    (score, mockup)
                })
                .collect();
            for (new_score, new_mockup) in results {
                if let Some(slot) = scored
                    .iter_mut()
                    .find(|(s, _)| s.transform == new_score.transform)
                {
                    *slot = (new_score, new_mockup);
                }
            }
        }
        scored.sort_by(|a, b| {
            b.0.score
                .partial_cmp(&a.0.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let mut scores = Vec::with_capacity(n);
    let mut mockups = Vec::new();
    for (score, mockup) in scored {
        if let Some(m) = mockup {
            mockups.push((score.transform, m));
        }
        scores.push(score);
    }

    AlignmentSearch {
        result: AlignmentResult {
            best: scores.first().map(|s| s.transform).unwrap_or(CropTransform {
                zoom: 0,
                shift_x: 0,
                shift_y: 0,
            }),
            steps,
            scores,
        },
        mockups,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr3;

    fn test_chars() -> Chars {
        // blank, half-dark vertical band, dark block
        arr3(&[
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ])
    }

    #[test]
    fn test_candidate_transforms_count() {
        assert_eq!(candidate_transforms(4).len(), 64);
        assert_eq!(candidate_transforms(1).len(), 1);
    }

    #[test]
    fn test_apply_crop_identity() {
        let target = Image::from_elem((16, 12), 0.5);
        let t = CropTransform {
            zoom: 0,
            shift_x: 0,
            shift_y: 0,
        };
        let (out, score_region) = apply_crop(&target, &t, (4, 4), (0, 16, 0, 12), 4);
        assert_eq!(out.dim(), (16, 12));
        assert_eq!(score_region, (0, 16, 0, 12));
        assert!(out.iter().zip(target.iter()).all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn test_apply_crop_shift_moves_content_and_pads_white() {
        // Black left half → after shifting left by 1/4 char (4 px / 4 = 1 px...
        // char is 4 wide, steps 4 → 1 px per step), the right edge gains white.
        let target = Image::from_shape_fn((8, 8), |(_, x)| if x < 4 { 0.0 } else { 1.0 });
        let t = CropTransform {
            zoom: 0,
            shift_x: 2,
            shift_y: 1,
        };
        // shift_x = 2*4/4 = 2 px left, shift_y = 1*4/4 = 1 px up
        let (out, score_region) = apply_crop(&target, &t, (4, 4), (0, 8, 0, 8), 4);
        assert_eq!(out.dim(), (8, 8), "container size is preserved");
        assert_eq!(score_region, (0, 7, 0, 6), "score region excludes shifted-in white");
        // Content moved left by 2: black now occupies columns 0..2
        assert_eq!(out[[0, 0]], 0.0);
        assert_eq!(out[[0, 1]], 0.0);
        assert_eq!(out[[0, 2]], 1.0);
        // Columns beyond the score region are white fill
        assert_eq!(out[[0, 7]], 1.0);
    }

    #[test]
    fn test_apply_crop_score_rect_excludes_padding() {
        // 12x16 container with 8x10 of image content: the padding border must
        // never be part of the score region.
        let target = Image::from_elem((12, 16), 0.5);
        let content = (2, 10, 3, 13);

        // Identity: score region == exactly the content
        let t = CropTransform { zoom: 0, shift_x: 0, shift_y: 0 };
        let (_, score) = apply_crop(&target, &t, (4, 4), content, 4);
        assert_eq!(score, (2, 10, 3, 13));

        // Shift (1 step = 1 px for a 4 px char): content moves up/left with it
        let t = CropTransform { zoom: 0, shift_x: 1, shift_y: 1 };
        let (_, score) = apply_crop(&target, &t, (4, 4), content, 4);
        assert_eq!(score, (1, 9, 2, 12));

        // Zoom (4 steps: rows scale by 16/12, cols by 20/16): content scales
        // with the transform; the grown bottom edge is clipped to the container
        // (10 * 16/12 = 13.3 → clipped to 12)
        let t = CropTransform { zoom: 4, shift_x: 0, shift_y: 0 };
        let (_, score) = apply_crop(&target, &t, (4, 4), content, 4);
        assert_eq!(score, (3, 12, 4, 16));
    }

    #[test]
    fn test_search_returns_sorted_scores() {
        let chars = test_chars();
        let target = Image::from_elem((8, 8), 0.6);
        let config = AlignConfig {
            enabled: true,
            steps: 2,
            passes: 1,
            ssim_weight: 4.0,
            refine_top: 3,
        };
        let completed = std::sync::atomic::AtomicUsize::new(0);
        let progress = |c: usize, _t: usize| {
            completed.store(c, std::sync::atomic::Ordering::SeqCst);
        };
        // Two layers → two-stage sweep: 8 screen evals + 3 refined evals
        let result = search(
            &target,
            &chars,
            &[(0, 0), (2, 2)],
            &config,
            0.1,
            (0, 8, 0, 8),
            true,
            Some(&progress),
        );

        assert_eq!(result.result.steps, 2);
        assert_eq!(result.result.scores.len(), 8);
        assert_eq!(result.mockups.len(), 8);
        assert_eq!(completed.load(std::sync::atomic::Ordering::SeqCst), 11);
        assert_eq!(result.result.best, result.result.scores[0].transform);
        // Top 3 were rescored with the full layer stack
        assert!(result.result.scores[..3].iter().all(|s| s.refined));
        assert!(result.result.scores[3..].iter().all(|s| !s.refined));
        for pair in result.result.scores.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
    }

    #[test]
    fn test_search_single_layer_skips_refinement() {
        let chars = test_chars();
        let target = Image::from_elem((8, 8), 0.6);
        let config = AlignConfig {
            enabled: true,
            steps: 2,
            passes: 1,
            ssim_weight: 4.0,
            refine_top: 3,
        };
        // One layer → screening is already the full evaluation
        let result = search(&target, &chars, &[(0, 0)], &config, 0.1, (0, 8, 0, 8), false, None);
        assert_eq!(result.result.scores.len(), 8);
        assert!(result.result.scores.iter().all(|s| !s.refined));
    }

    #[test]
    fn test_quick_mockup_shape() {
        let chars = test_chars();
        let target = Image::from_elem((8, 8), 0.6);
        let mockup = quick_mockup(&target, &chars, &[(0, 0), (2, 2)], 1, 0.1);
        assert_eq!(mockup.dim(), (10, 10), "includes max layer offsets");
    }
}
