//! Pre-optimization auto-alignment ("optimized cropping").
//!
//! Port of the `autoCrop` routine from v1 `kword.py` (removed from `src/` in
//! commit 85cba3d; described in the GI 2021 paper §3.4 "Optimized cropping").
//!
//! Characters can only be placed on a fixed character grid, so how the target
//! image happens to line up with that grid strongly affects shape matching —
//! this is "the primary factor when the typed image is small". Before the
//! iterative optimization starts, we therefore try a range of small zoom and
//! shift values, run a quick greedy optimization cycle for each cropped target,
//! and keep the crop that maximizes `ssim * ssim_weight + psnr`
//! (the paper's `SSIM × 4 + PSNR`).
//!
//! The crop is a scale + translate of the *grid-aligned* target within the
//! same fixed-size container (white fill):
//! - zoom: the container grows (or shrinks, when negative) by `zoom/steps` of
//!   a character in each dimension — the paper's scale `s = (c + n) / n`.
//! - shift: content moves by `(shift_x, shift_y)/steps` of a character
//!   (positive = left/up, the paper's `x = -a * charWidth`; negative moves it
//!   right/down, with white fill on the other side).
//!
//! The search space is signed `translate + scale` in character fractions, and
//! candidates are evaluated in an approximate-index style cascade:
//! 1. **Screen** a coarse grid (¼-char translation phases × signed ¾-char
//!    scale) with a single placement pass — v1 parity.
//! 2. **Refine** the neighborhoods of the top few seeds at ⅛-char fine steps
//!    (±1 step in every parameter), still screened cheaply.
//! 3. **Rescore** the top [`AlignConfig::refine_top`] with the full layer stack
//!    (one cycle) and rank by that.
//! 4. Optional **deluxe rerank** ([`AlignConfig::rerank_top`]): deeper
//!    optimization of the top few, winner picked by that — see
//!    [`crate::optimize`], which continues the winner instead of restarting.
//!
//! Deliberate deviations from the v1 code:
//! - The parameter space is signed and searched coarse-to-fine; v1's was
//!   one-sided (shifts only left/up, zoom only growing) at fixed ¼-char steps.
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
use crate::render::{Ink, combine_layers, compute_background, render_layer};
use crate::search::optimize_layer_pass;
use crate::types::{
    AlignConfig, AlignmentResult, AlignmentScore, AlignmentSearch, CenterFocus, Chars,
    CropTransform, Image, LayerOffset, Layers, SearchAlgorithm,
};

/// Progress callback: `(completed_candidates, total_candidates)`.
///
/// Called after each candidate finishes (from parallel worker threads).
pub type AlignProgressFn<'a> = dyn Fn(usize, usize) + Sync + 'a;

/// Number of coarse candidates whose neighborhoods are refined at fine steps.
pub(crate) const FINE_SEEDS: usize = 3;

/// Coarse candidate grid: shifts at quarter-character phases `{0, ¼, ½, ¾}`
/// char (the paper's `a, b ∈ {0, 0.25, 0.5, 0.75}`) crossed with signed scale
/// `{-¾, 0, +¾}` char (v1 could only grow). Parameter values are in fine
/// steps: `v` = `v/steps` of a character.
pub fn coarse_candidates(steps: u32) -> Vec<CropTransform> {
    let steps = steps.max(4) as i32;
    let q = (steps / 4).max(1); // quarter char in fine steps
    let phases = [0, q, 2 * q, 3 * q];
    let zooms = [-3 * q, 0, 3 * q];
    let mut out = Vec::with_capacity(zooms.len() * phases.len() * phases.len());
    for &zoom in &zooms {
        for &shift_y in &phases {
            for &shift_x in &phases {
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

/// Fine candidates around each seed: every combination of ±1 fine step in
/// `(zoom, shift_x, shift_y)`, deduplicated across seeds and excluding the
/// seeds themselves (already scored).
pub fn fine_candidates(seeds: &[CropTransform]) -> Vec<CropTransform> {
    let mut seen: std::collections::HashSet<(i32, i32, i32)> = seeds
        .iter()
        .map(|t| (t.zoom, t.shift_x, t.shift_y))
        .collect();
    let mut out = Vec::new();
    for s in seeds {
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let t = CropTransform {
                        zoom: s.zoom + dz,
                        shift_x: s.shift_x + dx,
                        shift_y: s.shift_y + dy,
                    };
                    if seen.insert((t.zoom, t.shift_x, t.shift_y)) {
                        out.push(t);
                    }
                }
            }
        }
    }
    out
}

/// Apply one crop transform to a grid-aligned target.
///
/// The target is scaled by `zoom/steps` of a character in each dimension
/// (negative shrinks), translated by `(shift_x, shift_y)/steps` of a character
/// (positive = left/up), then cropped or white-padded back to the original
/// container size (v1 parity: the whole container including the padding
/// border is transformed; white fill enters from the side content moved away
/// from).
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
    let steps = i64::from(steps.max(1));
    let (h0, w0) = target.dim();

    // Zoom: signed boundary growth of zoom/steps of a character per dimension
    let (h1, w1) = (
        (h0 as i64 + (i64::from(t.zoom) * char_h as i64) / steps).max(1) as usize,
        (w0 as i64 + (i64::from(t.zoom) * char_w as i64) / steps).max(1) as usize,
    );
    let scaled = crate::resize::resize_area_f32(target, w1, h1);

    // Shift: signed translation; positive moves content left/up (v1),
    // negative right/down
    let shift_y = (i64::from(t.shift_y) * char_h as i64) / steps;
    let shift_x = (i64::from(t.shift_x) * char_w as i64) / steps;

    // Place the scaled image at (-shift_y, -shift_x) in the container,
    // cropping or white-padding (v1 parity: the padding border transforms too)
    let clip = |off: i64, span: usize, max: usize| {
        let start = off.max(0);
        let end = (off + span as i64).min(max as i64).max(0);
        (start as usize, end.max(start) as usize)
    };
    let (y0c, y1c) = clip(-shift_y, h1, h0);
    let (x0c, x1c) = clip(-shift_x, w1, w0);
    let mut out = Image::from_elem((h0, w0), 1.0);
    if y1c > y0c && x1c > x0c {
        let (sy0, sx0) = ((y0c as i64 + shift_y) as usize, (x0c as i64 + shift_x) as usize);
        out.slice_mut(s![y0c..y1c, x0c..x1c])
            .assign(&scaled.slice(s![sy0..(sy0 + (y1c - y0c)), sx0..(sx0 + (x1c - x0c))]));
    }

    // Where the image content lands after zoom + shift, clipped to the
    // container. Scaling maps source row y to dest row y * (h1/h0).
    let (cy0, cy1, cx0, cx1) = content;
    let (sy, sx) = (h1 as f64 / h0 as f64, w1 as f64 / w0 as f64);
    let map = |start: usize, end: usize, s: f64, shift: i64, max: usize| {
        let lo = ((start as f64 * s).round() as i64 - shift).clamp(0, max as i64) as usize;
        let hi = ((end as f64 * s).round() as i64 - shift).clamp(0, max as i64) as usize;
        (lo, hi.max(lo))
    };
    let (y0, y1) = map(cy0, cy1, sy, shift_y, h0);
    let (x0, x1) = map(cx0, cx1, sx, shift_x, w0);

    // Degenerate content mapping (pathological inputs): fall back to the
    // overlap of the shifted content with the container (v1's minShape).
    let score = if y1 <= y0 || x1 <= x0 {
        (0, y1c, 0, x1c)
    } else {
        (y0, y1, x0, x1)
    };

    (out, score)
}

/// Score a mockup against its target over `rect` = `(y0, y1, x0, x1)`:
/// returns `(score, psnr, ssim)` with `score = ssim * ssim_weight + psnr`,
/// weighted by the center-focus profile over the rect when enabled.
pub fn score_mockup(
    target: &Image,
    mockup: &Image,
    rect: (usize, usize, usize, usize),
    focus: CenterFocus,
    ssim_weight: f32,
) -> (f32, f32, f32) {
    let (y0, y1, x0, x1) = rect;
    if y1.saturating_sub(y0) < 3 || x1.saturating_sub(x0) < 3 {
        // Too little content left visible to score (and below SSIM's 3x3
        // window): rank such crops last. (Finite sentinel — JSON-safe.)
        return (-1.0e9, 0.0, 0.0);
    }
    let mockup_scored = mockup.slice(s![y0..y1, x0..x1]).to_owned();
    let target_scored = target.slice(s![y0..y1, x0..x1]).to_owned();

    // Score the content only, with the same "center focus" weighting the
    // optimization uses (None = uniform).
    let score_weights = crate::weights::focus_map((y1 - y0, x1 - x0), focus);
    let (psnr, ssim) = match score_weights {
        Some(ref w) => (
            metrics::psnr_weighted(&target_scored, &mockup_scored, w, 1.0),
            metrics::ssim_weighted(&target_scored, &mockup_scored, w, 1.0),
        ),
        None => (
            metrics::psnr(&target_scored, &mockup_scored, 1.0),
            metrics::ssim(&target_scored, &mockup_scored, 1.0),
        ),
    };
    (ssim * ssim_weight + psnr, psnr, ssim)
}

/// Run the quick greedy evaluation for one candidate target.
///
/// One blank-init greedy coordinate-descent cycle over all layers — the v2
/// equivalent of v1's single `initOnly` placement pass. `passes` repeats the
/// cycle over all layers. The loss uses the same "center focus" weighting as
/// the main optimization, over `content` = `(y0, y1, x0, x1)` in target
/// coordinates.
pub fn quick_mockup(
    target: &Image,
    chars: &Chars,
    ink: &Ink,
    layer_offsets: &[LayerOffset],
    passes: u32,
    asymmetry: f32,
    focus: CenterFocus,
    content: (usize, usize, usize, usize),
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
    let weights = crate::weights::container_map(out_shape, content, focus);

    for _ in 0..passes.max(1) {
        for (layer_num, &layer_offset) in layer_offsets.iter().enumerate() {
            let bg = compute_background(&layers, layer_num, ink);
            let mut layer_choices = choices.index_axis(ndarray::Axis(0), layer_num).to_owned();
            optimize_layer_pass(
                &bg,
                &mut mockup,
                &target_padded,
                ink,
                &mut layer_choices,
                layer_offset,
                asymmetry,
                SearchAlgorithm::Greedy,
                0.0,
                weights.as_ref(),
                0, // greedy pass: RNG unused
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
            mockup = combine_layers(&layers, ink);
        }
    }

    combine_layers(&layers, ink)
}

/// Score one candidate: crop the target, run the quick greedy pass, and
/// compare mockup and target over the transformed **content** region only
/// (the padding border is excluded — see module docs).
fn evaluate_candidate(
    target: &Image,
    chars: &Chars,
    ink: &Ink,
    layer_offsets: &[LayerOffset],
    config: &AlignConfig,
    asymmetry: f32,
    content: (usize, usize, usize, usize),
    focus: CenterFocus,
    t: &CropTransform,
    keep_mockup: bool,
) -> (AlignmentScore, Option<Image>) {
    let char_shape = (chars.shape()[1], chars.shape()[2]);
    let (candidate_target, (y0, y1, x0, x1)) =
        apply_crop(target, t, char_shape, content, config.steps);
    let mockup = quick_mockup(
        &candidate_target,
        chars,
        ink,
        layer_offsets,
        config.passes,
        asymmetry,
        focus,
        (y0, y1, x0, x1),
    );

    let (score, psnr, ssim) = score_mockup(
        &candidate_target,
        &mockup,
        (y0, y1, x0, x1),
        focus,
        config.ssim_weight,
    );

    (
        AlignmentScore {
            transform: *t,
            psnr,
            ssim,
            score,
            refined: false,
        },
        if keep_mockup {
            Some(mockup.slice(s![y0..y1, x0..x1]).to_owned())
        } else {
            None
        },
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
    ink: &Ink,
    layer_offsets: &[LayerOffset],
    config: &AlignConfig,
    asymmetry: f32,
    content: (usize, usize, usize, usize),
    focus: CenterFocus,
    keep_mockups: bool,
    progress: Option<&AlignProgressFn<'_>>,
) -> AlignmentSearch {
    let steps = config.steps.max(4);
    let coarse = coarse_candidates(steps);

    let refine_top = if layer_offsets.len() > 1 {
        (config.refine_top as usize).min(coarse.len() + FINE_SEEDS * 27)
    } else {
        0
    };
    let done = std::sync::atomic::AtomicUsize::new(0);
    // Progress total: coarse grid + upper bound on fine candidates + refines
    let total_est = coarse.len() + FINE_SEEDS * 27 + refine_top;
    let advance = || {
        let completed = done.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        if let Some(progress) = progress {
            progress(completed, total_est);
        }
    };

    // Waves of at most one candidate per thread: full parallelism, bounded memory.
    let wave_size = rayon::current_num_threads().max(1);
    let screen_offsets = &layer_offsets[..1.min(layer_offsets.len())];
    let screen_all = |candidates: &[CropTransform]| {
        let mut out = Vec::with_capacity(candidates.len());
        for wave in candidates.chunks(wave_size) {
            let results: Vec<_> = wave
                .par_iter()
                .map(|t| {
                    let r = evaluate_candidate(
                        target,
                        chars,
                        ink,
                        screen_offsets,
                        config,
                        asymmetry,
                        content,
                        focus,
                        t,
                        keep_mockups,
                    );
                    advance();
                    r
                })
                .collect();
            out.extend(results);
        }
        out
    };

    // Stage 1a: screen the coarse grid with a single placement pass.
    let mut scored: Vec<(AlignmentScore, Option<Image>)> = screen_all(&coarse);
    scored.sort_by(|a, b| {
        b.0.score
            .partial_cmp(&a.0.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Stage 1b: screen fine neighborhoods around the top coarse seeds.
    let seeds: Vec<CropTransform> = scored
        .iter()
        .take(FINE_SEEDS)
        .map(|(s, _)| s.transform)
        .collect();
    let fine = fine_candidates(&seeds);
    if !fine.is_empty() {
        scored.extend(screen_all(&fine));
        scored.sort_by(|a, b| {
            b.0.score
                .partial_cmp(&a.0.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

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
                        ink,
                        layer_offsets,
                        config,
                        asymmetry,
                        content,
                        focus,
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

    let mut scores = Vec::with_capacity(scored.len());
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
            rerank: None,
        },
        mockups,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::InkFloor;
    use ndarray::arr3;

    fn test_ink(chars: &Chars) -> Ink {
        Ink::new(InkFloor::Fixed(0.0), chars)
    }

    fn test_chars() -> Chars {
        // blank, half-dark vertical band, dark block
        arr3(&[
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ])
    }

    #[test]
    fn test_coarse_candidates_grid() {
        // 3 signed zooms × 4 shift phases × 4 shift phases = 48
        assert_eq!(coarse_candidates(8).len(), 48);
        assert_eq!(coarse_candidates(4).len(), 48);
        // Quarter-char phases (fine steps of 1/8 char): 0, 2, 4, 6
        let c = coarse_candidates(8);
        assert!(c.contains(&CropTransform { zoom: -6, shift_x: 2, shift_y: 4 }));
        assert!(c.contains(&CropTransform { zoom: 6, shift_x: 6, shift_y: 0 }));
    }

    #[test]
    fn test_fine_candidates_neighborhood() {
        let seeds = vec![
            CropTransform { zoom: 0, shift_x: 0, shift_y: 0 },
            CropTransform { zoom: 0, shift_x: 1, shift_y: 0 },
        ];
        let fine = fine_candidates(&seeds);
        // ±1 fine step in every parameter around each seed, minus the seeds
        // (already scored) and shared cells
        assert!(fine.len() >= 27 && fine.len() < 54);
        assert!(!fine.contains(&seeds[0]));
        assert!(fine.contains(&CropTransform { zoom: -1, shift_x: -1, shift_y: 1 }));
        let unique: std::collections::HashSet<_> =
            fine.iter().map(|t| (t.zoom, t.shift_x, t.shift_y)).collect();
        assert_eq!(unique.len(), fine.len(), "no duplicates");
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
    fn test_apply_crop_negative_shift_moves_right_down() {
        let target = Image::from_shape_fn((8, 8), |(_, x)| if x < 4 { 0.0 } else { 1.0 });
        let t = CropTransform { zoom: 0, shift_x: -1, shift_y: 0 };
        let (out, score) = apply_crop(&target, &t, (4, 4), (0, 8, 0, 8), 4);
        // Content moves right by 1 px (1/4 char): black now at columns 1..5,
        // white fill enters from the left
        assert_eq!(out[[0, 0]], 1.0);
        assert_eq!(out[[0, 1]], 0.0);
        assert_eq!(out[[0, 4]], 0.0);
        assert_eq!(out[[0, 5]], 1.0);
        // Score rect excludes the white fill column
        assert_eq!(score, (0, 8, 1, 8));
    }

    #[test]
    fn test_apply_crop_negative_zoom_shrinks() {
        let target = Image::from_elem((12, 16), 0.5);
        let t = CropTransform { zoom: -4, shift_x: 0, shift_y: 0 };
        let (out, score) = apply_crop(&target, &t, (4, 4), (2, 10, 3, 13), 4);
        assert_eq!(out.dim(), (12, 16), "container size is preserved");
        // Rows scale by 8/12 → content (2, 10) maps to (1.33, 6.67) ≈ (1, 7);
        // cols scale by 12/16 → (3, 13) maps to (2.25, 9.75) ≈ (2, 10)
        assert_eq!(score, (1, 7, 2, 10));
        // White fill enters at the bottom/right of the shrunk content
        assert_eq!(out[[11, 15]], 1.0);
    }

    #[test]
    fn test_search_returns_sorted_scores() {
        let chars = test_chars();
        let target = Image::from_elem((8, 8), 0.6);
        let config = AlignConfig {
            enabled: true,
            steps: 4,
            passes: 1,
            ssim_weight: 4.0,
            refine_top: 3,
            rerank_top: 0,
            rerank_loops: 0,
        };
        let completed = std::sync::atomic::AtomicUsize::new(0);
        let progress = |c: usize, _t: usize| {
            completed.store(c, std::sync::atomic::Ordering::SeqCst);
        };
        // Two layers → coarse + fine screen (single placement pass), then the
        // top 3 rescored with the full layer stack
        let ink = test_ink(&chars);
        let result = search(
            &target,
            &chars,
            &ink,
            &[(0, 0), (2, 2)],
            &config,
            0.1,
            (0, 8, 0, 8),
            CenterFocus::Off,
            true,
            Some(&progress),
        );

        assert_eq!(result.result.steps, 4);
        assert!(result.result.scores.len() >= 48, "coarse grid is screened");
        assert_eq!(result.mockups.len(), result.result.scores.len());
        assert_eq!(
            completed.load(std::sync::atomic::Ordering::SeqCst),
            result.result.scores.len() + 3,
            "one eval per screened candidate + 3 refinements"
        );
        assert_eq!(result.result.best, result.result.scores[0].transform);
        assert!(result.result.rerank.is_none());
        // Exactly 3 candidates were rescored with the full layer stack
        assert_eq!(result.result.scores.iter().filter(|s| s.refined).count(), 3);
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
            steps: 4,
            passes: 1,
            ssim_weight: 4.0,
            refine_top: 3,
            rerank_top: 0,
            rerank_loops: 0,
        };
        // One layer → screening is already the full evaluation
        let ink = test_ink(&chars);
        let result = search(
            &target,
            &chars,
            &ink,
            &[(0, 0)],
            &config,
            0.1,
            (0, 8, 0, 8),
            CenterFocus::Off,
            false,
            None,
        );
        assert!(result.result.scores.len() >= 48);
        assert!(result.result.scores.iter().all(|s| !s.refined));
    }

    #[test]
    fn test_search_with_center_focus() {
        // Weighted scoring path: scores must be finite and sorted
        let chars = test_chars();
        let ink = test_ink(&chars);
        let target = Image::from_elem((8, 8), 0.6);
        let config = AlignConfig {
            enabled: true,
            steps: 4,
            passes: 1,
            ssim_weight: 4.0,
            refine_top: 0,
            rerank_top: 0,
            rerank_loops: 0,
        };
        let result = search(
            &target,
            &chars,
            &ink,
            &[(0, 0)],
            &config,
            0.1,
            (0, 8, 0, 8),
            CenterFocus::Strong,
            false,
            None,
        );
        assert!(result.result.scores.len() >= 48);
        assert!(result.result.scores.iter().all(|s| s.score.is_finite()));
    }

    #[test]
    fn test_quick_mockup_shape() {
        let chars = test_chars();
        let ink = test_ink(&chars);
        let target = Image::from_elem((8, 8), 0.6);
        let mockup = quick_mockup(
            &target,
            &chars,
            &ink,
            &[(0, 0), (2, 2)],
            1,
            0.1,
            CenterFocus::Off,
            (0, 8, 0, 8),
        );
        assert_eq!(mockup.dim(), (10, 10), "includes max layer offsets");
    }
}
