//! Dithering for light tones.
//!
//! Pointwise MSE cannot dither. For a tile at 4 % ink, blank beats scattering
//! the lightest mark by ~8× on MSE (the mark's own pixels are far too dark),
//! even though the scattered pattern matches the tone and blank does not.
//! The benefit of dithering lives in the **local mean**, which the per-cell
//! objective never sees — so flat white wins and light gradients vanish.
//!
//! This finishing stage supplies the missing mechanism in two phases:
//!
//! 1. **Seeding** — the canvas is covered by a *composite lattice*: every
//!    layer's cell grid lands on it at its offset, so with quarter-char
//!    offsets the lattice is quarter-char dense. Each lattice point computes
//!    its ink demand `u` in *mark units* (how many marks' worth of ink the
//!    local area still needs) and marks its cells against one shared screen
//!    (`bayer4`/`bayer8`/`bluenoise`, or `fs` error diffusion on the
//!    lattice): the k-th cell at a point is marked when `u − k > screen[r, c]`.
//!    Because the screen is sampled at lattice coordinates, all layers share
//!    one coherent pattern — per-layer screens would composite into noise.
//!    Demand is divided by the lattice density, so a tone of `t` ink lands at
//!    `t / (mark_ink · lattice_per_tile)` coverage instead of saturating.
//! 2. **Flip refinement** — greedy passes over the band, flipping cells
//!    between blank and the mark whenever it lowers the **blurred** error
//!    (a box blur of ~1.5 char cells). This settles the local density to the
//!    target tone and arbitrates overlaps between layers, with the true ink
//!    composite under the hood. Cells that carry optimized content may only
//!    be flipped where the target tile is *flat* — dithering is for tone;
//!    structured light areas (highlights, textures) keep their detail.
//!
//! Scattered marks keep a **minimum distance** derived from the mark glyph's
//! own ink footprint: the lattice is finer than the mark (a period is ~5 px
//! on a 2.5 px lattice), so unconstrained marks fuse into worms. The rule is
//! the classic stippling constraint — overstrike copies at one point may
//! still stack.
//!
//! Greedy flips strictly decrease the flip objective, so the stage is
//! deterministic and cannot oscillate. The stage runs after the optimization
//! loops on their best state and is not part of the alignment sweep.

use std::collections::{HashMap, HashSet};

use ndarray::{s, Array2, Axis};

use crate::render::{compute_background, Ink};
use crate::types::{
    Chars, Choices, DitherConfig, DitherMark, DitherMode, Image, LayerOffset, Layers,
};

/// Target tiles with brightness standard deviation below this count as flat
/// tone (ditherable) rather than structure (protected).
const FLAT_STD: f32 = 0.05;

/// Outcome of the dither finishing stage.
#[derive(Debug, Clone, Copy, Default)]
pub struct DitherStats {
    /// Marks placed by the seeding phase.
    pub seeded: u32,
    /// Flips accepted by the refinement phase.
    pub flipped: u32,
    /// Blurred (low-frequency) error before and after the stage.
    pub lf_before: f32,
    pub lf_after: f32,
}

/// Standard 4×4 Bayer threshold matrix (values / 16).
const BAYER4: [[u8; 4]; 4] = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
];

/// Standard 8×8 Bayer threshold matrix (values / 64).
const BAYER8: [[u8; 8]; 8] = [
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
];

/// Progressive blue-noise threshold matrix (farthest-point ranking,
/// 16×16 periodic, rank 0 = densest). Threshold at `rank / 256`.
pub const BLUE_NOISE_16: [[u8; 16]; 16] = [
    [176, 102, 224, 80, 213, 70, 255, 65, 154, 82, 239, 95, 180, 123, 159, 103],
    [47, 160, 27, 242, 63, 140, 17, 132, 42, 182, 22, 175, 53, 152, 31, 208],
    [202, 71, 221, 75, 184, 97, 209, 99, 225, 127, 136, 81, 249, 79, 216, 86],
    [0, 169, 61, 196, 14, 129, 41, 253, 2, 186, 57, 241, 15, 200, 45, 174],
    [164, 92, 219, 76, 243, 108, 222, 114, 251, 105, 193, 91, 149, 78, 163, 107],
    [62, 232, 18, 230, 50, 252, 19, 157, 44, 201, 23, 235, 49, 177, 29, 226],
    [215, 126, 238, 121, 189, 109, 217, 93, 237, 74, 192, 125, 194, 83, 181, 85],
    [10, 151, 46, 183, 7, 145, 36, 203, 8, 205, 43, 143, 6, 130, 52, 173],
    [168, 67, 185, 118, 240, 104, 179, 94, 135, 116, 210, 64, 233, 96, 190, 124],
    [34, 191, 21, 165, 60, 247, 24, 139, 59, 128, 20, 153, 35, 161, 25, 188],
    [167, 110, 227, 84, 195, 117, 137, 101, 158, 115, 207, 90, 204, 120, 141, 72],
    [3, 214, 39, 212, 13, 245, 54, 197, 1, 254, 32, 234, 11, 250, 37, 134],
    [150, 87, 147, 100, 199, 119, 231, 69, 229, 77, 162, 106, 218, 73, 236, 88],
    [56, 246, 30, 187, 55, 156, 28, 166, 51, 206, 16, 178, 40, 244, 26, 155],
    [142, 98, 228, 89, 133, 66, 220, 68, 171, 112, 131, 122, 138, 113, 146, 111],
    [12, 172, 33, 223, 4, 148, 48, 170, 9, 211, 38, 144, 5, 198, 58, 248],
];

/// Run the dither finishing stage in place.
///
/// `mockup`, `choices` and `layers` are updated together (the layers and the
/// mockup are re-rendered for every accepted change), `target` is the padded
/// optimization target. Returns statistics for logging.
#[allow(clippy::too_many_arguments)]
pub fn run(
    mockup: &mut Image,
    choices: &mut Choices,
    layers: &mut Layers,
    target: &Image,
    chars: &Chars,
    ink: &Ink,
    layer_offsets: &[LayerOffset],
    num_cols: usize,
    config: &DitherConfig,
) -> DitherStats {
    if config.mode == DitherMode::None {
        return DitherStats::default();
    }
    let num_layers = layer_offsets.len();
    let num_positions = choices.shape()[1];
    let char_h = chars.shape()[1];
    let char_w = chars.shape()[2];
    let (img_h, img_w) = mockup.dim();

    let blank_idx = least_ink_char(chars);
    let Some(mark_idx) = resolve_mark(chars, config.mark) else {
        log::warn!("Dither: charset has no non-blank glyph to scatter, skipping");
        return DitherStats::default();
    };
    if mark_idx == blank_idx {
        return DitherStats::default();
    }

    // ---- Composite lattice ------------------------------------------------
    // Every layer's cell grid lands on one shared lattice at its offset.
    // `k_y`/`k_x` = lattice points per char cell per axis; layers sharing an
    // offset are overstrike copies of the same lattice point.
    let mut ys: Vec<usize> = layer_offsets.iter().map(|o| o.0).collect();
    let mut xs: Vec<usize> = layer_offsets.iter().map(|o| o.1).collect();
    ys.sort_unstable();
    ys.dedup();
    xs.sort_unstable();
    xs.dedup();
    let k_y = ys.len().max(1);
    let k_x = xs.len().max(1);
    let n_lat = k_y * k_x; // lattice points per char tile
    let lattice = |y0: usize, x0: usize| -> (usize, usize) {
        (
            (y0 * k_y + char_h / 2) / char_h,
            (x0 * k_x + char_w / 2) / char_w,
        )
    };
    log::info!(
        "Dither: mode {:?}, band ≤ {:.2}, mark glyph {} ({:.4} ink), lattice {}×{} per cell ({} copies), refine {} passes",
        config.mode,
        config.band_max,
        mark_idx,
        mean_ink(chars, mark_idx as usize),
        k_y,
        k_x,
        num_layers / n_lat.max(1),
        config.refine_passes,
    );

    let tile_rect = |p: usize, offset: LayerOffset| -> (usize, usize, usize, usize) {
        let row = p / num_cols;
        let col = p % num_cols;
        let y0 = row * char_h + offset.0;
        let x0 = col * char_w + offset.1;
        (y0, y0 + char_h, x0, x0 + char_w)
    };

    // Minimum spacing between scattered marks (see module docs): derived
    // from the mark glyph's ink footprint so dots cannot fuse into worms.
    let exclusion = mark_exclusion(
        &ink.chars_t.index_axis(Axis(0), mark_idx as usize),
        char_h as f32 / k_y as f32,
        char_w as f32 / k_x as f32,
    );
    let mut occupied: HashSet<(usize, usize)> = HashSet::new();
    let blocked = |occupied: &HashSet<(usize, usize)>, r: usize, c: usize| -> bool {
        exclusion.iter().any(|&(dr, dc)| {
            let rr = r as isize + dr;
            let cc = c as isize + dc;
            rr >= 0 && cc >= 0 && occupied.contains(&(rr as usize, cc as usize))
        })
    };

    // Ink demand of one mark at this cell: (1−f)·bg_t·(1−τ(mark)), averaged
    // over the tile (blank contributes τ(1) = 1, i.e. nothing).
    let mark_delta = |bg_t: &Array2<f32>, y0: usize, y1: usize, x0: usize, x1: usize| -> f32 {
        let mark_t = ink.chars_t.index_axis(Axis(0), mark_idx as usize);
        let mut total = 0.0f32;
        for y in y0..y1 {
            for x in x0..x1 {
                total += (1.0 - ink.floor) * bg_t[[y, x]] * (1.0 - mark_t[[y - y0, x - x0]]);
            }
        }
        total / ((y1 - y0) * (x1 - x0)) as f32
    };

    // Coverage demand at one cell, in marks per lattice point: the tile's
    // missing ink spread over the lattice points covering it.
    let demand =
        |mockup: &Image,
         target: &Image,
         bg_t: &Array2<f32>,
         y0: usize,
         y1: usize,
         x0: usize,
         x1: usize|
         -> Option<f32> {
            let target_tile = target.slice(s![y0..y1, x0..x1]);
            let target_ink = 1.0 - target_tile.mean().unwrap_or(1.0);
            if target_ink > config.band_max {
                return None;
            }
            let mockup_tile = mockup.slice(s![y0..y1, x0..x1]);
            let need = target_ink - (1.0 - mockup_tile.mean().unwrap_or(1.0));
            if need <= 0.0 {
                return None;
            }
            let delta = mark_delta(bg_t, y0, y1, x0, x1);
            if delta <= 1e-6 {
                return None;
            }
            Some(need / (delta * n_lat as f32))
        };

    // ---- Phase 1: seeding on the composite lattice ------------------------
    // lattice point -> cells (layer, position) in layer order = overstrike order
    let mut points: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    let mut points_all: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for l in 0..num_layers {
        for p in 0..num_positions {
            let (y0, _, x0, _) = tile_rect(p, layer_offsets[l]);
            let key = lattice(y0, x0);
            points_all.entry(key).or_default().push((l, p));
            if choices[[l, p]] == blank_idx {
                points.entry(key).or_default().push((l, p));
            }
        }
    }

    let mut seeded = 0u32;

    if config.mode == DitherMode::FloydSteinberg {
        // Error diffusion over the lattice's demand grid.
        let rows = ((num_positions + num_cols - 1) / num_cols) * k_y;
        let cols = num_cols * k_x;
        let mut u_grid = Array2::<f32>::zeros((rows, cols));
        let mut bg_cache: HashMap<usize, Array2<f32>> = HashMap::new();
        for ((r, c), cells) in points.iter() {
            let &(l, p) = cells.first().unwrap();
            let (y0, y1, x0, x1) = tile_rect(p, layer_offsets[l]);
            let bg_t = bg_cache
                .entry(l)
                .or_insert_with(|| compute_background(layers, l, ink))
                .clone();
            if let Some(u) = demand(mockup, target, &bg_t, y0, y1, x0, x1) {
                u_grid[[*r, *c]] = u;
            }
        }
        let mut err = Array2::<f32>::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                let acc = u_grid[[r, c]] + err[[r, c]];
                let copies = points.get(&(r, c)).map_or(0, |v| v.len()) as f32;
                let out = acc.clamp(0.0, copies).round();
                let e = acc - out;
                let mut diffuse = |rr: isize, cc: isize, w: f32| {
                    if rr >= 0 && (rr as usize) < rows && cc >= 0 && (cc as usize) < cols {
                        err[[rr as usize, cc as usize]] += e * w;
                    }
                };
                diffuse(r as isize, c as isize + 1, 7.0 / 16.0);
                diffuse(r as isize + 1, c as isize - 1, 3.0 / 16.0);
                diffuse(r as isize + 1, c as isize, 5.0 / 16.0);
                diffuse(r as isize + 1, c as isize + 1, 1.0 / 16.0);
                if out > 0.0 && !blocked(&occupied, r, c) {
                    if let Some(cells) = points.get(&(r, c)) {
                        let mut placed = 0u32;
                        for &(l, p) in cells.iter() {
                            if placed as f32 >= out {
                                break;
                            }
                            if choices[[l, p]] != blank_idx {
                                continue;
                            }
                            let (y0, y1, x0, x1) = tile_rect(p, layer_offsets[l]);
                            let bg_t = bg_cache
                                .entry(l)
                                .or_insert_with(|| compute_background(layers, l, ink));
                            apply_choice(
                                mockup,
                                choices,
                                layers,
                                bg_t,
                                chars,
                                ink,
                                l,
                                p,
                                mark_idx,
                                y0,
                                y1,
                                x0,
                                x1,
                            );
                            update_bg_cache(&mut bg_cache, ink, mark_idx, l, y0, y1, x0, x1);
                            occupied.insert((r, c));
                            seeded += 1;
                            placed += 1;
                        }
                    }
                }
            }
        }
    } else {
        // Ordered screens: raster over lattice points so the pattern is a
        // coherent halftone across all layers.
        let mut ordered: Vec<((usize, usize), Vec<(usize, usize)>)> = points.into_iter().collect();
        ordered.sort_by_key(|((r, c), _)| (*r, *c));
        let mut bg_cache: HashMap<usize, Array2<f32>> = HashMap::new();
        for ((r, c), cells) in ordered {
            for &(l, p) in cells.iter() {
                if choices[[l, p]] != blank_idx {
                    continue;
                }
                let (y0, y1, x0, x1) = tile_rect(p, layer_offsets[l]);
                let bg_t = bg_cache
                    .entry(l)
                    .or_insert_with(|| compute_background(layers, l, ink))
                    .clone();
                // Demand is recomputed from the live mockup, so it already
                // accounts for marks placed at this lattice point by earlier
                // overstrike copies: mark while it beats the screen.
                let Some(u) = demand(mockup, target, &bg_t, y0, y1, x0, x1) else {
                    continue;
                };
                if u > screen_threshold(config.mode, r, c) && !blocked(&occupied, r, c) {
                    apply_choice(
                        mockup,
                        choices,
                        layers,
                        &bg_t,
                        chars,
                        ink,
                        l,
                        p,
                        mark_idx,
                        y0,
                        y1,
                        x0,
                        x1,
                    );
                    update_bg_cache(&mut bg_cache, ink, mark_idx, l, y0, y1, x0, x1);
                    occupied.insert((r, c));
                    seeded += 1;
                }
            }
        }
    }

    // ---- Phase 2: low-frequency flip refinement ---------------------------
    // Blur scale: "the local mean" at the scale the eye integrates halftone
    // texture at (~1.5 char cells).
    let rx = (((char_w as f32) * 1.5).round() as usize).max(1);
    let ry = (((char_h as f32) * 1.5).round() as usize).max(1);
    let lf_target = blur_box(target, rx, ry);
    let lf_before = mean_sq_err(&lf_target, &blur_box(mockup, rx, ry));
    let mut lf_mockup = blur_box(mockup, rx, ry);
    let mut flipped = 0u32;

    for _ in 0..config.refine_passes {
        for l in 0..num_layers {
            let bg_t = compute_background(layers, l, ink);
            for p in 0..num_positions {
                let cur = choices[[l, p]];
                let (y0, y1, x0, x1) = tile_rect(p, layer_offsets[l]);
                let target_tile = target.slice(s![y0..y1, x0..x1]);
                let target_ink = 1.0 - target_tile.mean().unwrap_or(1.0);
                if target_ink > config.band_max {
                    continue;
                }
                // Dithering is for tone. Optimized content may only be
                // reshaped where the target tile is flat; structured light
                // areas (highlights, textures) keep their detail.
                if cur != blank_idx && cur != mark_idx && !is_flat(&target_tile) {
                    continue;
                }
                let (lat_r, lat_c) = lattice(y0, x0);

                // Window = tile ⊕ blur halo; the blurred delta of a tile
                // change has exactly this support.
                let wy0 = y0.saturating_sub(ry);
                let wy1 = (y1 + ry).min(img_h);
                let wx0 = x0.saturating_sub(rx);
                let wx1 = (x1 + rx).min(img_w);

                let mut best: Option<(u16, Array2<f32>, f32)> = None;
                for cand in [blank_idx, mark_idx] {
                    if cand == cur {
                        continue;
                    }
                    if cand == mark_idx && blocked(&occupied, lat_r, lat_c) {
                        continue; // spacing: no fusing with a neighboring mark
                    }
                    let tile_new = composed_tile(ink, &bg_t, cand, y0, y1, x0, x1);
                    let delta_tile = &tile_new - &mockup.slice(s![y0..y1, x0..x1]);
                    let delta_blur =
                        blur_delta(&delta_tile, y0 - wy0, x0 - wx0, wy1 - wy0, wx1 - wx0, rx, ry);
                    let t = lf_target.slice(s![wy0..wy1, wx0..wx1]);
                    let m = lf_mockup.slice(s![wy0..wy1, wx0..wx1]);
                    // Δ = Σ [d² − 2·(t − m)·d]
                    let mut delta = 0.0f32;
                    for ((tv, mv), dv) in t.iter().zip(m.iter()).zip(delta_blur.iter()) {
                        let e = tv - mv;
                        delta += *dv * *dv - 2.0 * e * *dv;
                    }
                    if best.as_ref().map_or(true, |b| delta < b.2) {
                        best = Some((cand, delta_blur, delta));
                    }
                }

                if let Some((cand, delta_blur, delta)) = best {
                    if delta < -1e-12 {
                        let tile_new = composed_tile(ink, &bg_t, cand, y0, y1, x0, x1);
                        apply_choice_with_tile(
                            mockup,
                            choices,
                            layers,
                            chars,
                            l,
                            p,
                            cand,
                            y0,
                            y1,
                            x0,
                            x1,
                            &tile_new,
                        );
                        lf_mockup
                            .slice_mut(s![wy0..wy1, wx0..wx1])
                            .zip_mut_with(&delta_blur, |m, d| *m += d);
                        if cand == mark_idx {
                            occupied.insert((lat_r, lat_c));
                        } else if cur == mark_idx
                            && !points_all
                                .get(&(lat_r, lat_c))
                                .map_or(false, |cells| {
                                    cells.iter().any(|&(l2, p2)| choices[[l2, p2]] == mark_idx)
                                })
                        {
                            occupied.remove(&(lat_r, lat_c));
                        }
                        flipped += 1;
                    }
                }
            }
        }
    }

    let lf_after = mean_sq_err(&lf_target, &lf_mockup);
    DitherStats {
        seeded,
        flipped,
        lf_before,
        lf_after,
    }
}

/// Keep cached τ-space backgrounds in sync after placing a mark on layer
/// `layer`: the other layers' backgrounds pick up the new glyph's ink, so
/// overstrike copies see their true (diminished) demand.
fn update_bg_cache(
    bg_cache: &mut HashMap<usize, Array2<f32>>,
    ink: &Ink,
    mark_idx: u16,
    layer: usize,
    y0: usize,
    y1: usize,
    x0: usize,
    x1: usize,
) {
    let mark_t = ink.chars_t.index_axis(Axis(0), mark_idx as usize);
    for (&l2, bg2) in bg_cache.iter_mut() {
        if l2 == layer {
            continue;
        }
        bg2.slice_mut(s![y0..y1, x0..x1])
            .zip_mut_with(&mark_t, |b, m| *b *= m);
    }
}

/// Lattice offsets at which a second mark would fuse with one placed at the
/// origin: any offset whose spacing is below the mark glyph's ink footprint
/// on both axes. (0, 0) is excluded — overstrike copies stack there.
fn mark_exclusion(glyph: &ndarray::ArrayView2<f32>, sy: f32, sx: f32) -> Vec<(isize, isize)> {
    let (h, w) = glyph.dim();
    let (mut y0, mut y1) = (h, 0usize);
    let (mut x0, mut x1) = (w, 0usize);
    for y in 0..h {
        for x in 0..w {
            if 1.0 - glyph[[y, x]] > 0.02 {
                y0 = y0.min(y);
                y1 = y1.max(y);
                x0 = x0.min(x);
                x1 = x1.max(x);
            }
        }
    }
    if y1 < y0 {
        return Vec::new();
    }
    let fh = (y1 - y0 + 1) as f32;
    let fw = (x1 - x0 + 1) as f32;
    let ry = (fh / sy).ceil().max(1.0) as isize;
    let rx = (fw / sx).ceil().max(1.0) as isize;
    let mut out = Vec::new();
    for dr in -ry..=ry {
        for dc in -rx..=rx {
            if dr == 0 && dc == 0 {
                continue;
            }
            if (dr as f32 * sy).abs() < fh && (dc as f32 * sx).abs() < fw {
                out.push((dr, dc));
            }
        }
    }
    out
}

/// A target tile is "flat tone" when its brightness varies less than
/// [`FLAT_STD`] — the only place where optimized content may be replaced by
/// dither marks.
fn is_flat(target_tile: &ndarray::ArrayView2<f32>) -> bool {
    let n = target_tile.len() as f32;
    let mean = target_tile.mean().unwrap_or(0.0);
    let var = target_tile.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    var.sqrt() < FLAT_STD
}

/// Threshold of the chosen screen at one composite-lattice cell, in [0, 1).
fn screen_threshold(mode: DitherMode, row: usize, col: usize) -> f32 {
    match mode {
        DitherMode::Bayer4 => BAYER4[row % 4][col % 4] as f32 / 16.0,
        DitherMode::Bayer8 => BAYER8[row % 8][col % 8] as f32 / 64.0,
        DitherMode::BlueNoise => BLUE_NOISE_16[row % 16][col % 16] as f32 / 256.0,
        DitherMode::FloydSteinberg | DitherMode::None => 0.5,
    }
}

/// The scattered mark: the lightest non-blank glyph (by mean ink), or an
/// explicit index.
fn resolve_mark(chars: &Chars, mark: DitherMark) -> Option<u16> {
    match mark {
        DitherMark::Char(c) if (c as usize) < chars.shape()[0] => Some(c),
        DitherMark::Char(_) => None,
        DitherMark::Auto => {
            let blank = least_ink_char(chars);
            let mut best: Option<(u16, f32)> = None;
            for c in 0..chars.shape()[0] {
                if c as u16 == blank {
                    continue;
                }
                let ink = mean_ink(chars, c);
                if ink > 1e-4 && best.map_or(true, |(_, b)| ink < b) {
                    best = Some((c as u16, ink));
                }
            }
            best.map(|(c, _)| c)
        }
    }
}

/// The glyph with the least ink (the blank key), lowest index on ties.
fn least_ink_char(chars: &Chars) -> u16 {
    let mut best = 0usize;
    let mut best_ink = f32::INFINITY;
    for c in 0..chars.shape()[0] {
        let ink = mean_ink(chars, c);
        if ink < best_ink {
            best_ink = ink;
            best = c;
        }
    }
    best as u16
}

fn mean_ink(chars: &Chars, c: usize) -> f32 {
    let g = chars.index_axis(Axis(0), c);
    let total: f32 = g.iter().map(|v| 1.0 - v).sum();
    total / g.len() as f32
}

/// The mockup tile that results from placing `cand` at this cell.
fn composed_tile(
    ink: &Ink,
    bg_t: &Array2<f32>,
    cand: u16,
    y0: usize,
    y1: usize,
    x0: usize,
    x1: usize,
) -> Array2<f32> {
    let mark_t = ink.chars_t.index_axis(Axis(0), cand as usize);
    let mut tile = Array2::<f32>::zeros((y1 - y0, x1 - x0));
    for y in y0..y1 {
        for x in x0..x1 {
            tile[[y - y0, x - x0]] = ink.compose(bg_t[[y, x]], mark_t[[y - y0, x - x0]]);
        }
    }
    tile
}

#[allow(clippy::too_many_arguments)]
fn apply_choice(
    mockup: &mut Image,
    choices: &mut Choices,
    layers: &mut Layers,
    bg_t: &Array2<f32>,
    chars: &Chars,
    ink: &Ink,
    layer: usize,
    position: usize,
    cand: u16,
    y0: usize,
    y1: usize,
    x0: usize,
    x1: usize,
) {
    let tile = composed_tile(ink, bg_t, cand, y0, y1, x0, x1);
    apply_choice_with_tile(
        mockup, choices, layers, chars, layer, position, cand, y0, y1, x0, x1, &tile,
    );
}

#[allow(clippy::too_many_arguments)]
fn apply_choice_with_tile(
    mockup: &mut Image,
    choices: &mut Choices,
    layers: &mut Layers,
    chars: &Chars,
    layer: usize,
    position: usize,
    cand: u16,
    y0: usize,
    y1: usize,
    x0: usize,
    x1: usize,
    tile: &Array2<f32>,
) {
    choices[[layer, position]] = cand;
    let glyph = chars.index_axis(Axis(0), cand as usize);
    layers.slice_mut(s![layer, y0..y1, x0..x1]).assign(&glyph);
    mockup.slice_mut(s![y0..y1, x0..x1]).assign(tile);
}

/// Separable box blur with zero padding (linear — incremental updates via
/// [`blur_delta`] are exact).
fn blur_box(img: &Array2<f32>, rx: usize, ry: usize) -> Array2<f32> {
    let (h, w) = img.dim();
    let mut tmp = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        let mut cs = vec![0f32; w + 1];
        for x in 0..w {
            cs[x + 1] = cs[x] + img[[y, x]];
        }
        for x in 0..w {
            let lo = x.saturating_sub(rx);
            let hi = (x + rx + 1).min(w);
            tmp[[y, x]] = (cs[hi] - cs[lo]) / (2 * rx + 1) as f32;
        }
    }
    let mut out = Array2::<f32>::zeros((h, w));
    for x in 0..w {
        let mut cs = vec![0f32; h + 1];
        for y in 0..h {
            cs[y + 1] = cs[y] + tmp[[y, x]];
        }
        for y in 0..h {
            let lo = y.saturating_sub(ry);
            let hi = (y + ry + 1).min(h);
            out[[y, x]] = (cs[hi] - cs[lo]) / (2 * ry + 1) as f32;
        }
    }
    out
}

/// Blur of a tile-sized delta, sampled on a `(win_h, win_w)` window whose
/// origin sits `(off_y, off_x)` from the tile origin. Exact: the delta's
/// support is the tile, so the blur's support is the tile ⊕ halo.
fn blur_delta(
    delta_tile: &Array2<f32>,
    off_y: usize,
    off_x: usize,
    win_h: usize,
    win_w: usize,
    rx: usize,
    ry: usize,
) -> Array2<f32> {
    let mut extended = Array2::<f32>::zeros((win_h, win_w));
    let (dh, dw) = delta_tile.dim();
    extended
        .slice_mut(s![off_y..off_y + dh, off_x..off_x + dw])
        .assign(delta_tile);
    blur_box(&extended, rx, ry)
}

fn mean_sq_err(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    let mut total = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y) as f64;
        total += d * d;
    }
    (total / a.len() as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::InkFloor;
    use ndarray::{arr2, arr3};

    /// 3 glyphs: blank (no ink), a small mark (0.25 ink), a solid dark glyph.
    fn test_chars() -> Chars {
        arr3(&[
            [[1.0f32, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ])
    }

    /// One layer of 2×2 glyphs on a `rows*2 × cols*2` canvas, all blank.
    fn blank_state(rows: usize, cols: usize) -> (Choices, Layers, Image) {
        let h = rows * 2;
        let w = cols * 2;
        let choices = Choices::zeros((1, rows * cols));
        let layers = Layers::from_elem((1, h, w), 1.0f32);
        let mockup = Image::from_elem((h, w), 1.0f32);
        (choices, layers, mockup)
    }

    fn cfg(mode: DitherMode) -> DitherConfig {
        DitherConfig {
            mode,
            band_max: 0.15,
            refine_passes: 2,
            mark: DitherMark::Auto,
        }
    }

    #[test]
    fn test_ordered_seeding_hits_requested_coverage() {
        // 4×4 cells of a uniform 0.05-ink target; mark ink is 0.25 →
        // demand 0.2 marks/lattice-point. Bayer4 screens below 0.2 at
        // exactly 4 of 16 cells.
        let chars = test_chars();
        let ink = Ink::new(InkFloor::Fixed(0.0), &chars);
        let (mut choices, mut layers, mut mockup) = blank_state(4, 4);
        let target = Image::from_elem((8, 8), 0.95f32);

        let stats = run(
            &mut mockup,
            &mut choices,
            &mut layers,
            &target,
            &chars,
            &ink,
            &[(0, 0)],
            4,
            &DitherConfig {
                refine_passes: 0,
                ..cfg(DitherMode::Bayer4)
            },
        );
        let marks = choices.iter().filter(|&&c| c == 1).count();
        assert_eq!(marks, 4, "demand 0.2 on 16 cells → 4 marks");
        assert_eq!(stats.seeded, 4);
    }

    #[test]
    fn test_floyd_steinberg_preserves_mean_density() {
        // Uniform 0.10-ink target → demand 0.4 → ~40 % marks over 64 cells.
        let chars = test_chars();
        let ink = Ink::new(InkFloor::Fixed(0.0), &chars);
        let (mut choices, mut layers, mut mockup) = blank_state(8, 8);
        let target = Image::from_elem((16, 16), 0.90f32);

        run(
            &mut mockup,
            &mut choices,
            &mut layers,
            &target,
            &chars,
            &ink,
            &[(0, 0)],
            8,
            &DitherConfig {
                refine_passes: 0,
                ..cfg(DitherMode::FloydSteinberg)
            },
        );
        let marks = choices.iter().filter(|&&c| c == 1).count();
        assert!(
            (22..=30).contains(&marks),
            "demand 0.4 on 64 cells → ~26 marks, got {marks}"
        );
    }

    #[test]
    fn test_dark_targets_are_not_dithered() {
        // Target ink 0.5 is outside the light band — untouched.
        let chars = test_chars();
        let ink = Ink::new(InkFloor::Fixed(0.0), &chars);
        let (mut choices, mut layers, mut mockup) = blank_state(4, 4);
        let target = Image::from_elem((8, 8), 0.5f32);

        let stats = run(
            &mut mockup,
            &mut choices,
            &mut layers,
            &target,
            &chars,
            &ink,
            &[(0, 0)],
            4,
            &cfg(DitherMode::Bayer4),
        );
        assert_eq!(stats.seeded, 0);
        assert_eq!(stats.flipped, 0);
    }

    #[test]
    fn test_gradient_dithers_to_local_mean() {
        // Left half 0.02 ink, right half 0.10 ink (both in band): the stage
        // must lower the blurred error and put more marks on the darker side.
        let chars = test_chars();
        let ink = Ink::new(InkFloor::Fixed(0.0), &chars);
        let (mut choices, mut layers, mut mockup) = blank_state(8, 12);
        let mut target = Image::from_elem((16, 24), 0.98f32);
        target.slice_mut(s![.., 12..]).fill(0.90);

        let stats = run(
            &mut mockup,
            &mut choices,
            &mut layers,
            &target,
            &chars,
            &ink,
            &[(0, 0)],
            12,
            &cfg(DitherMode::Bayer4),
        );
        assert!(stats.lf_after < stats.lf_before, "LF error must drop");
        // The reported LF error must be the truth, not an incrementally
        // maintained fiction: recompute it from the returned mockup.
        let rx = (((2usize as f32) * 1.5).round() as usize).max(1);
        let ry = rx;
        let recomputed = mean_sq_err(&blur_box(&target, rx, ry), &blur_box(&mockup, rx, ry));
        assert!(
            (recomputed - stats.lf_after).abs() < 1e-6,
            "lf_after {recomputed} vs reported {}",
            stats.lf_after
        );
        let left: usize = (0..8)
            .map(|r| (0..6).filter(|&c| choices[[0, r * 12 + c]] == 1).count())
            .sum();
        let right: usize = (0..8)
            .map(|r| (6..12).filter(|&c| choices[[0, r * 12 + c]] == 1).count())
            .sum();
        assert!(right > left, "darker half needs more marks ({right} vs {left})");
        assert!(stats.flipped > 0, "refinement must adjust the seeded pattern");
    }

    #[test]
    fn test_dither_is_deterministic() {
        let chars = test_chars();
        let ink = Ink::new(InkFloor::Fixed(0.0), &chars);
        let run_once = || {
            let (mut choices, mut layers, mut mockup) = blank_state(6, 6);
            let mut target = Image::from_elem((12, 12), 0.93f32);
            target.slice_mut(s![.., 6..]).fill(0.97);
            run(
                &mut mockup,
                &mut choices,
                &mut layers,
                &target,
                &chars,
                &ink,
                &[(0, 0)],
                6,
                &cfg(DitherMode::BlueNoise),
            );
            choices
        };
        assert_eq!(run_once(), run_once());
    }

    #[test]
    fn test_mark_can_be_explicit() {
        // Explicit mark index 2 (solid dark glyph): heavy ink → fewer cells.
        let chars = test_chars();
        let ink = Ink::new(InkFloor::Fixed(0.0), &chars);
        let (mut choices, mut layers, mut mockup) = blank_state(4, 4);
        let target = Image::from_elem((8, 8), 0.95f32);
        let config = DitherConfig {
            mode: DitherMode::Bayer4,
            mark: DitherMark::Char(2),
            refine_passes: 0,
            ..cfg(DitherMode::Bayer4)
        };
        run(
            &mut mockup,
            &mut choices,
            &mut layers,
            &target,
            &chars,
            &ink,
            &[(0, 0)],
            4,
            &config,
        );
        // Mark ink 1.0 → demand 0.05 → only the 0.0-threshold cell.
        assert_eq!(choices.iter().filter(|&&c| c == 2).count(), 1);
    }

    #[test]
    fn test_blur_delta_matches_full_blur() {
        // Linearity check: blur(m + δ) = blur(m) + blur_delta(δ).
        let m = arr2(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let delta = arr2(&[[0.5f32, -1.0, 0.25], [0.0, 2.0, -0.5]]);
        let (rx, ry) = (1, 1);
        let sum = blur_box(&(&m + &delta), rx, ry);
        let base = blur_box(&m, rx, ry);
        let inc = blur_delta(&delta, 0, 0, 2, 3, rx, ry);
        for ((s, b), i) in sum.iter().zip(base.iter()).zip(inc.iter()) {
            assert!((s - (b + i)).abs() < 1e-5);
        }
    }

    /// Overstrike copies at one lattice point share the demand instead of
    /// each claiming it. A grey key (0.75 brightness) struck twice lands at
    /// 0.5625; the target (ink 0.375) wants 1.5 strikes-worth per cell, so
    /// copy 1 marks everywhere and copy 2 marks where the screen is below
    /// 2/3 (11 of 16 cells) — 27 marks whose mean ink ≈ the target tone.
    #[test]
    fn test_overstrike_copies_share_the_demand() {
        let chars: Chars = arr3(&[
            [[1.0f32, 1.0], [1.0, 1.0]],   // blank
            [[0.75, 0.75], [0.75, 0.75]],  // grey key
            [[0.0, 0.0], [0.0, 0.0]],      // dark
        ]);
        let ink = Ink::new(InkFloor::Fixed(0.0), &chars);
        let rows = 4;
        let cols = 4;
        let mut choices = Choices::zeros((2, rows * cols));
        let mut layers = Layers::from_elem((2, rows * 2, cols * 2), 1.0f32);
        let mut mockup = Image::from_elem((rows * 2, cols * 2), 1.0f32);
        let target = Image::from_elem((rows * 2, cols * 2), 0.625f32); // ink 0.375

        run(
            &mut mockup,
            &mut choices,
            &mut layers,
            &target,
            &chars,
            &ink,
            &[(0, 0), (0, 0)],
            cols,
            &DitherConfig {
                refine_passes: 0,
                band_max: 0.5,
                ..cfg(DitherMode::Bayer4)
            },
        );
        let marks = choices.iter().filter(|&&c| c == 1).count();
        assert_eq!(marks, 27, "overstrike copies must share the demand");
    }

    /// The screen is sampled on the composite lattice, so marks follow one
    /// coherent pattern across layers instead of per-layer noise.
    #[test]
    fn test_screen_is_coherent_across_layers() {
        let chars = test_chars();
        let ink = Ink::new(InkFloor::Fixed(0.0), &chars);
        // 2×2 cells per layer at pixel offsets (0,0),(0,1),(1,0),(1,1):
        // lattice = the pixel grid itself (4×4 points, one cell each).
        // Canvas carries the offset margin like the real pipeline's padding.
        let mut choices = Choices::zeros((4, 4));
        let mut layers = Layers::from_elem((4, 5, 5), 1.0f32);
        let mut mockup = Image::from_elem((5, 5), 1.0f32);
        let target = Image::from_elem((5, 5), 0.9f32); // ink 0.10
        let offsets = [(0usize, 0usize), (0, 1), (1, 0), (1, 1)];

        run(
            &mut mockup,
            &mut choices,
            &mut layers,
            &target,
            &chars,
            &ink,
            &offsets,
            2,
            &DitherConfig {
                refine_passes: 0,
                ..cfg(DitherMode::Bayer4)
            },
        );
        // demand = 0.10 / (0.25 × 4) = 0.1 → cells where Bayer4 < 0.1:
        // lattice (0,0) = 0/16 and (2,2) = 1/16 → layer offsets (0,0) at
        // positions (0,0) and (1,1) of that layer.
        for l in 0..4 {
            for p in 0..4 {
                let row = p / 2;
                let col = p % 2;
                let expect = if l == 0 && ((row == 0 && col == 0) || (row == 1 && col == 1)) {
                    1
                } else {
                    0
                };
                assert_eq!(
                    choices[[l, p]], expect,
                    "layer {l} pos {p}: coherent screen must place marks only at (0,0)/(2,2)"
                );
            }
        }
    }

    /// Marks never fuse: the exclusion spacing derived from the mark's
    /// footprint holds even at full demand.
    #[test]
    fn test_marks_keep_their_distance() {
        // Mark footprint is 2 px wide; the lattice is 1 px apart in x
        // (two layers offset by one pixel), so neighbors must be skipped.
        let chars: Chars = arr3(&[
            [[1.0f32, 1.0], [1.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]);
        let ink = Ink::new(InkFloor::Fixed(0.0), &chars);
        let rows = 1;
        let cols = 4;
        let mut choices = Choices::zeros((2, rows * cols));
        let mut layers = Layers::from_elem((2, 2, 2 * cols + 1), 1.0f32);
        let mut mockup = Image::from_elem((2, 2 * cols + 1), 1.0f32); // blank start
        let target = Image::from_elem((2, 2 * cols + 1), 0.0f32); // ink 1.0

        run(
            &mut mockup,
            &mut choices,
            &mut layers,
            &target,
            &chars,
            &ink,
            &[(0, 0), (0, 1)],
            cols,
            &DitherConfig {
                refine_passes: 0,
                band_max: 1.0,
                ..cfg(DitherMode::Bayer4)
            },
        );
        let marks = choices.iter().filter(|&&c| c == 1).count();
        assert!(marks > 0, "some marks must be placed");
        assert!(
            marks <= 4,
            "spacing must thin full-demand placement, got {marks}"
        );
        // Pairwise invariant: no two marked cells within the exclusion kernel.
        let mut pts: Vec<(isize, isize)> = Vec::new();
        for l in 0..2 {
            for p in 0..rows * cols {
                if choices[[l, p]] == 1 {
                    let row = (p / cols) as isize;
                    let col = (p % cols) as isize;
                    pts.push((row * 2, col * 2 + l as isize));
                }
            }
        }
        for (i, a) in pts.iter().enumerate() {
            for b in pts.iter().skip(i + 1) {
                let (dr, dc) = (a.0 - b.0, a.1 - b.1);
                let fuse = (dr * 2).abs() < 1 && dc.abs() < 2;
                assert!(!fuse, "marks at lattice {:?} and {:?} would fuse", a, b);
            }
        }
    }

    /// Dithering is for tone: optimized content may only be replaced where
    /// the target tile is flat. A structured light tile keeps its content.
    #[test]
    fn test_content_flips_protect_structure() {
        let chars = test_chars();
        let ink = Ink::new(InkFloor::Fixed(0.0), &chars);

        let run_one = |target: Image| {
            let mut choices = Choices::zeros((1, 1));
            choices[[0, 0]] = 2; // the dark "content" glyph fills the tile
            let mut layers = Layers::from_elem((1, 2, 2), 1.0f32);
            layers
                .slice_mut(s![0, .., ..])
                .assign(&chars.index_axis(Axis(0), 2));
            let mut mockup = Image::from_elem((2, 2), 0.0f32);
            run(
                &mut mockup,
                &mut choices,
                &mut layers,
                &target,
                &chars,
                &ink,
                &[(0, 0)],
                1,
                &DitherConfig {
                    refine_passes: 2,
                    band_max: 0.2,
                    ..cfg(DitherMode::Bayer4)
                },
            );
            choices[[0, 0]]
        };

        // Flat 0.15-ink target: the content glyph (ink 1.0) is wildly too
        // dark — reshaping it is right and the flip must happen.
        let flat = Image::from_elem((2, 2), 0.85f32);
        assert_ne!(run_one(flat), 2, "flat tone tile: content should be reshaped");

        // Same mean ink but structured (alternating): detail must survive.
        let structured = arr2(&[[0.7f32, 1.0], [1.0, 0.7]]);
        assert_eq!(
            run_one(structured),
            2,
            "structured tile: content must be protected"
        );
    }
}
