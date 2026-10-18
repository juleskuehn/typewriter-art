//! Search algorithms for character selection.
//!
//! Implements greedy and simulated annealing search strategies.
//! Uses batch character evaluation for SIMD-friendly performance:
//! all character composites are computed at once, then candidates are
//! evaluated in random order (preserving SA semantics).

use ndarray::{s, Array1, Axis};
use rand::SeedableRng;
use rayon::prelude::*;

use crate::chars::evaluate_all_chars;
use crate::types::{Chars, Image, SearchAlgorithm};

/// Optimize all positions in a single layer pass.
///
/// This is the core optimization function. For each position in the layer:
/// 1. Extract the target and background slices
/// 2. Batch-evaluate all characters against the background
/// 3. Select the best character using the chosen search strategy
/// 4. Update the mockup with the chosen composite
///
/// Positions are processed in parallel with Rayon.
///
/// `weights` is an optional per-pixel loss weight map covering the whole
/// canvas ("center focus"); each position uses its slice of it. Pass `None`
/// for uniform weighting.
///
/// `rng_seed` seeds the per-position RNGs used by simulated annealing, so a
/// run is reproducible given the same seed (positions run in parallel, so each
/// derives its own stream from `rng_seed + position`).
///
/// Returns updated choices, updated mockup, total comparisons, and mean error.
pub fn optimize_layer_pass(
    bg: &Image,
    mockup: &mut Image,
    target: &Image,
    chars: &Chars,
    choices: &mut Array1<u16>,
    layer_offset: (usize, usize),
    asymmetry: f32,
    mode: SearchAlgorithm,
    temperature: f32,
    weights: Option<&Image>,
    rng_seed: u64,
) -> (u64, f32) {
    let num_positions = choices.len();
    let num_cols = target.ncols() / chars.shape()[2];
    let char_h = chars.shape()[1];
    let char_w = chars.shape()[2];

    // Collect results per position in parallel
    let results: Vec<(u32, u16, f32, Image)> = (0..num_positions)
        .into_par_iter()
        .map(|i| {
            let prev_choice = choices[i];
            let row = i / num_cols;
            let col = i % num_cols;

            // Extract slices for this position
            let y_start = row * char_h + layer_offset.0;
            let x_start = col * char_w + layer_offset.1;
            let y_end = y_start + char_h;
            let x_end = x_start + char_w;

            let target_slice = target
                .slice(s![y_start..y_end, x_start..x_end])
                .to_owned();
            let bg_slice = bg
                .slice(s![y_start..y_end, x_start..x_end])
                .to_owned();
            let mockup_slice = mockup
                .slice(s![y_start..y_end, x_start..x_end])
                .to_owned();
            let weight_slice = weights.map(|w| {
                w.slice(s![y_start..y_end, x_start..x_end])
                    .to_owned()
            });

            // Compute current error
            let cur_composite = mockup_slice;
            let cur_amse = asymmetric_mse(
                &target_slice,
                &cur_composite,
                asymmetry,
                weight_slice.as_ref(),
            );

            // Batch-evaluate all characters
            let all_mses =
                evaluate_all_chars(&bg_slice, &target_slice, chars, asymmetry, weight_slice.as_ref());

            // Select character using search strategy
            let (new_choice, new_composite, comparisons) = match mode {
                SearchAlgorithm::Greedy => {
                    greedy_search(&all_mses, cur_amse, prev_choice, &bg_slice, chars)
                }
                SearchAlgorithm::SimAnneal => {
                    let mut rng = rand::rngs::StdRng::seed_from_u64(rng_seed.wrapping_add(i as u64));
                    simulated_annealing(
                        &all_mses,
                        cur_amse,
                        prev_choice,
                        &bg_slice,
                        chars,
                        temperature,
                        &mut rng,
                    )
                }
            };

            (comparisons, new_choice, all_mses[new_choice as usize], new_composite)
        })
        .collect();

    // Apply results sequentially (safe: no two positions overlap)
    let mut total_comparisons: u64 = 0;
    let mut total_error: f64 = 0.0;

    for (i, (comparisons, new_choice, mse, composite)) in results.into_iter().enumerate() {
        let row = i / num_cols;
        let col = i % num_cols;

        let y_start = row * char_h + layer_offset.0;
        let x_start = col * char_w + layer_offset.1;
        let y_end = y_start + char_h;
        let x_end = x_start + char_w;

        choices[i] = new_choice;
        mockup
            .slice_mut(s![y_start..y_end, x_start..x_end])
            .assign(&composite);

        total_comparisons += comparisons as u64;
        total_error += mse as f64;
    }

    let mean_error = total_error / num_positions as f64;
    (total_comparisons, mean_error as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr3;

    fn test_chars() -> Chars {
        // 3 chars: blank (all white), half-dark, all-dark
        arr3(&[
            [[1.0, 1.0], [1.0, 1.0]],
            [[1.0, 0.5], [1.0, 0.5]],
            [[0.5, 0.5], [0.5, 0.5]],
        ])
    }

    #[test]
    fn test_greedy_keeps_current_when_best() {
        // Current char (index 1) is already the best choice:
        // bg * char 1 matches the target exactly, so no candidate
        // should beat it and greedy must keep it (v1 parity).
        let chars = test_chars();
        let bg = Image::from_elem((2, 2), 1.0);
        let target = Image::from_shape_vec((2, 2), vec![1.0, 0.5, 1.0, 0.5]).unwrap();

        let all_mses = evaluate_all_chars(&bg, &target, &chars, 0.0, None);
        let cur_amse = all_mses[1];

        let (choice, _composite, comparisons) =
            greedy_search(&all_mses, cur_amse, 1, &bg, &chars);
        assert_eq!(choice, 1);
        assert_eq!(comparisons, 3);
    }

    #[test]
    fn test_greedy_switches_when_better() {
        let chars = test_chars();
        let bg = Image::from_elem((2, 2), 1.0);
        // Target matches char 2 (all 0.5)
        let target = Image::from_elem((2, 2), 0.5);

        let all_mses = evaluate_all_chars(&bg, &target, &chars, 0.0, None);
        let cur_amse = all_mses[1];

        let (choice, _composite, _comparisons) =
            greedy_search(&all_mses, cur_amse, 1, &bg, &chars);
        assert_eq!(choice, 2);
    }

    #[test]
    fn test_sa_counts_examined_comparisons() {
        // With temperature 0, SA only accepts strict improvements.
        // Current char is best → all candidates examined, no switch.
        let chars = test_chars();
        let bg = Image::from_elem((2, 2), 1.0);
        let target = Image::from_shape_vec((2, 2), vec![1.0, 0.5, 1.0, 0.5]).unwrap();

        let all_mses = evaluate_all_chars(&bg, &target, &chars, 0.0, None);
        let cur_amse = all_mses[1];

        let (choice, _composite, comparisons) =
            simulated_annealing(&all_mses, cur_amse, 1, &bg, &chars, 0.0, &mut rand::rngs::StdRng::seed_from_u64(1));
        assert_eq!(choice, 1);
        assert_eq!(comparisons, 3);
    }

    #[test]
    fn test_seeded_passes_are_reproducible() {
        let chars = test_chars();
        let bg = Image::from_elem((2, 2), 1.0);
        let target = Image::from_shape_vec((2, 2), vec![0.4, 0.5, 0.6, 0.5]).unwrap();

        let run = |rng_seed: u64| {
            let mut mockup = Image::from_elem((2, 2), 1.0);
            // One 2x2 char fits a 2x2 target: a single position
            let mut choices = Array1::<u16>::zeros(1);
            optimize_layer_pass(
                &bg,
                &mut mockup,
                &target,
                &chars,
                &mut choices,
                (0, 0),
                0.1,
                SearchAlgorithm::SimAnneal,
                0.001,
                None,
                rng_seed,
            );
            choices
        };
        assert_eq!(run(7), run(7), "same seed → identical choices");
    }
}

/// Greedy search: pick the character with the lowest MSE.
///
/// Port of v1 semantics: only switch away from the current character when a
/// candidate is *strictly better* than the current composite's error
/// (`new_amse < cur_amse`); otherwise keep the current character.
/// Returns the character index, its composite image, and comparisons made.
pub fn greedy_search(
    all_mses: &Array1<f32>,
    cur_amse: f32,
    prev_choice: u16,
    bg_slice: &Image,
    chars: &Chars,
) -> (u16, Image, u32) {
    let num_chars = chars.shape()[0] as u16;
    let mut best_idx = prev_choice;
    let mut best_mse = cur_amse;
    let mut comparisons: u32 = 0;

    for idx in 0..num_chars {
        comparisons += 1;
        if idx == prev_choice {
            continue;
        }
        let mse = all_mses[idx as usize];
        if mse < best_mse {
            best_mse = mse;
            best_idx = idx;
        }
    }

    let composite = bg_slice * &chars.index_axis(Axis(0), best_idx as usize);
    (best_idx, composite.to_owned(), comparisons)
}

/// Simulated annealing search with random candidate evaluation order.
///
/// Precomputed MSEs are evaluated in random order, accepting the first
/// candidate that passes the SA test. This preserves SA semantics while
/// using batch-computed errors. `comparisons` counts the candidates actually
/// examined (matching v1's early-exit counting). The caller supplies the RNG
/// for reproducibility.
pub fn simulated_annealing(
    all_mses: &Array1<f32>,
    cur_amse: f32,
    prev_choice: u16,
    bg_slice: &Image,
    chars: &Chars,
    temperature: f32,
    rng: &mut (impl rand::Rng + ?Sized),
) -> (u16, Image, u32) {
    let num_chars = chars.shape()[0] as u16;

    // Generate random permutation of character indices
    let mut indices: Vec<u16> = (0..num_chars).collect();
    // Shuffle in-place
    use rand::seq::SliceRandom;
    indices.shuffle(rng);

    let mut comparisons: u32 = 0;
    for &new_idx in &indices {
        comparisons += 1;
        if new_idx == prev_choice {
            continue;
        }

        let new_amse = all_mses[new_idx as usize];
        let delta = cur_amse - new_amse;

        // Always accept improvements
        if delta > 0.0 {
            let composite = bg_slice * &chars.index_axis(Axis(0), new_idx as usize);
            return (new_idx, composite.to_owned(), comparisons);
        }

        // Probabilistically accept worse moves (SA)
        if temperature > 0.0 {
            let acceptance_prob = (delta / temperature).exp();
            if acceptance_prob > rng.gen::<f32>() {
                let composite = bg_slice * &chars.index_axis(Axis(0), new_idx as usize);
                return (new_idx, composite.to_owned(), comparisons);
            }
        }
    }

    // No replacement found — keep current character
    let composite = bg_slice * &chars.index_axis(Axis(0), prev_choice as usize);
    (prev_choice, composite.to_owned(), comparisons)
}

/// Compute asymmetric mean squared error.
///
/// Errors where the mockup is too light (err > 0) are penalized by `(1+asymmetry)`.
/// This is critical because the multiplicative model can only darken pixels;
/// once a region is too light, later layers can't fix it. `weight` is an
/// optional per-pixel weight slice (same shape) applied before averaging.
#[inline]
pub fn asymmetric_mse(
    target: &Image,
    composite: &Image,
    asymmetry: f32,
    weight: Option<&Image>,
) -> f32 {
    let mut total: f64 = 0.0;
    let n = target.len();

    // Single-pass computation for better performance
    match weight {
        Some(w) => {
            for ((&x, &y), &w) in target.iter().zip(composite.iter()).zip(w.iter()) {
                let err = x - y;
                let weighted = if err > 0.0 {
                    err * (1.0 + asymmetry)
                } else {
                    err
                };
                total += (weighted * weighted) as f64 * w as f64;
            }
        }
        None => {
            for (&x, &y) in target.iter().zip(composite.iter()) {
                let err = x - y;
                let weighted = if err > 0.0 {
                    err * (1.0 + asymmetry)
                } else {
                    err
                };
                total += (weighted * weighted) as f64;
            }
        }
    }

    (total / n as f64) as f32
}
