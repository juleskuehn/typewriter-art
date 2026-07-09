//! Search algorithms for character selection.
//!
//! Implements greedy and simulated annealing search strategies.
//! Uses batch character evaluation for SIMD-friendly performance:
//! all character composites are computed at once, then candidates are
//! evaluated in random order (preserving SA semantics).

use ndarray::{s, Array1, Axis};
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

            // Compute current error
            let cur_composite = mockup_slice;
            let cur_amse = asymmetric_mse(&target_slice, &cur_composite, asymmetry);

            // Batch-evaluate all characters
            let all_mses = evaluate_all_chars(&bg_slice, &target_slice, chars, asymmetry);

            // Select character using search strategy
            let (new_choice, new_composite, comparisons) = match mode {
                SearchAlgorithm::Greedy => {
                    greedy_search(&all_mses, prev_choice, &bg_slice, chars)
                }
                SearchAlgorithm::SimAnneal => {
                    simulated_annealing(
                        &all_mses,
                        cur_amse,
                        prev_choice,
                        &bg_slice,
                        chars,
                        temperature,
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

/// Greedy search: pick the character with the lowest MSE.
///
/// Never picks the same character as current (to force exploration).
/// Returns the character index, its composite image, and comparisons made.
pub fn greedy_search(
    all_mses: &Array1<f32>,
    prev_choice: u16,
    bg_slice: &Image,
    chars: &Chars,
) -> (u16, Image, u32) {
    let num_chars = chars.shape()[0] as u16;
    let mut best_idx = prev_choice;
    let mut best_mse = f32::MAX;
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
/// using batch-computed errors.
pub fn simulated_annealing(
    all_mses: &Array1<f32>,
    cur_amse: f32,
    prev_choice: u16,
    bg_slice: &Image,
    chars: &Chars,
    temperature: f32,
) -> (u16, Image, u32) {
    let num_chars = chars.shape()[0] as u16;

    // Generate random permutation of character indices
    let mut indices: Vec<u16> = (0..num_chars).collect();
    // Shuffle in-place
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    indices.shuffle(&mut rng);

    for &new_idx in &indices {
        if new_idx == prev_choice {
            continue;
        }

        let new_amse = all_mses[new_idx as usize];
        let delta = cur_amse - new_amse;

        // Always accept improvements
        if delta > 0.0 {
            let composite = bg_slice * &chars.index_axis(Axis(0), new_idx as usize);
            return (new_idx, composite.to_owned(), indices.len() as u32);
        }

        // Probabilistically accept worse moves (SA)
        if temperature > 0.0 {
            let acceptance_prob = (delta / temperature).exp();
            if acceptance_prob > rand::random::<f32>() {
                let composite = bg_slice * &chars.index_axis(Axis(0), new_idx as usize);
                return (new_idx, composite.to_owned(), indices.len() as u32);
            }
        }
    }

    // No replacement found — keep current character
    let composite = bg_slice * &chars.index_axis(Axis(0), prev_choice as usize);
    (prev_choice, composite.to_owned(), indices.len() as u32)
}

/// Compute asymmetric mean squared error.
///
/// Errors where the mockup is too light (err > 0) are penalized by `(1+asymmetry)`.
/// This is critical because the multiplicative model can only darken pixels;
/// once a region is too light, later layers can't fix it.
#[inline]
pub fn asymmetric_mse(target: &Image, composite: &Image, asymmetry: f32) -> f32 {
    let diff = target - composite;
    let mut total: f64 = 0.0;
    let n = diff.len();

    // Single-pass computation for better performance
    for (&x, &y) in target.iter().zip(composite.iter()) {
        let err = x - y;
        let weighted = if err > 0.0 {
            err * (1.0 + asymmetry)
        } else {
            err
        };
        total += (weighted * weighted) as f64;
    }

    (total / n as f64) as f32
}
