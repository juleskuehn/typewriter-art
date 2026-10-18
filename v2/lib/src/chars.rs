//! Batch character evaluation.
//!
//! The key performance optimization: instead of computing composite images
//! one character at a time, we compute all characters at once using SIMD-friendly
//! array operations.

use ndarray::{Array1, Array2, Axis};

use crate::render::Ink;

/// Batch-evaluate all characters against a background slice.
///
/// Computes the asymmetric MSE for every character in the charset against
/// the given background and target slice, under the ink model
/// ([`Ink::compose`]). `bg_slice` is the τ-space background of the other
/// layers ([`crate::render::compute_background`]); the glyphs come from
/// `ink.chars_t`.
///
/// `weight` is an optional per-pixel
/// weight slice (same shape, e.g. "center focus") applied to the squared
/// error before averaging; pass `None` for uniform weighting.
///
/// Returns an array of MSE values, one per character: `(num_chars,)`.
pub fn evaluate_all_chars(
    bg_slice: &Array2<f32>,
    target_slice: &Array2<f32>,
    ink: &Ink,
    asymmetry: f32,
    weight: Option<&Array2<f32>>,
) -> Array1<f32> {
    let num_chars = ink.chars_t.shape()[0];
    let h = ink.chars_t.shape()[1];
    let w = ink.chars_t.shape()[2];
    let n = (h * w) as f32;

    let mut mses = Array1::zeros(num_chars);

    for c in 0..num_chars {
        let char_slice = ink.chars_t.index_axis(Axis(0), c);
        let mut total: f64 = 0.0;

        for y in 0..h {
            for x in 0..w {
                let bg_val = bg_slice[[y, x]];
                let char_val = char_slice[[y, x]];
                let target_val = target_slice[[y, x]];

                let composite = ink.compose(bg_val, char_val);
                let err = target_val - composite;
                let weighted = if err > 0.0 {
                    err * (1.0 + asymmetry)
                } else {
                    err
                };
                let w = weight.map_or(1.0, |m| m[[y, x]]);
                total += (weighted * weighted) as f64 * w as f64;
            }
        }

        mses[c] = (total / n as f64) as f32;
    }

    mses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Chars, InkFloor};
    use ndarray::arr3;

    #[test]
    fn test_evaluate_all_chars_basic() {
        // Simple 2-character, 2x2 test
        let bg = Array2::from_shape_vec((2, 2), vec![1.0, 0.5, 1.0, 0.5]).unwrap();
        let target = Array2::from_shape_vec((2, 2), vec![0.5, 0.3, 0.5, 0.3]).unwrap();
        let chars: Chars = arr3(&[
            [[1.0, 0.0], [1.0, 0.0]], // dark on right
            [[0.0, 1.0], [0.0, 1.0]], // dark on left
        ]);
        let ink = Ink::new(InkFloor::Fixed(0.0), &chars);

        let mses = evaluate_all_chars(&bg, &target, &ink, 0.0, None);
        assert_eq!(mses.len(), 2);
        // Both should produce non-negative MSE
        assert!(mses[0] >= 0.0);
        assert!(mses[1] >= 0.0);
        // They should differ
        assert!(mses[0] != mses[1]);
    }

    /// With a floor > 0 the overlap of two 0.5 strikes is the saturating
    /// value, not the plain product (0.25): the candidate matching that
    /// value must win.
    #[test]
    fn test_overlap_uses_saturating_composite() {
        let chars: Chars = arr3(&[[[0.5f32, 0.5], [0.5, 0.5]]]);
        let ink = Ink::new(InkFloor::Fixed(0.1), &chars);
        // One other layer already contributes a 0.5 strike: τ-space bg.
        let bg_t = Array2::from_elem((2, 2), ink.tau(0.5));
        let expected = ink.compose(ink.tau(0.5), ink.tau(0.5));
        assert!((expected - 0.2777778).abs() < 1e-5, "got {expected}");

        let target = Array2::from_elem((2, 2), expected);
        let mses = evaluate_all_chars(&bg_t, &target, &ink, 0.0, None);
        assert!(mses[0] < 1e-10, "saturating composite must match exactly");

        // The naive product target (0.25) does NOT match under the model.
        let naive = Array2::from_elem((2, 2), 0.25);
        let mses = evaluate_all_chars(&bg_t, &naive, &ink, 0.0, None);
        assert!(mses[0] > 1e-4);
    }
}
