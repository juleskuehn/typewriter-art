//! Batch character evaluation.
//!
//! The key performance optimization: instead of computing composite images
//! one character at a time, we compute all characters at once using SIMD-friendly
//! array operations.

use ndarray::{Array1, Array2, Axis};

use crate::types::Chars;

/// Batch-evaluate all characters against a background slice.
///
/// Computes the asymmetric MSE for every character in the charset against
/// the given background and target slice.
///
/// Returns an array of MSE values, one per character: `(num_chars,)`.
pub fn evaluate_all_chars(
    bg_slice: &Array2<f32>,
    target_slice: &Array2<f32>,
    chars: &Chars,
    asymmetry: f32,
) -> Array1<f32> {
    let num_chars = chars.shape()[0];
    let h = chars.shape()[1];
    let w = chars.shape()[2];
    let n = (h * w) as f32;

    let mut mses = Array1::zeros(num_chars);

    for c in 0..num_chars {
        let char_slice = chars.index_axis(Axis(0), c);
        let mut total: f64 = 0.0;

        for y in 0..h {
            for x in 0..w {
                let bg_val = bg_slice[[y, x]];
                let char_val = char_slice[[y, x]];
                let target_val = target_slice[[y, x]];

                let composite = bg_val * char_val;
                let err = target_val - composite;
                let weighted = if err > 0.0 {
                    err * (1.0 + asymmetry)
                } else {
                    err
                };
                total += (weighted * weighted) as f64;
            }
        }

        mses[c] = (total / n as f64) as f32;
    }

    mses
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr3;

    #[test]
    fn test_evaluate_all_chars_basic() {
        // Simple 2-character, 2x2 test
        let bg = Array2::from_shape_vec((2, 2), vec![1.0, 0.5, 1.0, 0.5]).unwrap();
        let target = Array2::from_shape_vec((2, 2), vec![0.5, 0.3, 0.5, 0.3]).unwrap();
        let chars = arr3(&[
            [[1.0, 0.0], [1.0, 0.0]], // dark on right
            [[0.0, 1.0], [0.0, 1.0]], // dark on left
        ]);

        let mses = evaluate_all_chars(&bg, &target, &chars, 0.0);
        assert_eq!(mses.len(), 2);
        // Both should produce non-negative MSE
        assert!(mses[0] >= 0.0);
        assert!(mses[1] >= 0.0);
        // They should differ
        assert!(mses[0] != mses[1]);
    }
}
