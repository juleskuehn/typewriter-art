//! Image quality metrics.
//!
//! Ports of `skimage.metrics.peak_signal_noise_ratio` and
//! `skimage.metrics.structural_similarity` with their default parameters, so
//! that auto-alignment scores are comparable with the v1 `autocropScores.json`
//! output (v1 scored uint8 images; these functions work on [`Image`] values in
//! [0.0, 1.0] with `data_range = 1.0`, which yields the same numbers).

use crate::types::Image;

/// Peak signal-to-noise ratio in dB.
///
/// `data_range` is 1.0 for images in [0.0, 1.0]. Returns `f32::INFINITY` for
/// identical images (skimage behavior).
pub fn psnr(a: &Image, b: &Image, data_range: f32) -> f32 {
    assert_eq!(a.dim(), b.dim(), "images must have the same shape");
    let mut sum = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = f64::from(x) - f64::from(y);
        sum += d * d;
    }
    if sum == 0.0 {
        return f32::INFINITY;
    }
    let mse = sum / a.len() as f64;
    (10.0 * (f64::from(data_range).powi(2) / mse).log10()) as f32
}

/// Mean structural similarity index (SSIM).
///
/// Matches `skimage.metrics.structural_similarity` defaults: 7×7 uniform
/// (box) window, sample covariance, `K1 = 0.01`, `K2 = 0.03`, and the window
/// border cropped before averaging. `data_range` is 1.0 for images in
/// [0.0, 1.0].
pub fn ssim(a: &Image, b: &Image, data_range: f32) -> f32 {
    assert_eq!(a.dim(), b.dim(), "images must have the same shape");
    let (h, w) = a.dim();
    let mut win = 7.min(h.min(w));
    if win % 2 == 0 {
        win -= 1;
    }
    assert!(win >= 3, "SSIM requires images of at least 3x3");

    let np = (win * win) as f64;
    // skimage's `use_sample_covariance=True` default
    let cov_norm = np / (np - 1.0);

    let ux = uniform_filter(a, win);
    let uy = uniform_filter(b, win);
    let uxx = uniform_filter(&(a * a), win);
    let uyy = uniform_filter(&(b * b), win);
    let uxy = uniform_filter(&(a * b), win);

    let c1 = (0.01 * f64::from(data_range)).powi(2);
    let c2 = (0.03 * f64::from(data_range)).powi(2);

    // Crop the window border before averaging (skimage crops `pad` on all sides)
    let pad = (win - 1) / 2;
    let mut total = 0.0f64;
    let mut count = 0usize;
    for y in pad..(h - pad) {
        for x in pad..(w - pad) {
            let ux = f64::from(ux[[y, x]]);
            let uy = f64::from(uy[[y, x]]);
            let vx = cov_norm * (f64::from(uxx[[y, x]]) - ux * ux);
            let vy = cov_norm * (f64::from(uyy[[y, x]]) - uy * uy);
            let vxy = cov_norm * (f64::from(uxy[[y, x]]) - ux * uy);

            let num = (2.0 * ux * uy + c1) * (2.0 * vxy + c2);
            let den = (ux * ux + uy * uy + c1) * (vx + vy + c2);
            total += num / den;
            count += 1;
        }
    }
    (total / count as f64) as f32
}

/// Separable box filter with reflect (half-sample symmetric) borders —
/// the equivalent of `scipy.ndimage.uniform_filter` defaults used by skimage.
fn uniform_filter(img: &Image, win: usize) -> Image {
    let (h, w) = img.dim();
    let r = (win / 2) as isize;
    let n = win as f64;

    // scipy 'reflect': ... c b | a b c d | c b ...
    let reflect = |i: isize, len: usize| -> usize {
        let len = len as isize;
        if len == 1 {
            return 0;
        }
        let mut i = i;
        while i < 0 || i >= len {
            if i < 0 {
                i = -i - 1;
            } else {
                i = 2 * len - i - 1;
            }
        }
        i as usize
    };

    // Horizontal pass
    let mut tmp = Image::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for k in -r..=r {
                s += f64::from(img[[y, reflect(x as isize + k, w)]]);
            }
            tmp[[y, x]] = (s / n) as f32;
        }
    }

    // Vertical pass
    let mut out = Image::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for k in -r..=r {
                s += f64::from(tmp[[reflect(y as isize + k, h), x]]);
            }
            out[[y, x]] = (s / n) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psnr_identical_is_infinite() {
        let a = Image::from_elem((8, 8), 0.5);
        assert_eq!(psnr(&a, &a, 1.0), f32::INFINITY);
    }

    #[test]
    fn test_psnr_known_value() {
        // Constant offset 0.5 → mse = 0.25 → 10*log10(1/0.25) = 6.0206 dB
        let a = Image::from_elem((8, 8), 0.0);
        let b = Image::from_elem((8, 8), 0.5);
        let p = psnr(&a, &b, 1.0);
        assert!((p - 6.0206).abs() < 1e-3, "psnr = {}", p);
    }

    #[test]
    fn test_ssim_identical_is_one() {
        let a = Image::from_elem((12, 10), 0.42);
        assert!((ssim(&a, &a, 1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_ssim_constant_images_match_formula() {
        // Two constant images have zero variance/covariance, so SSIM collapses
        // to (2*ux*uy + C1) / (ux^2 + uy^2 + C1) at every window.
        // (Constants are binary-exact so f32 rounding can't perturb the
        // variance terms against the tiny C2.)
        let a = Image::from_elem((12, 12), 0.5);
        let b = Image::from_elem((12, 12), 0.25);
        let c1 = (0.01f64).powi(2);
        let expected = ((2.0 * 0.5 * 0.25 + c1) / (0.5f64.powi(2) + 0.25f64.powi(2) + c1)) as f32;
        let got = ssim(&a, &b, 1.0);
        assert!((got - expected).abs() < 1e-5, "got {}, expected {}", got, expected);
    }

    #[test]
    fn test_ssim_penalizes_structure_change() {
        // Half-white/half-black vs its horizontal flip: tone matches,
        // structure does not.
        let a = Image::from_shape_fn((16, 16), |(_, x)| if x < 8 { 0.0 } else { 1.0 });
        let b = Image::from_shape_fn((16, 16), |(_, x)| if x < 8 { 1.0 } else { 0.0 });
        let s = ssim(&a, &b, 1.0);
        assert!(s < 0.5, "ssim should be low for flipped structure, got {}", s);
    }

    #[test]
    fn test_uniform_filter_constant() {
        let a = Image::from_elem((9, 9), 0.3);
        let f = uniform_filter(&a, 7);
        assert!(f.iter().all(|&v| (v - 0.3).abs() < 1e-6));
    }

    /// Parity fixture: values and expected scores generated with
    /// `skimage.metrics.structural_similarity` / `peak_signal_noise_ratio`
    /// from scikit-image 0.26 (default parameters, data_range=255).
    #[test]
    fn test_parity_with_skimage() {
        #[rustfmt::skip]
        const A: [u8; 120] = [
            136,  38, 217,  22, 205, 251,  33, 198, 193, 255,
            145, 167,  97,  86,  90, 112,  36,  22, 218, 110,
            194,  18, 205, 219, 141, 136,   0,  22,  14, 182,
            134, 178, 235, 129, 147,  51, 181,   1,  28,  24,
            140,  83, 199, 134, 237,  98, 194, 249, 113,  68,
             90, 188,  47,  13, 218, 194,  64, 151, 172, 183,
            163, 130,  59, 201,  19, 209,  98, 131,  22,  14,
            204,  32, 110, 189, 249, 214,  41, 126,  76, 115,
             41,  16,  23, 128, 139, 158, 236,  94, 175, 146,
            187,  46, 102, 120,  64, 237,   6, 206,  20, 200,
             49,  88, 212, 164, 182, 160,   4, 103,  77, 129,
            160, 210,  35,  62, 161, 139, 202, 151, 131, 113,
        ];
        #[rustfmt::skip]
        const B: [u8; 120] = [
             97,  59, 149,  68, 117, 166,  38, 128, 113, 155,
            147,  95, 146,  77, 169, 126, 105, 106, 147, 168,
            210,  50, 145, 117, 125, 193,  92, 120, 127, 172,
            178, 194, 139, 157, 182,  60,  98,  97, 104,  92,
            192,  99, 137,  88, 216,  53, 106, 221,  72,  94,
             87, 183, 118,  33, 206, 142, 100, 101, 135, 100,
            136, 150,  93, 224,  39, 178,  61, 122, 134,  81,
            183, 130, 143, 107, 194, 193,  28,  81,  72, 157,
             28, 106,  79, 161, 151, 186, 223,  71,  99, 194,
            113,  69,  62,  63,  93, 178,  31, 137,  65, 163,
             55, 136, 133,  87, 171, 205, 119, 121,  55, 130,
            143, 124,  39, 144, 119, 164, 206, 113, 175, 143,
        ];
        let to_img = |m: [u8; 120]| {
            Image::from_shape_vec(
                (12, 10),
                m.iter().map(|&v| v as f32 / 255.0).collect(),
            )
            .unwrap()
        };
        let (a, b) = (to_img(A), to_img(B));

        let s = ssim(&a, &b, 1.0);
        let p = psnr(&a, &b, 1.0);
        assert!((s - 0.674806621).abs() < 1e-4, "ssim = {}, expected 0.674806621", s);
        assert!((p - 13.3884709).abs() < 1e-3, "psnr = {}, expected 13.3884709", p);
    }
}
