//! Mockup and layer rendering.
//!
//! Functions for composing layers into a mockup under the ink model,
//! rendering individual layers from character assignments, and computing the
//! background (combined effect of all other layers) for coordinate descent
//! optimization.
//!
//! # Ink model
//!
//! Plain multiplicative compositing (the paper's model, and v1's) lets
//! overstrikes reach black: eight strikes of a 0.5-grey pixel compose to
//! 0.004. Real typing cannot — repeated strikes of the same key saturate at
//! the ribbon's ink density. [`Ink`] implements floor-shifted compositing:
//!
//! ```text
//! τ(c) = clamp((c − f) / (1 − f), 0, 1)      // "effective whiteness"
//! composite = f + (1 − f) · Π τ(layers)
//! ```
//!
//! - a single strike renders exactly as scanned (identity on one layer),
//! - k overstrikes of value c approach f exponentially, never below it,
//! - `f = 0` is exactly plain multiplicative compositing.

use crate::types::{Chars, Image, InkFloor, LayerOffset, Layers};

/// Resolved ink model: the saturation floor and τ-space character glyphs.
///
/// The τ-transformed glyphs are precomputed once so the hot evaluation loop
/// only pays one fused multiply-add per pixel.
#[derive(Debug, Clone)]
pub struct Ink {
    /// Brightness floor: the darkest achievable pixel (ink density).
    pub floor: f32,
    /// τ-space glyphs: `τ(c)` per pixel, same shape as the source charset.
    pub chars_t: Chars,
}

impl Ink {
    /// Resolve the configured floor against the loaded charset and precompute
    /// the τ-space glyphs.
    ///
    /// `Auto` takes the darkest glyph pixel as the floor: a fully inked pixel
    /// measures the ribbon's ink density, and no amount of overstriking can
    /// get below it. A `Fixed` floor above that is clamped down (otherwise
    /// single strikes of dark glyph pixels would render lighter than the
    /// scan) — the clamp is logged.
    pub fn new(floor: InkFloor, chars: &Chars) -> Ink {
        let min_pixel = chars
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min)
            .min(1.0);
        let min_pixel = if min_pixel.is_finite() {
            min_pixel
        } else {
            0.0
        };
        let mut f = match floor {
            InkFloor::Auto => {
                log::info!(
                    "Ink floor: {:.3} (auto — darkest glyph pixel = ribbon ink density)",
                    min_pixel
                );
                min_pixel
            }
            InkFloor::Fixed(v) => {
                if v > min_pixel {
                    log::warn!(
                        "Ink floor {:.3} clamped to {:.3} (darkest glyph pixel)",
                        v,
                        min_pixel
                    );
                    min_pixel
                } else {
                    log::info!("Ink floor: {:.3} (fixed)", v);
                    v
                }
            }
        };
        f = f.clamp(0.0, 0.99);
        let chars_t = chars.mapv(|c| ((c - f) / (1.0 - f)).clamp(0.0, 1.0));
        Ink { floor: f, chars_t }
    }

    /// τ: effective whiteness of one raw brightness value.
    pub fn tau(&self, c: f32) -> f32 {
        ((c - self.floor) / (1.0 - self.floor)).clamp(0.0, 1.0)
    }

    /// Map a τ-space product back to brightness.
    pub fn from_tau(&self, t: f32) -> f32 {
        self.floor + (1.0 - self.floor) * t
    }

    /// Compose one candidate pixel: τ-space background × τ-space glyph value.
    pub fn compose(&self, bg_t: f32, char_t: f32) -> f32 {
        self.floor + (1.0 - self.floor) * bg_t * char_t
    }
}

/// Compute the full mockup from all layers under the ink model.
///
/// Layers are mapped through τ, multiplied, and mapped back to brightness
/// (see the module docs). With `floor = 0` this is the plain product.
pub fn combine_layers(layers: &Layers, ink: &Ink) -> Image {
    if layers.shape()[0] == 0 {
        return Image::from_elem((1, 1), 1.0);
    }

    let mut product = layers
        .index_axis(ndarray::Axis(0), 0)
        .mapv(|c| ink.tau(c));
    for l in 1..layers.shape()[0] {
        let other = layers.index_axis(ndarray::Axis(0), l).mapv(|c| ink.tau(c));
        product = product * &other;
    }
    product.mapv(|t| ink.from_tau(t))
}

/// Compute the background for a specific layer: the combined effect of all
/// OTHER layers, in τ-space (effective-whiteness product).
///
/// This is used in coordinate descent optimization: when optimizing layer `i`,
/// all other layers are held fixed and treated as the "background". Compose a
/// candidate glyph into it with [`Ink::compose`].
pub fn compute_background(layers: &Layers, exclude_layer: usize, ink: &Ink) -> Image {
    let num_layers = layers.shape()[0];
    if num_layers <= 1 {
        return Image::from_elem((layers.shape()[1], layers.shape()[2]), 1.0);
    }

    // Start with the first layer that isn't the excluded one
    let mut bg: Option<Image> = None;
    for l in 0..num_layers {
        if l != exclude_layer {
            let other = layers
                .index_axis(ndarray::Axis(0), l)
                .mapv(|c| ink.tau(c));
            bg = Some(match bg {
                None => other,
                Some(current) => current * &other,
            });
        }
    }

    bg.unwrap_or_else(|| Image::from_elem((layers.shape()[1], layers.shape()[2]), 1.0))
}

/// Render a single layer from its character choices.
///
/// Places each character glyph at its grid position with the given offset.
/// All other pixels remain white (1.0).
pub fn render_layer(
    choices: &[u16],
    chars: &Chars,
    num_cols: usize,
    layer_offset: LayerOffset,
    output_shape: (usize, usize),
) -> Image {
    let char_h = chars.shape()[1];
    let char_w = chars.shape()[2];
    let mut layer = Image::from_elem(output_shape, 1.0f32);

    for (i, &choice) in choices.iter().enumerate() {
        let choice = choice as usize;
        let row = i / num_cols;
        let col = i % num_cols;

        let y_start = row * char_h + layer_offset.0;
        let x_start = col * char_w + layer_offset.1;
        let y_end = y_start + char_h;
        let x_end = x_start + char_w;

        let char_glyph = chars.index_axis(ndarray::Axis(0), choice);
        let mut region = layer.slice_mut(ndarray::s![y_start..y_end, x_start..x_end]);
        region.assign(&char_glyph);
    }

    layer
}

/// Render all layers from their character choices.
pub fn render_all_layers(
    choices: &ndarray::Array2<u16>,
    chars: &Chars,
    num_cols: usize,
    layer_offsets: &[LayerOffset],
    output_shape: (usize, usize),
) -> Layers {
    use ndarray::Array3;

    let num_layers = choices.shape()[0];
    let mut layers = Array3::from_elem((num_layers, output_shape.0, output_shape.1), 1.0f32);

    for l in 0..num_layers {
        let row = choices.index_axis(ndarray::Axis(0), l);
        let slice: Vec<u16> = row.iter().copied().collect();
        let layer_img = render_layer(
            &slice,
            chars,
            num_cols,
            layer_offsets[l],
            output_shape,
        );
        layers
            .index_axis_mut(ndarray::Axis(0), l)
            .assign(&layer_img);
    }

    layers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::InkFloor;
    use ndarray::arr3;

    /// With floor 0 the model must reproduce the plain product exactly
    /// (bit-for-bit behaviour parity with the paper's multiplicative model).
    #[test]
    fn test_zero_floor_is_plain_product() {
        let chars: Chars = arr3(&[[[0.5f32]]]);
        let ink = Ink::new(InkFloor::Fixed(0.0), &chars);
        let layers = Layers::from_shape_vec((3, 1, 1), vec![0.5, 0.4, 0.8]).unwrap();
        let mockup = combine_layers(&layers, &ink);
        assert_eq!(mockup[[0, 0]], 0.5 * 0.4 * 0.8);
    }

    /// One layer renders exactly as scanned, whatever the floor.
    #[test]
    fn test_single_strike_is_exact() {
        let chars: Chars = arr3(&[[[0.3f32]]]);
        let ink = Ink::new(InkFloor::Auto, &chars);
        let layers = Layers::from_shape_vec((1, 1, 1), vec![0.7]).unwrap();
        let mockup = combine_layers(&layers, &ink);
        assert!((mockup[[0, 0]] - 0.7).abs() < 1e-6);
    }

    /// Overstrikes of a grey pixel darken toward the floor and stop there —
    /// the regression this model exists for (the plain product reaches black).
    #[test]
    fn test_overstrikes_saturate_at_floor() {
        let chars: Chars = arr3(&[[[0.5f32]]]);
        let ink = Ink::new(InkFloor::Fixed(0.1), &chars);
        let tau = ink.tau(0.5);

        let mut vals = Vec::new();
        for k in 1..=8 {
            let layers = Layers::from_elem((k, 1, 1), 0.5f32);
            vals.push(combine_layers(&layers, &ink)[[0, 0]]);
        }

        // Single strike unchanged
        assert!((vals[0] - 0.5).abs() < 1e-6);
        // Closed form: f + (1 − f)·τ^k
        for (k, v) in vals.iter().enumerate() {
            let expected = 0.1 + 0.9 * tau.powi(k as i32 + 1);
            assert!((v - expected).abs() < 1e-6, "k = {}", k + 1);
        }
        // Monotonically darker, always above the floor, converging to it
        for w in vals.windows(2) {
            assert!(w[1] < w[0]);
        }
        assert!(vals[7] > 0.1);
        assert!(
            vals[7] - 0.1 < 0.01,
            "8 overstrikes must sit at the floor, got {}",
            vals[7]
        );
        // The plain product would have been 0.5^8 = 0.0039 ≈ black
        assert!(vals[7] > 0.0039 * 2.0);
    }

    /// `Auto` measures the ribbon's ink density from the scan.
    #[test]
    fn test_auto_floor_is_darkest_glyph_pixel() {
        let chars: Chars = arr3(&[
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.35, 0.9], [1.0, 0.22]],
        ]);
        let ink = Ink::new(InkFloor::Auto, &chars);
        assert!((ink.floor - 0.22).abs() < 1e-6);
    }

    /// A fixed floor darker than the darkest glyph pixel is clamped —
    /// otherwise single strikes would render lighter than the scan.
    #[test]
    fn test_fixed_floor_clamped_to_glyph_min() {
        let chars: Chars = arr3(&[[[0.4f32, 0.6], [0.7, 0.8]]]);
        let ink = Ink::new(InkFloor::Fixed(0.6), &chars);
        assert!((ink.floor - 0.4).abs() < 1e-6);
    }

    /// Backgrounds are τ-space products: composing the excluded layer into
    /// the background reproduces the full mockup.
    #[test]
    fn test_compute_background_composes_consistently() {
        let chars: Chars = arr3(&[[[0.5f32]]]);
        let ink = Ink::new(InkFloor::Fixed(0.1), &chars);
        let layers = Layers::from_shape_vec((2, 1, 1), vec![0.5, 0.5]).unwrap();
        let bg = compute_background(&layers, 0, &ink);
        assert!((bg[[0, 0]] - ink.tau(0.5)).abs() < 1e-6);

        let composed = ink.compose(bg[[0, 0]], ink.tau(0.5));
        let mockup = combine_layers(&layers, &ink);
        assert!((composed - mockup[[0, 0]]).abs() < 1e-6);
    }
}
