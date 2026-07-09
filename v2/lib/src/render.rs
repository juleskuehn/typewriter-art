//! Mockup and layer rendering.
//!
//! Functions for composing layers into a mockup, rendering individual layers
//! from character assignments, and computing the background (product of all
//! other layers) for coordinate descent optimization.


use crate::types::{Chars, Image, LayerOffset, Layers};

/// Compute the full mockup as the product of all layers.
///
/// Each pixel in the mockup is the product of the corresponding pixels across
/// all layers. Values are in [0.0, 1.0] where 1.0 = white.
pub fn combine_layers(layers: &Layers) -> Image {
    if layers.shape()[0] == 0 {
        return Image::from_elem((1, 1), 1.0);
    }

    let mut result = layers.index_axis(ndarray::Axis(0), 0).to_owned();
    for l in 1..layers.shape()[0] {
        result = result * layers.index_axis(ndarray::Axis(0), l);
    }
    result
}

/// Compute the background for a specific layer: the product of all OTHER layers.
///
/// This is used in coordinate descent optimization: when optimizing layer `i`,
/// all other layers are held fixed and treated as the "background".
pub fn compute_background(layers: &Layers, exclude_layer: usize) -> Image {
    let num_layers = layers.shape()[0];
    if num_layers <= 1 {
        return Image::from_elem(
            (layers.shape()[1], layers.shape()[2]),
            1.0,
        );
    }

    // Start with the first layer that isn't the excluded one
    let mut bg: Option<Image> = None;
    for l in 0..num_layers {
        if l != exclude_layer {
            match bg {
                None => bg = Some(layers.index_axis(ndarray::Axis(0), l).to_owned()),
                Some(ref current) => {
                    let other = layers.index_axis(ndarray::Axis(0), l);
                    let next = current * &other;
                    bg = Some(next);
                }
            }
        }
    }

    bg.unwrap_or_else(|| {
        Image::from_elem(
            (layers.shape()[1], layers.shape()[2]),
            1.0,
        )
    })
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
