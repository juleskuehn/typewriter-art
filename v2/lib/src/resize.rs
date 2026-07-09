//! Target image preprocessing.
//!
//! Port of `resizeTarget` from utils.py.
//! Resizes and pads the target image to align with the character grid.

use crate::types::{CharShape, Image, Padding};

/// Resize and pad a target image to align with the character grid.
///
/// Port of `resizeTarget` from utils.py.
///
/// Steps:
/// 1. Resize to `row_length * char_width` wide, maintaining aspect ratio
///    corrected by the character change ratio
/// 2. Pad height to be a multiple of char_height
/// 3. Add a border of 1/2 char dimensions on all sides
/// 4. Remove bottom border if it's entirely white
pub fn resize_target(
    im: &image::GrayImage,
    row_length: u32,
    char_shape: CharShape,
    char_change: (f32, f32),
) -> (Image, Padding) {
    let (char_height, char_width) = char_shape;
    let (x_change, y_change) = char_change;

    let in_height = im.height() as f32;
    let in_width = im.width() as f32;

    let out_width = row_length * char_width as u32;

    // Scale height to maintain aspect ratio, corrected for char change
    let out_height = ((out_width as f32 / in_width)
        * in_height
        * (x_change / y_change))
        .round() as u32;

    // Resize the image
    let mut resized = image::imageops::resize(
        im,
        out_width,
        out_height,
        image::imageops::FilterType::Triangle,
    );

    // Pad height to be a multiple of char_height
    let mut padded_h = out_height as usize;
    if padded_h % char_height != 0 {
        let new_height = ((padded_h / char_height) + 1) * char_height;
        let mut blank = image::GrayImage::new(out_width, new_height as u32);
        // Fill with white (255)
        for p in blank.pixels_mut() {
            p.0[0] = 255;
        }
        // Copy resized image into top of blank
        image::imageops::overlay(&mut blank, &resized, 0, 0);
        resized = blank;
        padded_h = new_height;
    }

    // Convert to ndarray float [0.0, 1.0]
    let mut array = gray_to_float(&resized);

    // Pad with 1/2 char dimension on each side (white border)
    let pad_top = char_height / 2;
    let pad_bottom = char_height / 2;
    let pad_left = char_width / 2;
    let pad_right = char_width / 2;

    array = pad_image(&array, pad_top, pad_bottom, pad_left, pad_right, 1.0);

    // If the bottom char_height rows are entirely white, remove them
    let h = array.nrows();
    if h > char_height {
        let bottom_rows = array.slice(ndarray::s![(h - char_height).., ..]);
        if bottom_rows.iter().all(|&x| x >= 0.999) {
            array = array
                .slice(ndarray::s![..(h - char_height), ..])
                .to_owned();
        }
    }

    let final_h = array.nrows();
    let padding = Padding {
        top: pad_top,
        right: pad_right,
        bottom: pad_bottom + (final_h - padded_h),
        left: pad_left,
    };

    (array, padding)
}

/// Convert a grayscale image to a float ndarray [0.0, 1.0].
fn gray_to_float(img: &image::GrayImage) -> Image {
    let (w, h) = img.dimensions();
    let pixels: Vec<f32> = img.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
    ndarray::Array2::from_shape_vec((h as usize, w as usize), pixels)
        .expect("pixel count matches dimensions")
}

/// Pad an image evenly on all sides with a constant value.
fn pad_image(img: &Image, top: usize, bottom: usize, left: usize, right: usize, value: f32) -> Image {
    let (h, w) = (img.nrows(), img.ncols());
    let new_h = h + top + bottom;
    let new_w = w + left + right;

    let mut padded = Image::from_elem((new_h, new_w), value);
    let mut view = padded.slice_mut(ndarray::s![top..(top + h), left..(left + w)]);
    view.assign(img);

    padded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_image() {
        let img = Image::from_elem((4, 4), 0.5);
        let padded = pad_image(&img, 1, 2, 3, 4, 1.0);
        assert_eq!(padded.nrows(), 7); // 4 + 1 + 2
        assert_eq!(padded.ncols(), 11); // 4 + 3 + 4
        // Top-left corner should be the pad value
        assert!((padded[[0, 0]] - 1.0).abs() < 1e-6);
        // Center should be original
        assert!((padded[[1, 3]] - 0.5).abs() < 1e-6);
    }
}
