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
    // (v1 uses cv2.INTER_AREA; `resize_area` below is the exact equivalent)
    let mut resized = crate::resize::resize_area(
        im,
        out_width,
        out_height,
    );

    // Pad height to be a multiple of char_height
    let padded_h = out_height as usize;
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
    // v1 parity: bottom padding = white rows below the resized content,
    // i.e. final_h - (content top offset + out_height), clamped at 0.
    // (v1: `newHeight - outHeight + charHeight // 2`)
    let bottom = final_h.saturating_sub(pad_top + out_height as usize);
    let padding = Padding {
        top: pad_top,
        right: pad_right,
        bottom,
        left: pad_left,
    };

    (array, padding)
}

/// Resize a grayscale image using area averaging (box filter with exact
/// fractional coverage).
///
/// This is the equivalent of OpenCV's `cv2.INTER_AREA`, which v1 used for both
/// the target image and the charset sprite sheet. When downscaling it computes
/// the true average of the source pixels covered by each output pixel
/// (no aliasing); when upscaling it interpolates linearly between the covered
/// source pixels.
pub fn resize_area(im: &image::GrayImage, new_w: u32, new_h: u32) -> image::GrayImage {
    let (in_w, in_h) = im.dimensions();
    if (new_w, new_h) == (in_w, in_h) {
        return im.clone();
    }

    let sx = in_w as f64 / new_w as f64;
    let sy = in_h as f64 / new_h as f64;

    let mut out = image::GrayImage::new(new_w, new_h);
    for oy in 0..new_h {
        let y0 = f64::from(oy) * sy;
        let y1 = f64::from(oy + 1) * sy;
        let iy0 = y0.floor() as u32;
        let iy1 = y1.ceil() as u32; // exclusive

        for ox in 0..new_w {
            let x0 = f64::from(ox) * sx;
            let x1 = f64::from(ox + 1) * sx;
            let ix0 = x0.floor() as u32;
            let ix1 = x1.ceil() as u32; // exclusive

            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            for iy in iy0..iy1.min(in_h) {
                // Vertical overlap of the source pixel with the output box
                let wy = (y1.min(f64::from(iy + 1)) - y0.max(f64::from(iy))).max(0.0);
                if wy <= 0.0 {
                    continue;
                }
                for ix in ix0..ix1.min(in_w) {
                    // Horizontal overlap of the source pixel with the output box
                    let wx = (x1.min(f64::from(ix + 1)) - x0.max(f64::from(ix))).max(0.0);
                    if wx <= 0.0 {
                        continue;
                    }
                    let w = wx * wy;
                    sum += w * f64::from(im.get_pixel(ix, iy).0[0]);
                    weight_sum += w;
                }
            }

            let value = if weight_sum > 0.0 {
                (sum / weight_sum).round() as u8
            } else {
                255
            };
            out.put_pixel(ox, oy, image::Luma([value]));
        }
    }
    out
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

    fn gray(w: u32, h: u32, fill: u8) -> image::GrayImage {
        image::GrayImage::from_pixel(w, h, image::Luma([fill]))
    }

    #[test]
    fn test_resize_area_identity() {
        let img = gray(6, 4, 128);
        let out = resize_area(&img, 6, 4);
        assert_eq!(out.dimensions(), (6, 4));
        assert!(out.pixels().all(|p| p.0[0] == 128));
    }

    #[test]
    fn test_resize_area_integer_downscale_averages() {
        // 4x4 → 2x2: each output pixel averages a 2x2 block (cv2.INTER_AREA)
        let img = image::GrayImage::from_fn(4, 4, |x, y| {
            image::Luma([((y / 2) * 2 + (x / 2)) as u8 * 63])
        });
        // Blocks: [0,63; 126,189]
        let out = resize_area(&img, 2, 2);
        assert_eq!(out.get_pixel(0, 0).0[0], 0);
        assert_eq!(out.get_pixel(1, 0).0[0], 63);
        assert_eq!(out.get_pixel(0, 1).0[0], 126);
        assert_eq!(out.get_pixel(1, 1).0[0], 189);
    }

    #[test]
    fn test_resize_area_fractional_downscale_averages() {
        // 3px of values [0, 100, 200] → 2px: each covers 1.5 source px.
        // out[0] = (0 + 100/2) / 1.5 = 33; out[1] = (100/2 + 200) / 1.5 = 167
        let img = image::GrayImage::from_fn(3, 1, |x, _| image::Luma([x as u8 * 100]));
        let out = resize_area(&img, 2, 1);
        assert_eq!(out.get_pixel(0, 0).0[0], 33);
        assert_eq!(out.get_pixel(1, 0).0[0], 167);
    }

    #[test]
    fn test_resize_target_bottom_padding() {
        // Target that needs no height padding and has non-white bottom rows:
        // bottom padding should be char_height / 2 (border only)
        let img = gray(8, 16, 128); // exactly 2 chars tall, not white
        let (arr, padding) = resize_target(&img, 4, (8, 2), (1.0, 1.0));
        assert_eq!(arr.nrows() % 8, 0);
        assert_eq!(padding.top, 4);
        assert_eq!(padding.bottom, 4); // 16 rows content + 4 border - content ends 4 above bottom
    }
}
