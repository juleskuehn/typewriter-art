//! Spatial loss weighting ("center focus").
//!
//! A radial profile that emphasizes the middle of the image and de-emphasizes
//! its edges and margins. Used in two places (controlled by one toggle):
//!
//! - **Auto-alignment scoring**: weighted PSNR/SSIM over the content region, so
//!   the crop ranking reflects the middle of the picture. This is where the
//!   weighting has real leverage — the score is a global image metric.
//! - **Optimization loss**: per-pixel weights on the asymmetric MSE. Because
//!   each grid position optimizes its own tile independently, a slowly varying
//!   profile cannot trade edge quality for middle quality (there is no shared
//!   budget); what it does do is stop charging edge characters for ink in the
//!   padding border, letting them fit the content at the cost of fuzzier
//!   margins — and mildly bias characters toward the center-facing part of
//!   tiles that span the gradient.
//!
//! The profile is `w = floor + (1 - floor) * exp(-r² / (2σ²))`, where `r` is
//! the distance from the content center normalized so the rect's edge
//! midpoints sit at `r = 1`. Pixels outside the content (the white border and
//! bottom padding that `resize_target` adds) get the preset's `padding` weight.

use ndarray::s;

use crate::types::{CenterFocus, Image};

/// Radial "center focus" profile over a `shape` = (rows, cols) region.
///
/// Returns `None` for [`CenterFocus::Off`] (uniform weights) so callers can
/// stay on the unweighted code path exactly.
pub fn focus_map(shape: (usize, usize), focus: CenterFocus) -> Option<Image> {
    let preset = focus.preset()?;
    let (h, w) = shape;
    let (cy, cx) = ((h.saturating_sub(1)) as f64 / 2.0, (w.saturating_sub(1)) as f64 / 2.0);
    // Edge midpoints at r = 1 (corners at r = sqrt(2))
    let (ry, rx) = (h as f64 / 2.0, w as f64 / 2.0);
    let sigma2 = 2.0 * f64::from(preset.sigma) * f64::from(preset.sigma);

    let mut out = Image::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let dy = (y as f64 - cy) / ry.max(f64::EPSILON);
            let dx = (x as f64 - cx) / rx.max(f64::EPSILON);
            let g = (-(dy * dy + dx * dx) / sigma2).exp() as f32;
            out[[y, x]] = preset.floor + (1.0 - preset.floor) * g;
        }
    }
    Some(out)
}

/// Loss weight map over the full optimization canvas `container` = (rows, cols).
///
/// Places the [`focus_map`] profile over `content` = `(y0, y1, x0, x1)` and
/// fills everything outside it (the padding border and any layer-offset spill
/// area) with the preset's `padding` weight. Returns `None` for
/// [`CenterFocus::Off`].
pub fn container_map(
    container: (usize, usize),
    content: (usize, usize, usize, usize),
    focus: CenterFocus,
) -> Option<Image> {
    let preset = focus.preset()?;
    let (h, w) = container;
    let (y0, y1, x0, x1) = (
        content.0.min(h),
        content.1.min(h),
        content.2.min(w),
        content.3.min(w),
    );

    let mut out = Image::from_elem((h, w), preset.padding);
    if y1 > y0 && x1 > x0 {
        let profile = focus_map((y1 - y0, x1 - x0), focus).expect("preset is Some");
        out.slice_mut(s![y0..y1, x0..x1]).assign(&profile);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_focus_map_center_heavier_than_edges() {
        let map = focus_map((21, 21), CenterFocus::Soft).unwrap();
        let center = map[[10, 10]];
        let edge = map[[10, 20]];
        let corner = map[[20, 20]];
        assert!(center > edge, "center {} > edge {}", center, edge);
        assert!(edge > corner, "edge {} > corner {}", edge, corner);
        assert!(center <= 1.0);
        assert!(corner >= 0.25 - 1e-6, "corner respects floor");
    }

    #[test]
    fn test_focus_map_off_is_none() {
        assert!(focus_map((8, 8), CenterFocus::Off).is_none());
    }

    #[test]
    fn test_container_map_padding_region() {
        let map = container_map((13, 13), (3, 10, 3, 10), CenterFocus::Strong).unwrap();
        // Outside the content: padding weight
        assert_eq!(map[[0, 0]], 0.0);
        assert_eq!(map[[12, 12]], 0.0);
        // Exact content center: weight 1
        assert!(map[[6, 6]] > 0.95, "center weight {}", map[[6, 6]]);
    }
}
