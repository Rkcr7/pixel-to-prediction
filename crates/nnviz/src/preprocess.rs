//! Turning an arbitrary drawing into the exact kind of image the network was trained on.
//!
//! This follows the original MNIST normalisation, deliberately and exactly: crop to the ink,
//! scale the long side to 20 px preserving aspect ratio with area averaging, then place the
//! result in a 28x28 field so that the ink's centre of mass lands on the centre of the field.
//!
//! Note what is *not* here: no deskewing. MNIST itself is not deskewed, so deskewing user
//! input would push it off the distribution the model learned. Matching the training data
//! beats "improving" the input.

use crate::tensor::Vol;

pub const BOX: usize = 20; // MNIST's inner box
pub const FIELD: usize = 28;

/// Every intermediate Stage 2 needs to animate the preparation honestly.
pub struct Prepared {
    /// Bounding box of ink in source pixels: (x0, y0, w, h). Zero-size when the canvas is blank.
    pub bbox: (usize, usize, usize, usize),
    /// The cropped region, still at source resolution.
    pub cropped: Vol,
    /// After area-averaged scaling: at most 20x20, aspect preserved.
    pub boxed: Vol,
    /// Centre of mass of `boxed`, in its own pixel coordinates.
    pub com: (f32, f32),
    /// Integer translation applied when placing `boxed` into the 28x28 field.
    pub offset: (i32, i32),
    /// The finished 1x28x28 input. This is what the network sees.
    pub input: Vol,
    /// True when there was no ink at all.
    pub empty: bool,
}

/// Area-averaged (box filter) resample. This is what produces MNIST's characteristic soft
/// grey edges; a nearest-neighbour or bilinear resize gives visibly different stroke
/// statistics and measurably hurts accuracy on hand drawings.
pub fn resample_area(src: &Vol, dst_w: usize, dst_h: usize) -> Vol {
    assert_eq!(src.c, 1, "resample_area expects a single channel");
    let mut out = Vol::zeros(1, dst_h, dst_w);
    if dst_w == 0 || dst_h == 0 || src.w == 0 || src.h == 0 {
        return out;
    }
    let sx = src.w as f32 / dst_w as f32;
    let sy = src.h as f32 / dst_h as f32;

    for oy in 0..dst_h {
        let y0 = oy as f32 * sy;
        let y1 = y0 + sy;
        for ox in 0..dst_w {
            let x0 = ox as f32 * sx;
            let x1 = x0 + sx;

            let mut acc = 0.0f32;
            let mut wsum = 0.0f32;
            for iy in (y0.floor() as usize)..(y1.ceil() as usize).min(src.h) {
                let cy = (y1.min(iy as f32 + 1.0) - y0.max(iy as f32)).max(0.0);
                if cy <= 0.0 {
                    continue;
                }
                for ix in (x0.floor() as usize)..(x1.ceil() as usize).min(src.w) {
                    let cx = (x1.min(ix as f32 + 1.0) - x0.max(ix as f32)).max(0.0);
                    if cx <= 0.0 {
                        continue;
                    }
                    let wgt = cx * cy;
                    acc += src.at(0, iy, ix) * wgt;
                    wsum += wgt;
                }
            }
            out.set(0, oy, ox, if wsum > 0.0 { acc / wsum } else { 0.0 });
        }
    }
    out
}

/// Ink bounding box. `thresh` guards against stray antialiasing dust from the brush.
fn ink_bbox(src: &Vol, thresh: f32) -> Option<(usize, usize, usize, usize)> {
    let (mut x0, mut y0) = (usize::MAX, usize::MAX);
    let (mut x1, mut y1) = (0usize, 0usize);
    let mut found = false;
    for y in 0..src.h {
        for x in 0..src.w {
            if src.at(0, y, x) > thresh {
                found = true;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    if !found {
        return None;
    }
    Some((x0, y0, x1 - x0 + 1, y1 - y0 + 1))
}

fn centre_of_mass(v: &Vol) -> (f32, f32) {
    let mut sum = 0.0f32;
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    for y in 0..v.h {
        for x in 0..v.w {
            let m = v.at(0, y, x);
            sum += m;
            sx += m * x as f32;
            sy += m * y as f32;
        }
    }
    if sum <= 1e-6 {
        return (v.w as f32 / 2.0 - 0.5, v.h as f32 / 2.0 - 0.5);
    }
    (sx / sum, sy / sum)
}

/// `src` must be a single-channel image with ink as high values in [0, 1].
pub fn prepare(src: &Vol) -> Prepared {
    assert_eq!(src.c, 1, "prepare expects a single channel");

    let Some(bbox) = ink_bbox(src, 0.05) else {
        return Prepared {
            bbox: (0, 0, 0, 0),
            cropped: Vol::zeros(1, 1, 1),
            boxed: Vol::zeros(1, 1, 1),
            com: (0.0, 0.0),
            offset: (0, 0),
            input: Vol::zeros(1, FIELD, FIELD),
            empty: true,
        };
    };

    // 1. Crop to the ink.
    let (bx, by, bw, bh) = bbox;
    let mut cropped = Vol::zeros(1, bh, bw);
    for y in 0..bh {
        for x in 0..bw {
            cropped.set(0, y, x, src.at(0, by + y, bx + x));
        }
    }

    // 2. Scale the long side to 20, preserving aspect ratio.
    let scale = BOX as f32 / bw.max(bh) as f32;
    let tw = ((bw as f32 * scale).round() as usize).clamp(1, BOX);
    let th = ((bh as f32 * scale).round() as usize).clamp(1, BOX);
    let boxed = resample_area(&cropped, tw, th);

    // 3. Place so the centre of mass sits at the centre of the 28x28 field.
    let com = centre_of_mass(&boxed);
    let target = (FIELD as f32 - 1.0) / 2.0;
    let off_x = (target - com.0).round() as i32;
    let off_y = (target - com.1).round() as i32;

    let mut input = Vol::zeros(1, FIELD, FIELD);
    for y in 0..th {
        let dy = y as i32 + off_y;
        if dy < 0 || dy >= FIELD as i32 {
            continue;
        }
        for x in 0..tw {
            let dx = x as i32 + off_x;
            if dx < 0 || dx >= FIELD as i32 {
                continue;
            }
            input.set(0, dy as usize, dx as usize, boxed.at(0, y, x));
        }
    }

    Prepared {
        bbox,
        cropped,
        boxed,
        com,
        offset: (off_x, off_y),
        input,
        empty: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stroke(w: usize, h: usize, rect: (usize, usize, usize, usize)) -> Vol {
        let mut v = Vol::zeros(1, h, w);
        let (x0, y0, rw, rh) = rect;
        for y in y0..y0 + rh {
            for x in x0..x0 + rw {
                v.set(0, y, x, 1.0);
            }
        }
        v
    }

    #[test]
    fn blank_canvas_is_reported_empty_and_produces_a_black_field() {
        let p = prepare(&Vol::zeros(1, 100, 100));
        assert!(p.empty);
        assert_eq!(p.input.len(), 784);
        assert!(p.input.data.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn bbox_finds_the_ink() {
        let src = stroke(100, 100, (10, 20, 30, 40));
        let p = prepare(&src);
        assert_eq!(p.bbox, (10, 20, 30, 40));
    }

    #[test]
    fn long_side_is_scaled_to_twenty_and_aspect_is_preserved() {
        let src = stroke(200, 200, (50, 50, 40, 80)); // 1:2 aspect
        let p = prepare(&src);
        assert_eq!(p.boxed.h, 20);
        assert_eq!(p.boxed.w, 10);
    }

    #[test]
    fn centre_of_mass_lands_on_the_centre_of_the_field() {
        // Deliberately lopsided: a small blob far from the canvas centre.
        let src = stroke(200, 200, (10, 150, 20, 20));
        let p = prepare(&src);
        let com = centre_of_mass(&p.input);
        assert!((com.0 - 13.5).abs() < 1.0, "com.x was {}", com.0);
        assert!((com.1 - 13.5).abs() < 1.0, "com.y was {}", com.1);
    }

    #[test]
    fn output_is_always_a_28x28_single_channel_field() {
        for rect in [(0, 0, 200, 200), (99, 99, 2, 2), (0, 0, 1, 200)] {
            let p = prepare(&stroke(200, 200, rect));
            assert_eq!((p.input.c, p.input.h, p.input.w), (1, 28, 28));
        }
    }

    #[test]
    fn area_resample_conserves_average_intensity() {
        let src = stroke(40, 40, (10, 10, 20, 20)); // quarter of the area is ink
        let out = resample_area(&src, 10, 10);
        let mean_in: f32 = src.data.iter().sum::<f32>() / src.len() as f32;
        let mean_out: f32 = out.data.iter().sum::<f32>() / out.len() as f32;
        assert!((mean_in - mean_out).abs() < 0.02, "{mean_in} vs {mean_out}");
    }

    #[test]
    fn area_resample_antialiases_rather_than_dropping_pixels() {
        // A single-pixel line downscaled 4x must survive as grey, not vanish.
        let src = stroke(40, 40, (20, 0, 1, 40));
        let out = resample_area(&src, 10, 10);
        assert!(
            out.data.iter().any(|v| *v > 0.1 && *v < 0.9),
            "expected grey edge pixels"
        );
    }

    #[test]
    fn tiny_ink_still_fills_the_box() {
        // Two pixels of ink should still be normalised up to the 20px box.
        let p = prepare(&stroke(200, 200, (100, 100, 2, 2)));
        assert_eq!(p.boxed.w.max(p.boxed.h), 20);
    }
}
