//! MNIST loading and the affine augmentation that gets a 27k-parameter model past 99.3%.
//!
//! Native only: the browser never trains, it only runs what was trained here.

use crate::tensor::{Rng, Vol};
use std::fs;
use std::path::Path;

pub struct Dataset {
    pub images: Vec<Vol>, // each 1x28x28, values in [0, 1]
    pub labels: Vec<u8>,
}

impl Dataset {
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }
}

fn be_u32(b: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

pub fn load_idx_images(path: &Path) -> Result<Vec<Vol>, String> {
    let b = fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    if b.len() < 16 {
        return Err(format!("{} is too short to be an idx file", path.display()));
    }
    let magic = be_u32(&b, 0);
    if magic != 0x0000_0803 {
        return Err(format!(
            "{}: expected image magic 0x803, got {magic:#x}",
            path.display()
        ));
    }
    let n = be_u32(&b, 4) as usize;
    let rows = be_u32(&b, 8) as usize;
    let cols = be_u32(&b, 12) as usize;
    let need = 16 + n * rows * cols;
    if b.len() < need {
        return Err(format!(
            "{}: expected {need} bytes, found {}",
            path.display(),
            b.len()
        ));
    }

    Ok((0..n)
        .map(|i| {
            let start = 16 + i * rows * cols;
            let data = b[start..start + rows * cols]
                .iter()
                .map(|v| *v as f32 / 255.0)
                .collect();
            Vol::from_vec(1, rows, cols, data)
        })
        .collect())
}

pub fn load_idx_labels(path: &Path) -> Result<Vec<u8>, String> {
    let b = fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    if b.len() < 8 {
        return Err(format!("{} is too short to be an idx file", path.display()));
    }
    let magic = be_u32(&b, 0);
    if magic != 0x0000_0801 {
        return Err(format!(
            "{}: expected label magic 0x801, got {magic:#x}",
            path.display()
        ));
    }
    let n = be_u32(&b, 4) as usize;
    if b.len() < 8 + n {
        return Err(format!("{}: truncated label data", path.display()));
    }
    Ok(b[8..8 + n].to_vec())
}

pub fn load_split(dir: &Path, images: &str, labels: &str) -> Result<Dataset, String> {
    let images = load_idx_images(&dir.join(images))?;
    let labels = load_idx_labels(&dir.join(labels))?;
    if images.len() != labels.len() {
        return Err(format!(
            "{} images but {} labels",
            images.len(),
            labels.len()
        ));
    }
    Ok(Dataset { images, labels })
}

/// Random affine warp with bilinear sampling.
///
/// The parameters are deliberately modest. Large rotations start turning 6s into 9s and
/// teach the model something false; +/-12 degrees is the band that improves generalisation
/// on hand drawings without corrupting the labels.
pub fn augment(src: &Vol, r: &mut Rng) -> Vol {
    let angle = r.range(-12.0, 12.0) * std::f32::consts::PI / 180.0;
    let scale = r.range(0.90, 1.12);
    let shear = r.range(-0.14, 0.14);
    let tx = r.range(-2.0, 2.0);
    let ty = r.range(-2.0, 2.0);

    let (s, c) = (angle.sin(), angle.cos());
    // Inverse map: destination pixel -> source pixel.
    let inv = 1.0 / scale;
    let cx = (src.w as f32 - 1.0) / 2.0;
    let cy = (src.h as f32 - 1.0) / 2.0;

    let mut out = Vol::zeros(1, src.h, src.w);
    for y in 0..src.h {
        for x in 0..src.w {
            let dx = x as f32 - cx - tx;
            let dy = y as f32 - cy - ty;
            let ux = (c * dx + s * dy) * inv;
            let uy = (-s * dx + c * dy) * inv;
            let sx = ux - shear * uy + cx;
            let sy = uy + cy;
            out.set(0, y, x, sample_bilinear(src, sx, sy));
        }
    }
    out
}

fn sample_bilinear(v: &Vol, x: f32, y: f32) -> f32 {
    if x < -1.0 || y < -1.0 || x > v.w as f32 || y > v.h as f32 {
        return 0.0;
    }
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let get = |yy: f32, xx: f32| -> f32 {
        if yy < 0.0 || xx < 0.0 || yy >= v.h as f32 || xx >= v.w as f32 {
            0.0
        } else {
            v.at(0, yy as usize, xx as usize)
        }
    };
    let a = get(y0, x0);
    let b = get(y0, x0 + 1.0);
    let c = get(y0 + 1.0, x0);
    let d = get(y0 + 1.0, x0 + 1.0);
    a * (1.0 - fx) * (1.0 - fy) + b * fx * (1.0 - fy) + c * (1.0 - fx) * fy + d * fx * fy
}

/// Mean image per class, used as the ghost glyphs orbiting in Stage 5.
///
/// These are real: the average of every training example of that digit. They read
/// instantly as digits, which activation maximisation does not reliably do, and they are
/// honest data rather than a font.
pub fn class_means(ds: &Dataset) -> Vec<Vol> {
    // Accumulate in f64: summing sixty thousand f32 pixel values loses noticeable
    // precision in the low bits, and these means are shipped as a display asset.
    let mut sums: Vec<Vec<f64>> = (0..10).map(|_| vec![0.0f64; 784]).collect();
    let mut counts = [0usize; 10];
    for (img, lab) in ds.images.iter().zip(&ds.labels) {
        let k = *lab as usize;
        counts[k] += 1;
        for (acc, v) in sums[k].iter_mut().zip(&img.data) {
            *acc += *v as f64;
        }
    }
    (0..10)
        .map(|k| {
            let n = counts[k].max(1) as f64;
            let data: Vec<f32> = sums[k].iter().map(|v| (v / n) as f32).collect();
            // Contrast stretch: the raw mean is faint and washes out on screen.
            let hi = data.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
            Vol::from_vec(
                1,
                28,
                28,
                data.iter().map(|v| (v / hi).clamp(0.0, 1.0)).collect(),
            )
        })
        .collect()
}

/// Pack the ten prototypes as one byte per pixel: 7,840 bytes on the wire.
pub fn pack_prototypes(protos: &[Vol]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + 10 * 784);
    out.extend_from_slice(b"PROT");
    for p in protos {
        for v in &p.data {
            out.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn augment_preserves_shape_and_range() {
        let mut r = Rng::new(1);
        let mut src = Vol::zeros(1, 28, 28);
        for y in 8..20 {
            for x in 12..16 {
                src.set(0, y, x, 1.0);
            }
        }
        for _ in 0..50 {
            let a = augment(&src, &mut r);
            assert_eq!((a.c, a.h, a.w), (1, 28, 28));
            assert!(a.data.iter().all(|v| (0.0..=1.0).contains(v)));
            // The stroke must survive: augmentation should never erase the digit.
            assert!(a.data.iter().sum::<f32>() > 10.0);
        }
    }

    #[test]
    fn bilinear_sampling_is_exact_on_integer_coordinates() {
        let v = Vol::from_vec(1, 2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert!((sample_bilinear(&v, 0.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((sample_bilinear(&v, 1.0, 1.0) - 4.0).abs() < 1e-6);
        assert!((sample_bilinear(&v, 0.5, 0.0) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn sampling_outside_the_image_returns_zero() {
        let v = Vol::from_vec(1, 2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(sample_bilinear(&v, -5.0, 0.0), 0.0);
        assert_eq!(sample_bilinear(&v, 0.0, 99.0), 0.0);
    }

    #[test]
    fn prototypes_pack_to_one_byte_per_pixel() {
        let protos: Vec<Vol> = (0..10).map(|_| Vol::zeros(1, 28, 28)).collect();
        assert_eq!(pack_prototypes(&protos).len(), 4 + 7840);
    }

    #[test]
    fn idx_loader_rejects_a_wrong_magic_number() {
        let dir = std::env::temp_dir().join("nnviz-idx-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("bad-idx3-ubyte");
        std::fs::write(&p, [0u8, 0, 9, 9, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]).unwrap();
        assert!(load_idx_images(&p).is_err());
    }
}
