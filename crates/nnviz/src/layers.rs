//! Forward and backward for every operation in the network.
//!
//! Backward is not an afterthought here: the product's explanation engine runs a real
//! backward pass in the browser to find which pixels actually decided the answer. If these
//! gradients are wrong, every sentence the app shows a user is a lie. Hence `grad_check`.

use crate::tensor::Vol;

// ---------------------------------------------------------------------------
// Convolution (stride 1, symmetric padding)
// ---------------------------------------------------------------------------

/// Learned weights for one conv layer. `w` is laid out [oc][ic][kh][kw].
#[derive(Clone, Debug)]
pub struct ConvW {
    pub oc: usize,
    pub ic: usize,
    pub k: usize,
    pub pad: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

impl ConvW {
    pub fn new(oc: usize, ic: usize, k: usize, pad: usize) -> Self {
        ConvW {
            oc,
            ic,
            k,
            pad,
            w: vec![0.0; oc * ic * k * k],
            b: vec![0.0; oc],
        }
    }

    #[inline(always)]
    fn widx(&self, oc: usize, ic: usize, ky: usize, kx: usize) -> usize {
        ((oc * self.ic + ic) * self.k + ky) * self.k + kx
    }

    /// One kernel plane, for rendering the learned filters as images in Stage 3.
    pub fn kernel(&self, oc: usize, ic: usize) -> &[f32] {
        let n = self.k * self.k;
        let start = (oc * self.ic + ic) * n;
        &self.w[start..start + n]
    }
}

pub fn conv_forward(x: &Vol, cw: &ConvW) -> Vol {
    let oh = x.h + 2 * cw.pad - cw.k + 1;
    let ow = x.w + 2 * cw.pad - cw.k + 1;
    let mut out = Vol::zeros(cw.oc, oh, ow);

    for oc in 0..cw.oc {
        let bias = cw.b[oc];
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = bias;
                for ic in 0..cw.ic {
                    for ky in 0..cw.k {
                        let iy = oy + ky;
                        if iy < cw.pad || iy - cw.pad >= x.h {
                            continue;
                        }
                        let iy = iy - cw.pad;
                        for kx in 0..cw.k {
                            let ix = ox + kx;
                            if ix < cw.pad || ix - cw.pad >= x.w {
                                continue;
                            }
                            let ix = ix - cw.pad;
                            acc += x.at(ic, iy, ix) * cw.w[cw.widx(oc, ic, ky, kx)];
                        }
                    }
                }
                out.set(oc, oy, ox, acc);
            }
        }
    }
    out
}

/// Returns dx; accumulates into dw/db so a batch can share one gradient buffer.
pub fn conv_backward(x: &Vol, cw: &ConvW, dout: &Vol, dw: &mut [f32], db: &mut [f32]) -> Vol {
    let mut dx = Vol::zeros(x.c, x.h, x.w);

    for oc in 0..cw.oc {
        for oy in 0..dout.h {
            for ox in 0..dout.w {
                let g = dout.at(oc, oy, ox);
                if g == 0.0 {
                    continue;
                }
                db[oc] += g;
                for ic in 0..cw.ic {
                    for ky in 0..cw.k {
                        let iy = oy + ky;
                        if iy < cw.pad || iy - cw.pad >= x.h {
                            continue;
                        }
                        let iy = iy - cw.pad;
                        for kx in 0..cw.k {
                            let ix = ox + kx;
                            if ix < cw.pad || ix - cw.pad >= x.w {
                                continue;
                            }
                            let ix = ix - cw.pad;
                            let wi = cw.widx(oc, ic, ky, kx);
                            dw[wi] += x.at(ic, iy, ix) * g;
                            let di = dx.idx(ic, iy, ix);
                            dx.data[di] += cw.w[wi] * g;
                        }
                    }
                }
            }
        }
    }
    dx
}

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

pub fn relu_forward(x: &Vol) -> Vol {
    Vol::from_vec(x.c, x.h, x.w, x.data.iter().map(|v| v.max(0.0)).collect())
}

/// Gated by the *pre-activation* values, which the network keeps anyway because Stage 3
/// animates ReLU cutting the negative half of the field away.
pub fn relu_backward(pre: &Vol, dout: &Vol) -> Vol {
    Vol::from_vec(
        pre.c,
        pre.h,
        pre.w,
        pre.data
            .iter()
            .zip(&dout.data)
            .map(|(p, g)| if *p > 0.0 { *g } else { 0.0 })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Max pool 2x2, stride 2
// ---------------------------------------------------------------------------

/// `argmax[i]` is which of the four cells in the 2x2 window won, encoded ky*2+kx.
/// Stage 4 renders pooling as a competition, so the winner index is part of the output.
pub struct Pooled {
    pub out: Vol,
    pub argmax: Vec<u8>,
}

pub fn pool_forward(x: &Vol) -> Pooled {
    let oh = x.h / 2;
    let ow = x.w / 2;
    let mut out = Vol::zeros(x.c, oh, ow);
    let mut argmax = vec![0u8; x.c * oh * ow];

    for c in 0..x.c {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut best = f32::NEG_INFINITY;
                let mut best_k = 0u8;
                for ky in 0..2 {
                    for kx in 0..2 {
                        let v = x.at(c, oy * 2 + ky, ox * 2 + kx);
                        if v > best {
                            best = v;
                            best_k = (ky * 2 + kx) as u8;
                        }
                    }
                }
                let oi = out.idx(c, oy, ox);
                out.data[oi] = best;
                argmax[oi] = best_k;
            }
        }
    }
    Pooled { out, argmax }
}

pub fn pool_backward(x_shape: (usize, usize, usize), p: &Pooled, dout: &Vol) -> Vol {
    let (c, h, w) = x_shape;
    let mut dx = Vol::zeros(c, h, w);
    for ch in 0..c {
        for oy in 0..dout.h {
            for ox in 0..dout.w {
                let oi = dout.idx(ch, oy, ox);
                let k = p.argmax[oi];
                let (ky, kx) = ((k / 2) as usize, (k % 2) as usize);
                let di = dx.idx(ch, oy * 2 + ky, ox * 2 + kx);
                dx.data[di] += dout.data[oi];
            }
        }
    }
    dx
}

// ---------------------------------------------------------------------------
// Dense
// ---------------------------------------------------------------------------

/// `w` is laid out [out][in].
#[derive(Clone, Debug)]
pub struct DenseW {
    pub n_out: usize,
    pub n_in: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

impl DenseW {
    pub fn new(n_out: usize, n_in: usize) -> Self {
        DenseW {
            n_out,
            n_in,
            w: vec![0.0; n_out * n_in],
            b: vec![0.0; n_out],
        }
    }
}

pub fn dense_forward(x: &[f32], d: &DenseW) -> Vec<f32> {
    let mut out = d.b.clone();
    for o in 0..d.n_out {
        let row = &d.w[o * d.n_in..(o + 1) * d.n_in];
        let mut acc = out[o];
        for i in 0..d.n_in {
            acc += row[i] * x[i];
        }
        out[o] = acc;
    }
    out
}

pub fn dense_backward(
    x: &[f32],
    d: &DenseW,
    dout: &[f32],
    dw: &mut [f32],
    db: &mut [f32],
) -> Vec<f32> {
    let mut dx = vec![0.0; d.n_in];
    for o in 0..d.n_out {
        let g = dout[o];
        if g == 0.0 {
            continue;
        }
        db[o] += g;
        let base = o * d.n_in;
        for i in 0..d.n_in {
            dw[base + i] += x[i] * g;
            dx[i] += d.w[base + i] * g;
        }
    }
    dx
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

/// Numerically stable softmax. Also returns the shifted exponentials, because Stage 6
/// animates the two halves of softmax separately: exponentiate, then normalise.
pub fn softmax(logits: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs = exps.iter().map(|e| e / sum).collect();
    (exps, probs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Rng;

    fn rand_vol(r: &mut Rng, c: usize, h: usize, w: usize) -> Vol {
        Vol::from_vec(c, h, w, (0..c * h * w).map(|_| r.normal()).collect())
    }

    #[test]
    fn conv_with_identity_kernel_is_identity() {
        let mut cw = ConvW::new(1, 1, 3, 1);
        cw.w[1 * 3 + 1] = 1.0; // centre tap
        let x = Vol::from_vec(1, 3, 3, vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let y = conv_forward(&x, &cw);
        assert_eq!(y.data, x.data);
    }

    #[test]
    fn conv_preserves_spatial_size_with_same_padding() {
        let cw = ConvW::new(4, 1, 5, 2);
        let x = Vol::zeros(1, 28, 28);
        let y = conv_forward(&x, &cw);
        assert_eq!((y.c, y.h, y.w), (4, 28, 28));
    }

    #[test]
    fn relu_clips_negatives() {
        let x = Vol::from_vec(1, 1, 4, vec![-2.0, -0.001, 0.0, 3.5]);
        assert_eq!(relu_forward(&x).data, vec![0.0, 0.0, 0.0, 3.5]);
    }

    #[test]
    fn relu_backward_gates_on_preactivation() {
        // Stage 3 draws ReLU as "the negative half is extinguished". That is only
        // honest if the backward pass also keys off the pre-activation, not the
        // already-clipped output: gating on the post-ReLU field would treat a
        // genuine zero the same as a dead unit, and the attribution would lie.
        let pre = Vol::from_vec(1, 1, 4, vec![-2.0, -0.001, 0.0, 3.5]);
        let dout = Vol::from_vec(1, 1, 4, vec![1.0, 1.0, 1.0, 7.0]);
        assert_eq!(relu_backward(&pre, &dout).data, vec![0.0, 0.0, 0.0, 7.0]);
    }

    #[test]
    fn pool_takes_max_and_records_winner() {
        let x = Vol::from_vec(1, 2, 2, vec![1.0, 9.0, 3.0, 4.0]);
        let p = pool_forward(&x);
        assert_eq!(p.out.data, vec![9.0]);
        assert_eq!(p.argmax, vec![1]); // ky=0, kx=1
    }

    #[test]
    fn pool_backward_routes_gradient_only_to_winner() {
        let x = Vol::from_vec(1, 2, 2, vec![1.0, 9.0, 3.0, 4.0]);
        let p = pool_forward(&x);
        let dout = Vol::from_vec(1, 1, 1, vec![5.0]);
        let dx = pool_backward((1, 2, 2), &p, &dout);
        assert_eq!(dx.data, vec![0.0, 5.0, 0.0, 0.0]);
    }

    #[test]
    fn softmax_sums_to_one_and_survives_large_logits() {
        let (_, p) = softmax(&[1000.0, 1001.0, 999.0]);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum was {sum}");
        assert!(p.iter().all(|v| v.is_finite()));
        assert!(p[1] > p[0] && p[0] > p[2]);
    }

    #[test]
    fn softmax_is_shift_invariant() {
        let (_, a) = softmax(&[1.0, 2.0, 3.0]);
        let (_, b) = softmax(&[101.0, 102.0, 103.0]);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    /// Central-difference check of the analytic conv gradients. This is the test that
    /// protects the credibility of the whole explanation feature.
    #[test]
    fn conv_backward_matches_numeric_gradient() {
        let mut r = Rng::new(3);
        let mut cw = ConvW::new(2, 2, 3, 1);
        cw.w.iter_mut().for_each(|v| *v = r.normal() * 0.3);
        cw.b.iter_mut().for_each(|v| *v = r.normal() * 0.1);
        let x = rand_vol(&mut r, 2, 5, 5);

        // Scalar loss: sum(out * seed) with a fixed random seed volume.
        let seed = rand_vol(&mut r, 2, 5, 5);
        let loss = |cw: &ConvW, x: &Vol| -> f32 {
            conv_forward(x, cw)
                .data
                .iter()
                .zip(&seed.data)
                .map(|(a, b)| a * b)
                .sum()
        };

        let mut dw = vec![0.0; cw.w.len()];
        let mut db = vec![0.0; cw.b.len()];
        let dx = conv_backward(&x, &cw, &seed, &mut dw, &mut db);

        let eps = 1e-3;
        // Check a sample of weights.
        for &i in &[0usize, 5, 11, 17, 30] {
            let mut plus = cw.clone();
            plus.w[i] += eps;
            let mut minus = cw.clone();
            minus.w[i] -= eps;
            let num = (loss(&plus, &x) - loss(&minus, &x)) / (2.0 * eps);
            assert!(
                (num - dw[i]).abs() < 2e-2,
                "w[{i}]: analytic {} numeric {}",
                dw[i],
                num
            );
        }
        // Check a sample of input pixels.
        for &i in &[0usize, 7, 13, 24, 39] {
            let mut xp = x.clone();
            xp.data[i] += eps;
            let mut xm = x.clone();
            xm.data[i] -= eps;
            let num = (loss(&cw, &xp) - loss(&cw, &xm)) / (2.0 * eps);
            assert!(
                (num - dx.data[i]).abs() < 2e-2,
                "x[{i}]: analytic {} numeric {}",
                dx.data[i],
                num
            );
        }
        // And the biases.
        for o in 0..cw.b.len() {
            let mut plus = cw.clone();
            plus.b[o] += eps;
            let mut minus = cw.clone();
            minus.b[o] -= eps;
            let num = (loss(&plus, &x) - loss(&minus, &x)) / (2.0 * eps);
            assert!(
                (num - db[o]).abs() < 2e-2,
                "b[{o}]: analytic {} numeric {}",
                db[o],
                num
            );
        }
    }

    #[test]
    fn dense_backward_matches_numeric_gradient() {
        let mut r = Rng::new(11);
        let mut d = DenseW::new(4, 6);
        d.w.iter_mut().for_each(|v| *v = r.normal() * 0.4);
        d.b.iter_mut().for_each(|v| *v = r.normal() * 0.1);
        let x: Vec<f32> = (0..6).map(|_| r.normal()).collect();
        let seed: Vec<f32> = (0..4).map(|_| r.normal()).collect();

        let loss = |d: &DenseW, x: &[f32]| -> f32 {
            dense_forward(x, d)
                .iter()
                .zip(&seed)
                .map(|(a, b)| a * b)
                .sum()
        };

        let mut dw = vec![0.0; d.w.len()];
        let mut db = vec![0.0; d.b.len()];
        let dx = dense_backward(&x, &d, &seed, &mut dw, &mut db);

        let eps = 1e-3;
        for i in [0usize, 3, 9, 20] {
            let mut p = d.clone();
            p.w[i] += eps;
            let mut m = d.clone();
            m.w[i] -= eps;
            let num = (loss(&p, &x) - loss(&m, &x)) / (2.0 * eps);
            assert!((num - dw[i]).abs() < 1e-2, "w[{i}]");
        }
        for i in 0..x.len() {
            let mut xp = x.clone();
            xp[i] += eps;
            let mut xm = x.clone();
            xm[i] -= eps;
            let num = (loss(&d, &xp) - loss(&d, &xm)) / (2.0 * eps);
            assert!((num - dx[i]).abs() < 1e-2, "x[{i}]");
        }
    }
}
