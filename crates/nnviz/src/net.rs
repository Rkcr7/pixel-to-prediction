//! The network itself, plus the full activation record the visualisation is built from.
//!
//! `forward` keeps every intermediate tensor including the *pre*-activation values. A normal
//! inference engine throws those away; here they are the product. Stage 3 cannot animate ReLU
//! cutting the negative half of a field away unless it can see the negative half.

use crate::layers::*;
use crate::tensor::{Rng, Vol};

pub const IMG: usize = 28;
pub const C1: usize = 8; // 8 first-layer filters: the most that stay individually readable
pub const K1: usize = 5; // 5x5 so the learned kernels look like edge detectors on screen
pub const C2: usize = 16; // tiles as a satisfying 4x4 grid
pub const K2: usize = 3;
pub const FC1: usize = 32;
pub const CLASSES: usize = 10;

pub const FLAT: usize = C2 * (IMG / 4) * (IMG / 4); // 16 * 7 * 7 = 784

#[derive(Clone, Debug)]
pub struct Net {
    pub conv1: ConvW,
    pub conv2: ConvW,
    pub fc1: DenseW,
    pub fc2: DenseW,
}

/// Every tensor the visualisation is allowed to draw from. Nothing on screen may come
/// from anywhere else.
pub struct Acts {
    pub input: Vol,     // 1x28x28, post preprocessing
    pub conv1_pre: Vol, // 8x28x28, before ReLU (signed)
    pub conv1: Vol,     // 8x28x28, after ReLU
    pub pool1: Vol,     // 8x14x14
    pub pool1_arg: Vec<u8>,
    pub conv2_pre: Vol, // 16x14x14, signed
    pub conv2: Vol,     // 16x14x14
    pub pool2: Vol,     // 16x7x7
    pub pool2_arg: Vec<u8>,
    pub fc1_pre: Vec<f32>, // 32, signed
    pub fc1: Vec<f32>,     // 32, after ReLU
    pub logits: Vec<f32>,  // 10, signed
    pub exps: Vec<f32>,    // 10, exp(logit - max): the first half of softmax
    pub probs: Vec<f32>,   // 10
}

impl Acts {
    pub fn top_k(&self) -> Vec<(usize, f32)> {
        let mut v: Vec<(usize, f32)> = self.probs.iter().cloned().enumerate().collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    }

    pub fn predicted(&self) -> usize {
        self.top_k()[0].0
    }
}

impl Net {
    pub fn new_random(seed: u64) -> Self {
        let mut r = Rng::new(seed);
        let mut conv1 = ConvW::new(C1, 1, K1, K1 / 2);
        let mut conv2 = ConvW::new(C2, C1, K2, K2 / 2);
        let mut fc1 = DenseW::new(FC1, FLAT);
        let mut fc2 = DenseW::new(CLASSES, FC1);

        // He initialisation, appropriate for ReLU.
        let he = |r: &mut Rng, buf: &mut [f32], fan_in: usize| {
            let s = (2.0 / fan_in as f32).sqrt();
            buf.iter_mut().for_each(|v| *v = r.normal() * s);
        };
        he(&mut r, &mut conv1.w, K1 * K1);
        he(&mut r, &mut conv2.w, K2 * K2 * C1);
        he(&mut r, &mut fc1.w, FLAT);
        he(&mut r, &mut fc2.w, FC1);

        Net {
            conv1,
            conv2,
            fc1,
            fc2,
        }
    }

    pub fn num_params(&self) -> usize {
        self.conv1.w.len()
            + self.conv1.b.len()
            + self.conv2.w.len()
            + self.conv2.b.len()
            + self.fc1.w.len()
            + self.fc1.b.len()
            + self.fc2.w.len()
            + self.fc2.b.len()
    }

    /// Full forward pass, retaining everything.
    pub fn forward(&self, input: &Vol) -> Acts {
        let conv1_pre = conv_forward(input, &self.conv1);
        let conv1 = relu_forward(&conv1_pre);
        let p1 = pool_forward(&conv1);

        let conv2_pre = conv_forward(&p1.out, &self.conv2);
        let conv2 = relu_forward(&conv2_pre);
        let p2 = pool_forward(&conv2);

        let fc1_pre = dense_forward(&p2.out.data, &self.fc1);
        let fc1: Vec<f32> = fc1_pre.iter().map(|v| v.max(0.0)).collect();
        let logits = dense_forward(&fc1, &self.fc2);
        let (exps, probs) = softmax(&logits);

        Acts {
            input: input.clone(),
            conv1_pre,
            conv1,
            pool1: p1.out,
            pool1_arg: p1.argmax,
            conv2_pre,
            conv2,
            pool2: p2.out,
            pool2_arg: p2.argmax,
            fc1_pre,
            fc1,
            logits,
            exps,
            probs,
        }
    }

    /// Backward pass from an arbitrary gradient on the logits.
    ///
    /// Two callers matter:
    /// - training, where `dlogits = probs - onehot(label)`
    /// - the explanation engine, where `dlogits = onehot(top1)` for saliency or
    ///   `onehot(top1) - onehot(top2)` for the counterfactual "why not the runner-up"
    ///
    /// Returns the gradient with respect to the input image; accumulates parameter
    /// gradients into `g` when one is supplied.
    pub fn backward(&self, a: &Acts, dlogits: &[f32], g: Option<&mut Grads>) -> Vol {
        let mut scratch = Grads::zeros(self);
        let g = match g {
            Some(g) => g,
            None => &mut scratch,
        };

        let dfc1 = dense_backward(&a.fc1, &self.fc2, dlogits, &mut g.fc2_w, &mut g.fc2_b);
        let dfc1_pre: Vec<f32> = a
            .fc1_pre
            .iter()
            .zip(&dfc1)
            .map(|(p, d)| if *p > 0.0 { *d } else { 0.0 })
            .collect();

        let dflat = dense_backward(
            &a.pool2.data,
            &self.fc1,
            &dfc1_pre,
            &mut g.fc1_w,
            &mut g.fc1_b,
        );
        let dpool2 = Vol::from_vec(a.pool2.c, a.pool2.h, a.pool2.w, dflat);

        let dconv2 = pool_backward(
            (a.conv2.c, a.conv2.h, a.conv2.w),
            &Pooled {
                out: a.pool2.clone(),
                argmax: a.pool2_arg.clone(),
            },
            &dpool2,
        );
        let dconv2_pre = relu_backward(&a.conv2_pre, &dconv2);
        let dpool1 = conv_backward(
            &a.pool1,
            &self.conv2,
            &dconv2_pre,
            &mut g.conv2_w,
            &mut g.conv2_b,
        );

        let dconv1 = pool_backward(
            (a.conv1.c, a.conv1.h, a.conv1.w),
            &Pooled {
                out: a.pool1.clone(),
                argmax: a.pool1_arg.clone(),
            },
            &dpool1,
        );
        let dconv1_pre = relu_backward(&a.conv1_pre, &dconv1);

        conv_backward(
            &a.input,
            &self.conv1,
            &dconv1_pre,
            &mut g.conv1_w,
            &mut g.conv1_b,
        )
    }

    // -- serialisation ------------------------------------------------------

    /// Flat little-endian f32 dump, in a fixed order. Small enough (about 107 KB)
    /// that compressing it further is not worth the decode complexity.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.num_params() * 4 + 8);
        out.extend_from_slice(b"NNV1");
        let push = |out: &mut Vec<u8>, xs: &[f32]| {
            for v in xs {
                out.extend_from_slice(&v.to_le_bytes());
            }
        };
        push(&mut out, &self.conv1.w);
        push(&mut out, &self.conv1.b);
        push(&mut out, &self.conv2.w);
        push(&mut out, &self.conv2.b);
        push(&mut out, &self.fc1.w);
        push(&mut out, &self.fc1.b);
        push(&mut out, &self.fc2.w);
        push(&mut out, &self.fc2.b);
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 4 || &bytes[0..4] != b"NNV1" {
            return Err("bad magic: expected NNV1".into());
        }
        let mut net = Net::new_random(0);
        let expected = 4 + net.num_params() * 4;
        if bytes.len() != expected {
            return Err(format!("expected {expected} bytes, got {}", bytes.len()));
        }
        let mut off = 4usize;
        let mut pull = |dst: &mut [f32]| {
            for v in dst.iter_mut() {
                *v = f32::from_le_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]);
                off += 4;
            }
        };
        pull(&mut net.conv1.w);
        pull(&mut net.conv1.b);
        pull(&mut net.conv2.w);
        pull(&mut net.conv2.b);
        pull(&mut net.fc1.w);
        pull(&mut net.fc1.b);
        pull(&mut net.fc2.w);
        pull(&mut net.fc2.b);
        Ok(net)
    }
}

/// Parameter gradients, shaped exactly like the network.
pub struct Grads {
    pub conv1_w: Vec<f32>,
    pub conv1_b: Vec<f32>,
    pub conv2_w: Vec<f32>,
    pub conv2_b: Vec<f32>,
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
}

impl Grads {
    pub fn zeros(n: &Net) -> Self {
        Grads {
            conv1_w: vec![0.0; n.conv1.w.len()],
            conv1_b: vec![0.0; n.conv1.b.len()],
            conv2_w: vec![0.0; n.conv2.w.len()],
            conv2_b: vec![0.0; n.conv2.b.len()],
            fc1_w: vec![0.0; n.fc1.w.len()],
            fc1_b: vec![0.0; n.fc1.b.len()],
            fc2_w: vec![0.0; n.fc2.w.len()],
            fc2_b: vec![0.0; n.fc2.b.len()],
        }
    }

    pub fn clear(&mut self) {
        for s in self.slices_mut() {
            s.iter_mut().for_each(|v| *v = 0.0);
        }
    }

    pub fn add(&mut self, other: &Grads) {
        let mut a = self.slices_mut();
        let b = other.slices();
        for (x, y) in a.iter_mut().zip(b) {
            for (p, q) in x.iter_mut().zip(y) {
                *p += *q;
            }
        }
    }

    pub fn slices(&self) -> [&[f32]; 8] {
        [
            &self.conv1_w,
            &self.conv1_b,
            &self.conv2_w,
            &self.conv2_b,
            &self.fc1_w,
            &self.fc1_b,
            &self.fc2_w,
            &self.fc2_b,
        ]
    }

    pub fn slices_mut(&mut self) -> [&mut [f32]; 8] {
        [
            &mut self.conv1_w,
            &mut self.conv1_b,
            &mut self.conv2_w,
            &mut self.conv2_b,
            &mut self.fc1_w,
            &mut self.fc1_b,
            &mut self.fc2_w,
            &mut self.fc2_b,
        ]
    }
}

impl Net {
    pub fn param_slices_mut(&mut self) -> [&mut [f32]; 8] {
        [
            &mut self.conv1.w,
            &mut self.conv1.b,
            &mut self.conv2.w,
            &mut self.conv2.b,
            &mut self.fc1.w,
            &mut self.fc1.b,
            &mut self.fc2.w,
            &mut self.fc2.b,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_count_matches_the_spec() {
        let n = Net::new_random(1);
        assert_eq!(n.conv1.w.len() + n.conv1.b.len(), 5 * 5 * 1 * 8 + 8);
        assert_eq!(n.conv2.w.len() + n.conv2.b.len(), 3 * 3 * 8 * 16 + 16);
        assert_eq!(n.fc1.w.len() + n.fc1.b.len(), 784 * 32 + 32);
        assert_eq!(n.fc2.w.len() + n.fc2.b.len(), 32 * 10 + 10);
        assert_eq!(n.num_params(), 26_826);
    }

    #[test]
    fn forward_produces_the_expected_shapes() {
        let n = Net::new_random(2);
        let a = n.forward(&Vol::zeros(1, 28, 28));
        assert_eq!((a.conv1.c, a.conv1.h, a.conv1.w), (8, 28, 28));
        assert_eq!((a.pool1.c, a.pool1.h, a.pool1.w), (8, 14, 14));
        assert_eq!((a.conv2.c, a.conv2.h, a.conv2.w), (16, 14, 14));
        assert_eq!((a.pool2.c, a.pool2.h, a.pool2.w), (16, 7, 7));
        assert_eq!(a.pool2.len(), FLAT);
        assert_eq!(a.fc1.len(), 32);
        assert_eq!(a.probs.len(), 10);
        let sum: f32 = a.probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    /// The dense layer drawn as three panels (the unit's weights, the drawing's
    /// features, and their element-wise agreement) is only honest if the agreement panel
    /// actually sums to the number the unit reports. This pins that identity.
    #[test]
    fn hidden_unit_agreement_sums_to_its_preactivation() {
        let n = Net::new_random(21);
        let mut r = Rng::new(99);
        let x = Vol::from_vec(1, 28, 28, (0..784).map(|_| r.f32()).collect());
        let a = n.forward(&x);

        for unit in [0usize, 7, 19, 31] {
            let row = &n.fc1.w[unit * FLAT..(unit + 1) * FLAT];
            let agreement: f32 = row.iter().zip(&a.pool2.data).map(|(w, v)| w * v).sum();
            let total = agreement + n.fc1.b[unit];
            assert!(
                (total - a.fc1_pre[unit]).abs() < 1e-3,
                "unit {unit}: panel sums to {total} but the unit reports {}",
                a.fc1_pre[unit]
            );
        }
    }

    /// A hidden unit's weight row reshapes exactly onto the pooled feature stack, which
    /// is what lets the two be drawn in the same coordinates.
    #[test]
    fn hidden_unit_weight_row_matches_the_feature_stack_shape() {
        let n = Net::new_random(4);
        let a = n.forward(&Vol::zeros(1, 28, 28));
        assert_eq!(n.fc1.n_in, a.pool2.len());
        assert_eq!(a.pool2.c * a.pool2.h * a.pool2.w, C2 * 7 * 7);
        assert_eq!(n.fc1.n_in, FLAT);
    }

    #[test]
    fn weights_round_trip_through_bytes() {
        let n = Net::new_random(9);
        let bytes = n.to_bytes();
        let back = Net::from_bytes(&bytes).expect("round trip should succeed");
        assert_eq!(n.conv1.w, back.conv1.w);
        assert_eq!(n.fc2.b, back.fc2.b);
        assert_eq!(bytes.len(), 4 + 26_826 * 4);
    }

    #[test]
    fn from_bytes_rejects_corrupt_input() {
        assert!(Net::from_bytes(b"XXXX").is_err());
        assert!(Net::from_bytes(&[]).is_err());
        let mut good = Net::new_random(1).to_bytes();
        good.truncate(good.len() - 4);
        assert!(Net::from_bytes(&good).is_err());
    }

    /// The weights that actually ship. Every other test uses `new_random`, so a
    /// truncated or swapped model.bin would still go green and still be deployed.
    #[test]
    fn committed_model_loads_and_recognises_its_own_prototypes() {
        let bytes = include_bytes!("../../../web/public/model/model.bin");
        let net = Net::from_bytes(bytes).expect("committed model.bin should load");
        assert_eq!(net.num_params(), 26_826);

        let proto = include_bytes!("../../../web/public/model/prototypes.bin");
        assert_eq!(&proto[0..4], b"PROT", "prototypes.bin is missing its magic");
        assert_eq!(proto.len(), 4 + 10 * 784);

        for digit in 0..10 {
            let start = 4 + digit * 784;
            let data: Vec<f32> = proto[start..start + 784]
                .iter()
                .map(|b| *b as f32 / 255.0)
                .collect();
            let a = net.forward(&Vol::from_vec(1, 28, 28, data));
            assert_eq!(
                a.predicted(),
                digit,
                "prototype of {digit} was classified as {}",
                a.predicted()
            );
            assert!(
                a.probs[digit] > 0.5,
                "prototype of {digit} only scored {}",
                a.probs[digit]
            );
        }
    }

    /// End-to-end numeric gradient check through the whole network, on the exact quantity
    /// the explanation engine uses: d(logit)/d(input).
    #[test]
    fn input_gradient_matches_numeric_gradient() {
        let n = Net::new_random(5);
        let mut r = Rng::new(17);
        let x = Vol::from_vec(1, 28, 28, (0..784).map(|_| r.f32()).collect());

        let target = 3usize;
        let a = n.forward(&x);
        let mut dlogits = vec![0.0; CLASSES];
        dlogits[target] = 1.0;
        let dx = n.backward(&a, &dlogits, None);

        let eps = 1e-2;
        for &i in &[100usize, 250, 400, 543, 700] {
            let mut xp = x.clone();
            xp.data[i] += eps;
            let mut xm = x.clone();
            xm.data[i] -= eps;
            let num = (n.forward(&xp).logits[target] - n.forward(&xm).logits[target]) / (2.0 * eps);
            assert!(
                (num - dx.data[i]).abs() < 1e-2,
                "pixel {i}: analytic {} numeric {}",
                dx.data[i],
                num
            );
        }
    }

    #[test]
    fn parameter_gradients_match_numeric_gradient() {
        let n = Net::new_random(6);
        let mut r = Rng::new(23);
        let x = Vol::from_vec(1, 28, 28, (0..784).map(|_| r.f32()).collect());
        let label = 7usize;

        let loss_of = |net: &Net| -> f32 {
            let a = net.forward(&x);
            -(a.probs[label].max(1e-12)).ln()
        };

        let a = n.forward(&x);
        let dlogits: Vec<f32> = (0..CLASSES)
            .map(|i| a.probs[i] - if i == label { 1.0 } else { 0.0 })
            .collect();
        let mut g = Grads::zeros(&n);
        n.backward(&a, &dlogits, Some(&mut g));

        let eps = 1e-3;
        // conv1 weights, then fc2 weights: one early layer, one late.
        for &i in &[0usize, 37, 99, 150] {
            let mut p = n.clone();
            p.conv1.w[i] += eps;
            let mut m = n.clone();
            m.conv1.w[i] -= eps;
            let num = (loss_of(&p) - loss_of(&m)) / (2.0 * eps);
            assert!(
                (num - g.conv1_w[i]).abs() < 5e-2,
                "conv1.w[{i}]: {} vs {}",
                g.conv1_w[i],
                num
            );
        }
        for &i in &[0usize, 45, 200, 300] {
            let mut p = n.clone();
            p.fc2.w[i] += eps;
            let mut m = n.clone();
            m.fc2.w[i] -= eps;
            let num = (loss_of(&p) - loss_of(&m)) / (2.0 * eps);
            assert!(
                (num - g.fc2_w[i]).abs() < 5e-2,
                "fc2.w[{i}]: {} vs {}",
                g.fc2_w[i],
                num
            );
        }
    }
}
