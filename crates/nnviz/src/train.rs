//! Adam trainer. Native only, parallel over the batch with rayon.

use crate::data::{augment, Dataset};
use crate::net::{Grads, Net, CLASSES};
use crate::tensor::Rng;
use rayon::prelude::*;

pub struct Adam {
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: u32,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Adam {
    pub fn new(net: &Net, lr: f32) -> Self {
        let g = Grads::zeros(net);
        let shapes: Vec<usize> = g.slices().iter().map(|s| s.len()).collect();
        Adam {
            m: shapes.iter().map(|n| vec![0.0; *n]).collect(),
            v: shapes.iter().map(|n| vec![0.0; *n]).collect(),
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 1e-4,
        }
    }

    pub fn step(&mut self, net: &mut Net, grads: &Grads, scale: f32) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        let lr = self.lr;

        let mut params = net.param_slices_mut();
        let gs = grads.slices();

        for (k, param) in params.iter_mut().enumerate() {
            let g_slice = gs[k];
            let m = &mut self.m[k];
            let v = &mut self.v[k];
            // Biases are index 1, 3, 5, 7 in the slice order and are not decayed.
            let decay = if k % 2 == 0 { self.weight_decay } else { 0.0 };

            for i in 0..param.len() {
                let g = g_slice[i] * scale + decay * param[i];
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g;
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g;
                let mh = m[i] / bc1;
                let vh = v[i] / bc2;
                param[i] -= lr * mh / (vh.sqrt() + self.eps);
            }
        }
    }
}

pub struct EpochStats {
    pub loss: f32,
    pub train_acc: f32,
}

/// One pass over the training set. Returns mean loss and running training accuracy.
pub fn train_epoch(
    net: &mut Net,
    opt: &mut Adam,
    ds: &Dataset,
    order: &[usize],
    batch: usize,
    rng_seed: u64,
    augment_data: bool,
) -> EpochStats {
    let mut total_loss = 0.0f64;
    let mut correct = 0usize;
    let mut seen = 0usize;

    for (bi, chunk) in order.chunks(batch).enumerate() {
        let results: Vec<(Grads, f32, bool)> = chunk
            .par_iter()
            .enumerate()
            .map(|(i, &idx)| {
                let label = ds.labels[idx] as usize;
                let img = if augment_data {
                    let mut r = Rng::new(
                        rng_seed
                            .wrapping_mul(0x9E37_79B9)
                            .wrapping_add((bi as u64) << 20)
                            .wrapping_add(i as u64),
                    );
                    augment(&ds.images[idx], &mut r)
                } else {
                    ds.images[idx].clone()
                };

                let a = net.forward(&img);
                let loss = -(a.probs[label].max(1e-12)).ln();
                let ok = a.predicted() == label;

                let dlogits: Vec<f32> = (0..CLASSES)
                    .map(|c| a.probs[c] - if c == label { 1.0 } else { 0.0 })
                    .collect();
                let mut g = Grads::zeros(net);
                net.backward(&a, &dlogits, Some(&mut g));
                (g, loss, ok)
            })
            .collect();

        let mut acc = Grads::zeros(net);
        for (g, loss, ok) in &results {
            acc.add(g);
            total_loss += *loss as f64;
            correct += *ok as usize;
        }
        seen += results.len();

        opt.step(net, &acc, 1.0 / results.len() as f32);
    }

    EpochStats {
        loss: (total_loss / seen.max(1) as f64) as f32,
        train_acc: correct as f32 / seen.max(1) as f32,
    }
}

pub fn evaluate(net: &Net, ds: &Dataset) -> f32 {
    let correct: usize = ds
        .images
        .par_iter()
        .zip(&ds.labels)
        .map(|(img, lab)| (net.forward(img).predicted() == *lab as usize) as usize)
        .sum();
    correct as f32 / ds.len().max(1) as f32
}

/// Confusion matrix, rows = truth, cols = prediction. Used by the trainer's report so a
/// regression in one digit cannot hide behind a good headline number.
pub fn confusion(net: &Net, ds: &Dataset) -> [[u32; CLASSES]; CLASSES] {
    let preds: Vec<(usize, usize)> = ds
        .images
        .par_iter()
        .zip(&ds.labels)
        .map(|(img, lab)| (*lab as usize, net.forward(img).predicted()))
        .collect();
    let mut m = [[0u32; CLASSES]; CLASSES];
    for (t, p) in preds {
        m[t][p] += 1;
    }
    m
}

/// Cosine decay with a short linear warmup. Warmup matters here because Adam's second
/// moment is badly conditioned in the first few dozen steps on a net this small.
pub fn lr_at(epoch: usize, total: usize, base: f32) -> f32 {
    let warmup = 1usize;
    if epoch < warmup {
        return base * (epoch + 1) as f32 / (warmup + 1) as f32;
    }
    let p = (epoch - warmup) as f32 / (total - warmup).max(1) as f32;
    base * 0.5 * (1.0 + (std::f32::consts::PI * p).cos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Vol;

    #[test]
    fn learning_rate_warms_up_then_decays_to_zero() {
        let base = 2e-3;
        assert!(lr_at(0, 30, base) < base);
        assert!((lr_at(1, 30, base) - base).abs() < 1e-6);
        assert!(lr_at(29, 30, base) < base * 0.02);
        assert!(lr_at(29, 30, base) >= 0.0);
    }

    #[test]
    fn adam_reduces_loss_on_a_single_memorised_example() {
        let mut net = Net::new_random(3);
        let mut opt = Adam::new(&net, 3e-3);
        let mut r = crate::tensor::Rng::new(2);
        let x = Vol::from_vec(1, 28, 28, (0..784).map(|_| r.f32()).collect());
        let label = 4usize;

        let first = -(net.forward(&x).probs[label].max(1e-12)).ln();
        for _ in 0..60 {
            let a = net.forward(&x);
            let dlogits: Vec<f32> = (0..CLASSES)
                .map(|c| a.probs[c] - if c == label { 1.0 } else { 0.0 })
                .collect();
            let mut g = Grads::zeros(&net);
            net.backward(&a, &dlogits, Some(&mut g));
            opt.step(&mut net, &g, 1.0);
        }
        let last = -(net.forward(&x).probs[label].max(1e-12)).ln();
        assert!(last < first * 0.2, "loss went {first} -> {last}");
        assert_eq!(net.forward(&x).predicted(), label);
    }

    #[test]
    fn confusion_matrix_rows_sum_to_the_class_counts() {
        let net = Net::new_random(1);
        let mut r = crate::tensor::Rng::new(9);
        let ds = Dataset {
            images: (0..20)
                .map(|_| Vol::from_vec(1, 28, 28, (0..784).map(|_| r.f32()).collect()))
                .collect(),
            labels: (0..20).map(|i| (i % 10) as u8).collect(),
        };
        let m = confusion(&net, &ds);
        for (t, row) in m.iter().enumerate() {
            let n: u32 = row.iter().sum();
            assert_eq!(
                n,
                ds.labels.iter().filter(|l| **l as usize == t).count() as u32
            );
        }
    }
}
