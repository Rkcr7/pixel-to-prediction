//! The browser-facing surface.
//!
//! Design rule for this boundary: hand JavaScript *conclusions*, not raw material. The
//! fc1 weight matrix is 25,088 floats; ranking it every frame in JS would be wasteful and
//! would show the same static hairball for every drawing. Rust ranks by contribution
//! (weight x activation, which changes per drawing) and returns only the edges worth
//! drawing.

use wasm_bindgen::prelude::*;

use crate::explain::{explain, Explanation};
use crate::net::{Acts, Net};
use crate::preprocess::{prepare, Prepared};
use crate::tensor::Vol;

#[wasm_bindgen]
pub struct Engine {
    net: Net,
    prep: Option<Prepared>,
    acts: Option<Acts>,
    exp: Option<Explanation>,
}

#[wasm_bindgen]
impl Engine {
    /// `model_bytes` is the contents of model.bin produced by the trainer.
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: &[u8]) -> Result<Engine, JsValue> {
        let net = Net::from_bytes(model_bytes).map_err(|e| JsValue::from_str(&e))?;
        Ok(Engine {
            net,
            prep: None,
            acts: None,
            exp: None,
        })
    }

    /// Number of learned parameters, for the UI to quote honestly.
    #[wasm_bindgen(getter)]
    pub fn params(&self) -> usize {
        self.net.num_params()
    }

    /// Preprocess a grayscale buffer (ink as high values in [0,1]) and run the network.
    /// Returns false when the canvas has no ink.
    pub fn run(&mut self, src: &[f32], w: usize, h: usize) -> Result<bool, JsValue> {
        if src.len() != w * h {
            return Err(JsValue::from_str(&format!(
                "buffer is {} values but {w}x{h} needs {}",
                src.len(),
                w * h
            )));
        }
        let vol = Vol::from_vec(1, h, w, src.to_vec());
        let prep = prepare(&vol);
        if prep.empty {
            self.prep = Some(prep);
            self.acts = None;
            self.exp = None;
            return Ok(false);
        }
        let acts = self.net.forward(&prep.input);
        let exp = explain(&self.net, &acts);
        self.prep = Some(prep);
        self.acts = Some(acts);
        self.exp = Some(exp);
        Ok(true)
    }

    fn acts(&self) -> Result<&Acts, JsValue> {
        self.acts
            .as_ref()
            .ok_or_else(|| JsValue::from_str("no run yet"))
    }

    fn exp(&self) -> Result<&Explanation, JsValue> {
        self.exp
            .as_ref()
            .ok_or_else(|| JsValue::from_str("no run yet"))
    }

    // -- preprocessing snapshots (Stage 2) ----------------------------------

    /// [x0, y0, w, h] of the ink in source pixels.
    pub fn bbox(&self) -> Vec<u32> {
        match &self.prep {
            Some(p) => {
                let (x, y, w, h) = p.bbox;
                vec![x as u32, y as u32, w as u32, h as u32]
            }
            None => vec![0, 0, 0, 0],
        }
    }

    /// The aspect-preserved 20px-box image, flattened. Pair with `boxed_size`.
    pub fn boxed(&self) -> Vec<f32> {
        self.prep
            .as_ref()
            .map(|p| p.boxed.data.clone())
            .unwrap_or_default()
    }

    pub fn boxed_size(&self) -> Vec<u32> {
        self.prep
            .as_ref()
            .map(|p| vec![p.boxed.w as u32, p.boxed.h as u32])
            .unwrap_or_else(|| vec![0, 0])
    }

    /// Centre of mass of the boxed image, and the integer offset used to place it.
    pub fn centring(&self) -> Vec<f32> {
        self.prep
            .as_ref()
            .map(|p| vec![p.com.0, p.com.1, p.offset.0 as f32, p.offset.1 as f32])
            .unwrap_or_else(|| vec![0.0; 4])
    }

    // -- activations --------------------------------------------------------

    /// 28x28. What the network actually sees.
    pub fn input(&self) -> Result<Vec<f32>, JsValue> {
        Ok(self.acts()?.input.data.clone())
    }

    /// 8x28x28, signed. Stage 3 needs the negative half to show ReLU removing it.
    pub fn conv1_pre(&self) -> Result<Vec<f32>, JsValue> {
        Ok(self.acts()?.conv1_pre.data.clone())
    }

    /// 8x28x28, after ReLU.
    pub fn conv1(&self) -> Result<Vec<f32>, JsValue> {
        Ok(self.acts()?.conv1.data.clone())
    }

    /// 8x14x14.
    pub fn pool1(&self) -> Result<Vec<f32>, JsValue> {
        Ok(self.acts()?.pool1.data.clone())
    }

    /// Which cell of each 2x2 window won, encoded ky*2+kx. Stage 4 animates the contest.
    pub fn pool1_winners(&self) -> Result<Vec<u8>, JsValue> {
        Ok(self.acts()?.pool1_arg.clone())
    }

    /// 16x14x14, signed.
    pub fn conv2_pre(&self) -> Result<Vec<f32>, JsValue> {
        Ok(self.acts()?.conv2_pre.data.clone())
    }

    /// 16x14x14, after ReLU.
    pub fn conv2(&self) -> Result<Vec<f32>, JsValue> {
        Ok(self.acts()?.conv2.data.clone())
    }

    /// 16x7x7.
    pub fn pool2(&self) -> Result<Vec<f32>, JsValue> {
        Ok(self.acts()?.pool2.data.clone())
    }

    pub fn pool2_winners(&self) -> Result<Vec<u8>, JsValue> {
        Ok(self.acts()?.pool2_arg.clone())
    }

    /// 32 hidden units, signed (pre-ReLU).
    pub fn fc1_pre(&self) -> Result<Vec<f32>, JsValue> {
        Ok(self.acts()?.fc1_pre.clone())
    }

    /// 32 hidden units after ReLU.
    pub fn fc1(&self) -> Result<Vec<f32>, JsValue> {
        Ok(self.acts()?.fc1.clone())
    }

    /// 10 raw scores, signed. These do not sum to anything, which is the point of Stage 6.
    pub fn logits(&self) -> Result<Vec<f32>, JsValue> {
        Ok(self.acts()?.logits.clone())
    }

    /// exp(logit - max): the first half of softmax, before normalisation.
    pub fn exps(&self) -> Result<Vec<f32>, JsValue> {
        Ok(self.acts()?.exps.clone())
    }

    /// 10 probabilities, summing to 1.
    pub fn probs(&self) -> Result<Vec<f32>, JsValue> {
        Ok(self.acts()?.probs.clone())
    }

    /// Per-channel sum of squares, so Stage 4 can rank feature maps by how much they fired.
    pub fn conv1_energy(&self) -> Result<Vec<f32>, JsValue> {
        Ok(self.acts()?.conv1.channel_energy())
    }

    pub fn conv2_energy(&self) -> Result<Vec<f32>, JsValue> {
        Ok(self.acts()?.conv2.channel_energy())
    }

    // -- learned parameters -------------------------------------------------

    /// The eight 5x5 first-layer kernels, flattened. Rendered directly in Stage 3:
    /// these are the actual learned edge detectors, not an illustration of one.
    pub fn kernels1(&self) -> Vec<f32> {
        self.net.conv1.w.clone()
    }

    /// The 16x8 second-layer 3x3 kernels, flattened as [oc][ic][3][3].
    pub fn kernels2(&self) -> Vec<f32> {
        self.net.conv2.w.clone()
    }

    // -- contributions (Stage 5) --------------------------------------------

    /// All 320 fc1 -> logits contributions as weight x activation, laid out
    /// [class * 32 + unit]. Positive pushes the class up, negative pushes it down.
    /// Small enough to send whole; this is what drives the excitatory/inhibitory flow.
    pub fn fc2_contributions(&self) -> Result<Vec<f32>, JsValue> {
        let a = self.acts()?;
        let d = &self.net.fc2;
        let mut out = vec![0.0; d.n_out * d.n_in];
        for o in 0..d.n_out {
            for i in 0..d.n_in {
                out[o * d.n_in + i] = d.w[o * d.n_in + i] * a.fc1[i];
            }
        }
        Ok(out)
    }

    /// Top-`k` pool2 -> fc1 contributions per hidden unit, as flat triples
    /// [src_index, dst_unit, contribution, ...]. The full matrix is 25,088 edges;
    /// drawing all of them is a hairball and copying all of them is 100 KB per run.
    pub fn fc1_top_contributions(&self, k: usize) -> Result<Vec<f32>, JsValue> {
        let a = self.acts()?;
        let d = &self.net.fc1;
        let k = k.clamp(1, d.n_in);
        let mut out = Vec::with_capacity(d.n_out * k * 3);

        let mut scratch: Vec<(usize, f32)> = Vec::with_capacity(d.n_in);
        for o in 0..d.n_out {
            scratch.clear();
            let row = &d.w[o * d.n_in..(o + 1) * d.n_in];
            for i in 0..d.n_in {
                let c = row[i] * a.pool2.data[i];
                if c != 0.0 {
                    scratch.push((i, c));
                }
            }
            scratch.sort_by(|x, y| {
                y.1.abs()
                    .partial_cmp(&x.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (i, c) in scratch.iter().take(k) {
                out.push(*i as f32);
                out.push(o as f32);
                out.push(*c);
            }
        }
        Ok(out)
    }

    // -- explanation (Stage 7) ----------------------------------------------

    /// 28x28 in [0,1]: which ink produced the winning class.
    pub fn saliency(&self) -> Result<Vec<f32>, JsValue> {
        Ok(self.exp()?.saliency.data.clone())
    }

    /// 28x28 in [-1,1]: positive argued for the winner, negative argued for the
    /// runner-up. This is the "what nearly changed its mind" layer.
    pub fn counterfactual(&self) -> Result<Vec<f32>, JsValue> {
        Ok(self.exp()?.counter.data.clone())
    }

    /// Measured facts for the closing copy. Rust reports, the UI phrases.
    pub fn evidence(&self) -> Result<String, JsValue> {
        let e = self.exp()?;
        let regions = e
            .regions
            .iter()
            .map(|v| format!("{v:.4}"))
            .collect::<Vec<_>>()
            .join(",");
        let counter = e
            .counter_regions
            .iter()
            .map(|v| format!("{v:.4}"))
            .collect::<Vec<_>>()
            .join(",");
        Ok(format!(
            concat!(
                "{{\"top1\":{},\"top2\":{},\"p1\":{:.6},\"p2\":{:.6},\"margin\":{:.4},",
                "\"holes\":{},\"inkRatio\":{:.4},\"regions\":[{}],\"counterRegions\":[{}],",
                "\"strongestRegion\":{},\"contestedRegion\":{},\"diffuse\":{}}}"
            ),
            e.top1,
            e.top2,
            e.p1,
            e.p2,
            e.margin,
            e.holes,
            e.ink_ratio,
            regions,
            counter,
            e.strongest_region(),
            e.most_contested_region(),
            e.evidence_is_diffuse()
        ))
    }

    /// Gradient ascent on a blank canvas: the network's own idea of a digit.
    /// Generated live rather than shipped as an asset.
    pub fn dream(&self, class: usize, steps: usize) -> Result<Vec<f32>, JsValue> {
        if class >= 10 {
            return Err(JsValue::from_str("class must be 0..9"));
        }
        Ok(crate::explain::dream(&self.net, class, steps.min(400), 0xD1CE).data)
    }
}

/// Decode the packed prototype asset (magic "PROT" then 10 x 784 bytes) into floats.
#[wasm_bindgen]
pub fn decode_prototypes(bytes: &[u8]) -> Result<Vec<f32>, JsValue> {
    if bytes.len() != 4 + 10 * 784 || &bytes[0..4] != b"PROT" {
        return Err(JsValue::from_str("prototypes.bin is malformed"));
    }
    Ok(bytes[4..].iter().map(|b| *b as f32 / 255.0).collect())
}
