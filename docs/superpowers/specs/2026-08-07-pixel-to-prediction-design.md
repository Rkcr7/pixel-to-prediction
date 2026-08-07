# Pixel to Prediction — Design Spec

Date: 2026-08-07
Status: Approved (autonomous authority granted by Ritik)

## 1. What this is

A single-page web experience. Someone draws a digit. They press one button. They watch a
cinematic, seven-stage journey through what a convolutional neural network actually does to
that drawing, ending in a prediction, a confidence, and an explanation derived from real
gradients rather than a canned template.

The target feeling is "I did not know I was allowed to see this."

## 2. Non-negotiable qualities

| Quality | Concrete meaning |
| --- | --- |
| Real | Every pixel on screen is derived from live tensor data. Nothing is pre-rendered, faked, or approximated for looks. |
| Fast | Time from "Reveal" click to first frame under 30 ms. Sustained 60 fps on a mid-range phone. |
| Seek-safe | Any visual state is a pure function of timeline position. Scrubbing backward is identical to playing forward to that point. |
| Small | Total transfer under 300 KB gzipped including model weights and WASM. |
| Shareable | A screen recording of one run must look good enough to post without editing. |

## 3. The decision that shapes everything

**No third-party inference runtime.**

LiteRT.js, ONNX Runtime Web and TensorFlow.js all treat a model as a black box: input in,
logits out. This product is made entirely of the things they hide — per-layer activations,
the learned kernels themselves, pre-softmax logits, and gradients with respect to the input.

So the forward *and backward* pass are written from scratch in Rust. That one crate then does
three jobs from a single source of truth:

1. Trains the model natively (with rayon).
2. Runs inference in the browser via WASM, capturing every intermediate tensor.
3. Computes gradient saliency and counterfactual attribution live, in-browser.

There is no train/serve skew because it is literally the same code path.

## 4. Model architecture — designed for legibility

```
input      28x28x1
conv1      8  filters, 5x5, pad 2   -> 28x28x8   -> ReLU      (208 params)
pool1      max 2x2                  -> 14x14x8
conv2      16 filters, 3x3, pad 1   -> 14x14x16  -> ReLU    (1,168 params)
pool2      max 2x2                  -> 7x7x16
flatten                             -> 784
fc1        784 -> 32                -> ReLU    (25,120 params)
fc2        32 -> 10                 -> softmax    (330 params)
                                                total 26,826 params
```

Choices made for the visualization, not for the leaderboard:

- **5x5 kernels in conv1.** A 5x5 kernel rendered on screen reads as a recognizable oriented
  edge or blob detector. A 3x3 kernel is visual mush. This is the single most important
  architectural concession to the product.
- **8 filters in conv1.** The most you can display large and individually readable at once.
- **16 filters in conv2.** Tiles as a 4x4 grid, and reads as "more, and more abstract".
- **No batch norm, no dropout at inference, ReLU only.** Every operation in the graph must be
  explainable in one plain sentence. Batch norm is not.
- **32-unit fc1.** Small enough to draw all 32 units as discrete objects in Stage 5.

Accuracy target: >= 99.3% on the MNIST test set. Reached with affine augmentation.

## 5. Stage design

Each stage is an isolated module exposing `enter/update(t)/exit` and drawing only from tensors
the core produced.

| # | Name | Real data shown |
| --- | --- | --- |
| 1 | Your Digit | Raw canvas bitmap |
| 2 | Preparing the Input | Actual crop / scale / center-of-mass steps, each snapshotted |
| 3 | First Look | The 8 learned 5x5 kernels convolving; conv1 activations at 28x28 |
| 4 | Finding Shapes | 16 conv2 maps at 14x14, ranked by activation energy, with depth |
| 5 | Matching Possibilities | fc1 -> fc2 weight flow; 10 logits growing; class prototypes from live activation maximization |
| 6 | The Decision | Softmax normalization made visible; losers collapse into the winner |
| 7 | Final Answer | Prediction, confidence, gradient-derived explanation, counterfactual |

## 6. The explanation engine

Stage 7's sentence is the credibility test of the whole project. It is generated, never
templated:

1. Compute `d(logit_top1) / d(input)` — a real backward pass. Gradient x input gives
   per-pixel attribution.
2. Compute `d(logit_top1 - logit_top2) / d(input)` — the *counterfactual* gradient. This is
   the evidence that separated the winner from the runner-up.
3. Project attribution onto a coarse anatomical grid (upper-left, centre, lower-right, ...)
   and onto stroke topology (closed region count, endpoints).
4. Compose a sentence from the strongest real evidence, e.g.
   "The closed loop up top and the tail falling to the left is what made this a 9 and not a 4."

If the evidence is weak or contradictory the copy says so honestly rather than inventing a
reason. Low confidence gets its own voice.

**Class prototypes are generated live.** The ten candidate glyphs orbiting in Stage 5 are not
font characters. They are produced by gradient ascent on a blank input to maximize each class
logit — the network's own idea of what each digit looks like. Roughly 120 steps over 784
pixels; under 10 ms in WASM. This is real-time asset generation in the most literal sense.

## 7. Rendering

**WebGL2, custom renderer, no Three.js.**

WebGPU reaches roughly 70% of browsers in 2026 and still has gaps on Firefox/Linux and older
Android. A project whose success metric is virality cannot exclude a third of phones. The
actual GPU load here — 28x28 textures, a few thousand instanced particles, one bloom chain —
is trivial for WebGL2. WebGPU would add risk and buy nothing.

Three.js is rejected on bundle size and because a scene graph is the wrong abstraction: this
is a 2.5D compositing problem, not a 3D scene.

Pipeline:

- Activation tensors uploaded as R32F textures, one per layer, refreshed only when the run
  changes.
- A field shader maps activation to colour through a shared palette LUT, so "important" and
  "weak" mean the same thing in every stage.
- Instanced particle pass, positions rejection-sampled from activation magnitude, so particles
  literally flow along real signal.
- Two-pass separable bloom for glow.
- Text and controls live in DOM above the canvas — crisp, accessible, and free.

## 8. Timeline

A single deterministic clock. Every animated property is `f(t)` evaluated fresh each frame from
the timeline definition; nothing accumulates. Consequences: scrubbing is exact, speed changes
are free, stepping backward is correct, and a recorded render is frame-identical to a live one.

Hand-written, roughly 150 lines. GSAP is 70 KB and solves a harder problem than we have.

## 9. Stack

- `crates/nnviz` — Rust. Core net, preprocessing, saliency, native trainer binary, WASM lib.
- `web/` — Vite + TypeScript, no UI framework. One screen, animation-driven; a virtual DOM
  fights a 60 fps timeline for no benefit.
- Model weights ship as a quantized binary blob loaded by the WASM module.

## 10. Testing

- Rust: unit tests per layer, plus a numerical gradient check against the analytic backward
  pass. If backward is wrong, every explanation is a lie, so this is the highest-value test.
- Preprocessing: golden tests on known inputs.
- Accuracy: the trainer asserts the >= 99.3% gate before exporting weights.
- Browser: scripted run through all seven stages, console clean, frame budget measured.

## 11. Out of scope

Accounts, persistence, server, analytics, non-digit classes, model training in browser.
