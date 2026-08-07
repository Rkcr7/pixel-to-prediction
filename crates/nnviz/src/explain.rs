//! Why the network decided what it decided.
//!
//! Everything here is measured, not asserted. The app's closing sentence is composed from
//! this evidence, so the rule is: Rust reports facts, the UI does the phrasing. If the
//! evidence is weak, that is itself a fact and the copy is allowed to say so.

use crate::net::{Acts, Net, CLASSES};
use crate::tensor::Vol;

pub const GRID: usize = 3; // 3x3 anatomical regions over the 28x28 field
pub const REGIONS: usize = GRID * GRID;

/// Human-facing names for the 3x3 regions, row-major from the top-left.
pub const REGION_NAMES: [&str; REGIONS] = [
    "upper left",
    "top",
    "upper right",
    "left side",
    "centre",
    "right side",
    "lower left",
    "bottom",
    "lower right",
];

pub struct Explanation {
    /// Positive attribution for the winning class, normalised to [0, 1].
    pub saliency: Vol,
    /// Signed evidence separating the winner from the runner-up, normalised to [-1, 1].
    /// Positive means "this ink argued for the winner"; negative means it argued for the
    /// runner-up.
    pub counter: Vol,
    /// Share of total positive attribution falling in each region. Sums to 1.
    pub regions: [f32; REGIONS],
    /// Signed counterfactual mass per region, normalised by the largest magnitude.
    pub counter_regions: [f32; REGIONS],
    /// Enclosed background areas in the stroke: 0 for a 1 or 7, 1 for a 0 or 9, 2 for an 8.
    pub holes: u32,
    /// Fraction of the field carrying ink.
    pub ink_ratio: f32,
    pub top1: usize,
    pub top2: usize,
    pub p1: f32,
    pub p2: f32,
    /// Logit margin between winner and runner-up: how decisive the call was.
    pub margin: f32,
}

impl Explanation {
    /// Index of the region contributing most positive evidence for the winner.
    pub fn strongest_region(&self) -> usize {
        argmax_f32(&self.regions)
    }

    /// Region whose ink argued hardest *for the runner-up* — the "what nearly changed
    /// its mind" region.
    pub fn most_contested_region(&self) -> usize {
        let mut best = 0;
        let mut best_v = f32::INFINITY;
        for (i, v) in self.counter_regions.iter().enumerate() {
            if *v < best_v {
                best_v = *v;
                best = i;
            }
        }
        best
    }

    /// True when no region carries a clear majority of the evidence, which is the honest
    /// signal that the model is guessing from diffuse cues.
    pub fn evidence_is_diffuse(&self) -> bool {
        self.regions.iter().cloned().fold(0.0f32, f32::max) < 0.22
    }
}

fn argmax_f32(v: &[f32]) -> usize {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, x) in v.iter().enumerate() {
        if *x > bv {
            bv = *x;
            best = i;
        }
    }
    best
}

/// Gradient of a single logit with respect to the input, multiplied by the input.
///
/// Plain gradient answers "what would change the answer"; gradient x input answers
/// "what in the drawing actually produced the answer". The second question is the one a
/// person is asking.
fn grad_times_input(net: &Net, a: &Acts, dlogits: &[f32]) -> Vol {
    let g = net.backward(a, dlogits, None);
    Vol::from_vec(
        1,
        g.h,
        g.w,
        g.data
            .iter()
            .zip(&a.input.data)
            .map(|(gr, x)| gr * x)
            .collect(),
    )
}

fn onehot(i: usize) -> Vec<f32> {
    let mut v = vec![0.0; CLASSES];
    v[i] = 1.0;
    v
}

/// Where changing a pixel would push the answer from `from` towards `to`.
///
/// The raw input gradient of `logit(to) - logit(from)`, normalised to [-1, 1] and
/// deliberately **not** multiplied by the input.
///
/// That is the whole difference between this and [`Explanation::counter`]. Attribution
/// (`grad * input`) answers "which of the ink you drew argued for what", so it is exactly
/// zero everywhere you did not draw, and it therefore cannot suggest anywhere to put a
/// stroke that is not there yet. The bare gradient can: positive here means ink added at
/// that pixel moves the network towards `to`, negative means ink removed from it does.
///
/// It is a local, first-order answer. It is honest about one small change, not a recipe,
/// which is why the UI offers it as a hint to try rather than as an instruction.
pub fn flip_gradient(net: &Net, a: &Acts, to: usize, from: usize) -> Vol {
    let mut d = onehot(to);
    d[from] -= 1.0;
    let g = net.backward(a, &d, None);
    let m = g.abs_max().max(1e-8);
    Vol::from_vec(1, g.h, g.w, g.data.iter().map(|v| v / m).collect())
}

/// Count enclosed background regions by flooding the background inward from the border.
/// Anything left unvisited is a hole.
pub fn count_holes(input: &Vol, thresh: f32) -> u32 {
    let (h, w) = (input.h, input.w);
    let ink = |y: usize, x: usize| input.at(0, y, x) > thresh;

    let mut seen = vec![false; h * w];
    let mut stack: Vec<(usize, usize)> = Vec::new();

    for x in 0..w {
        for y in [0usize, h - 1] {
            if !ink(y, x) && !seen[y * w + x] {
                seen[y * w + x] = true;
                stack.push((y, x));
            }
        }
    }
    for y in 0..h {
        for x in [0usize, w - 1] {
            if !ink(y, x) && !seen[y * w + x] {
                seen[y * w + x] = true;
                stack.push((y, x));
            }
        }
    }

    while let Some((y, x)) = stack.pop() {
        let push = |ny: usize, nx: usize, stack: &mut Vec<(usize, usize)>, seen: &mut Vec<bool>| {
            if !ink(ny, nx) && !seen[ny * w + nx] {
                seen[ny * w + nx] = true;
                stack.push((ny, nx));
            }
        };
        if y > 0 {
            push(y - 1, x, &mut stack, &mut seen);
        }
        if y + 1 < h {
            push(y + 1, x, &mut stack, &mut seen);
        }
        if x > 0 {
            push(y, x - 1, &mut stack, &mut seen);
        }
        if x + 1 < w {
            push(y, x + 1, &mut stack, &mut seen);
        }
    }

    // Whatever the border flood could not reach is enclosed. Count those components,
    // ignoring the ones only a few pixels across: those are antialiasing noise between
    // two strokes, not a loop anyone drew.
    count_significant_holes(input, thresh, &seen)
}

fn count_significant_holes(input: &Vol, thresh: f32, border_reached: &[bool]) -> u32 {
    let (h, w) = (input.h, input.w);
    let ink = |y: usize, x: usize| input.at(0, y, x) > thresh;
    let mut visited = border_reached.to_vec();
    let mut holes = 0u32;
    for y in 0..h {
        for x in 0..w {
            if ink(y, x) || visited[y * w + x] {
                continue;
            }
            let mut size = 0usize;
            let mut s = vec![(y, x)];
            visited[y * w + x] = true;
            while let Some((cy, cx)) = s.pop() {
                size += 1;
                let neighbours = [
                    (cy.wrapping_sub(1), cx),
                    (cy + 1, cx),
                    (cy, cx.wrapping_sub(1)),
                    (cy, cx + 1),
                ];
                for (ny, nx) in neighbours {
                    if ny >= h || nx >= w {
                        continue;
                    }
                    if !ink(ny, nx) && !visited[ny * w + nx] {
                        visited[ny * w + nx] = true;
                        s.push((ny, nx));
                    }
                }
            }
            if size >= 3 {
                holes += 1;
            }
        }
    }
    holes
}

fn region_of(y: usize, x: usize, h: usize, w: usize) -> usize {
    let ry = (y * GRID / h).min(GRID - 1);
    let rx = (x * GRID / w).min(GRID - 1);
    ry * GRID + rx
}

pub fn explain(net: &Net, a: &Acts) -> Explanation {
    let ranked = a.top_k();
    let (top1, p1) = ranked[0];
    let (top2, p2) = ranked[1];

    let sal_raw = grad_times_input(net, a, &onehot(top1));

    let mut d = onehot(top1);
    d[top2] -= 1.0;
    let counter_raw = grad_times_input(net, a, &d);

    // Saliency: keep the positive half, normalise to [0, 1].
    let smax = sal_raw
        .data
        .iter()
        .cloned()
        .fold(0.0f32, f32::max)
        .max(1e-8);
    let saliency = Vol::from_vec(
        1,
        sal_raw.h,
        sal_raw.w,
        sal_raw
            .data
            .iter()
            .map(|v| (v.max(0.0) / smax).min(1.0))
            .collect(),
    );

    // Counterfactual: signed, normalised by the largest magnitude either way.
    let cmax = counter_raw.abs_max().max(1e-8);
    let counter = Vol::from_vec(
        1,
        counter_raw.h,
        counter_raw.w,
        counter_raw
            .data
            .iter()
            .map(|v| (v / cmax).clamp(-1.0, 1.0))
            .collect(),
    );

    let mut regions = [0.0f32; REGIONS];
    let mut counter_regions = [0.0f32; REGIONS];
    for y in 0..saliency.h {
        for x in 0..saliency.w {
            let r = region_of(y, x, saliency.h, saliency.w);
            regions[r] += saliency.at(0, y, x);
            counter_regions[r] += counter.at(0, y, x);
        }
    }
    // When every attribution for the winning class is non-positive — possible for a
    // sparse stroke whose winning logit is carried by biases — there is no distribution
    // to normalise. Spread it evenly rather than emitting nine zeros, which would
    // silently break the "shares sum to 1" contract this field advertises.
    let total: f32 = regions.iter().sum();
    if total > 1e-7 {
        regions.iter_mut().for_each(|v| *v /= total);
    } else {
        regions.iter_mut().for_each(|v| *v = 1.0 / REGIONS as f32);
    }
    let cnorm = counter_regions
        .iter()
        .cloned()
        .fold(0.0f32, |m, v| m.max(v.abs()))
        .max(1e-8);
    counter_regions.iter_mut().for_each(|v| *v /= cnorm);

    let ink_ratio =
        a.input.data.iter().filter(|v| **v > 0.15).count() as f32 / a.input.len() as f32;

    Explanation {
        saliency,
        counter,
        regions,
        counter_regions,
        holes: count_holes(&a.input, 0.35),
        ink_ratio,
        top1,
        top2,
        p1,
        p2,
        margin: a.logits[top1] - a.logits[top2],
    }
}

/// The network's own idea of a digit, found by gradient ascent on a blank canvas.
///
/// Unregularised activation maximisation on a small net produces high-frequency noise that
/// happens to excite the class. The blur/decay/jitter schedule below is what turns it into
/// something a person recognises, and is the standard recipe from the feature-visualisation
/// literature. Roughly 5 ms for 120 steps in WASM, so this is generated live rather than
/// shipped as an asset.
pub fn dream(net: &Net, class: usize, steps: usize, seed: u64) -> Vol {
    use crate::tensor::Rng;
    let mut r = Rng::new(seed);
    let mut x = Vol::from_vec(
        1,
        28,
        28,
        (0..784).map(|_| 0.5 + r.normal() * 0.02).collect(),
    );
    let d = onehot(class);

    for step in 0..steps {
        // Jitter by up to one pixel: prevents the optimiser latching onto a fixed grid
        // phase and produces noticeably more coherent strokes.
        let (jy, jx) = (r.below(3) as i32 - 1, r.below(3) as i32 - 1);
        let jittered = shift(&x, jy, jx);

        let a = net.forward(&jittered);
        let g = net.backward(&a, &d, None);
        let gmax = g.abs_max().max(1e-8);

        let back = shift(&g, -jy, -jx);
        for (xv, gv) in x.data.iter_mut().zip(&back.data) {
            *xv += 0.9 * (gv / gmax);
            *xv *= 0.985; // L2-ish decay, keeps the canvas from saturating
            *xv = xv.clamp(0.0, 1.0);
        }
        if step % 4 == 3 {
            x = blur3(&x);
        }
    }

    // Final contrast stretch so the result reads as a digit rather than a grey fog.
    let lo = x.data.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = x.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let span = (hi - lo).max(1e-6);
    Vol::from_vec(
        1,
        28,
        28,
        x.data
            .iter()
            .map(|v| ((v - lo) / span).clamp(0.0, 1.0))
            .collect(),
    )
}

fn shift(v: &Vol, dy: i32, dx: i32) -> Vol {
    let mut out = Vol::zeros(v.c, v.h, v.w);
    for c in 0..v.c {
        for y in 0..v.h {
            let sy = y as i32 - dy;
            if sy < 0 || sy >= v.h as i32 {
                continue;
            }
            for x in 0..v.w {
                let sx = x as i32 - dx;
                if sx < 0 || sx >= v.w as i32 {
                    continue;
                }
                out.set(c, y, x, v.at(c, sy as usize, sx as usize));
            }
        }
    }
    out
}

fn blur3(v: &Vol) -> Vol {
    const K: [f32; 3] = [0.25, 0.5, 0.25];
    let mut tmp = Vol::zeros(v.c, v.h, v.w);
    for c in 0..v.c {
        for y in 0..v.h {
            for x in 0..v.w {
                let mut acc = 0.0;
                let mut wsum = 0.0;
                for (i, k) in K.iter().enumerate() {
                    let sx = x as i32 + i as i32 - 1;
                    if sx < 0 || sx >= v.w as i32 {
                        continue;
                    }
                    acc += v.at(c, y, sx as usize) * k;
                    wsum += k;
                }
                tmp.set(c, y, x, acc / wsum.max(1e-6));
            }
        }
    }
    let mut out = Vol::zeros(v.c, v.h, v.w);
    for c in 0..v.c {
        for y in 0..v.h {
            for x in 0..v.w {
                let mut acc = 0.0;
                let mut wsum = 0.0;
                for (i, k) in K.iter().enumerate() {
                    let sy = y as i32 + i as i32 - 1;
                    if sy < 0 || sy >= v.h as i32 {
                        continue;
                    }
                    acc += tmp.at(c, sy as usize, x) * k;
                    wsum += k;
                }
                out.set(c, y, x, acc / wsum.max(1e-6));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Rng;

    fn ring(hole: bool) -> Vol {
        let mut v = Vol::zeros(1, 28, 28);
        for y in 8..20 {
            for x in 8..20 {
                let edge = y == 8 || y == 19 || x == 8 || x == 19;
                if edge {
                    v.set(0, y, x, 1.0);
                }
            }
        }
        if !hole {
            // Break the ring so the background can escape.
            for x in 8..20 {
                v.set(0, 8, x, 0.0);
            }
        }
        v
    }

    #[test]
    fn closed_ring_has_one_hole() {
        assert_eq!(count_holes(&ring(true), 0.35), 1);
    }

    #[test]
    fn broken_ring_has_no_holes() {
        assert_eq!(count_holes(&ring(false), 0.35), 0);
    }

    #[test]
    fn blank_field_has_no_holes() {
        assert_eq!(count_holes(&Vol::zeros(1, 28, 28), 0.35), 0);
    }

    #[test]
    fn two_rings_give_two_holes() {
        let mut v = Vol::zeros(1, 28, 28);
        for (oy, ox) in [(3usize, 3usize), (15, 15)] {
            for y in oy..oy + 8 {
                for x in ox..ox + 8 {
                    if y == oy || y == oy + 7 || x == ox || x == ox + 7 {
                        v.set(0, y, x, 1.0);
                    }
                }
            }
        }
        assert_eq!(count_holes(&v, 0.35), 2);
    }

    #[test]
    fn region_attribution_sums_to_one_and_is_finite() {
        let net = Net::new_random(4);
        let mut r = Rng::new(31);
        let x = Vol::from_vec(1, 28, 28, (0..784).map(|_| r.f32() * 0.8).collect());
        let a = net.forward(&x);
        let e = explain(&net, &a);
        let sum: f32 = e.regions.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "regions summed to {sum}");
        assert!(e.regions.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(e
            .counter_regions
            .iter()
            .all(|v| v.is_finite() && v.abs() <= 1.0 + 1e-5));
    }

    #[test]
    fn saliency_is_normalised_and_counter_is_signed() {
        let net = Net::new_random(8);
        let mut r = Rng::new(77);
        let x = Vol::from_vec(1, 28, 28, (0..784).map(|_| r.f32()).collect());
        let a = net.forward(&x);
        let e = explain(&net, &a);
        assert!(e.saliency.data.iter().all(|v| (0.0..=1.0).contains(v)));
        assert!(e.counter.data.iter().all(|v| (-1.0..=1.0).contains(v)));
        assert!(e.top1 != e.top2);
        assert!(e.p1 >= e.p2);
    }

    #[test]
    fn flip_gradient_speaks_where_there_is_no_ink() {
        // The entire reason this exists instead of reusing `counter`. Attribution is
        // grad * input, so it is identically zero on every blank pixel and can never
        // point at somewhere to add a stroke. If the bare gradient were silent there
        // too, the hint would have nothing to say and the feature would be a lie.
        let net = Net::new_random(21);
        let mut r = Rng::new(5);
        let x = Vol::from_vec(
            1,
            28,
            28,
            (0..784)
                .map(|i| if i % 37 == 0 { r.f32() } else { 0.0 })
                .collect(),
        );
        let a = net.forward(&x);
        let e = explain(&net, &a);
        let g = flip_gradient(&net, &a, e.top2, e.top1);

        assert!(g.data.iter().all(|v| (-1.0..=1.0).contains(v)));

        let blank: Vec<usize> = (0..784).filter(|&i| x.data[i] == 0.0).collect();
        assert!(
            blank.len() > 700,
            "input was not sparse enough to test this"
        );
        assert!(
            blank.iter().all(|&i| e.counter.data[i] == 0.0),
            "attribution must be zero wherever there is no ink",
        );
        let speaks = blank.iter().filter(|&&i| g.data[i].abs() > 1e-6).count();
        assert!(
            speaks > 100,
            "the flip gradient was silent on all but {speaks} of {} blank pixels",
            blank.len(),
        );
    }

    #[test]
    fn stepping_along_the_flip_gradient_closes_the_margin() {
        // The direction has to be right, not just non-zero. A sign error here would
        // produce a hint that confidently tells you to do the opposite of what works,
        // and nothing on screen would look wrong.
        let net = Net::new_random(9);
        let mut r = Rng::new(31);
        let x = Vol::from_vec(1, 28, 28, (0..784).map(|_| r.f32() * 0.6).collect());
        let a = net.forward(&x);
        let e = explain(&net, &a);
        let g = flip_gradient(&net, &a, e.top2, e.top1);

        let before = a.logits[e.top2] - a.logits[e.top1];
        let stepped = Vol::from_vec(
            1,
            28,
            28,
            x.data
                .iter()
                .zip(&g.data)
                .map(|(v, gr)| (v + 0.02 * gr).clamp(0.0, 1.0))
                .collect(),
        );
        let after = net.forward(&stepped);
        let closed = after.logits[e.top2] - after.logits[e.top1];
        assert!(
            closed > before,
            "one step along the gradient moved the margin the wrong way: {before} -> {closed}",
        );
    }

    #[test]
    fn dream_stays_in_range_and_has_structure() {
        let net = Net::new_random(12);
        let d = dream(&net, 3, 40, 5);
        assert_eq!(d.len(), 784);
        assert!(d.data.iter().all(|v| (0.0..=1.0).contains(v)));
        // Contrast stretch guarantees it spans the range rather than being flat.
        let mean = d.data.iter().sum::<f32>() / 784.0;
        let var = d.data.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 784.0;
        assert!(var > 1e-4, "dream was nearly flat, var {var}");
    }
}
