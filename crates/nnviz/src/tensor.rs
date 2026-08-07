//! Minimal CHW volume type. Deliberately tiny: the whole network is 27k parameters,
//! so clarity beats generality everywhere in this crate.

/// A 3D volume laid out channel-major: index = (c * h + y) * w + x.
#[derive(Clone, Debug, PartialEq)]
pub struct Vol {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Vol {
    pub fn zeros(c: usize, h: usize, w: usize) -> Self {
        Vol {
            c,
            h,
            w,
            data: vec![0.0; c * h * w],
        }
    }

    pub fn from_vec(c: usize, h: usize, w: usize, data: Vec<f32>) -> Self {
        debug_assert_eq!(data.len(), c * h * w, "Vol::from_vec shape/data mismatch");
        Vol { c, h, w, data }
    }

    #[inline(always)]
    pub fn idx(&self, c: usize, y: usize, x: usize) -> usize {
        (c * self.h + y) * self.w + x
    }

    #[inline(always)]
    pub fn at(&self, c: usize, y: usize, x: usize) -> f32 {
        self.data[self.idx(c, y, x)]
    }

    #[inline(always)]
    pub fn set(&mut self, c: usize, y: usize, x: usize, v: f32) {
        let i = self.idx(c, y, x);
        self.data[i] = v;
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Read-only slice of a single channel plane.
    pub fn plane(&self, c: usize) -> &[f32] {
        let n = self.h * self.w;
        &self.data[c * n..(c + 1) * n]
    }

    pub fn fill(&mut self, v: f32) {
        self.data.iter_mut().for_each(|d| *d = v);
    }

    /// Largest absolute value, used to normalise a layer for display so that
    /// every channel of a layer shares one scale (channels stay comparable).
    pub fn abs_max(&self) -> f32 {
        self.data.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    }

    /// Per-channel sum of squares. Stage 4 ranks feature maps by this "energy".
    pub fn channel_energy(&self) -> Vec<f32> {
        (0..self.c)
            .map(|c| self.plane(c).iter().map(|v| v * v).sum())
            .collect()
    }
}

/// xorshift128+ variant. Self-contained so the crate has zero RNG dependencies and
/// so training runs are exactly reproducible from a seed on any platform.
pub struct Rng(u64, u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // splitmix64 to spread a small seed across both words.
        let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        };
        Rng(next(), next() | 1)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut s1 = self.0;
        let s0 = self.1;
        let result = s0.wrapping_add(s1);
        self.0 = s0;
        s1 ^= s1 << 23;
        self.1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
        result
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Uniform in [lo, hi).
    #[inline]
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.f32() * (hi - lo)
    }

    /// Standard normal via Box-Muller (one of the pair, which is fine here).
    pub fn normal(&mut self) -> f32 {
        let u1 = (self.f32()).max(1e-7);
        let u2 = self.f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexing_is_channel_major() {
        let mut v = Vol::zeros(2, 3, 4);
        v.set(1, 2, 3, 7.0);
        assert_eq!(v.data[(1 * 3 + 2) * 4 + 3], 7.0);
        assert_eq!(v.at(1, 2, 3), 7.0);
    }

    #[test]
    fn plane_returns_one_channel() {
        let v = Vol::from_vec(2, 1, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.plane(0), &[1.0, 2.0]);
        assert_eq!(v.plane(1), &[3.0, 4.0]);
    }

    #[test]
    fn channel_energy_is_sum_of_squares() {
        let v = Vol::from_vec(2, 1, 2, vec![3.0, 4.0, 1.0, 0.0]);
        assert_eq!(v.channel_energy(), vec![25.0, 1.0]);
    }

    #[test]
    fn rng_is_deterministic_and_in_range() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            let x = a.f32();
            assert_eq!(x, b.f32());
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn rng_normal_has_roughly_unit_variance() {
        let mut r = Rng::new(7);
        let n = 20000;
        let xs: Vec<f32> = (0..n).map(|_| r.normal()).collect();
        let mean = xs.iter().sum::<f32>() / n as f32;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.05, "mean was {mean}");
        assert!((var - 1.0).abs() < 0.1, "var was {var}");
    }
}
