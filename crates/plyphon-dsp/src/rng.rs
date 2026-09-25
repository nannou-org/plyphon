//! A random number generator - plyphon's port of scsynth's `RGen` (Taus88).
//!
//! As in scsynth, the World holds a set of these streams and every synth draws from one of them.
//! scsynth seeds each from the clock; plyphon seeds stream `i` with `Rng::new(i)`, which is
//! scsynth's `RGen::init(i)`, so a render is the same every run.

/// A Taus88 combined Tausworthe generator (the algorithm scsynth uses).
///
/// `repr(C)` + `Pod`, so a unit can copy it into and out of its state as plain bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Rng {
    s1: u32,
    s2: u32,
    s3: u32,
}

impl Rng {
    /// A generator seeded as scsynth's `RGen::init(seed)` seeds one (see [`Rng::init`]).
    pub fn new(seed: u32) -> Self {
        let mut rng = Rng {
            s1: 0,
            s2: 0,
            s3: 0,
        };
        rng.init(seed);
        rng
    }

    /// Re-seed the generator exactly as scsynth's `RGen::init`: the seed is scrambled with [`hash`],
    /// then XORed into three fixed state constants, and a state word that would break its Taus88
    /// lower bound (`s1 > 1`, `s2 > 7`, `s3 > 15`) falls back to its constant.
    pub fn init(&mut self, seed: u32) {
        let seed = hash(seed as i32) as u32;
        self.s1 = 1_243_598_713 ^ seed;
        if self.s1 < 2 {
            self.s1 = 1_243_598_713;
        }
        self.s2 = 3_093_459_404 ^ seed;
        if self.s2 < 8 {
            self.s2 = 3_093_459_404;
        }
        self.s3 = 1_821_928_721 ^ seed;
        if self.s3 < 16 {
            self.s3 = 1_821_928_721;
        }
    }

    /// The next 32-bit random word.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.s1 = ((self.s1 & 0xFFFF_FFFE) << 12) ^ (((self.s1 << 13) ^ self.s1) >> 19);
        self.s2 = ((self.s2 & 0xFFFF_FFF8) << 4) ^ (((self.s2 << 2) ^ self.s2) >> 25);
        self.s3 = ((self.s3 & 0xFFFF_FFF0) << 17) ^ (((self.s3 << 3) ^ self.s3) >> 11);
        self.s1 ^ self.s2 ^ self.s3
    }

    /// A bipolar sample uniformly distributed in `[-1, 1)` - bit-exact with scsynth's
    /// `RGen::frand2` (`0x40000000 | (trand() >> 9)` reinterpreted as a float in `[2, 4)`, minus
    /// 3), so seeded noise renders match scsynth's under the same Taus88 state.
    #[inline]
    pub fn next_bipolar(&mut self) -> f32 {
        f32::from_bits(0x4000_0000 | (self.next_u32() >> 9)) - 3.0
    }

    /// A unipolar sample uniformly distributed in `[0, 1)` (scsynth's `RGen::frand`). Uses the top 23
    /// bits so the result lands exactly in `[0, 1)` at `f32` precision.
    #[inline]
    pub fn next_unipolar(&mut self) -> f32 {
        (self.next_u32() >> 9) as f32 * (1.0 / 8_388_608.0)
    }

    /// A double-precision sample uniformly distributed in `[0, 1)` from all 32 bits of the next
    /// word - bit-exact with scsynth's `RGen::drand` (the word as the low mantissa bits of a double
    /// in `[2^20, 2^20 + 1)`, minus `2^20`).
    #[inline]
    pub fn next_unipolar_f64(&mut self) -> f64 {
        f64::from_bits(0x4130_0000_0000_0000 | self.next_u32() as u64) - 1_048_576.0
    }

    /// A uniform integer in `[0, scale)` (scsynth's `RGen::irand`, `floor(scale * drand())` in double
    /// precision). A `scale` of `0` or less yields values in `[scale, 0]`; every call consumes one
    /// word.
    #[inline]
    pub fn next_irand(&mut self, scale: i32) -> i32 {
        crate::math::floor(scale as f64 * self.next_unipolar_f64()) as i32
    }

    /// A uniform integer in `[-scale, scale]` (scsynth's `RGen::irand2`,
    /// `floor((2 * scale + 1) * drand() - scale)` in double precision).
    #[inline]
    pub fn next_irand2(&mut self, scale: i32) -> i32 {
        crate::math::floor((2.0 * scale as f64 + 1.0) * self.next_unipolar_f64() - scale as f64)
            as i32
    }

    /// An exponential-distribution draw between `lo` and `hi` (scsynth's `RGen::exprandrng`,
    /// `lo * exp(log(hi / lo) * drand())` in double precision).
    #[inline]
    pub fn next_exprand(&mut self, lo: f64, hi: f64) -> f64 {
        lo * crate::math::exp(crate::math::ln(hi / lo) * self.next_unipolar_f64())
    }
}

/// Thomas Wang's integer hash (scsynth's `Hash(int32)`): it scrambles an [`Rng::init`] seed, and
/// the `Hasher` unit derives deterministic noise from a signal's bits with it.
pub fn hash(key: i32) -> i32 {
    let mut h = key as u32;
    h = h.wrapping_add(!(h << 15));
    h ^= h >> 10;
    h = h.wrapping_add(h << 3);
    h ^= h >> 6;
    h = h.wrapping_add(!(h << 11));
    h ^= h >> 16;
    h as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_and_decorrelated() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let mut sum = 0.0f64;
        let mut diff = 0usize;
        for _ in 0..10_000 {
            let (x, y) = (a.next_bipolar(), b.next_bipolar());
            assert!((-1.0..1.0).contains(&x));
            sum += x as f64;
            if x != y {
                diff += 1;
            }
        }
        // Roughly zero-mean and clearly different streams for different seeds.
        assert!((sum / 10_000.0).abs() < 0.1);
        assert!(diff > 9_000);
    }

    #[test]
    fn frand_bit_parity_with_scsynth() {
        // `frand`/`frand2` must reproduce scsynth's mantissa bit-tricks exactly, so a seeded
        // render matches scsynth's under the same Taus88 state.
        let mut a = Rng::new(42);
        for _ in 0..1_000 {
            let mut word_rng = a; // `Rng` is `Copy`; peek the next word without advancing `a`.
            let word = word_rng.next_u32();
            let mut uni_rng = a;
            let mut bi_rng = a;
            assert_eq!(
                uni_rng.next_unipolar().to_bits(),
                (f32::from_bits(0x3F80_0000 | (word >> 9)) - 1.0).to_bits(),
                "frand"
            );
            assert_eq!(
                bi_rng.next_bipolar().to_bits(),
                (f32::from_bits(0x4000_0000 | (word >> 9)) - 3.0).to_bits(),
                "frand2"
            );
            a.next_u32();
        }
    }
}
