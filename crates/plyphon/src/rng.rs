//! A per-unit random number generator - plyphon's port of scsynth's `RGen` (Taus88).
//!
//! scsynth seeds its generators from a process-global source; plyphon instead seeds each generator
//! from a value threaded down through the builder (see [`crate::synthdef::SynthDef::instantiate`]),
//! so there is no global RNG state and two instances of the same synth still decorrelate.

/// A Taus88 combined Tausworthe generator (the algorithm scsynth uses).
///
/// `repr(C)` + `Pod` so it embeds directly in a unit's pool-resident state (e.g. [`WhiteNoise`]).
///
/// [`WhiteNoise`]: crate::unit::WhiteNoise
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Rng {
    s1: u32,
    s2: u32,
    s3: u32,
}

impl Rng {
    /// Seed a generator. The seed is scrambled and the Taus88 constraints (`s1 > 1`, `s2 > 7`,
    /// `s3 > 15`) are enforced so the generator never collapses to a fixed point.
    pub fn new(seed: u64) -> Self {
        let mut x = seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            // xorshift64* to spread the seed bits before splitting into the three states.
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        Rng {
            s1: (next() as u32) | 2,
            s2: (next() as u32) | 8,
            s3: (next() as u32) | 16,
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

    /// A bipolar sample uniformly distributed in `[-1, 1)`.
    #[inline]
    pub fn next_bipolar(&mut self) -> f32 {
        (self.next_u32() as i32) as f32 * (1.0 / 2_147_483_648.0)
    }
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
}
