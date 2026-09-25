//! Seeded sequences pinned to scsynth's own `RGen` (`SC_RGen.h`), so the generator seeds and draws
//! bit for bit as the server does.

use plyphon_dsp::rng::{Rng, hash};

/// A generator seeded with `seed` the way scsynth's `RGen::init` does.
fn seeded(seed: u32) -> Rng {
    Rng::new(seed)
}

#[test]
fn init_matches_rgen_init() {
    let mut rng = seeded(42);
    let words: [u32; 8] = core::array::from_fn(|_| rng.next_u32());
    assert_eq!(
        words,
        [
            0x65b7_bbf7,
            0x98e8_b64e,
            0x2ee2_df4a,
            0xcd75_94aa,
            0x9b76_6d62,
            0xb869_04b6,
            0xafbb_1c82,
            0xf4c1_440e,
        ]
    );
}

#[test]
fn init_hashes_the_seed() {
    // `Hash(42)` XORed into the three state constants gives scsynth's seeded state.
    assert_eq!(hash(42) as u32 ^ 1_243_598_713, 0x9e66_4278);
    assert_eq!(hash(42) as u32 ^ 3_093_459_404, 0x6c1b_fccd);
    assert_eq!(hash(42) as u32 ^ 1_821_928_721, 0xb8e1_e010);
}

#[test]
fn drand_uses_the_whole_word() {
    let mut rng = seeded(42);
    let draws: [u64; 4] = core::array::from_fn(|_| rng.next_unipolar_f64().to_bits());
    assert_eq!(
        draws,
        [
            0x3fd9_6dee_fdc0_0000,
            0x3fe3_1d16_c9c0_0000,
            0x3fc7_716f_a500_0000,
            0x3fe9_aeb2_9540_0000,
        ]
    );
}

#[test]
fn irand_matches_rgen_irand() {
    let mut rng = seeded(42);
    let draws: [i32; 8] = core::array::from_fn(|_| rng.next_irand(10));
    assert_eq!(draws, [3, 5, 1, 8, 6, 7, 6, 9]);

    let mut rng = seeded(42);
    let draws: [i32; 8] = core::array::from_fn(|_| rng.next_irand(-3));
    assert_eq!(draws, [-2, -2, -1, -3, -2, -3, -3, -3]);
}

#[test]
fn irand_of_zero_still_consumes_a_word() {
    let mut rng = seeded(42);
    for _ in 0..4 {
        assert_eq!(rng.next_irand(0), 0);
    }
    assert_eq!(rng.next_u32(), 0x9b76_6d62);
}

#[test]
fn irand2_matches_rgen_irand2() {
    let mut rng = seeded(42);
    let draws: [i32; 8] = core::array::from_fn(|_| rng.next_irand2(5));
    assert_eq!(draws, [-1, 1, -3, 3, 1, 2, 2, 5]);
}

#[test]
fn exprand_matches_rgen_exprandrng() {
    let mut rng = seeded(42);
    let draws: [u64; 4] = core::array::from_fn(|_| rng.next_exprand(100.0, 200.0).to_bits());
    assert_eq!(
        draws,
        [
            0x4060_76a1_d428_d199,
            0x4062_e93a_2ba9_25f3,
            0x405c_624d_25d3_a850,
            0x4065_cd7a_74fc_cfd2,
        ]
    );
}

#[test]
fn linrand_bilinrand_sum3rand_match_rgen() {
    let draws = |f: fn(&mut Rng) -> f32| {
        let mut rng = seeded(42);
        core::array::from_fn::<u32, 4, _>(|_| f(&mut rng).to_bits())
    };
    assert_eq!(
        draws(Rng::next_linrand),
        [0x3ecb_6f74, 0x3e3b_8b78, 0x3f1b_766c, 0x3f2f_bb1c]
    );
    assert_eq!(
        draws(Rng::next_bilinrand),
        [0xbe4c_c3f0, 0xbf1e_92b6, 0xbde7_94c0, 0xbe8a_0c50]
    );
    assert_eq!(
        draws(Rng::next_sum3rand),
        [0xbe5b_f730, 0x3ed7_1c05, 0x3e88_755d, 0x3ddc_3620]
    );
}
