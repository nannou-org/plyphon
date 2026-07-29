//! Source-parity checks for the public random generator.

use plyphon_dsp::{math, rng::Rng};

#[test]
fn reseed_matches_scsynth_rgen_init() {
    let mut rng = Rng::new(0);
    rng.reseed(42);
    assert_eq!(
        core::array::from_fn::<_, 8, _>(|_| rng.next_u32()),
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
fn exprand_uses_scsynth_double_precision_path() {
    let mut rng = Rng::new(0);
    rng.reseed(42);
    let draw = f64::from_bits(0x4130_0000_65b7_bbf7) - 1_048_576.0;
    let expected = 100.0 * math::exp(math::ln(2.0) * draw);
    assert_eq!(
        rng.next_exprand_f64(100.0, 200.0).to_bits(),
        expected.to_bits()
    );
}
