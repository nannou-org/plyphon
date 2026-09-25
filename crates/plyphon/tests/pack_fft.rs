//! The spectrum <-> value-list bridge: `Unpack1FFT` reads one packed slot per demand pull and
//! `PackFFT` writes a whole magnitude/phase list back into the chain buffer.
//!
//! The chain buffer is pre-filled with a known packed spectrum and the chain signal is supplied
//! directly (a constant buffer number, or a control bus for the hop-gating cases), so a frame is
//! ready on exactly the blocks the test chooses - no `FFT` analysis in the way. Each def ends in a
//! frame counter driving a non-interpolating `BufRd` over the same buffer, so one rendered control
//! block reads the whole packed frame back as it stands after that block's spectral units ran.
//!
//! Requires the default `fft` feature (the spectral units are gated on it).

use plyphon::{
    AddAction, Buffer, BuildError, Controller, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef,
    SynthNewError, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
/// Samples per control block, chosen equal to [`FFT_SIZE`] so one block reads the whole frame.
const BLOCK: usize = 64;
/// Chain-buffer frames: the smallest FFT size plyphon plans for.
const FFT_SIZE: usize = 64;
/// Bins in a packed `FFT_SIZE` frame, which holds `[dc, nyq, bins...]`.
const NUM_BINS: usize = (FFT_SIZE - 2) / 2;
/// The packed slot of the Nyquist term in `Unpack1FFT`'s one-based numbering.
const NYQ_SLOT: i32 = (FFT_SIZE / 2) as i32;
/// The control bus carrying the chain signal in the bus-driven tests.
const CHAIN_BUS: u32 = 0;

/// A constant input.
fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

/// Output 0 of unit `unit`.
fn u(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

/// A packed spectrum with pairwise-distinct, non-zero terms throughout, so a transposed, dropped or
/// swapped slot cannot pass unnoticed. Every value is an exact binary fraction, so a magnitude/phase
/// round trip is the only source of rounding in these tests.
fn test_spectrum() -> Vec<f32> {
    let mut data = vec![0.0f32; FFT_SIZE];
    data[0] = 0.8125;
    data[1] = -0.4375;
    for i in 0..NUM_BINS {
        data[2 + 2 * i] = 0.5 + i as f32 / 32.0;
        data[3 + 2 * i] = -0.25 - i as f32 / 64.0;
    }
    data
}

/// A bin after a magnitude/phase round trip, which is what `Unpack1FFT` -> `PackFFT` reconstructs.
fn round_trip(re: f32, im: f32) -> (f32, f32) {
    let mag = im.hypot(re);
    let phase = im.atan2(re);
    (mag * phase.cos(), mag * phase.sin())
}

/// `Unpack1FFT.dr(chain, FFT_SIZE, binindex, whichmeasure)` reading the chain buffer.
fn unpack(chain: InputRef, binindex: i32, whichmeasure: f32) -> UnitSpec {
    UnitSpec::new(
        "Unpack1FFT",
        Rate::Demand,
        vec![
            chain,
            c(FFT_SIZE as f32),
            c(binindex as f32),
            c(whichmeasure),
        ],
        1,
    )
}

/// `PackFFT.kr(chain, FFT_SIZE, frombin, tobin, zeroothers, numinvals, payload...)`.
fn pack(
    chain: InputRef,
    frombin: i32,
    tobin: i32,
    zeroothers: f32,
    payload: Vec<InputRef>,
) -> UnitSpec {
    let mut inputs = vec![
        chain,
        c(FFT_SIZE as f32),
        c(frombin as f32),
        c(tobin as f32),
        c(zeroothers),
        c(payload.len() as f32),
    ];
    inputs.extend(payload);
    UnitSpec::new("PackFFT", Rate::Control, inputs, 1)
}

/// An `Unpack1FFT` magnitude/phase pair per packed slot in `frombin..=tobin`, followed by the
/// `PackFFT` that writes them back - the shape sclang's `pvcollect` emits.
fn unpack_pack(chain: InputRef, frombin: i32, tobin: i32, zeroothers: f32) -> Vec<UnitSpec> {
    let mut units = Vec::new();
    let mut payload = Vec::new();
    for slot in frombin..=tobin {
        for measure in [0.0f32, 1.0] {
            payload.push(u(units.len() as u32));
            units.push(unpack(chain, slot, measure));
        }
    }
    units.push(pack(chain, frombin, tobin, zeroothers, payload));
    units
}

/// Append the frame read-back to `units`: a counter driving a non-interpolating `BufRd` over the
/// chain buffer into output channel 0, and `extra` into the channels after it.
fn read_back(mut units: Vec<UnitSpec>, extra: Vec<InputRef>) -> Vec<UnitSpec> {
    let phasor = units.len() as u32;
    units.push(UnitSpec::new(
        "Phasor",
        Rate::Audio,
        vec![c(0.0), c(1.0), c(0.0), c(FFT_SIZE as f32), c(0.0)],
        1,
    ));
    units.push(UnitSpec::new(
        "BufRd",
        Rate::Audio,
        vec![c(0.0), u(phasor), c(1.0), c(1.0)],
        1,
    ));
    let mut out = vec![c(0.0), u(phasor + 1)];
    out.extend(extra);
    units.push(UnitSpec::new("Out", Rate::Audio, out, 0));
    units
}

/// An engine whose buffer 0 holds `spectrum`, running `units` as one synth over `channels` outputs.
fn start(units: Vec<UnitSpec>, spectrum: &[f32], channels: usize) -> (Controller, World) {
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: channels,
        ..Options::default()
    });
    controller
        .buffer_set(
            0,
            Box::new(Buffer::from_interleaved(spectrum.to_vec(), 1, SR)),
        )
        .expect("buffer_set");
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    (controller, world)
}

/// Render one control block of `channels`-channel output.
fn block(world: &mut World, channels: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; BLOCK * channels];
    world.fill(&mut buf, channels);
    buf
}

/// Channel `ch` of an interleaved block.
fn channel(buf: &[f32], channels: usize, ch: usize) -> Vec<f32> {
    buf.iter().skip(ch).step_by(channels).copied().collect()
}

/// Assert `got` matches `want` slot by slot, to the tolerance a magnitude/phase round trip needs.
fn assert_frame(got: &[f32], want: &[f32], what: &str) {
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() < 1e-5,
            "{what}: packed slot {i} is {g}, expected {w}"
        );
    }
}

/// A def whose only unit is `unit`, so a rejected build surfaces as that unit's own error.
fn reject(unit: UnitSpec) -> SynthNewError {
    let (mut controller, _nrt, _world) = engine(Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "bad".to_string(),
        params: vec![],
        units: vec![unit],
    });
    controller
        .synth_new("bad", ROOT_GROUP_ID, AddAction::Tail)
        .expect_err("the def should have been rejected")
}

#[test]
fn pack_fft_roundtrip_identity_and_zeroothers() {
    let spectrum = test_spectrum();

    // Unpacking every packed slot and packing it straight back reconstructs the frame: the DC and
    // Nyquist terms exactly (they travel as magnitudes), the bins through a magnitude/phase round
    // trip. `tobin` is the Nyquist slot, so the whole spectrum is covered.
    let mut want = spectrum.clone();
    for i in 0..NUM_BINS {
        let (re, im) = round_trip(spectrum[2 + 2 * i], spectrum[3 + 2 * i]);
        want[2 + 2 * i] = re;
        want[3 + 2 * i] = im;
    }
    let units = read_back(unpack_pack(c(0.0), 0, NYQ_SLOT, 0.0), vec![]);
    let (_c, mut world) = start(units, &spectrum, 1);
    let got = block(&mut world, 1);
    assert_frame(&got, &want, "full-range round trip");
    assert_eq!(got[0], spectrum[0], "the DC term travels exactly");
    assert_eq!(got[1], spectrum[1], "the Nyquist term travels exactly");
    assert!(
        got[2..].iter().all(|s| s.abs() > 1e-3),
        "the reconstructed bins must all be non-zero for this to be discriminating"
    );

    // Packing slots 2..=4 covers bins 1..=3: `frombin` is one-based over the bins, so the payload's
    // first bin is `frombin - 1`. With `zeroothers` the DC and Nyquist terms and every bin outside
    // that window are zeroed, and bin 1 - the boundary a `frombin`-based window would have dropped -
    // survives.
    let mut want = vec![0.0f32; FFT_SIZE];
    for i in 1..=3 {
        let (re, im) = round_trip(spectrum[2 + 2 * i], spectrum[3 + 2 * i]);
        want[2 + 2 * i] = re;
        want[3 + 2 * i] = im;
    }
    let units = read_back(unpack_pack(c(0.0), 2, 4, 1.0), vec![]);
    let (_c, mut world) = start(units, &spectrum, 1);
    let got = block(&mut world, 1);
    assert_frame(&got, &want, "zeroothers window");
    assert!(
        want[4].abs() > 1e-3 && spectrum[2].abs() > 1e-3,
        "bin 1 must survive while bin 0, which was non-zero, is wiped - otherwise the boundary at \
         `frombin - 1` would not be discriminating"
    );

    // The same window without `zeroothers` leaves everything it does not cover exactly as it was.
    let mut want = spectrum.clone();
    for i in 1..=3 {
        let (re, im) = round_trip(spectrum[2 + 2 * i], spectrum[3 + 2 * i]);
        want[2 + 2 * i] = re;
        want[3 + 2 * i] = im;
    }
    let units = read_back(unpack_pack(c(0.0), 2, 4, 0.0), vec![]);
    let (_c, mut world) = start(units, &spectrum, 1);
    let got = block(&mut world, 1);
    assert_frame(&got, &want, "window without zeroothers");
}

#[test]
fn pack_fft_runs_a_payload_that_disagrees_with_its_range() {
    // scsynth never checks `numinvals`, `frombin` or `tobin` against the payload; it reads past its
    // inputs or its spectrum when they lie. Here each such def builds and runs, a read past the
    // inputs or a write past the spectrum is skipped, and the slots the payload does address are
    // still written.
    let spectrum = test_spectrum();
    let full = 2 * (NYQ_SLOT as usize + 1);
    let cases = [
        // `numinvals` claims two more values than the payload holds, so the Nyquist magnitude's
        // input index lands just past the last input.
        (0, NYQ_SLOT, full, full + 2),
        // The range needs six magnitude/phase pairs but the payload holds three.
        (0, 5, 6, 6),
        // The payload is longer than the range covers.
        (0, 1, 6, 6),
        // An inverted range writes no bin.
        (5, 2, 0, 0),
    ];
    for (frombin, tobin, len, numinvals) in cases {
        let mut unit = pack(c(0.0), frombin, tobin, 0.0, vec![c(0.5); len]);
        unit.inputs[5] = c(numinvals as f32);
        let (_c, mut world) = start(read_back(vec![unit], vec![]), &spectrum, 1);
        let got = block(&mut world, 1);
        assert!(
            got.iter().all(|s| s.is_finite()),
            "frombin {frombin}, tobin {tobin}: the frame stays finite"
        );
        if frombin == 0 {
            assert_eq!(
                got[0], 0.5,
                "frombin {frombin}, tobin {tobin}: DC is packed"
            );
        }
    }
}

#[test]
fn pack_fft_reads_its_range_when_the_synth_starts() {
    // The reference's constructor reads `frombin`, `tobin`, `zeroothers` and `numinvals` from its
    // inputs, so a wired value works exactly like the constant.
    let spectrum = test_spectrum();
    let constant = {
        let units = read_back(unpack_pack(c(0.0), 2, 4, 1.0), vec![]);
        let (_c, mut world) = start(units, &spectrum, 1);
        block(&mut world, 1)
    };
    // A `DC.kr(2)` first, so it has run when `PackFFT` reads it; every other unit shifts up by one.
    let mut units = vec![UnitSpec::new("DC", Rate::Control, vec![c(2.0)], 1)];
    for mut unit in unpack_pack(c(0.0), 2, 4, 1.0) {
        for input in &mut unit.inputs {
            if let InputRef::Unit { unit, .. } = input {
                *unit += 1;
            }
        }
        units.push(unit);
    }
    units.last_mut().expect("the PackFFT").inputs[2] = u(0);
    let (_c, mut world) = start(read_back(units, vec![]), &spectrum, 1);
    assert_frame(&block(&mut world, 1), &constant, "a wired frombin");
}

#[test]
fn pack_fft_hop_gating_pulls_nothing_off_hop() {
    // A `Dseq` supplies the DC magnitude. Its *phase* companion (payload slot 1) is wired to the
    // same sequence, so the sequence advances by one per hop only if `PackFFT` reads exactly the
    // slots the reference reads: the DC phase is never one of them.
    let items: Vec<f32> = (1..=8).map(|k| k as f32).collect();
    let mut seq_inputs = vec![c(f32::INFINITY)];
    seq_inputs.extend(items.iter().map(|&v| c(v)));
    let units = vec![
        // 0: the chain signal, driven from a control bus so a frame is ready only when the test says.
        UnitSpec::new("In", Rate::Control, vec![c(CHAIN_BUS as f32)], 1),
        // 1: the payload sequence.
        UnitSpec::new("Dseq", Rate::Demand, seq_inputs, 1),
        // 2: pack only the DC term.
        pack(u(0), 0, 0, 0.0, vec![u(1), u(1)]),
        // 3-4: read the packed frame back into channel 0.
        UnitSpec::new(
            "Phasor",
            Rate::Audio,
            vec![c(0.0), c(1.0), c(0.0), c(FFT_SIZE as f32), c(0.0)],
            1,
        ),
        UnitSpec::new("BufRd", Rate::Audio, vec![c(0.0), u(3), c(1.0), c(1.0)], 1),
        // 5: the chain index `PackFFT` passes on, into channel 1. Adding zero at audio rate carries
        // a control value into a block exactly, where `K2A` would interpolate across the step.
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![u(2), c(0.0)],
            num_outputs: 1,
            special_index: 0,
        },
        UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(4), u(5)], 0),
    ];

    let (mut controller, mut world) = start(units, &vec![0.0f32; FFT_SIZE], 2);

    // Frame ready: one pull, so the DC term takes the sequence's first item.
    controller.set_control_bus(CHAIN_BUS, 0.0).expect("set bus");
    let out = block(&mut world, 2);
    assert_eq!(channel(&out, 2, 0)[0], items[0], "first frame packs item 0");
    assert_eq!(
        channel(&out, 2, 1)[0],
        0.0,
        "a ready frame passes its chain index on"
    );

    // Two blocks between frames: nothing is pulled and the packed DC term is left alone.
    controller
        .set_control_bus(CHAIN_BUS, -1.0)
        .expect("set bus");
    for k in 0..2 {
        let out = block(&mut world, 2);
        assert_eq!(
            channel(&out, 2, 0)[0],
            items[0],
            "off-hop block {k} must not repack"
        );
        assert_eq!(
            channel(&out, 2, 1)[0],
            -1.0,
            "off-hop block {k} emits -1 on the chain"
        );
    }

    // The next frame resumes at the sequence's *second* item: the off-hop blocks advanced nothing,
    // and the unread phase companion advanced nothing either.
    controller.set_control_bus(CHAIN_BUS, 0.0).expect("set bus");
    let out = block(&mut world, 2);
    assert_eq!(
        channel(&out, 2, 0)[0],
        items[1],
        "the second frame packs item 1"
    );
}

#[test]
fn unpack1fft_reads_bins_dc_nyq_and_caches_within_block() {
    let spectrum = test_spectrum();
    let mags: Vec<f32> = (0..NUM_BINS)
        .map(|i| spectrum[3 + 2 * i].hypot(spectrum[2 + 2 * i]))
        .collect();

    // `binindex` is one-based over the bins: 0 is the DC term, `FFT_SIZE / 2` the Nyquist term, and
    // every value between reads packed bin `binindex - 1`. Reading the whole range back through a
    // `PackFFT` that writes each slot's magnitude into the bin below it would transpose the frame,
    // so the magnitudes are read one at a time instead, each into its own output.
    for (binindex, want) in [
        (0i32, spectrum[0]),
        (1, mags[0]),
        (2, mags[1]),
        (NUM_BINS as i32, mags[NUM_BINS - 1]),
        (NYQ_SLOT, spectrum[1]),
    ] {
        let got = demanded(&spectrum, unpack(c(0.0), binindex, 0.0), None);
        assert!(
            (got - want).abs() < 1e-6,
            "binindex {binindex} magnitude is {got}, expected {want}"
        );
    }

    // Phases follow the same routing, except that the DC and Nyquist terms are purely real and read
    // as a constant zero.
    for (binindex, want) in [
        (0i32, 0.0f32),
        (1, spectrum[3].atan2(spectrum[2])),
        (NYQ_SLOT, 0.0),
    ] {
        let got = demanded(&spectrum, unpack(c(0.0), binindex, 1.0), None);
        assert!(
            (got - want).abs() < 1e-6,
            "binindex {binindex} phase is {got}, expected {want}"
        );
    }

    // A polar predecessor: `PV_MagAbove` at threshold 0 changes nothing but leaves the frame in
    // polar form, so a reader that took the stored pair for a Cartesian one would report the phase
    // as a magnitude. `Unpack1FFT` converts first and still reports bin 0's magnitude.
    let polar = UnitSpec::new("PV_MagAbove", Rate::Control, vec![c(0.0), c(0.0)], 1);
    let got = demanded(&spectrum, unpack(u(0), 1, 0.0), Some(polar));
    assert!(
        (got - mags[0]).abs() < 1e-5,
        "after a polar predecessor bin 0's magnitude is {got}, expected {}",
        mags[0]
    );

    // Two consumers pulling the same source in one block, with the frame rewritten between them:
    // the second pull re-emits the first pull's value, because the source computes once per block.
    let source = unpack(c(0.0), 1, 0.0);
    let units = vec![
        source,
        UnitSpec::new("Demand", Rate::Audio, vec![c(1.0), c(0.0), u(0)], 1),
        // Squaring every magnitude changes what a second read of the frame would see.
        UnitSpec::new("PV_MagSquared", Rate::Control, vec![c(0.0)], 1),
        UnitSpec::new("Demand", Rate::Audio, vec![c(1.0), c(0.0), u(0)], 1),
        UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1), u(3)], 0),
    ];
    let (_c, mut world) = start(units, &spectrum, 2);
    let out = block(&mut world, 2);
    let (first, second) = (channel(&out, 2, 0)[0], channel(&out, 2, 1)[0]);
    assert!(
        (first - mags[0]).abs() < 1e-6,
        "the first pull reads bin 0's magnitude, got {first}"
    );
    assert_eq!(
        second, first,
        "the second pull in the same block must re-emit the cached value"
    );
    assert!(
        (mags[0] * mags[0] - mags[0]).abs() > 1e-3,
        "squaring must change the magnitude for the cache check to discriminate"
    );
}

/// The value `source` yields on its first pull, held on output 0 for the whole block. `predecessor`,
/// if given, runs before the pull and becomes unit 0, so the source can be chained behind it.
fn demanded(spectrum: &[f32], source: UnitSpec, predecessor: Option<UnitSpec>) -> f32 {
    let mut units = Vec::new();
    if let Some(unit) = predecessor {
        units.push(unit);
    }
    let source_index = units.len() as u32;
    units.push(source);
    units.push(UnitSpec::new(
        "Demand",
        Rate::Audio,
        vec![c(1.0), c(0.0), u(source_index)],
        1,
    ));
    let demand = units.len() as u32 - 1;
    units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![c(0.0), u(demand)],
        0,
    ));
    let (_c, mut world) = start(units, spectrum, 1);
    block(&mut world, 1)[0]
}

#[test]
fn unpack1fft_reset_matches_produce_and_no_exhaustion() {
    let spectrum = test_spectrum();
    let mag = spectrum[3].hypot(spectrum[2]);

    // A reset takes the same path as a produce: the reference's calc functions ignore the reset
    // flag. The first consumer only ever resets the source, then the frame is rewritten, then the
    // second consumer produces from it - and gets the value the *reset* computed, so the reset both
    // read the frame and recorded the block.
    let units = vec![
        unpack(c(0.0), 1, 0.0),
        UnitSpec::new("Demand", Rate::Audio, vec![c(0.0), c(1.0), u(0)], 1),
        UnitSpec::new("PV_MagSquared", Rate::Control, vec![c(0.0)], 1),
        UnitSpec::new("Demand", Rate::Audio, vec![c(1.0), c(0.0), u(0)], 1),
        UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(3)], 0),
    ];
    let (_c, mut world) = start(units, &spectrum, 1);
    let after_reset = block(&mut world, 1)[0];
    assert!(
        (after_reset - mag).abs() < 1e-6,
        "a reset must compute and record the block: got {after_reset}, expected {mag}"
    );

    // The source never runs out: pulled on every block for many blocks, it keeps yielding the
    // frame's magnitude rather than the exhaustion signal a sequence source would emit (which the
    // `Demand` consumer would show by freezing at its initial 0).
    let units = vec![
        unpack(c(0.0), 1, 0.0),
        UnitSpec::new("In", Rate::Control, vec![c(CHAIN_BUS as f32)], 1),
        UnitSpec::new("Demand", Rate::Audio, vec![u(1), c(0.0), u(0)], 1),
        UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(2)], 0),
    ];
    let (mut controller, mut world) = start(units, &spectrum, 1);
    for k in 0..16 {
        // Toggling the trigger gives one rising edge - one pull - every other block.
        controller
            .set_control_bus(CHAIN_BUS, ((k + 1) % 2) as f32)
            .expect("set bus");
        let got = block(&mut world, 1)[0];
        assert!(
            (got - mag).abs() < 1e-6,
            "pull {k} yielded {got}, expected {mag}"
        );
    }
}

/// Every unit this family adds rejects a short input list at build time, through the shared
/// `WrongInputCount` error, rather than reading past its inputs on the audio thread.
#[test]
fn new_units_reject_wrong_input_counts() {
    let short = [
        UnitSpec::new("PackFFT", Rate::Control, vec![c(0.0); 5], 1),
        UnitSpec::new("Unpack1FFT", Rate::Demand, vec![c(0.0); 3], 1),
        UnitSpec::new("PV_BinShift", Rate::Control, vec![c(0.0); 3], 1),
        UnitSpec::new("PV_MagSmear", Rate::Control, vec![c(0.0); 1], 1),
        UnitSpec::new("PV_RectComb", Rate::Control, vec![c(0.0); 3], 1),
    ];
    for unit in short {
        let name = unit.name.clone();
        let rate = unit.rate;
        assert!(
            matches!(
                reject(unit),
                SynthNewError::Build(BuildError::WrongInputCount)
            ),
            "{name} at {rate:?} should reject a short input list"
        );
    }
}
