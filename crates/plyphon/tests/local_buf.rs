//! Graph-owned buffers (`LocalBuf`, `MaxLocalBufs`, `SetBuf`, `ClearBuf`): the encoded buffer
//! number, transparent resolution through every buffer consumer (`BufWr`/`BufRd`, `SetBuf`,
//! `ClearBuf`, `BufFrames`/`BufChannels`, the FFT chain), per-instance isolation of the storage,
//! and world buffers continuing to resolve below the table capacity.

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
/// Samples per control block at the default engine options.
const BLOCK: usize = 64;
/// Buffer-table capacity for these tests; local buffer numbers start here.
const MAX_BUFFERS: usize = 32;

fn opts() -> Options {
    Options {
        sample_rate: SR,
        output_channels: 1,
        max_buffers: MAX_BUFFERS,
        ..Options::default()
    }
}

/// `Out.ar(0, Unit{src})`.
fn out(src: u32) -> UnitSpec {
    UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![
            InputRef::Constant(0.0),
            InputRef::Unit {
                unit: src,
                output: 0,
            },
        ],
        0,
    )
}

/// Output 0 of unit `unit`.
fn u(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

/// A constant input.
fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

/// `LocalBuf(numChannels, numFrames)` - scalar, constant shape.
fn local_buf(channels: f32, frames: f32) -> UnitSpec {
    UnitSpec::new("LocalBuf", Rate::Scalar, vec![c(channels), c(frames)], 1)
}

/// `Phasor.ar(0, 1, 0, end, 0)`: the frame counter 0, 1, ..., end-1, wrapping.
fn frame_phasor(end: f32) -> UnitSpec {
    UnitSpec::new(
        "Phasor",
        Rate::Audio,
        vec![c(0.0), c(1.0), c(0.0), c(end), c(0.0)],
        1,
    )
}

/// `BufRd.ar(1, bufnum, phase, loop: 1, interpolation: 1)` reading buffer `buf` at unit `phase`.
fn buf_rd(buf: InputRef, phase: InputRef) -> UnitSpec {
    UnitSpec::new("BufRd", Rate::Audio, vec![buf, phase, c(1.0), c(1.0)], 1)
}

/// Build a world playing `units` as the def `t`, and render `blocks` control blocks.
fn render(units: Vec<UnitSpec>, blocks: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut buf = vec![0.0f32; BLOCK * blocks];
    world.fill(&mut buf, 1);
    buf
}

/// The first output sample of the world's next block.
fn first_sample(world: &mut World) -> f32 {
    let mut buf = vec![0.0f32; BLOCK];
    world.fill(&mut buf, 1);
    buf[0]
}

#[test]
fn local_buf_outputs_capacity_plus_declaration_index() {
    // Two LocalBufs in one def: the first's number is the table capacity, the second's is one past
    // it (scsynth's `world->mNumSndBufs + i`, assigned in unit order).
    for (which, expected) in [(0u32, MAX_BUFFERS as f32), (1, MAX_BUFFERS as f32 + 1.0)] {
        let buf = render(
            vec![
                local_buf(1.0, 4.0),
                local_buf(2.0, 8.0),
                UnitSpec::new("K2A", Rate::Audio, vec![u(which)], 1),
                out(2),
            ],
            2,
        );
        assert!(
            buf.iter().all(|&s| s == expected),
            "LocalBuf {which} outputs {expected}, got {}",
            buf[0]
        );
    }
}

#[test]
fn buf_wr_writes_and_buf_rd_reads_a_local_buffer() {
    // One shared frame counter drives BufWr (writing the counter itself: buffer[i] = i) and, later
    // in the unit list, BufRd - so the read-back of every frame equals the ramp 0..64 in the very
    // first block, and again after the phasor wraps.
    let buf = render(
        vec![
            // 0: the graph-local buffer, 1 channel x 64 frames.
            local_buf(1.0, 64.0),
            // 1: frame counter 0..64.
            frame_phasor(64.0),
            // 2: BufWr.ar([phasor], bufnum: local, phase: phasor, loop: 1).
            UnitSpec::new("BufWr", Rate::Audio, vec![u(0), u(1), c(1.0), u(1)], 1),
            // 3: BufRd at the same phase, after the write.
            buf_rd(u(0), u(1)),
            out(3),
        ],
        2,
    );
    for (i, &s) in buf.iter().enumerate() {
        let expected = (i % 64) as f32;
        assert_eq!(
            s, expected,
            "sample {i}: ramp read back from the local buffer"
        );
    }
}

#[test]
fn set_buf_values_read_back_at_the_offset() {
    // SetBuf(local, offset: 2, numValues: 3, 5, 6, 7) into an 8-frame LocalBuf; BufRd cycles the 8
    // frames, so the output repeats [0, 0, 5, 6, 7, 0, 0, 0] (the storage is zeroed at spawn).
    let buf = render(
        vec![
            local_buf(1.0, 8.0),
            UnitSpec::new(
                "SetBuf",
                Rate::Scalar,
                vec![u(0), c(2.0), c(3.0), c(5.0), c(6.0), c(7.0)],
                1,
            ),
            frame_phasor(8.0),
            buf_rd(u(0), u(2)),
            out(3),
        ],
        1,
    );
    let expected = [0.0, 0.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0];
    for (i, &s) in buf.iter().enumerate() {
        assert_eq!(s, expected[i % 8], "sample {i}");
    }
}

#[test]
fn clear_buf_after_set_buf_yields_zeros() {
    // SetBuf earlier in the unit list, ClearBuf later: on the first block the writes land in unit
    // order (as scsynth's ctors run in unit order), so the clear wins and the read-back is silent.
    // BufRd takes its buffer number from the LocalBuf itself, as sclang's `.set`/`.clear` chaining
    // does (SetBuf/ClearBuf output a constant 0, like scsynth).
    let buf = render(
        vec![
            local_buf(1.0, 8.0),
            UnitSpec::new(
                "SetBuf",
                Rate::Scalar,
                vec![u(0), c(0.0), c(3.0), c(5.0), c(6.0), c(7.0)],
                1,
            ),
            UnitSpec::new("ClearBuf", Rate::Scalar, vec![u(0)], 1),
            frame_phasor(8.0),
            buf_rd(u(0), u(3)),
            out(4),
        ],
        2,
    );
    assert!(
        buf.iter().all(|&s| s == 0.0),
        "cleared local buffer reads silent, got {buf:?}"
    );
}

#[test]
fn max_local_bufs_is_a_declaration_no_op() {
    // sclang emits MaxLocalBufs(count) automatically; plyphon sizes the storage from the LocalBuf
    // units themselves, so the unit only consumes its input and outputs 0 - and the LocalBufs it
    // "declares" work regardless of the count it names.
    let buf = render(
        vec![
            UnitSpec::new("MaxLocalBufs", Rate::Scalar, vec![c(1.0)], 1),
            local_buf(1.0, 8.0),
            UnitSpec::new(
                "SetBuf",
                Rate::Scalar,
                vec![u(1), c(0.0), c(1.0), c(0.25)],
                1,
            ),
            frame_phasor(8.0),
            buf_rd(u(1), u(3)),
            out(4),
        ],
        1,
    );
    assert_eq!(buf[0], 0.25, "the declared LocalBuf still resolves");
    assert_eq!(buf[1], 0.0, "the rest of the buffer is zeroed storage");
}

#[test]
fn two_instances_of_one_def_do_not_share_local_buffers() {
    // Each instance records one instance-distinct Rand draw into frame 0 of its LocalBuf on its
    // first block (RecordBuf, non-looping over a 1-frame buffer, stops after that), then reads it
    // back every block. If the two instances shared storage, the second's write would clobber the
    // first's, and from the next block both would read the second draw (sum = 2 * v2); with
    // per-instance storage the sum stays v1 + v2.
    let units = vec![
        // 0: the 1-frame local buffer.
        local_buf(1.0, 1.0),
        // 1: one uniform draw in [0, 1), distinct per instance (per-instance RNG seeds).
        UnitSpec::new("Rand", Rate::Scalar, vec![c(0.0), c(1.0)], 1),
        // 2: the draw as an audio signal.
        UnitSpec::new("K2A", Rate::Audio, vec![u(1)], 1),
        // 3: RecordBuf.ar([draw], bufnum: local, offset 0, rec 1, pre 0, run 1, loop 0, trig 0,
        //    done 0) - writes frame 0 on the instance's first block, then holds.
        UnitSpec::new(
            "RecordBuf",
            Rate::Audio,
            vec![
                u(0),
                c(0.0),
                c(1.0),
                c(0.0),
                c(1.0),
                c(0.0),
                c(0.0),
                c(0.0),
                u(2),
            ],
            1,
        ),
        // 4: read frame 0 back every block.
        buf_rd(u(0), c(0.0)),
        out(4),
    ];
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("first synth");
    let v1 = first_sample(&mut world);
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("second synth");
    let sum_a = first_sample(&mut world);
    let sum_b = first_sample(&mut world);
    let v2 = sum_a - v1;
    assert!(
        (v2 - v1).abs() > 1e-7,
        "instance draws decorrelate: {v1} then {v2}"
    );
    // Shared storage would read 2 * v2 from the second block on; per-instance storage holds v1 + v2.
    assert_eq!(sum_b, sum_a, "both instances keep reading their own draw");
    assert!(
        (sum_b - 2.0 * v2).abs() > 1e-7,
        "the second write did not clobber the first ({sum_b} vs 2 * {v2})"
    );
}

#[test]
fn buf_info_reports_the_local_shape() {
    // BufFrames/BufChannels/BufSamples resolve the local buffer's compiled shape; BufSampleRate
    // reports the graph's audio rate (scsynth sets a local buffer's samplerate to FULLRATE).
    for (unit, expected) in [
        ("BufFrames", 17.0f32),
        ("BufChannels", 2.0),
        ("BufSamples", 34.0),
        ("BufSampleRate", SR as f32),
    ] {
        let buf = render(
            vec![
                local_buf(2.0, 17.0),
                UnitSpec::new(unit, Rate::Control, vec![u(0)], 1),
                UnitSpec::new("K2A", Rate::Audio, vec![u(1)], 1),
                out(2),
            ],
            1,
        );
        assert_eq!(buf[0], expected, "{unit} on a LocalBuf(2, 17)");
    }
}

#[test]
fn world_buffers_below_capacity_still_resolve() {
    // A table-installed buffer keeps resolving by its (below-capacity) number, and SetBuf writes it
    // through the same seam: buffer 0 holds [1, 2, 3, 4], SetBuf overwrites the middle two.
    let (mut controller, _nrt, mut world) = engine(opts());
    controller
        .buffer_set(
            0,
            Box::new(Buffer::from_interleaved(vec![1.0, 2.0, 3.0, 4.0], 1, SR)),
        )
        .expect("buffer_set");
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SetBuf",
                Rate::Scalar,
                vec![c(0.0), c(1.0), c(2.0), c(9.0), c(8.0)],
                1,
            ),
            frame_phasor(4.0),
            buf_rd(c(0.0), u(1)),
            out(2),
        ],
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut buf = vec![0.0f32; BLOCK];
    world.fill(&mut buf, 1);
    let expected = [1.0, 9.0, 8.0, 4.0];
    for (i, &s) in buf.iter().enumerate() {
        assert_eq!(s, expected[i % 4], "sample {i}");
    }
}

/// Goertzel magnitude of `freq` in `samples` - a single-bin DTFT for cheap pitch checks.
fn goertzel(samples: &[f32], freq: f32) -> f32 {
    let n = samples.len();
    let k = (0.5 + n as f32 * freq / SR as f32).floor();
    let w = 2.0 * std::f32::consts::PI * k / n as f32;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f32, 0.0f32);
    for &x in samples {
        let s = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0).sqrt() / n as f32
}

#[test]
fn fft_ifft_reconstructs_a_sine_over_a_local_buf() {
    // The idiomatic sclang chain `IFFT(FFT(LocalBuf(n), in))`: the packed-spectrum chain buffer is
    // a LocalBuf instead of a `/b_alloc`'d table slot, and the resynthesis still reconstructs the
    // (bin-aligned) tone. Mirrors the world-buffer chain test in `fft.rs`.
    const FFT_SIZE: usize = 1024;
    let bin = SR as f32 / FFT_SIZE as f32;
    let freq = 20.0 * bin;
    let units = vec![
        // 0: the chain buffer, 1 channel x FFT_SIZE frames, graph-local.
        local_buf(1.0, FFT_SIZE as f32),
        // 1: SinOsc.ar(freq).
        UnitSpec::new("SinOsc", Rate::Audio, vec![c(freq), c(0.0)], 1),
        // 2: SinOsc * 0.5.
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![u(1), c(0.5)],
            num_outputs: 1,
            special_index: 2, // multiply
        },
        // 3: FFT(local chain buffer, in, hop 0.5, wintype 0, active 1, winsize FFT_SIZE).
        UnitSpec::new(
            "FFT",
            Rate::Control,
            vec![u(0), u(2), c(0.5), c(0.0), c(1.0), c(FFT_SIZE as f32)],
            1,
        ),
        // 4: IFFT(fbufnum, wintype 0, winsize FFT_SIZE).
        UnitSpec::new(
            "IFFT",
            Rate::Audio,
            vec![u(3), c(0.0), c(FFT_SIZE as f32)],
            1,
        ),
        out(4),
    ];
    let output = render(units, 12_288 / BLOCK);
    let tail = &output[8_192..];
    assert!(
        tail.iter().any(|s| s.abs() > 0.05),
        "resynthesis was silent"
    );
    assert!(
        tail.iter().all(|s| s.abs() <= 1.0),
        "resynthesis left [-1, 1]"
    );
    let at = goertzel(tail, freq);
    let off = goertzel(tail, freq * 2.0);
    assert!(
        at > 8.0 * off,
        "expected {freq} Hz to dominate (at={at:.4}, off={off:.4})"
    );
    assert!(
        (0.1..1.0).contains(&at),
        "reconstruction amplitude {at:.4} unreasonable for input amp 0.5"
    );
}

#[test]
fn set_buf_and_clear_buf_output_zero() {
    // scsynth writes `OUT0(0) = 0.f` for both units; consumers take the buffer number from the
    // LocalBuf itself.
    let writer = |name: &str, inputs: Vec<InputRef>| {
        let buf = render(
            vec![
                local_buf(1.0, 8.0),
                UnitSpec::new(name, Rate::Scalar, inputs, 1),
                UnitSpec::new("K2A", Rate::Audio, vec![u(1)], 1),
                out(2),
            ],
            1,
        );
        assert!(
            buf.iter().all(|&s| s == 0.0),
            "{name} must output a constant 0, got {buf:?}"
        );
    };
    writer("ClearBuf", vec![u(0)]);
    writer("SetBuf", vec![u(0), c(0.0), c(1.0), c(0.5)]);
}
