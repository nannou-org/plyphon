//! Offline (non-real-time) score rendering - plyphon's port of `scsynth -N`.
//!
//! Reads a binary OSC *score* (a sequence of `[i32 len][time-tagged OSC bundle]` records, the same
//! command file scsynth's NRT mode consumes), drives the engine faster than real time on a
//! deterministic clock via [`plyphon::Render`], and writes the rendered audio to a WAV file. An
//! optional input WAV is fed into the input buses for `In.ar`. Each note onsets on its exact sample
//! via the scheduler and `OffsetOut`, just as in the real-time `schedule` example - but with no
//! audio device and bit-identical output across runs.
//!
//! ```console
//! # write a demo score (a pentatonic phrase of `blip`s), then render it to a WAV:
//! cargo run -p example-render -- gen phrase.osc
//! cargo run -p example-render -- render phrase.osc _ phrase.wav 48000
//! # re-render and the bytes are identical (determinism):
//! cargo run -p example-render -- render phrase.osc _ phrase2.wav 48000
//! ```
//!
//! The score may also carry its own SynthDefs via `/d_recv`; this example additionally registers a
//! built-in `blip` def so the demo score plays without one.

use std::io::Cursor;
use std::time::Duration;

use hound::{SampleFormat, WavSpec, WavWriter};
use plyphon::{
    InputRef, Options, Param, ROOT_GROUP_ID, Rate, Render, RenderUntil, SynthDef, UnitSpec, engine,
};
use plyphon_osc::{OscDispatcher, parse_score, render_osc_score};
use rosc::{OscBundle, OscMessage, OscPacket, OscTime, OscType};

/// OSC/NTP fixed-point units in one second (OSC time is 32.32 fixed point).
const OSC_UNITS_PER_SEC: f64 = 4_294_967_296.0;
/// A pentatonic scale (Hz) the demo phrase walks through.
const SCALE: [f32; 5] = [261.63, 311.13, 349.23, 392.00, 466.16];
/// Seconds between demo beats.
const BEAT_SECS: f64 = 0.16;
/// Number of beats in the demo phrase.
const NUM_BEATS: usize = 24;
/// Peak amplitude of each demo blip.
const AMP: f32 = 0.3;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("gen") => gen_score(&args[2..]),
        Some("render") => render(&args[2..]),
        _ => {
            usage();
            std::process::exit(2);
        }
    };
    if let Err(err) = result {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn usage() {
    eprintln!(
        "usage:\n  \
         example-render render <score> <in.wav|_> <out.wav> <sample-rate> [out-channels] [tail-secs] [sample-format]\n  \
         example-render gen <out-score>\n\
         \n  \
         sample-format: f32 (default) | i16 | i24"
    );
}

// -- the `render` subcommand --------------------------------------------------------------------

fn render(args: &[String]) -> Result<(), String> {
    let score_path = args.first().ok_or("missing <score>")?;
    let in_path = args.get(1).ok_or("missing <in.wav|_>")?;
    let out_path = args.get(2).ok_or("missing <out.wav>")?;
    let sample_rate: f64 = args
        .get(3)
        .ok_or("missing <sample-rate>")?
        .parse()
        .map_err(|_| "bad <sample-rate>")?;
    let out_channels: usize = match args.get(4) {
        Some(s) => s.parse().map_err(|_| "bad [out-channels]")?,
        None => 1,
    };
    let tail_secs: f64 = match args.get(5) {
        Some(s) => s.parse().map_err(|_| "bad [tail-secs]")?,
        None => 0.5,
    };
    let format: SampleFmt = match args.get(6) {
        Some(s) => s.parse()?,
        None => SampleFmt::Float,
    };

    let score_bytes = std::fs::read(score_path).map_err(|e| format!("reading score: {e}"))?;
    let (score, max_time) = parse_score(&score_bytes).map_err(|e| e.to_string())?;

    let input = if in_path == "_" {
        None
    } else {
        let bytes = std::fs::read(in_path).map_err(|e| format!("reading input wav: {e}"))?;
        let wav = decode_wav(&bytes)?;
        if wav.sample_rate != sample_rate {
            eprintln!(
                "warning: input is {} Hz but rendering at {sample_rate} Hz; \
                 input is fed frame-for-frame (no resampling)",
                wav.sample_rate
            );
        }
        Some(wav)
    };
    let in_channels = input.as_ref().map_or(0, |w| w.channels);

    let options = Options {
        sample_rate,
        output_channels: out_channels,
        input_channels: in_channels,
        ..Options::default()
    };
    let (mut controller, nrt, world) = engine(options);
    // Register the demo def so a score using `/s_new blip` plays; scores may also `/d_recv` their own.
    controller.add_synthdef(blip_def(out_channels));
    let mut dispatcher = OscDispatcher::new(controller);
    let mut render = Render::new(world, nrt, &options);

    let mut writer = WavWriter::create(out_path, format.spec(out_channels, sample_rate))
        .map_err(|e| e.to_string())?;

    let mut feed = WavInput::new(input);
    render_osc_score(
        &mut render,
        &mut dispatcher,
        &score,
        Some(&mut |block: &mut [f32]| feed.fill(block)),
        |block| {
            for &s in block {
                format.write(&mut writer, s);
            }
        },
        RenderUntil::EndOfScore {
            tail: Duration::from_secs_f64(tail_secs),
        },
    )
    .map_err(|e| e.to_string())?;
    writer.finalize().map_err(|e| e.to_string())?;

    let secs = osc_units_to_secs(max_time) + tail_secs;
    println!(
        "rendered {} commands -> {out_path} (~{secs:.2}s)",
        score.len()
    );
    Ok(())
}

/// Feeds a decoded input WAV into the render one control block at a time, zero-padded past its end.
struct WavInput {
    wav: Option<Wav>,
    /// Next input frame to emit.
    frame: usize,
}

impl WavInput {
    fn new(wav: Option<Wav>) -> Self {
        WavInput { wav, frame: 0 }
    }

    /// Fill one interleaved input block (already zeroed by the caller) from the WAV at the cursor.
    fn fill(&mut self, block: &mut [f32]) {
        let Some(wav) = &self.wav else { return };
        if wav.channels == 0 {
            return;
        }
        let frames = block.len() / wav.channels;
        let total = wav.samples.len() / wav.channels;
        let avail = total.saturating_sub(self.frame).min(frames);
        let src = &wav.samples[self.frame * wav.channels..(self.frame + avail) * wav.channels];
        block[..src.len()].copy_from_slice(src);
        self.frame += avail;
    }
}

// -- the `gen` subcommand -----------------------------------------------------------------------

fn gen_score(args: &[String]) -> Result<(), String> {
    let out_path = args.first().ok_or("missing <out-score>")?;
    let bytes = demo_score();
    std::fs::write(out_path, &bytes).map_err(|e| format!("writing score: {e}"))?;
    println!("wrote a {NUM_BEATS}-beat demo score -> {out_path}");
    Ok(())
}

/// A demo score: `NUM_BEATS` `blip`s walking a pentatonic scale, one bundle per beat, encoded as a
/// binary OSC score. Deliberately emitted out of time order to show the engine fires by time tag.
fn demo_score() -> Vec<u8> {
    let beat_units = (BEAT_SECS * OSC_UNITS_PER_SEC) as u64;
    let mut packets = Vec::new();
    for k in (0..NUM_BEATS).rev() {
        let time = k as u64 * beat_units;
        let freq = SCALE[k % SCALE.len()];
        packets.push(beat_bundle(time, 1000 + k as i32, freq));
    }
    encode_score(&packets)
}

/// A time-tagged bundle that starts one `blip` at `freq`, scheduled for OSC/NTP `time`.
fn beat_bundle(time: u64, id: i32, freq: f32) -> OscPacket {
    OscPacket::Bundle(OscBundle {
        timetag: OscTime {
            seconds: (time >> 32) as u32,
            fractional: time as u32,
        },
        content: vec![OscPacket::Message(OscMessage {
            addr: "/s_new".into(),
            args: vec![
                OscType::String("blip".into()),
                OscType::Int(id),
                OscType::Int(1),             // addAction: tail
                OscType::Int(ROOT_GROUP_ID), // target: root group
                OscType::String("freq".into()),
                OscType::Float(freq),
            ],
        })],
    })
}

/// Encode `packets` as a binary OSC score: each record is a big-endian `i32` byte length then the
/// encoded bundle.
fn encode_score(packets: &[OscPacket]) -> Vec<u8> {
    let mut out = Vec::new();
    for packet in packets {
        let bytes = rosc::encoder::encode(packet).expect("encode bundle");
        out.extend_from_slice(&(bytes.len() as i32).to_be_bytes());
        out.extend_from_slice(&bytes);
    }
    out
}

// -- the demo synthdef --------------------------------------------------------------------------

/// `SinOsc.ar(freq) * EnvGen.kr(Env.perc, doneAction: 2)`, written via `OffsetOut` so a scheduled
/// note onsets at exactly its sample (ported from the `schedule` example).
fn blip_def(channels: usize) -> SynthDef {
    let channels = channels.max(1);
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 2, output: 0 });
    }
    SynthDef {
        name: "blip".into(),
        params: vec![Param::control("freq", 440.0)],
        units: vec![
            UnitSpec::new("EnvGen", Rate::Control, perc_env_inputs(), 1),
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Param(0), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec {
                name: "BinaryOpUGen".into(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                num_outputs: 1,
                special_index: 2, // multiply
            },
            UnitSpec::new("OffsetOut", Rate::Audio, out_inputs, 0),
        ],
    }
}

/// `Env.perc(0.001, 0.13, AMP)` unrolled for `EnvGen`: near-instant attack, short exponential decay,
/// freeing the synth when faded.
fn perc_env_inputs() -> Vec<InputRef> {
    [
        1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, -99.0, -99.0, // gate..loopNode
        AMP, 0.001, 1.0, 0.0, // attack: -> AMP over 1 ms, linear
        0.0, 0.13, 5.0, -4.0, // decay: -> 0 over 130 ms, exponential
    ]
    .into_iter()
    .map(InputRef::Constant)
    .collect()
}

// -- WAV helpers --------------------------------------------------------------------------------

/// The output WAV sample format (scsynth's `<sample-format>`).
#[derive(Clone, Copy)]
enum SampleFmt {
    /// 32-bit float (scsynth's default; lossless, but some basic players reject it).
    Float,
    /// 16-bit signed PCM (the most widely compatible).
    Int16,
    /// 24-bit signed PCM.
    Int24,
}

impl std::str::FromStr for SampleFmt {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "f32" | "float" => Ok(SampleFmt::Float),
            "i16" | "int16" => Ok(SampleFmt::Int16),
            "i24" | "int24" => Ok(SampleFmt::Int24),
            other => Err(format!("bad [sample-format] '{other}' (f32 | i16 | i24)")),
        }
    }
}

impl SampleFmt {
    fn spec(self, channels: usize, sample_rate: f64) -> WavSpec {
        let (bits, sample_format) = match self {
            SampleFmt::Float => (32, SampleFormat::Float),
            SampleFmt::Int16 => (16, SampleFormat::Int),
            SampleFmt::Int24 => (24, SampleFormat::Int),
        };
        WavSpec {
            channels: channels as u16,
            sample_rate: sample_rate as u32,
            bits_per_sample: bits,
            sample_format,
        }
    }

    /// Write one sample, converting and clamping to the integer formats (the inverse of `decode_wav`).
    fn write<W: std::io::Write + std::io::Seek>(self, writer: &mut WavWriter<W>, s: f32) {
        let result = match self {
            SampleFmt::Float => writer.write_sample(s),
            SampleFmt::Int16 => writer.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16),
            SampleFmt::Int24 => writer.write_sample((s.clamp(-1.0, 1.0) * 8_388_607.0) as i32),
        };
        result.expect("write sample");
    }
}

/// Interleaved-`f32` WAV contents.
struct Wav {
    samples: Vec<f32>,
    channels: usize,
    sample_rate: f64,
}

/// Decode WAV bytes (any bit depth, PCM or float) into interleaved `f32` (reused from the `sampler`
/// example).
fn decode_wav(bytes: &[u8]) -> Result<Wav, String> {
    let reader = hound::WavReader::new(Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?,
        SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()
                .map_err(|e| e.to_string())?
        }
    };
    Ok(Wav {
        samples,
        channels: spec.channels.max(1) as usize,
        sample_rate: spec.sample_rate as f64,
    })
}

fn osc_units_to_secs(units: u64) -> f64 {
    units as f64 / OSC_UNITS_PER_SEC
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 48_000.0;
    const BLOCK: usize = 64;

    fn inc() -> u64 {
        (BLOCK as f64 * OSC_UNITS_PER_SEC / SR) as u64
    }

    /// The OSC tag firing at exactly global sample `s` on the free-running clock (non-block-aligned).
    fn time_for_sample(s: usize) -> u64 {
        let block = (s / BLOCK) as u64;
        let off = (s % BLOCK) as f64;
        block * inc() + (off * (OSC_UNITS_PER_SEC / SR)).round() as u64
    }

    /// A click voice: 0.5 held for 5 ms then self-freed, onset placed by `OffsetOut`.
    fn click_def() -> SynthDef {
        SynthDef {
            name: "click".into(),
            params: vec![],
            units: vec![
                UnitSpec::new(
                    "Line",
                    Rate::Audio,
                    vec![
                        InputRef::Constant(0.5),
                        InputRef::Constant(0.5),
                        InputRef::Constant(0.005),
                        InputRef::Constant(2.0),
                    ],
                    1,
                ),
                UnitSpec::new(
                    "OffsetOut",
                    Rate::Audio,
                    vec![
                        InputRef::Constant(0.0),
                        InputRef::Unit { unit: 0, output: 0 },
                    ],
                    0,
                ),
            ],
        }
    }

    fn click_bundle(time: u64, id: i32) -> OscPacket {
        OscPacket::Bundle(OscBundle {
            timetag: OscTime {
                seconds: (time >> 32) as u32,
                fractional: time as u32,
            },
            content: vec![OscPacket::Message(OscMessage {
                addr: "/s_new".into(),
                args: vec![
                    OscType::String("click".into()),
                    OscType::Int(id),
                    OscType::Int(1),
                    OscType::Int(ROOT_GROUP_ID),
                ],
            })],
        })
    }

    /// Render a click score (mono) offline and return the interleaved samples.
    fn render_clicks(targets: &[usize]) -> Vec<f32> {
        let options = Options {
            sample_rate: SR,
            output_channels: 1,
            input_channels: 0,
            ..Options::default()
        };
        let (mut controller, nrt, world) = engine(options);
        controller.add_synthdef(click_def());
        let mut dispatcher = OscDispatcher::new(controller);
        let mut render = Render::new(world, nrt, &options);

        let packets: Vec<OscPacket> = targets
            .iter()
            .enumerate()
            .map(|(i, &s)| click_bundle(time_for_sample(s), 1000 + i as i32))
            .collect();
        let (score, _) = parse_score(&encode_score(&packets)).expect("parse");

        let mut out = Vec::new();
        let mut no_input = |_: &mut [f32]| {};
        render_osc_score(
            &mut render,
            &mut dispatcher,
            &score,
            Some(&mut no_input),
            |block| out.extend_from_slice(block),
            RenderUntil::EndOfScore {
                tail: Duration::from_millis(20),
            },
        )
        .expect("render");
        out
    }

    fn assert_onsets(out: &[f32], targets: &[usize]) {
        let mut from = 0;
        for (k, &s) in targets.iter().enumerate() {
            let onset = (from..out.len())
                .find(|&i| out[i] != 0.0)
                .unwrap_or_else(|| panic!("click {k} never sounded"));
            assert_eq!(onset, s, "click {k} should onset at {s}, got {onset}");
            from = onset;
            while from < out.len() && out[from] != 0.0 {
                from += 1;
            }
        }
    }

    #[test]
    fn clicks_render_to_exact_samples() {
        let targets = [600usize, 1503, 2305, 3100];
        assert_onsets(&render_clicks(&targets), &targets);
    }

    #[test]
    fn render_is_byte_deterministic() {
        let targets = [600usize, 1503, 2305, 3100];
        assert_eq!(render_clicks(&targets), render_clicks(&targets));
    }

    #[test]
    fn wav_round_trips_through_hound() {
        // Render clicks to an in-memory float WAV, decode it back, and assert the exact onsets
        // survive the WAV encode/decode (the example's output path, minus the file).
        let targets = [600usize, 1503];
        let out = render_clicks(&targets);
        let spec = WavSpec {
            channels: 1,
            sample_rate: SR as u32,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut buf = Cursor::new(Vec::<u8>::new());
        {
            let mut writer = WavWriter::new(&mut buf, spec).expect("writer");
            for &s in &out {
                writer.write_sample(s).expect("write");
            }
            writer.finalize().expect("finalize");
        }
        let decoded = decode_wav(buf.get_ref()).expect("decode");
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.sample_rate, SR);
        assert_onsets(&decoded.samples, &targets);
    }

    #[test]
    fn demo_score_round_trips() {
        let (score, max_time) = parse_score(&demo_score()).expect("parse demo");
        assert_eq!(score.len(), NUM_BEATS);
        // The last beat is at (NUM_BEATS-1) beat units (matching `demo_score`'s integer arithmetic).
        let beat_units = (BEAT_SECS * OSC_UNITS_PER_SEC) as u64;
        assert_eq!(max_time, (NUM_BEATS as u64 - 1) * beat_units);
    }

    /// `In.ar(input-bus) -> Out.ar(0)`: a one-to-one passthrough of the input bus to output 0.
    fn passthrough_def(input_bus: f32) -> SynthDef {
        SynthDef {
            name: "thru".into(),
            params: vec![],
            units: vec![
                UnitSpec::new("In", Rate::Audio, vec![InputRef::Constant(input_bus)], 1),
                UnitSpec::new(
                    "Out",
                    Rate::Audio,
                    vec![
                        InputRef::Constant(0.0),
                        InputRef::Unit { unit: 0, output: 0 },
                    ],
                    0,
                ),
            ],
        }
    }

    #[test]
    fn duplex_input_passes_through() {
        // Mono in, mono out: input channel 0 lands on input bus index `output_channels` (= 1).
        let options = Options {
            sample_rate: SR,
            output_channels: 1,
            input_channels: 1,
            ..Options::default()
        };
        let (mut controller, nrt, world) = engine(options);
        controller.add_synthdef(passthrough_def(options.output_channels as f32));
        let mut dispatcher = OscDispatcher::new(controller);
        let mut render = Render::new(world, nrt, &options);

        // Start the passthrough immediately (time tag 0 = immediate), before any audio block.
        let start = encode_score(&[OscPacket::Bundle(OscBundle {
            timetag: OscTime {
                seconds: 0,
                fractional: 0,
            },
            content: vec![OscPacket::Message(OscMessage {
                addr: "/s_new".into(),
                args: vec![
                    OscType::String("thru".into()),
                    OscType::Int(2000),
                    OscType::Int(1),
                    OscType::Int(ROOT_GROUP_ID),
                ],
            })],
        })]);
        let (score, _) = parse_score(&start).expect("parse");

        // A ramp input, a whole number of blocks long.
        let blocks = 8;
        let input: Vec<f32> = (0..blocks * BLOCK).map(|i| (i as f32) / 1000.0).collect();
        let mut wav_in = WavInput::new(Some(Wav {
            samples: input.clone(),
            channels: 1,
            sample_rate: SR,
        }));

        let mut out = Vec::new();
        render_osc_score(
            &mut render,
            &mut dispatcher,
            &score,
            Some(&mut |block: &mut [f32]| wav_in.fill(block)),
            |block| out.extend_from_slice(block),
            RenderUntil::Duration(Duration::from_secs_f64(blocks as f64 * BLOCK as f64 / SR)),
        )
        .expect("render");

        // The passthrough is created and audible from block 0, so output mirrors input block-aligned.
        assert!(out.len() >= input.len());
        assert_eq!(&out[..input.len()], &input[..]);
    }
}
