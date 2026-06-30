//! `plyphon render` - offline (non-real-time) score rendering, plyphon's `scsynth -N`.
//!
//! Reads a binary OSC score, drives the engine faster than real time on a deterministic clock via
//! [`plyphon::Render`], and writes the rendered audio to a WAV file. An optional input WAV is fed
//! into the input buses for `In.ar`. Output is bit-identical across runs.

use std::time::Duration;

use hound::WavWriter;
use plyphon::{Render, RenderUntil, engine};
use plyphon_osc::{OscDispatcher, parse_score, render_osc_score};

use crate::cli::RenderArgs;
use crate::defs::load_dir;
use crate::options::engine_options;
use crate::wav::{self, WavInput};

/// OSC/NTP fixed-point units in one second (OSC time is 32.32 fixed point).
const OSC_UNITS_PER_SEC: f64 = 4_294_967_296.0;

pub fn run(args: RenderArgs) -> Result<(), String> {
    let score_bytes = std::fs::read(&args.score).map_err(|e| format!("reading score: {e}"))?;
    let (score, max_time) = parse_score(&score_bytes).map_err(|e| e.to_string())?;

    // Decode the optional input WAV, warning if its rate differs (it is fed frame-for-frame).
    let input = match &args.input {
        None => None,
        Some(path) => {
            let bytes = std::fs::read(path).map_err(|e| format!("reading input wav: {e}"))?;
            let decoded = wav::decode(&bytes)?;
            if decoded.sample_rate != args.sample_rate {
                eprintln!(
                    "warning: input is {} Hz but rendering at {} Hz; \
                     input is fed frame-for-frame (no resampling)",
                    decoded.sample_rate, args.sample_rate
                );
            }
            Some(decoded)
        }
    };
    let in_channels = input.as_ref().map_or(0, |w| w.channels);

    let options = engine_options(&args.engine, args.sample_rate, args.channels, in_channels);
    let (mut controller, nrt, world) = engine(options);
    if let Some(dir) = &args.engine.load_dir {
        load_dir(&mut controller, dir)?;
    }
    let mut dispatcher = OscDispatcher::new();
    let mut render = Render::new(world, nrt, &options);

    let mut writer = WavWriter::create(
        &args.output,
        wav::spec(args.sample_format, args.channels, args.sample_rate),
    )
    .map_err(|e| e.to_string())?;

    let mut feed = WavInput::new(input);
    render_osc_score(
        &mut render,
        &mut dispatcher,
        &mut controller,
        &score,
        Some(&mut |block: &mut [f32]| feed.fill(block)),
        |block| {
            for &s in block {
                wav::write_sample(args.sample_format, &mut writer, s);
            }
        },
        RenderUntil::EndOfScore {
            tail: Duration::from_secs_f64(args.tail),
        },
    )
    .map_err(|e| e.to_string())?;
    writer.finalize().map_err(|e| e.to_string())?;

    let secs = max_time as f64 / OSC_UNITS_PER_SEC + args.tail;
    println!(
        "rendered {} commands -> {} (~{secs:.2}s)",
        score.len(),
        args.output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use rosc::{OscBundle, OscMessage, OscPacket, OscTime, OscType};

    use crate::cli::{EngineArgs, RenderArgs, SampleFormat};

    /// Engine flags with scsynth defaults, plus an optional `--load-dir`.
    fn engine_args(load_dir: Option<PathBuf>) -> EngineArgs {
        EngineArgs {
            block_size: 64,
            audio_buses: 128,
            control_buses: 4096,
            max_nodes: 1024,
            max_buffers: 1024,
            max_synthdefs: 1024,
            rt_memory_kib: 8192,
            load_dir,
        }
    }

    /// A unique, freshly-created temp dir for a test's files.
    fn temp_dir(tag: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("plyphon-cli-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// SCgf bytes (exactly an `.scsyndef` file's contents) for `SinOsc.ar(440) -> Out.ar(0)`.
    fn sine_scsyndef() -> Vec<u8> {
        use scgf::{Input, Rate, SynthDef, SynthDefFile, Ugen};
        let file = SynthDefFile {
            version: 2,
            defs: vec![SynthDef {
                name: "sine".to_string(),
                constants: vec![440.0, 0.0], // freq, phase/bus
                param_values: vec![],
                param_names: vec![],
                ugens: vec![
                    Ugen {
                        name: "SinOsc".to_string(),
                        rate: Rate::Audio,
                        special_index: 0,
                        inputs: vec![Input::Constant { index: 0 }, Input::Constant { index: 1 }],
                        outputs: vec![Rate::Audio],
                    },
                    Ugen {
                        name: "Out".to_string(),
                        rate: Rate::Audio,
                        special_index: 0,
                        inputs: vec![
                            Input::Constant { index: 1 },
                            Input::Ugen { ugen: 0, output: 0 },
                        ],
                        outputs: vec![],
                    },
                ],
                variants: vec![],
                ..Default::default()
            }],
        };
        scgf::encode(&file).expect("encode SCgf")
    }

    /// A one-bundle binary OSC score that starts `sine` at time 0.
    fn sine_score() -> Vec<u8> {
        let packet = OscPacket::Bundle(OscBundle {
            timetag: OscTime {
                seconds: 0,
                fractional: 0,
            },
            content: vec![OscPacket::Message(OscMessage {
                addr: "/s_new".to_string(),
                args: vec![
                    OscType::String("sine".to_string()),
                    OscType::Int(1000),
                    OscType::Int(0), // addAction: head
                    OscType::Int(0), // target: root group
                ],
            })],
        });
        let bytes = rosc::encoder::encode(&packet).expect("encode bundle");
        let mut score = (bytes.len() as i32).to_be_bytes().to_vec();
        score.extend_from_slice(&bytes);
        score
    }

    #[test]
    fn render_loads_a_def_and_sounds_deterministically() {
        let dir = temp_dir("render");
        std::fs::write(dir.join("sine.scsyndef"), sine_scsyndef()).unwrap();
        let score = dir.join("phrase.osc");
        std::fs::write(&score, sine_score()).unwrap();

        let render_to = |out: PathBuf| {
            super::run(RenderArgs {
                score: score.clone(),
                output: out,
                input: None,
                sample_rate: 48_000.0,
                channels: 1,
                tail: 0.2,
                sample_format: SampleFormat::F32,
                engine: engine_args(Some(dir.clone())),
            })
            .unwrap();
        };

        let a = dir.join("a.wav");
        let b = dir.join("b.wav");
        render_to(a.clone());
        render_to(b.clone());

        // The loaded def sounded...
        let decoded = crate::wav::decode(&std::fs::read(&a).unwrap()).unwrap();
        assert!(
            decoded.samples.iter().any(|&s| s.abs() > 0.01),
            "rendered audio should be non-silent"
        );
        // ...and the render is byte-deterministic.
        assert_eq!(
            std::fs::read(&a).unwrap(),
            std::fs::read(&b).unwrap(),
            "two renders of the same score should be byte-identical"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn render_writes_a_16_bit_wav() {
        let dir = temp_dir("render-i16");
        std::fs::write(dir.join("sine.scsyndef"), sine_scsyndef()).unwrap();
        let score = dir.join("phrase.osc");
        std::fs::write(&score, sine_score()).unwrap();
        let out = dir.join("out.wav");

        super::run(RenderArgs {
            score,
            output: out.clone(),
            input: None,
            sample_rate: 48_000.0,
            channels: 1,
            tail: 0.2,
            sample_format: SampleFormat::I16,
            engine: engine_args(Some(dir.clone())),
        })
        .unwrap();

        let spec = hound::WavReader::open(&out).unwrap().spec();
        assert_eq!(spec.bits_per_sample, 16);
        assert_eq!(spec.sample_format, hound::SampleFormat::Int);
        let decoded = crate::wav::decode(&std::fs::read(&out).unwrap()).unwrap();
        assert!(decoded.samples.iter().any(|&s| s.abs() > 0.01));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
