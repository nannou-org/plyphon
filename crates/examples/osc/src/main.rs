//! Driving the engine through SuperCollider OSC packets - with no sockets - and printing the
//! conversation, natively and on the web.
//!
//! This is the same wire protocol an `sclang` client speaks to `scsynth`, but instead of a UDP
//! socket the example encodes each command to OSC bytes and hands them straight to a
//! [`plyphon_osc::OscDispatcher`] (`apply_bytes`) - exactly what a real transport would do after
//! receiving a datagram. Replies flow back the same way: buffer queries and `/done` acknowledgements
//! arrive as queued OSC packets, and node lifecycle events (a synth starting, or freeing itself)
//! are surfaced as `/n_go`/`/n_end` notifications via `OscDispatcher::notify`. Both directions are
//! printed (natively), so you can see OSC controlling the audio and reporting back; wiring up an
//! actual UDP/TCP transport is then just swapping `apply_bytes`/`take_replies` for socket I/O.
//!
//! The scripted session: load a `tone` SynthDef over `/d_recv`, start it (`/s_new`), bend its pitch
//! (`/n_set`), allocate and query a buffer (`/b_alloc`, `/b_query`), trace the running synth's
//! per-unit inputs and outputs (`/n_trace`, dumped to a text sink, like scsynth's stdout), then free
//! it (`/n_free`). The `plyphon` engine driving the audio is pure Rust and identical on native and
//! web; only the control-plane ticking differs (a thread loop vs a timer).

use plyphon::{Controller, Nrt, Options, World, engine};
use plyphon_osc::OscDispatcher;
use rosc::{OscMessage, OscPacket, OscType};

/// How often to tick the control plane, in milliseconds (one scripted command per tick).
const TICK_MS: u32 = 800;
/// Number of commands in the scripted session.
const SCRIPT_LEN: usize = 7;
/// Tick once per scripted command, plus a few trailing ticks so the final `/n_end` drains.
const TRAILING_TICKS: u32 = 4;
/// Master gain applied in the audio callback (the `tone` def already scales by 0.2).
const GAIN: f32 = 1.0;

/// The control plane: a dispatcher applying OSC to the engine's `Controller` (owned here, lent per
/// call), plus the `Nrt`, ticked off the audio thread. Each tick sends the next scripted OSC command
/// and prints the replies the dispatcher has queued - including the node notifications synthesized
/// from `Nrt` events.
struct Session {
    dispatcher: OscDispatcher,
    controller: Controller,
    nrt: Nrt,
    /// The next scripted command to send.
    step: usize,
    /// The `tone` SynthDef, compiled to SCgf bytes, sent over `/d_recv`.
    tone: Vec<u8>,
}

impl Session {
    /// One tick: drop finished synths, send the next scripted command, then print every queued reply.
    fn tick(&mut self) {
        // Drop the boxes of freed synths here, off the audio thread.
        self.nrt.process();
        if self.step < SCRIPT_LEN {
            self.send_step(self.step);
            self.step += 1;
        }
        for message in self.collect_reports() {
            trace(&format!("  <- {}", format_msg(&message)));
        }
    }

    /// Encode scripted command `step` to OSC bytes and feed it to the dispatcher, as a transport
    /// would with a received datagram.
    fn send_step(&mut self, step: usize) {
        let (label, message) = match step {
            0 => (
                format!(
                    "/d_recv  (a compiled 'tone' SynthDef, {} bytes)",
                    self.tone.len()
                ),
                msg("/d_recv", vec![OscType::Blob(self.tone.clone())]),
            ),
            // name, node id, addAction (1 = tail), target group (0 = root).
            1 => (
                "/s_new tone 1000 1 0".to_string(),
                msg(
                    "/s_new",
                    vec![
                        OscType::String("tone".to_string()),
                        OscType::Int(1000),
                        OscType::Int(1),
                        OscType::Int(0),
                    ],
                ),
            ),
            2 => (
                "/n_set 1000 freq 440".to_string(),
                msg(
                    "/n_set",
                    vec![
                        OscType::Int(1000),
                        OscType::String("freq".to_string()),
                        OscType::Float(440.0),
                    ],
                ),
            ),
            // Dump the running synth's per-unit inputs/outputs to the trace sink (no OSC reply). The
            // dump arrives a render later over the reply ring and prints below; tracing here (while the
            // synth lives through the rest of the script) lets the dispatcher resolve unit names from
            // its def, which it could not once the node is freed.
            3 => (
                "/n_trace 1000".to_string(),
                msg("/n_trace", vec![OscType::Int(1000)]),
            ),
            // bufnum, frames, channels.
            4 => (
                "/b_alloc 0 1024 1".to_string(),
                msg(
                    "/b_alloc",
                    vec![OscType::Int(0), OscType::Int(1024), OscType::Int(1)],
                ),
            ),
            5 => (
                "/b_query 0".to_string(),
                msg("/b_query", vec![OscType::Int(0)]),
            ),
            6 => (
                "/n_free 1000".to_string(),
                msg("/n_free", vec![OscType::Int(1000)]),
            ),
            _ => return,
        };
        trace(&format!("-> {label}"));
        let bytes = rosc::encoder::encode(&OscPacket::Message(message)).expect("encode OSC packet");
        if let Err(err) = self.dispatcher.apply_bytes(&mut self.controller, &bytes) {
            trace(&format!("   (dispatcher rejected the packet: {err})"));
        }
    }

    /// Turn pending engine events into OSC node notifications, drain the reply ring (so async answers
    /// like the `/n_trace` dump reach their sinks), then take every queued reply.
    fn collect_reports(&mut self) -> Vec<OscMessage> {
        while let Some(event) = self.nrt.poll() {
            self.dispatcher.notify(event);
        }
        while let Some(reply) = self.nrt.poll_reply() {
            self.dispatcher.reply(&self.controller, reply);
        }
        self.dispatcher
            .take_replies()
            .into_iter()
            .filter_map(|packet| match packet {
                OscPacket::Message(message) => Some(message),
                OscPacket::Bundle(_) => None,
            })
            .collect()
    }
}

/// The `tone` SynthDef - `SinOsc.ar(freq) * 0.2 -> Out.ar(0)` - compiled to SCgf bytes, as an
/// `sclang` client would send over `/d_recv`. SuperCollider models the named `freq` parameter as a
/// `Control` UGen whose output feeds the graph; plyphon folds that back into a parameter on load.
fn tone_scgf() -> Vec<u8> {
    use scgf::{Input, ParamName, Rate, SynthDef, SynthDefFile, Ugen};
    let file = SynthDefFile {
        version: 2,
        defs: vec![SynthDef {
            name: "tone".to_string(),
            constants: vec![0.0, 0.2], // 0: SinOsc phase and Out bus; 1: amplitude.
            param_values: vec![330.0], // freq default.
            param_names: vec![ParamName {
                name: "freq".to_string(),
                index: 0,
            }],
            ugens: vec![
                // 0: Control.kr -> freq
                Ugen {
                    name: "Control".to_string(),
                    rate: Rate::Control,
                    special_index: 0,
                    inputs: vec![],
                    outputs: vec![Rate::Control],
                },
                // 1: SinOsc.ar(freq, phase = 0)
                Ugen {
                    name: "SinOsc".to_string(),
                    rate: Rate::Audio,
                    special_index: 0,
                    inputs: vec![
                        Input::Ugen { ugen: 0, output: 0 },
                        Input::Constant { index: 0 },
                    ],
                    outputs: vec![Rate::Audio],
                },
                // 2: SinOsc * 0.2 (BinaryOpUGen, special index 2 = multiply)
                Ugen {
                    name: "BinaryOpUGen".to_string(),
                    rate: Rate::Audio,
                    special_index: 2,
                    inputs: vec![
                        Input::Ugen { ugen: 1, output: 0 },
                        Input::Constant { index: 1 },
                    ],
                    outputs: vec![Rate::Audio],
                },
                // 3: Out.ar(0, signal)
                Ugen {
                    name: "Out".to_string(),
                    rate: Rate::Audio,
                    special_index: 0,
                    inputs: vec![
                        Input::Constant { index: 0 },
                        Input::Ugen { ugen: 2, output: 0 },
                    ],
                    outputs: vec![],
                },
            ],
            variants: vec![],
            ..Default::default()
        }],
    };
    scgf::encode(&file).expect("encode tone SynthDef")
}

/// Build the control plane (kept alive and ticked by the host) and the `World` (the audio source).
/// The SynthDef is *not* registered here - it arrives over OSC via `/d_recv`, like everything else.
fn build(sample_rate: f32, channels: usize) -> (Session, World) {
    let channels = channels.max(1);
    let (controller, nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels,
        ..Options::default()
    });
    // `/n_trace` has no OSC reply; it dumps to a host text sink (scsynth prints to stdout). Print each
    // line in the same `<-` reply style so it reads as part of the conversation.
    let mut dispatcher = OscDispatcher::new();
    dispatcher.set_trace_sink(Box::new(|text| {
        for line in text.lines() {
            trace(&format!("  <- {line}"));
        }
    }));
    let session = Session {
        dispatcher,
        controller,
        nrt,
        step: 0,
        tone: tone_scgf(),
    };
    (session, world)
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    // cpal's AudioWorklet backend re-instantiates this module on the audio thread, re-running
    // `main` there; only set up audio on the main browser thread.
    if example_audio::on_worklet_thread() {
        return;
    }

    #[cfg(not(target_arch = "wasm32"))]
    println!(
        "OSC session driving the plyphon engine (packets handed straight to the dispatcher):\n"
    );

    let (stream, mut session) = example_audio::play_with(GAIN, |sample_rate, channels| {
        let (session, mut world) = build(sample_rate as f32, channels);
        (
            move |out: &mut [f32], channels: usize| world.fill(out, channels),
            session,
        )
    });
    // One tick per scripted command, plus a few trailing ticks so the final `/n_end` drains; web
    // ticks past the script are harmless no-ops.
    let total_ms = (SCRIPT_LEN as u32 + TRAILING_TICKS) * TICK_MS;
    example_audio::run_control(stream, total_ms, TICK_MS, move || session.tick());
}

/// Build an `OscMessage` from an address and arguments.
fn msg(addr: &str, args: Vec<OscType>) -> OscMessage {
    OscMessage {
        addr: addr.to_string(),
        args,
    }
}

/// Format an `OscMessage` as a readable `addr arg arg ...` line.
fn format_msg(message: &OscMessage) -> String {
    let mut line = message.addr.clone();
    for arg in &message.args {
        line.push(' ');
        match arg {
            OscType::Int(i) => line.push_str(&i.to_string()),
            OscType::Float(f) => line.push_str(&format!("{f}")),
            OscType::String(s) => line.push_str(s),
            OscType::Blob(bytes) => line.push_str(&format!("<{} bytes>", bytes.len())),
            other => line.push_str(&format!("{other:?}")),
        }
    }
    line
}

/// Print a line of the OSC trace (native only; the web demo still plays the controlled audio).
fn trace(line: &str) {
    #[cfg(not(target_arch = "wasm32"))]
    println!("{line}");
    #[cfg(target_arch = "wasm32")]
    let _ = line;
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn goertzel(samples: &[f32], freq: f32) -> f32 {
        let n = samples.len();
        let k = (0.5 + n as f32 * freq / SR).floor();
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

    fn render(world: &mut World, frames: usize) -> Vec<f32> {
        let sizes = [64usize, 100, 128, 480, 512, 333];
        let mut out = Vec::with_capacity(frames + 512);
        let mut buf = Vec::new();
        let mut i = 0;
        while out.len() < frames {
            buf.clear();
            buf.resize(sizes[i % sizes.len()], 0.0);
            i += 1;
            world.fill(&mut buf, 1);
            out.extend_from_slice(&buf);
        }
        out.truncate(frames);
        out
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt()
    }

    /// Has any reported message with this address and a leading int argument equal to `id`.
    fn reported(reports: &[OscMessage], addr: &str, id: i32) -> bool {
        reports
            .iter()
            .any(|m| m.addr == addr && matches!(m.args.first(), Some(OscType::Int(i)) if *i == id))
    }

    /// The whole scripted session, deterministic: every command applied, audio rendered between
    /// steps, and the OSC replies/notifications collected and asserted.
    #[test]
    fn osc_session_controls_audio_and_reports_back() {
        let (mut session, mut world) = build(SR, 1);
        let mut reports: Vec<OscMessage> = Vec::new();

        // Override the demo's printing trace sink with one that captures, so the /n_trace dump can be
        // asserted on (the dump has no OSC reply - it only reaches the sink).
        let traced = std::rc::Rc::new(std::cell::RefCell::new(String::new()));
        {
            let sink = std::rc::Rc::clone(&traced);
            session
                .dispatcher
                .set_trace_sink(Box::new(move |text| sink.borrow_mut().push_str(text)));
        }

        // /d_recv the def, then /s_new the synth; a render lets the World link it and emit /n_go.
        session.send_step(0);
        session.send_step(1);
        let _ = render(&mut world, 4096);
        reports.extend(session.collect_reports());
        assert!(
            reported(&reports, "/n_go", 1000),
            "starting the synth should report /n_go 1000, got {reports:?}"
        );

        // /n_set the frequency to 440 Hz; the rendered tone should be dominated by 440.
        session.send_step(2);
        let tone = render(&mut world, SR as usize / 4);
        assert!(rms(&tone) > 0.05, "the tone should be audible");
        assert!(
            goertzel(&tone, 440.0) > 3.0 * goertzel(&tone, 330.0),
            "/n_set freq 440 should retune the tone to 440 Hz"
        );

        // /n_trace 1000: the running synth dumps its per-unit inputs/outputs to the trace sink. The
        // dump arrives a render later over the reply ring; collect_reports drains it into the sink. The
        // unit names resolve from the (still-live) node's def.
        session.send_step(3);
        let _ = render(&mut world, 4096);
        reports.extend(session.collect_reports());
        {
            let trace_text = traced.borrow();
            assert!(
                trace_text.contains("TRACE node 1000 (tone)"),
                "/n_trace should dump the node's units, got {trace_text:?}"
            );
            assert!(
                trace_text.contains("SinOsc") && trace_text.contains("Out"),
                "the trace should name the synth's units, got {trace_text:?}"
            );
            // SinOsc reads freq 440 (the /n_set value flowing through the folded Control param).
            assert!(
                trace_text.contains("SinOsc  in: [440"),
                "the trace should show 440 Hz flowing into SinOsc, got {trace_text:?}"
            );
        }

        // /b_alloc then /b_query: the dispatcher acknowledges with /done and reports /b_info.
        session.send_step(4);
        reports.extend(session.collect_reports());
        session.send_step(5);
        reports.extend(session.collect_reports());
        assert!(
            reports.iter().any(|m| m.addr == "/done"
                && m.args.first() == Some(&OscType::String("/b_alloc".to_string()))),
            "/b_alloc should be acknowledged with /done"
        );
        let info = reports
            .iter()
            .find(|m| m.addr == "/b_info")
            .expect("/b_query should report /b_info");
        assert_eq!(
            info.args,
            vec![
                OscType::Int(0),
                OscType::Int(1024),
                OscType::Int(1),
                OscType::Float(SR),
            ],
            "/b_info should report the buffer's dimensions"
        );

        // /n_free the synth; a render lets the World free it and emit /n_end, then it falls silent.
        session.send_step(6);
        let _ = render(&mut world, 4096);
        reports.extend(session.collect_reports());
        assert!(
            reported(&reports, "/n_end", 1000),
            "freeing the synth should report /n_end 1000"
        );
        let silent = render(&mut world, 4096);
        assert!(
            rms(&silent) < 1e-6,
            "the engine should be silent after the synth is freed"
        );
    }
}
