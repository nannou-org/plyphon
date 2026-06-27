//! Drive the engine entirely through OSC: receive a SynthDef with `/d_recv`, start it with
//! `/s_new`, retune it with `/n_set` (by control name), and stop it with `/n_free` - checking the
//! audio out of a `World` at each step.

use plyphon::{Options, ROOT_GROUP_ID, engine};
use plyphon_osc::OscDispatcher;
use rosc::{OscMessage, OscPacket, OscType};
use scgf::{Input, ParamName, Rate, SynthDef, SynthDefFile, Ugen};

const SR: f32 = 48_000.0;

fn sine_scgf() -> Vec<u8> {
    let file = SynthDefFile {
        version: 2,
        defs: vec![SynthDef {
            name: "sine".to_string(),
            constants: vec![0.0, 0.0],
            param_values: vec![440.0],
            param_names: vec![ParamName {
                name: "freq".to_string(),
                index: 0,
            }],
            ugens: vec![
                Ugen {
                    name: "Control".to_string(),
                    rate: Rate::Control,
                    special_index: 0,
                    inputs: vec![],
                    outputs: vec![Rate::Control],
                },
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
                Ugen {
                    name: "Out".to_string(),
                    rate: Rate::Audio,
                    special_index: 0,
                    inputs: vec![
                        Input::Constant { index: 1 },
                        Input::Ugen { ugen: 1, output: 0 },
                    ],
                    outputs: vec![],
                },
            ],
            variants: vec![],
        }],
    };
    scgf::encode(&file).expect("encode SCgf")
}

/// Like [`sine_scgf`] but with two controls - `freq` (440) and `freq2` (660) - summed through two
/// `Out`s, so range setters (`/n_setn`, `/n_fill`) can be checked against two distinct partials.
fn dual_sine_scgf() -> Vec<u8> {
    let sin_osc = |ctrl_out| Ugen {
        name: "SinOsc".to_string(),
        rate: Rate::Audio,
        special_index: 0,
        inputs: vec![
            Input::Ugen {
                ugen: 0,
                output: ctrl_out,
            },
            Input::Constant { index: 0 }, // phase 0.0
        ],
        outputs: vec![Rate::Audio],
    };
    let out = |sin_ugen| Ugen {
        name: "Out".to_string(),
        rate: Rate::Audio,
        special_index: 0,
        inputs: vec![
            Input::Constant { index: 1 }, // bus 0.0
            Input::Ugen {
                ugen: sin_ugen,
                output: 0,
            },
        ],
        outputs: vec![],
    };
    let file = SynthDefFile {
        version: 2,
        defs: vec![SynthDef {
            name: "dual".to_string(),
            constants: vec![0.0, 0.0],
            param_values: vec![440.0, 660.0],
            param_names: vec![
                ParamName {
                    name: "freq".to_string(),
                    index: 0,
                },
                ParamName {
                    name: "freq2".to_string(),
                    index: 1,
                },
            ],
            ugens: vec![
                Ugen {
                    name: "Control".to_string(),
                    rate: Rate::Control,
                    special_index: 0,
                    inputs: vec![],
                    outputs: vec![Rate::Control, Rate::Control],
                },
                sin_osc(0),
                sin_osc(1),
                out(1),
                out(2),
            ],
            variants: vec![],
        }],
    };
    scgf::encode(&file).expect("encode SCgf")
}

fn osc(addr: &str, args: Vec<OscType>) -> Vec<u8> {
    rosc::encoder::encode(&OscPacket::Message(OscMessage {
        addr: addr.to_string(),
        args,
    }))
    .expect("encode OSC")
}

/// Take the queued replies, keeping only messages (the way a transport would after a command burst).
fn take_msgs(dispatcher: &mut OscDispatcher) -> Vec<OscMessage> {
    dispatcher
        .take_replies()
        .into_iter()
        .filter_map(|packet| match packet {
            OscPacket::Message(message) => Some(message),
            OscPacket::Bundle(_) => None,
        })
        .collect()
}

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

/// Surface the engine's pending node events as OSC notifications (`/n_go`, `/n_end`, ...) and take
/// them, the way a transport would after polling the `Nrt`.
fn drain_notifications(dispatcher: &mut OscDispatcher, nrt: &mut plyphon::Nrt) -> Vec<OscMessage> {
    nrt.process();
    while let Some(event) = nrt.poll() {
        dispatcher.notify(event);
    }
    dispatcher
        .take_replies()
        .into_iter()
        .filter_map(|packet| match packet {
            OscPacket::Message(message) => Some(message),
            OscPacket::Bundle(_) => None,
        })
        .collect()
}

fn render(world: &mut plyphon::World, frames: usize) -> Vec<f32> {
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

#[test]
fn drives_engine_over_osc() {
    let (controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    let mut dispatcher = OscDispatcher::new(controller);

    // Receive the SynthDef and start a synth (node 1000) at the tail of the root group.
    dispatcher
        .apply_bytes(&osc("/d_recv", vec![OscType::Blob(sine_scgf())]))
        .expect("/d_recv");
    dispatcher
        .apply_bytes(&osc(
            "/s_new",
            vec![
                OscType::String("sine".to_string()),
                OscType::Int(1000),
                OscType::Int(1), // addToTail
                OscType::Int(ROOT_GROUP_ID),
            ],
        ))
        .expect("/s_new");

    let a = render(&mut world, SR as usize / 2);
    assert!(
        goertzel(&a, 440.0) > 5.0 * goertzel(&a, 880.0),
        "expected 440 Hz"
    );

    // Retune by control name.
    dispatcher
        .apply_bytes(&osc(
            "/n_set",
            vec![
                OscType::Int(1000),
                OscType::String("freq".to_string()),
                OscType::Float(330.0),
            ],
        ))
        .expect("/n_set");
    let _ = render(&mut world, 512);
    let b = render(&mut world, SR as usize / 2);
    assert!(
        goertzel(&b, 330.0) > 5.0 * goertzel(&b, 660.0),
        "expected 330 Hz"
    );

    // Free the node; after flushing the in-flight block the output is silent.
    dispatcher
        .apply_bytes(&osc("/n_free", vec![OscType::Int(1000)]))
        .expect("/n_free");
    let _ = render(&mut world, 1024);
    let c = render(&mut world, SR as usize / 4);
    assert!(
        c.iter().all(|s| s.abs() < 1e-6),
        "expected silence after /n_free"
    );

    nrt.process();
}

#[test]
fn notifies_node_lifecycle_over_osc() {
    let (controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    let mut dispatcher = OscDispatcher::new(controller);

    dispatcher
        .apply_bytes(&osc("/d_recv", vec![OscType::Blob(sine_scgf())]))
        .expect("/d_recv");
    dispatcher
        .apply_bytes(&osc(
            "/s_new",
            vec![
                OscType::String("sine".to_string()),
                OscType::Int(1000),
                OscType::Int(1),
                OscType::Int(ROOT_GROUP_ID),
            ],
        ))
        .expect("/s_new");

    // The World links the synth and emits a NodeStarted event; notify turns it into /n_go.
    let _ = render(&mut world, 1024);
    let started = drain_notifications(&mut dispatcher, &mut nrt);
    assert!(
        started
            .iter()
            .any(|m| m.addr == "/n_go" && m.args == vec![OscType::Int(1000)]),
        "expected an /n_go 1000 notification, got {started:?}"
    );

    // Free it; the World emits NodeEnded, surfaced as /n_end.
    dispatcher
        .apply_bytes(&osc("/n_free", vec![OscType::Int(1000)]))
        .expect("/n_free");
    let _ = render(&mut world, 1024);
    let ended = drain_notifications(&mut dispatcher, &mut nrt);
    assert!(
        ended
            .iter()
            .any(|m| m.addr == "/n_end" && m.args == vec![OscType::Int(1000)]),
        "expected an /n_end 1000 notification, got {ended:?}"
    );
}

#[test]
fn maps_and_sets_control_buses_over_osc() {
    let (controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    let mut dispatcher = OscDispatcher::new(controller);

    dispatcher
        .apply_bytes(&osc("/d_recv", vec![OscType::Blob(sine_scgf())]))
        .expect("/d_recv");
    dispatcher
        .apply_bytes(&osc(
            "/s_new",
            vec![
                OscType::String("sine".to_string()),
                OscType::Int(1000),
                OscType::Int(1),
                OscType::Int(ROOT_GROUP_ID),
            ],
        ))
        .expect("/s_new");

    // Map the `freq` control (by name) to control bus 3, then drive that bus with /c_set: the
    // synth ignores its 440 default and follows the bus.
    dispatcher
        .apply_bytes(&osc(
            "/n_map",
            vec![
                OscType::Int(1000),
                OscType::String("freq".to_string()),
                OscType::Int(3),
            ],
        ))
        .expect("/n_map");
    dispatcher
        .apply_bytes(&osc("/c_set", vec![OscType::Int(3), OscType::Float(220.0)]))
        .expect("/c_set");
    let _ = render(&mut world, 512);
    let a = render(&mut world, SR as usize / 2);
    assert!(
        goertzel(&a, 220.0) > 5.0 * goertzel(&a, 440.0),
        "expected the mapped control to follow bus 3 at 220 Hz"
    );

    // Move the bus; the mapped control follows.
    dispatcher
        .apply_bytes(&osc("/c_set", vec![OscType::Int(3), OscType::Float(660.0)]))
        .expect("/c_set");
    let _ = render(&mut world, 512);
    let b = render(&mut world, SR as usize / 2);
    assert!(
        goertzel(&b, 660.0) > 5.0 * goertzel(&b, 220.0),
        "expected the mapped control to follow bus 3 to 660 Hz"
    );
}

#[test]
fn n_mapa_dispatches_over_osc() {
    // `/n_mapa`/`/n_mapan` on a control parameter are no-ops, but must parse and dispatch cleanly -
    // this guards the dispatch entries and handlers. (The audio-bus behaviour is covered by the
    // `audio_control` integration test.)
    let (controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    let mut dispatcher = OscDispatcher::new(controller);
    dispatcher
        .apply_bytes(&osc("/d_recv", vec![OscType::Blob(sine_scgf())]))
        .expect("/d_recv");
    dispatcher
        .apply_bytes(&osc(
            "/s_new",
            vec![
                OscType::String("sine".to_string()),
                OscType::Int(1000),
                OscType::Int(1),
                OscType::Int(ROOT_GROUP_ID),
            ],
        ))
        .expect("/s_new");
    dispatcher
        .apply_bytes(&osc(
            "/n_mapa",
            vec![
                OscType::Int(1000),
                OscType::String("freq".to_string()),
                OscType::Int(0),
            ],
        ))
        .expect("/n_mapa");
    dispatcher
        .apply_bytes(&osc(
            "/n_mapan",
            vec![
                OscType::Int(1000),
                OscType::String("freq".to_string()),
                OscType::Int(0),
                OscType::Int(1),
            ],
        ))
        .expect("/n_mapan");
    let _ = render(&mut world, 64);
}

/// Build a one-channel engine wrapped in a dispatcher.
fn engine_1ch() -> (OscDispatcher, plyphon::Nrt, plyphon::World) {
    let (controller, nrt, world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    (OscDispatcher::new(controller), nrt, world)
}

/// Receive the `sine` def and start node 1000 at the root's tail.
fn start_sine(dispatcher: &mut OscDispatcher) {
    dispatcher
        .apply_bytes(&osc("/d_recv", vec![OscType::Blob(sine_scgf())]))
        .expect("/d_recv");
    dispatcher
        .apply_bytes(&osc(
            "/s_new",
            vec![
                OscType::String("sine".to_string()),
                OscType::Int(1000),
                OscType::Int(1),
                OscType::Int(ROOT_GROUP_ID),
            ],
        ))
        .expect("/s_new");
}

#[test]
fn n_run_pauses_and_resumes() {
    let (mut d, mut nrt, mut world) = engine_1ch();
    start_sine(&mut d);
    let a = render(&mut world, SR as usize / 4);
    assert!(
        goertzel(&a, 440.0) > 0.01,
        "synth should sound before pause"
    );

    // Pause: after flushing the in-flight block the synth is silent, and /n_off is notified.
    d.apply_bytes(&osc("/n_run", vec![OscType::Int(1000), OscType::Int(0)]))
        .expect("/n_run pause");
    let _ = render(&mut world, 1024);
    let paused = render(&mut world, SR as usize / 4);
    assert!(
        paused.iter().all(|s| s.abs() < 1e-6),
        "paused synth should be silent"
    );
    let msgs = drain_notifications(&mut d, &mut nrt);
    assert!(
        msgs.iter()
            .any(|m| m.addr == "/n_off" && m.args == vec![OscType::Int(1000)]),
        "expected /n_off 1000, got {msgs:?}"
    );

    // Resume: the tone returns and /n_on is notified.
    d.apply_bytes(&osc("/n_run", vec![OscType::Int(1000), OscType::Int(1)]))
        .expect("/n_run resume");
    let _ = render(&mut world, 512);
    let resumed = render(&mut world, SR as usize / 4);
    assert!(
        goertzel(&resumed, 440.0) > 0.01,
        "resumed synth should sound again"
    );
    let msgs = drain_notifications(&mut d, &mut nrt);
    assert!(
        msgs.iter()
            .any(|m| m.addr == "/n_on" && m.args == vec![OscType::Int(1000)]),
        "expected /n_on 1000, got {msgs:?}"
    );
}

#[test]
fn c_fill_drives_a_mapped_control() {
    let (mut d, _nrt, mut world) = engine_1ch();
    start_sine(&mut d);
    // Map freq to bus 5, then fill buses 5..8 with 330 in one command.
    d.apply_bytes(&osc(
        "/n_map",
        vec![
            OscType::Int(1000),
            OscType::String("freq".to_string()),
            OscType::Int(5),
        ],
    ))
    .expect("/n_map");
    d.apply_bytes(&osc(
        "/c_fill",
        vec![OscType::Int(5), OscType::Int(3), OscType::Float(330.0)],
    ))
    .expect("/c_fill");
    let _ = render(&mut world, 512);
    let a = render(&mut world, SR as usize / 2);
    assert!(
        goertzel(&a, 330.0) > 5.0 * goertzel(&a, 440.0),
        "mapped control should follow the filled bus to 330 Hz"
    );
}

#[test]
fn n_setn_and_n_fill_set_control_ranges() {
    let (mut d, _nrt, mut world) = engine_1ch();
    d.apply_bytes(&osc("/d_recv", vec![OscType::Blob(dual_sine_scgf())]))
        .expect("/d_recv");
    d.apply_bytes(&osc(
        "/s_new",
        vec![
            OscType::String("dual".to_string()),
            OscType::Int(1000),
            OscType::Int(1),
            OscType::Int(ROOT_GROUP_ID),
        ],
    ))
    .expect("/s_new");

    // /n_setn by name: set the two partials to 300 and 500 in one contiguous range.
    d.apply_bytes(&osc(
        "/n_setn",
        vec![
            OscType::Int(1000),
            OscType::String("freq".to_string()),
            OscType::Int(2),
            OscType::Float(300.0),
            OscType::Float(500.0),
        ],
    ))
    .expect("/n_setn");
    let _ = render(&mut world, 512);
    let a = render(&mut world, SR as usize / 2);
    assert!(
        goertzel(&a, 300.0) > 3.0 * goertzel(&a, 440.0)
            && goertzel(&a, 500.0) > 3.0 * goertzel(&a, 660.0),
        "n_setn should retune both partials to 300 and 500 Hz"
    );

    // /n_fill by index: fill the same range with 400, so both partials land on 400 Hz.
    d.apply_bytes(&osc(
        "/n_fill",
        vec![
            OscType::Int(1000),
            OscType::Int(0),
            OscType::Int(2),
            OscType::Float(400.0),
        ],
    ))
    .expect("/n_fill");
    let _ = render(&mut world, 512);
    let b = render(&mut world, SR as usize / 2);
    assert!(
        goertzel(&b, 400.0) > 5.0 * goertzel(&b, 300.0),
        "n_fill should collapse both partials onto 400 Hz"
    );
}

#[test]
fn error_mode_gates_fail_replies() {
    let (mut d, _nrt, _world) = engine_1ch();
    let zero_unknown = || osc("/b_zero", vec![OscType::Int(99)]);

    // Default: a failing command queues /fail.
    d.apply_bytes(&zero_unknown()).expect("/b_zero");
    assert!(
        take_msgs(&mut d).iter().any(|m| m.addr == "/fail"),
        "default error mode should queue /fail"
    );

    // /error 0 suppresses /fail; /error 1 restores it.
    d.apply_bytes(&osc("/error", vec![OscType::Int(0)]))
        .expect("/error 0");
    d.apply_bytes(&zero_unknown()).expect("/b_zero");
    assert!(
        !take_msgs(&mut d).iter().any(|m| m.addr == "/fail"),
        "/error 0 should suppress /fail"
    );
    d.apply_bytes(&osc("/error", vec![OscType::Int(1)]))
        .expect("/error 1");
    d.apply_bytes(&zero_unknown()).expect("/b_zero");
    assert!(
        take_msgs(&mut d).iter().any(|m| m.addr == "/fail"),
        "/error 1 should restore /fail"
    );

    // A bundle-local /error -2 suppresses within the bundle only - it must not leak past it.
    let bundle = OscPacket::Bundle(rosc::OscBundle {
        timetag: rosc::OscTime {
            seconds: 0,
            fractional: 1,
        },
        content: vec![
            OscPacket::Message(OscMessage {
                addr: "/error".to_string(),
                args: vec![OscType::Int(-2)],
            }),
            OscPacket::Message(OscMessage {
                addr: "/b_zero".to_string(),
                args: vec![OscType::Int(99)],
            }),
        ],
    });
    d.apply(&bundle).expect("error bundle");
    assert!(
        !take_msgs(&mut d).iter().any(|m| m.addr == "/fail"),
        "/error -2 should suppress /fail inside its bundle"
    );
    d.apply_bytes(&zero_unknown()).expect("/b_zero");
    assert!(
        take_msgs(&mut d).iter().any(|m| m.addr == "/fail"),
        "the bundle-local /error -2 must not leak past the bundle"
    );
}

#[test]
fn s_noid_detaches_name_resolution() {
    let (mut d, _nrt, mut world) = engine_1ch();
    start_sine(&mut d);
    let _ = render(&mut world, 512);

    d.apply_bytes(&osc("/s_noid", vec![OscType::Int(1000)]))
        .expect("/s_noid");
    // By-name control resolution now fails (the def association is gone)...
    assert!(
        d.apply_bytes(&osc(
            "/n_set",
            vec![
                OscType::Int(1000),
                OscType::String("freq".to_string()),
                OscType::Float(330.0),
            ],
        ))
        .is_err(),
        "/n_set by name should fail after /s_noid"
    );
    // ...but the still-running node remains addressable by control index.
    d.apply_bytes(&osc(
        "/n_set",
        vec![OscType::Int(1000), OscType::Int(0), OscType::Float(330.0)],
    ))
    .expect("/n_set by index");
    let _ = render(&mut world, 512);
    let a = render(&mut world, SR as usize / 2);
    assert!(
        goertzel(&a, 330.0) > 5.0 * goertzel(&a, 440.0),
        "index control should still retune the noid'd node to 330 Hz"
    );
}

#[test]
fn p_new_creates_an_addressable_group() {
    let (mut d, _nrt, mut world) = engine_1ch();
    d.apply_bytes(&osc("/d_recv", vec![OscType::Blob(sine_scgf())]))
        .expect("/d_recv");
    d.apply_bytes(&osc(
        "/p_new",
        vec![
            OscType::Int(2000),
            OscType::Int(1),
            OscType::Int(ROOT_GROUP_ID),
        ],
    ))
    .expect("/p_new");
    d.apply_bytes(&osc(
        "/s_new",
        vec![
            OscType::String("sine".to_string()),
            OscType::Int(1000),
            OscType::Int(1),
            OscType::Int(2000), // into the parallel group
        ],
    ))
    .expect("/s_new into p_new group");
    let a = render(&mut world, SR as usize / 2);
    assert!(
        goertzel(&a, 440.0) > 0.01,
        "a synth in the /p_new group should sound"
    );
}

#[test]
fn d_free_removes_a_def() {
    let (mut d, _nrt, _world) = engine_1ch();
    start_sine(&mut d); // /d_recv sine + /s_new 1000

    // Free the def; a later /s_new of it fails until it is re-sent.
    d.apply_bytes(&osc("/d_free", vec![OscType::String("sine".to_string())]))
        .expect("/d_free");
    assert!(
        d.apply_bytes(&osc(
            "/s_new",
            vec![
                OscType::String("sine".to_string()),
                OscType::Int(1001),
                OscType::Int(1),
                OscType::Int(ROOT_GROUP_ID),
            ],
        ))
        .is_err(),
        "/s_new of a freed def should fail"
    );

    // Re-receiving the def makes it usable again (the def slot is reused).
    d.apply_bytes(&osc("/d_recv", vec![OscType::Blob(sine_scgf())]))
        .expect("/d_recv again");
    d.apply_bytes(&osc(
        "/s_new",
        vec![
            OscType::String("sine".to_string()),
            OscType::Int(1002),
            OscType::Int(1),
            OscType::Int(ROOT_GROUP_ID),
        ],
    ))
    .expect("/s_new after re-recv");
}
