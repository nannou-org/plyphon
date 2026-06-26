//! `plyphon server` - the real-time OSC server, behaving like `scsynth`.
//!
//! The engine's `World` plays on the audio thread (a cpal output stream). This control thread owns
//! the [`OscDispatcher`] (wrapping the `Controller`) and the [`Nrt`], and pumps a single channel fed
//! by the UDP/TCP listener threads (see [`crate::transport`]). Each loop it: applies received OSC,
//! sends each command's replies back to its sender, surfaces node-lifecycle [`plyphon::Event`]s as
//! `/n_go`/`/n_end`/... to clients that registered via `/notify`, and services queued buffer loads.
//!
//! `scsynth` splits "server commands" from "engine commands"; so do we - `/notify`, `/dumpOSC`,
//! `/version`, and `/quit` (the ones that concern the connection/process, not the engine) are handled
//! here; everything else is forwarded to the dispatcher. The engine-state getters (`/status`,
//! `/sync`, `/c_get`, `/n_query`, …) and the async buffer loads answer *later*, so the server tags the
//! dispatcher with the issuing client (a [`ReplyTarget::Requester`] handle) before each `apply`; the
//! dispatcher stamps every reply it produces with that tag and replays it across the async gap. The
//! server then routes purely by the tag (see [`flush_replies`]): `Requester` to the one client that
//! asked, `Broadcast` to every `/notify` subscriber. No reply is classified by inspecting its address.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::net::{TcpStream, UdpSocket};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use plyphon::{Nrt, engine};
use plyphon_osc::{OscDispatcher, ReplyTarget};
use rosc::{OscMessage, OscPacket, OscType};

use crate::audio;
use crate::bufsource::{FsSource, block_on};
use crate::cli::ServerArgs;
use crate::defs::load_dir;
use crate::options::engine_options;
use crate::transport::{self, Client, FromNet, Transport};

/// Control-loop cadence: how often to service NRT cleanup and flush notifications when idle.
const TICK: Duration = Duration::from_millis(5);

/// The server's control-side state, owned and ticked by the single control thread.
struct Server {
    dispatcher: OscDispatcher,
    nrt: Nrt,
    /// Cloned UDP socket used to send replies/notifications to UDP clients.
    udp: UdpSocket,
    /// Per-connection TCP writers for sending replies to TCP clients.
    tcp_writers: HashMap<u64, TcpStream>,
    /// Clients that registered for node notifications via `/notify 1`.
    notified: HashSet<Client>,
    /// Whether to log incoming OSC (`/dumpOSC`).
    dump_osc: bool,
    /// Maps each known client to its stable [`ReplyTarget::Requester`] handle (assigned by `token_for`)
    /// so the dispatcher can tag replies with an opaque id; `token_clients` is the reverse map used to
    /// route a tagged reply back. UDP entries persist for the run (clients are few); TCP entries drop on
    /// disconnect.
    client_tokens: HashMap<Client, u64>,
    token_clients: HashMap<u64, Client>,
    next_token: u64,
}

pub fn run(args: ServerArgs) -> Result<(), String> {
    let audio = audio::resolve(&args.audio)?;
    let options = engine_options(
        &args.engine,
        audio.sample_rate,
        audio.channels,
        args.audio.input_channels,
    );
    let (mut controller, nrt, world) = engine(options);
    if let Some(dir) = &args.engine.load_dir {
        let count = load_dir(&mut controller, dir)?;
        eprintln!("loaded {count} synthdef(s) from {}", dir.display());
    }
    let dispatcher = OscDispatcher::with_buffer_source(controller, Box::new(FsSource));

    // The World plays on the audio thread (output-only for v1); keep the stream alive for the run.
    let mut world = world;
    let _stream = audio::play_output(&audio, move |out, channels| world.fill(out, channels))?;

    let Transport { events, udp } =
        transport::start(args.net.bind, args.net.udp_port, args.net.tcp_port)?;
    eprintln!(
        "plyphon server: UDP {}:{}{}",
        args.net.bind,
        args.net.udp_port,
        match args.net.tcp_port {
            Some(port) => format!(", TCP {}:{port}", args.net.bind),
            None => String::new(),
        }
    );
    eprintln!(
        "audio: {} ch @ {} Hz ({})",
        audio.channels, audio.sample_rate, audio.sample_format
    );

    let mut server = Server {
        dispatcher,
        nrt,
        udp,
        tcp_writers: HashMap::new(),
        notified: HashSet::new(),
        dump_osc: false,
        client_tokens: HashMap::new(),
        token_clients: HashMap::new(),
        next_token: 0,
    };
    control_loop(&mut server, &events);
    Ok(())
}

/// Drive the server until a client sends `/quit` (or the listeners hang up).
fn control_loop(server: &mut Server, events: &Receiver<FromNet>) {
    loop {
        match events.recv_timeout(TICK) {
            Ok(FromNet::Packet { client, bytes }) => {
                if handle_packet(server, client, &bytes) {
                    break;
                }
            }
            Ok(FromNet::TcpConnected { id, writer }) => {
                server.tcp_writers.insert(id, writer);
            }
            Ok(FromNet::TcpDisconnected { id }) => {
                server.tcp_writers.remove(&id);
                let client = Client::Tcp(id);
                server.notified.remove(&client);
                if let Some(token) = server.client_tokens.remove(&client) {
                    server.token_clients.remove(&token);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
        service(server);
    }
    // Flush any final replies (e.g. the `/quit` ack and last notifications).
    service(server);
}

/// Decode and act on one received packet. Returns `true` if it was `/quit` (stop the server).
fn handle_packet(server: &mut Server, client: Client, bytes: &[u8]) -> bool {
    let packet = match rosc::decoder::decode_udp(bytes) {
        Ok((_, packet)) => packet,
        Err(err) => {
            eprintln!("warning: bad OSC packet from {client:?}: {err}");
            return false;
        }
    };
    if server.dump_osc {
        eprintln!("recv {client:?}: {packet:?}");
    }

    // Server-level commands are handled here, not by the engine front-end.
    if let OscPacket::Message(message) = &packet {
        match message.addr.as_str() {
            "/quit" => {
                reply(server, client, message_packet("/done", vec![]));
                return true;
            }
            "/notify" => {
                handle_notify(server, client, &message.args);
                return false;
            }
            "/dumpOSC" => {
                handle_dump(server, client, &message.args);
                return false;
            }
            "/version" => {
                handle_version(server, client);
                return false;
            }
            // `/status`, `/sync`, `/rtMemoryStatus` are engine-state queries: forwarded to the
            // dispatcher and routed back asynchronously, below.
            _ => {}
        }
    }

    // Everything else is an engine command. Tag the dispatcher with this client so every reply it
    // produces - now (a synchronous command's replies) or later (a getter answer, a load's `/done`, or
    // anything a load's completion message emits) - is stamped for this requester and routed back by
    // `flush_replies`. Notifications the engine raises are tagged `Broadcast` by the dispatcher itself.
    let token = token_for(server, client);
    server
        .dispatcher
        .set_reply_target(ReplyTarget::Requester(token));
    if let Err(err) = server.dispatcher.apply(&packet) {
        reply(
            server,
            client,
            fail_packet(command_addr(&packet), &err.to_string()),
        );
        return false;
    }
    flush_replies(server);
    false
}

/// The stable routing handle for `client`, assigned on first contact and reused thereafter, so the
/// dispatcher can tag replies with an opaque id the server maps back to a client in `flush_replies`.
fn token_for(server: &mut Server, client: Client) -> u64 {
    if let Some(&token) = server.client_tokens.get(&client) {
        return token;
    }
    let token = server.next_token;
    server.next_token += 1;
    server.client_tokens.insert(client, token);
    server.token_clients.insert(token, client);
    token
}

/// Run NRT cleanup, surface node notifications and async query/load answers, then route every queued
/// reply by its tag (see [`flush_replies`]).
fn service(server: &mut Server) {
    server.nrt.process();
    while let Some(event) = server.nrt.poll() {
        server.dispatcher.notify(event);
    }
    while let Some(trigger) = server.nrt.poll_trigger() {
        server.dispatcher.notify_trigger(trigger);
    }
    while let Some(reply) = server.nrt.poll_reply() {
        server.dispatcher.reply(reply);
    }
    // Buffer loads queued by `apply` (`/b_allocRead`/`/b_read`); the fs source is ready at once.
    block_on(server.dispatcher.run_pending());

    flush_replies(server);
}

/// Send each queued dispatcher reply to its tagged destination: `Broadcast` to every `/notify`
/// subscriber, `Requester` to the one client the handle names (scsynth's stored reply address). A
/// handle with no live client (one that vanished mid-flight) is logged and dropped.
fn flush_replies(server: &mut Server) {
    for (target, packet) in server.dispatcher.take_replies_targeted() {
        match target {
            ReplyTarget::Broadcast => {
                for client in server.notified.iter().copied().collect::<Vec<_>>() {
                    send(&server.udp, &server.tcp_writers, client, &packet);
                }
            }
            ReplyTarget::Requester(token) => match server.token_clients.get(&token).copied() {
                Some(client) => send(&server.udp, &server.tcp_writers, client, &packet),
                None => eprintln!("warning: reply for unknown client token {token}: {packet:?}"),
            },
        }
    }
}

/// Register (`flag != 0`) or unregister the client for node notifications, then ack.
fn handle_notify(server: &mut Server, client: Client, args: &[OscType]) {
    if int_arg(args.first()).unwrap_or(0) != 0 {
        server.notified.insert(client);
    } else {
        server.notified.remove(&client);
    }
    // scsynth replies `/done /notify <clientID> <maxLogins>`; plyphon has no logins, so report id 0.
    reply(
        server,
        client,
        message_packet(
            "/done",
            vec![OscType::String("/notify".to_string()), OscType::Int(0)],
        ),
    );
}

/// Reply with `/version.reply`: program name, major, minor, patch, branch, commit. The server is the
/// program scsynth's `/version` describes, so plyphon reports its own identity here (branch/commit
/// are left empty - the build is not coupled to git).
fn handle_version(server: &Server, client: Client) {
    let major = env!("CARGO_PKG_VERSION_MAJOR").parse::<i32>().unwrap_or(0);
    let minor = env!("CARGO_PKG_VERSION_MINOR").parse::<i32>().unwrap_or(0);
    reply(
        server,
        client,
        message_packet(
            "/version.reply",
            vec![
                OscType::String("plyphon".to_string()),
                OscType::Int(major),
                OscType::Int(minor),
                OscType::String(env!("CARGO_PKG_VERSION_PATCH").to_string()),
                OscType::String(String::new()),
                OscType::String(String::new()),
            ],
        ),
    );
}

/// Toggle OSC logging and ack.
fn handle_dump(server: &mut Server, client: Client, args: &[OscType]) {
    server.dump_osc = int_arg(args.first()).unwrap_or(0) != 0;
    reply(
        server,
        client,
        message_packet("/done", vec![OscType::String("/dumpOSC".to_string())]),
    );
}

/// Send `packet` to one client.
fn reply(server: &Server, client: Client, packet: OscPacket) {
    send(&server.udp, &server.tcp_writers, client, &packet);
}

/// Encode and send `packet` to `client` over its transport (UDP datagram or length-prefixed TCP).
fn send(
    udp: &UdpSocket,
    tcp_writers: &HashMap<u64, TcpStream>,
    client: Client,
    packet: &OscPacket,
) {
    match client {
        Client::Udp(addr) => {
            if let Ok(bytes) = rosc::encoder::encode(packet) {
                let _ = udp.send_to(&bytes, addr);
            }
        }
        Client::Tcp(id) => {
            if let Some(mut writer) = tcp_writers.get(&id)
                && let Ok(bytes) = rosc::encoder::encode_tcp(packet)
            {
                let _ = writer.write_all(&bytes);
            }
        }
    }
}

/// An OSC message packet from an address and args.
fn message_packet(addr: &str, args: Vec<OscType>) -> OscPacket {
    OscPacket::Message(OscMessage {
        addr: addr.to_string(),
        args,
    })
}

/// A `/fail` reply naming the failed command and the error.
fn fail_packet(command: &str, error: &str) -> OscPacket {
    message_packet(
        "/fail",
        vec![
            OscType::String(command.to_string()),
            OscType::String(error.to_string()),
        ],
    )
}

/// The command address used to label a `/fail`.
fn command_addr(packet: &OscPacket) -> &str {
    match packet {
        OscPacket::Message(message) => &message.addr,
        OscPacket::Bundle(_) => "/bundle",
    }
}

/// The `i32` value of an OSC argument, if it is an int.
fn int_arg(arg: Option<&OscType>) -> Option<i32> {
    match arg {
        Some(OscType::Int(value)) => Some(*value),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::Instant;

    use plyphon::{InputRef, Options, Param, Rate, SynthDef, UnitSpec};

    use super::*;
    use crate::transport::Transport;

    /// A self-freeing note: `SinOsc.ar(freq) * Line.kr(0.2, 0, 0.05, doneAction: 2) -> Out`.
    fn note_def() -> SynthDef {
        SynthDef {
            name: "note".to_string(),
            params: vec![Param {
                name: "freq".to_string(),
                default: 440.0,
            }],
            units: vec![
                UnitSpec {
                    name: "Line".to_string(),
                    rate: Rate::Control,
                    inputs: vec![
                        InputRef::Constant(0.2),
                        InputRef::Constant(0.0),
                        InputRef::Constant(0.05),
                        InputRef::Constant(2.0), // doneAction 2 = free the synth
                    ],
                    num_outputs: 1,
                    special_index: 0,
                },
                UnitSpec::new(
                    "SinOsc",
                    Rate::Audio,
                    vec![InputRef::Param(0), InputRef::Constant(0.0)],
                    1,
                ),
                UnitSpec {
                    name: "BinaryOpUGen".to_string(),
                    rate: Rate::Audio,
                    inputs: vec![
                        InputRef::Unit { unit: 1, output: 0 },
                        InputRef::Unit { unit: 0, output: 0 },
                    ],
                    num_outputs: 1,
                    special_index: 2, // multiply
                },
                UnitSpec::new(
                    "Out",
                    Rate::Audio,
                    vec![
                        InputRef::Constant(0.0),
                        InputRef::Unit { unit: 2, output: 0 },
                    ],
                    0,
                ),
            ],
        }
    }

    fn send_msg(sock: &UdpSocket, to: SocketAddr, addr: &str, args: Vec<OscType>) {
        let packet = message_packet(addr, args);
        let bytes = rosc::encoder::encode(&packet).unwrap();
        sock.send_to(&bytes, to).unwrap();
    }

    /// Drive an `/s_new` over real UDP (with the `World` filled on a stand-in audio thread) and
    /// confirm the self-freeing note's `/n_go` and `/n_end` come back to a registered client.
    #[test]
    fn s_new_over_udp_reports_node_lifecycle() {
        let options = Options {
            sample_rate: 48_000.0,
            output_channels: 1,
            input_channels: 0,
            ..Options::default()
        };
        let (mut controller, nrt, mut world) = plyphon::engine(options);
        controller.add_synthdef(note_def());
        let dispatcher = OscDispatcher::with_buffer_source(controller, Box::new(FsSource));

        // Drive the World on a background thread, standing in for the audio callback.
        let stop = Arc::new(AtomicBool::new(false));
        let stop_driver = stop.clone();
        let driver = thread::spawn(move || {
            let mut buf = vec![0f32; 64];
            while !stop_driver.load(Ordering::Relaxed) {
                world.fill(&mut buf, 1);
                thread::sleep(Duration::from_micros(500));
            }
        });

        // Server transport on an ephemeral UDP port. The `OscDispatcher` is deliberately `!Send`
        // (its buffer source need not be), so the control loop stays on this thread - exactly as the
        // real `run` does - and the client drives it from a spawned thread instead.
        let Transport { events, udp } =
            transport::start("127.0.0.1".parse().unwrap(), 0, None).unwrap();
        let server_addr = udp.local_addr().unwrap();
        let mut server = Server {
            dispatcher,
            nrt,
            udp,
            tcp_writers: HashMap::new(),
            notified: HashSet::new(),
            dump_osc: false,
            client_tokens: HashMap::new(),
            token_clients: HashMap::new(),
            next_token: 0,
        };

        let client = thread::spawn(move || {
            let sock = UdpSocket::bind("127.0.0.1:0").unwrap();
            sock.set_read_timeout(Some(Duration::from_millis(200)))
                .unwrap();
            // Register for notifications, then (once registration has landed) start a note.
            send_msg(&sock, server_addr, "/notify", vec![OscType::Int(1)]);
            thread::sleep(Duration::from_millis(50));
            send_msg(
                &sock,
                server_addr,
                "/s_new",
                vec![
                    OscType::String("note".to_string()),
                    OscType::Int(1000),
                    OscType::Int(0), // addAction: head
                    OscType::Int(0), // target: root group
                ],
            );

            let (mut got_go, mut got_end) = (false, false);
            let mut buf = [0u8; 65_536];
            let deadline = Instant::now() + Duration::from_secs(3);
            while (!got_go || !got_end) && Instant::now() < deadline {
                if let Ok(n) = sock.recv(&mut buf)
                    && let Ok((_, OscPacket::Message(message))) =
                        rosc::decoder::decode_udp(&buf[..n])
                {
                    match message.addr.as_str() {
                        "/n_go" => got_go = true,
                        "/n_end" => got_end = true,
                        _ => {}
                    }
                }
            }
            // Stop the server's control loop, then report what we observed.
            send_msg(&sock, server_addr, "/quit", vec![]);
            (got_go, got_end)
        });

        // Run the control loop here; it returns when the client sends `/quit`.
        control_loop(&mut server, &events);
        let (got_go, got_end) = client.join().unwrap();
        stop.store(true, Ordering::Relaxed);
        driver.join().unwrap();

        assert!(got_go, "expected an /n_go node-start notification");
        assert!(got_end, "expected an /n_end node-end notification");
    }

    /// `/version` over UDP returns a `/version.reply` identifying the plyphon server.
    #[test]
    fn version_over_udp_replies() {
        let options = Options {
            sample_rate: 48_000.0,
            output_channels: 1,
            input_channels: 0,
            ..Options::default()
        };
        let (controller, nrt, _world) = plyphon::engine(options);
        let dispatcher = OscDispatcher::with_buffer_source(controller, Box::new(FsSource));

        let Transport { events, udp } =
            transport::start("127.0.0.1".parse().unwrap(), 0, None).unwrap();
        let server_addr = udp.local_addr().unwrap();
        let mut server = Server {
            dispatcher,
            nrt,
            udp,
            tcp_writers: HashMap::new(),
            notified: HashSet::new(),
            dump_osc: false,
            client_tokens: HashMap::new(),
            token_clients: HashMap::new(),
            next_token: 0,
        };

        let client = thread::spawn(move || {
            let sock = UdpSocket::bind("127.0.0.1:0").unwrap();
            sock.set_read_timeout(Some(Duration::from_millis(200)))
                .unwrap();
            send_msg(&sock, server_addr, "/version", vec![]);

            let mut reply = None;
            let mut buf = [0u8; 65_536];
            let deadline = Instant::now() + Duration::from_secs(3);
            while reply.is_none() && Instant::now() < deadline {
                if let Ok(n) = sock.recv(&mut buf)
                    && let Ok((_, OscPacket::Message(message))) =
                        rosc::decoder::decode_udp(&buf[..n])
                    && message.addr == "/version.reply"
                {
                    reply = Some(message);
                }
            }
            // Stop the server's control loop, then report what came back.
            send_msg(&sock, server_addr, "/quit", vec![]);
            reply
        });

        control_loop(&mut server, &events);
        let reply = client.join().unwrap().expect("expected a /version.reply");
        assert_eq!(reply.args.len(), 6, "version reply has six fields");
        assert_eq!(reply.args[0], OscType::String("plyphon".to_string()));
    }

    /// Build a server + a background `World` driver on an ephemeral UDP port.
    fn server_with_driver() -> (
        Server,
        Receiver<FromNet>,
        SocketAddr,
        Arc<AtomicBool>,
        thread::JoinHandle<()>,
    ) {
        let options = Options {
            sample_rate: 48_000.0,
            output_channels: 1,
            input_channels: 0,
            ..Options::default()
        };
        let (mut controller, nrt, mut world) = plyphon::engine(options);
        controller.add_synthdef(note_def());
        let dispatcher = OscDispatcher::with_buffer_source(controller, Box::new(FsSource));

        let stop = Arc::new(AtomicBool::new(false));
        let stop_driver = stop.clone();
        let driver = thread::spawn(move || {
            let mut buf = vec![0f32; 64];
            while !stop_driver.load(Ordering::Relaxed) {
                world.fill(&mut buf, 1);
                thread::sleep(Duration::from_micros(500));
            }
        });

        let Transport { events, udp } =
            transport::start("127.0.0.1".parse().unwrap(), 0, None).unwrap();
        let server_addr = udp.local_addr().unwrap();
        let server = Server {
            dispatcher,
            nrt,
            udp,
            tcp_writers: HashMap::new(),
            notified: HashSet::new(),
            dump_osc: false,
            client_tokens: HashMap::new(),
            token_clients: HashMap::new(),
            next_token: 0,
        };
        (server, events, server_addr, stop, driver)
    }

    fn client_socket() -> UdpSocket {
        let sock = UdpSocket::bind("127.0.0.1:0").unwrap();
        sock.set_read_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        sock
    }

    /// A getter (`/c_set` then `/c_get`) over UDP answers the *requesting* client.
    #[test]
    fn c_get_over_udp_routes_to_requester() {
        let (mut server, events, server_addr, stop, driver) = server_with_driver();
        let client = thread::spawn(move || {
            let sock = client_socket();
            send_msg(
                &sock,
                server_addr,
                "/c_set",
                vec![OscType::Int(5), OscType::Float(0.5)],
            );
            send_msg(&sock, server_addr, "/c_get", vec![OscType::Int(5)]);

            let mut reply = None;
            let mut buf = [0u8; 65_536];
            let deadline = Instant::now() + Duration::from_secs(3);
            while reply.is_none() && Instant::now() < deadline {
                if let Ok(n) = sock.recv(&mut buf)
                    && let Ok((_, OscPacket::Message(m))) = rosc::decoder::decode_udp(&buf[..n])
                    && m.addr == "/c_set"
                {
                    reply = Some(m);
                }
            }
            send_msg(&sock, server_addr, "/quit", vec![]);
            reply
        });

        control_loop(&mut server, &events);
        let reply = client.join().unwrap().expect("/c_set reply");
        stop.store(true, Ordering::Relaxed);
        driver.join().unwrap();
        assert_eq!(reply.args, vec![OscType::Int(5), OscType::Float(0.5)]);
    }

    /// An async load whose completion message itself emits a reply (`/b_query` -> `/b_info`) routes
    /// that reply back to the loader. Before destination-tagging the server dropped it ("getter reply
    /// with no pending requester"); now the dispatcher tags it for the loader and `flush_replies`
    /// delivers it.
    #[test]
    fn completion_b_query_routes_to_the_loader() {
        // A tiny valid WAV the FsSource can read back.
        let path =
            std::env::temp_dir().join(format!("plyphon_completion_{}.wav", std::process::id()));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..240 {
            writer
                .write_sample(((i as f32 * 0.1).sin() * 1000.0) as i16)
                .unwrap();
        }
        writer.finalize().unwrap();
        let wav_path = path.to_str().unwrap().to_string();

        let (mut server, events, server_addr, stop, driver) = server_with_driver();
        let client = thread::spawn(move || {
            let sock = client_socket();
            // The completion queries the just-loaded buffer; its /b_info must come back to us.
            let completion =
                rosc::encoder::encode(&message_packet("/b_query", vec![OscType::Int(0)])).unwrap();
            send_msg(
                &sock,
                server_addr,
                "/b_allocRead",
                vec![
                    OscType::Int(0),
                    OscType::String(wav_path),
                    OscType::Blob(completion),
                ],
            );

            let mut info = None;
            let mut buf = [0u8; 65_536];
            let deadline = Instant::now() + Duration::from_secs(3);
            while info.is_none() && Instant::now() < deadline {
                if let Ok(n) = sock.recv(&mut buf)
                    && let Ok((_, OscPacket::Message(m))) = rosc::decoder::decode_udp(&buf[..n])
                    && m.addr == "/b_info"
                {
                    info = Some(m);
                }
            }
            send_msg(&sock, server_addr, "/quit", vec![]);
            info
        });

        control_loop(&mut server, &events);
        let info = client
            .join()
            .unwrap()
            .expect("/b_info from the completion routed to the loader");
        stop.store(true, Ordering::Relaxed);
        driver.join().unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            info.args.first(),
            Some(&OscType::Int(0)),
            "/b_info for buffer 0"
        );
    }

    /// A getter reply reaches only the requester, while node notifications reach only `/notify`
    /// subscribers - the two never cross.
    #[test]
    fn getter_reply_and_notifications_route_separately() {
        let (mut server, events, server_addr, stop, driver) = server_with_driver();

        // Client A registers for notifications and collects what it receives.
        let a = thread::spawn(move || {
            let sock = client_socket();
            send_msg(&sock, server_addr, "/notify", vec![OscType::Int(1)]);
            let (mut saw_go, mut saw_cset) = (false, false);
            let mut buf = [0u8; 65_536];
            let deadline = Instant::now() + Duration::from_millis(800);
            while Instant::now() < deadline {
                if let Ok(n) = sock.recv(&mut buf)
                    && let Ok((_, OscPacket::Message(m))) = rosc::decoder::decode_udp(&buf[..n])
                {
                    match m.addr.as_str() {
                        "/n_go" => saw_go = true,
                        "/c_set" => saw_cset = true,
                        _ => {}
                    }
                }
            }
            send_msg(&sock, server_addr, "/quit", vec![]);
            (saw_go, saw_cset)
        });

        // Client B (not registered) issues a getter and starts a self-freeing note.
        let b = thread::spawn(move || {
            let sock = client_socket();
            thread::sleep(Duration::from_millis(100)); // let A register first
            send_msg(&sock, server_addr, "/c_get", vec![OscType::Int(0)]);
            send_msg(
                &sock,
                server_addr,
                "/s_new",
                vec![
                    OscType::String("note".to_string()),
                    OscType::Int(1000),
                    OscType::Int(0),
                    OscType::Int(0),
                ],
            );
            let (mut saw_go, mut saw_cset) = (false, false);
            let mut buf = [0u8; 65_536];
            let deadline = Instant::now() + Duration::from_millis(600);
            while Instant::now() < deadline {
                if let Ok(n) = sock.recv(&mut buf)
                    && let Ok((_, OscPacket::Message(m))) = rosc::decoder::decode_udp(&buf[..n])
                {
                    match m.addr.as_str() {
                        "/n_go" => saw_go = true,
                        "/c_set" => saw_cset = true,
                        _ => {}
                    }
                }
            }
            (saw_go, saw_cset)
        });

        control_loop(&mut server, &events);
        let (a_go, a_cset) = a.join().unwrap();
        let (b_go, b_cset) = b.join().unwrap();
        stop.store(true, Ordering::Relaxed);
        driver.join().unwrap();

        assert!(
            a_go,
            "registered client A should receive the /n_go broadcast"
        );
        assert!(!a_cset, "client A must NOT receive client B's getter reply");
        assert!(b_cset, "requester client B should receive its /c_set reply");
        assert!(
            !b_go,
            "unregistered client B must NOT receive node notifications"
        );
    }
}
