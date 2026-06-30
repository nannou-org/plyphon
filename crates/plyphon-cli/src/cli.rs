//! The `plyphon` command-line interface, defined with clap's derive API.
//!
//! `scsynth` selects its mode with flags (`-u`/`-t` for the real-time server, `-N` for offline
//! rendering); plyphon instead exposes one subcommand per mode - the more clap-idiomatic shape. The
//! engine-construction options (`--block-size`, bus counts, pool size, ...) are gathered in
//! [`EngineArgs`] and the real-time audio-device options in [`AudioArgs`], each flattened into the
//! subcommands that need them, so every mode shares one consistent flag set mapped onto
//! [`plyphon::Options`].

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

/// An scsynth-compatible OSC synthesis server and offline renderer. (The `about` text shown in
/// `--help` is taken from the crate's `description`; this doc comment is for readers of the code.)
#[derive(Debug, Parser)]
#[command(name = "plyphon", version, about, long_about = None, propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    /// Increase log verbosity; repeat for more detail (`-vv`).
    #[arg(short = 'v', long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    /// Silence all non-error output.
    #[arg(short = 'q', long, global = true, conflicts_with = "verbose")]
    pub quiet: bool,
}

/// The plyphon subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the real-time OSC server (UDP and/or TCP), like `scsynth`.
    Server(ServerArgs),
    /// Render a time-tagged OSC score offline to a WAV file (like `scsynth -N`).
    Render(RenderArgs),
    /// Play a time-tagged OSC score to the audio device in real time.
    Play(PlayArgs),
    /// List the available audio devices.
    Devices,
    /// Print a shell completion script to stdout.
    Completions(CompletionsArgs),
}

/// `plyphon server` - the real-time OSC server.
#[derive(Debug, Args)]
pub struct ServerArgs {
    #[command(flatten)]
    pub net: NetArgs,
    #[command(flatten)]
    pub audio: AudioArgs,
    #[command(flatten)]
    pub engine: EngineArgs,
}

/// `plyphon render` - offline (non-real-time) score rendering.
#[derive(Debug, Args)]
pub struct RenderArgs {
    /// The binary OSC score to render (`[i32 len][time-tagged bundle]...`).
    pub score: PathBuf,
    /// The WAV file to write.
    pub output: PathBuf,

    /// An optional input WAV fed frame-for-frame into the input buses for `In.ar`.
    #[arg(short = 'i', long)]
    pub input: Option<PathBuf>,
    /// Render sample rate in Hz (`scsynth -N`'s `<sample-rate>`; offline, so required).
    #[arg(short = 'S', long)]
    pub sample_rate: f64,
    /// Number of output channels to render.
    #[arg(short = 'o', long = "channels", default_value_t = 1)]
    pub channels: usize,
    /// Seconds of silence to render past the last scheduled command (lets tails ring out).
    #[arg(long, default_value_t = 0.5)]
    pub tail: f64,
    /// Output sample format (`scsynth -N`'s `<sample-format>`).
    #[arg(long = "sample-format", value_enum, default_value_t = SampleFormat::F32)]
    pub sample_format: SampleFormat,

    #[command(flatten)]
    pub engine: EngineArgs,
}

/// `plyphon play` - real-time score playback to the audio device.
#[derive(Debug, Args)]
pub struct PlayArgs {
    /// The binary OSC score to play (`[i32 len][time-tagged bundle]...`).
    pub score: PathBuf,
    /// Seconds to keep playing past the last scheduled command (lets tails ring out).
    #[arg(long, default_value_t = 0.5)]
    pub tail: f64,

    #[command(flatten)]
    pub audio: AudioArgs,
    #[command(flatten)]
    pub engine: EngineArgs,
}

/// `plyphon completions` - shell completion generation.
#[derive(Debug, Args)]
pub struct CompletionsArgs {
    /// The shell to generate a completion script for.
    #[arg(value_enum)]
    pub shell: clap_complete::Shell,
}

/// Network options for the real-time server (`scsynth -u`/`-t`).
#[derive(Debug, Args)]
pub struct NetArgs {
    /// UDP port to listen on (`scsynth -u`). SuperCollider's conventional `scsynth` port.
    #[arg(short = 'u', long, default_value_t = 57110)]
    pub udp_port: u16,
    /// Also listen for length-prefixed OSC over TCP on this port (`scsynth -t`).
    #[arg(short = 't', long)]
    pub tcp_port: Option<u16>,
    /// Address to bind the listening sockets to.
    #[arg(long, default_value = "127.0.0.1")]
    pub bind: std::net::IpAddr,
}

/// Real-time audio-device options, shared by `server` and `play`.
#[derive(Debug, Args)]
pub struct AudioArgs {
    /// Output audio device name (`scsynth -H`); the host default if omitted.
    #[arg(short = 'H', long)]
    pub device: Option<String>,
    /// Request a specific sample rate in Hz (`scsynth -S`); the device default if omitted.
    #[arg(short = 'S', long)]
    pub sample_rate: Option<f64>,
    /// Number of output channels (`scsynth -o`); the device default if omitted.
    #[arg(short = 'o', long)]
    pub output_channels: Option<usize>,
    /// Number of input channels (`scsynth -i`); none if omitted. A non-zero value enables live
    /// hardware-input capture for `In.ar` (server only).
    #[arg(short = 'i', long, default_value_t = 0)]
    pub input_channels: usize,
    /// Input audio device name; the host default input device if omitted. Used only when
    /// `--input-channels` > 0.
    #[arg(short = 'I', long = "input-device")]
    pub input_device: Option<String>,
    /// Input jitter-buffer depth in control blocks - the latency/underrun trade-off for bridging the
    /// independent input and output clocks (cpal has no duplex stream). Used only when
    /// `--input-channels` > 0.
    #[arg(long = "input-latency-blocks", default_value_t = 3)]
    pub input_latency_blocks: usize,
    /// Request a fixed hardware buffer size in frames (`scsynth -Z`).
    #[arg(short = 'Z', long)]
    pub hardware_buffer_size: Option<u32>,
}

/// Engine-construction options, mapped onto [`plyphon::Options`]. Defaults match scsynth's.
#[derive(Debug, Args)]
pub struct EngineArgs {
    /// Samples per control block (`scsynth -z`).
    #[arg(short = 'z', long, default_value_t = 64)]
    pub block_size: usize,
    /// Number of private audio bus channels for routing between synths (`scsynth -a`).
    #[arg(short = 'a', long = "audio-buses", default_value_t = 128)]
    pub audio_buses: usize,
    /// Number of control bus channels (`scsynth -c`).
    #[arg(short = 'c', long = "control-buses", default_value_t = 4096)]
    pub control_buses: usize,
    /// Maximum number of live nodes (`scsynth -n`).
    #[arg(short = 'n', long, default_value_t = 1024)]
    pub max_nodes: usize,
    /// Number of buffer table slots (`scsynth -b`).
    #[arg(short = 'b', long = "buffers", default_value_t = 1024)]
    pub max_buffers: usize,
    /// Number of synthdef table slots (`scsynth -d`).
    #[arg(short = 'd', long = "max-synthdefs", default_value_t = 1024)]
    pub max_synthdefs: usize,
    /// Real-time memory pool size in KiB (`scsynth -m`).
    #[arg(short = 'm', long = "rt-memory", default_value_t = 8192)]
    pub rt_memory_kib: usize,
    /// Load every `.scsyndef` SynthDef in this directory at startup (adapts `scsynth -D`).
    #[arg(long = "load-dir")]
    pub load_dir: Option<PathBuf>,
}

/// The output WAV sample format (`scsynth -N`'s `<sample-format>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SampleFormat {
    /// 32-bit float (scsynth's default; lossless, but some basic players reject it).
    F32,
    /// 16-bit signed PCM (the most widely compatible).
    I16,
    /// 24-bit signed PCM.
    I24,
}
