//! Load a sample through a `BufferSource` and play it with `PlayBuf`, via cpal, native and web.
//!
//! This is the downstream side of plyphon's buffer model: the engine only installs finished
//! buffers, so *loading* sample data is the application's job, expressed by implementing
//! [`plyphon_buffers::BufferSource`]. The same checked-in `assets/tone.wav` is read the way each
//! platform actually reads a bundled asset - from the filesystem natively, and over HTTP (`fetch`)
//! on the web - decoded with `hound`. Only the body of `load` differs between the two.
//!
//! Loading is async, so the example is *build-then-load*: the (initially silent) audio stream starts
//! immediately, the sample is loaded off to the side, and the `PlayBuf` synth is started once the
//! buffer is installed. Natively the filesystem read resolves at once (driven by a small `block_on`);
//! on the web the `fetch` genuinely awaits, driven by `spawn_local`.
//!
//! To run the web build, serve it with the asset, e.g.
//! `trunk serve crates/examples/sampler/web/index.html`.

use std::io::Cursor;

use plyphon::{
    AddAction, Controller, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World,
    engine,
};
use plyphon_buffers::{BufFuture, BufferData, BufferSource, LoadError, ReadRegion};

/// The asset key, resolved per platform (a path under `assets/` natively, a URL on the web).
const SAMPLE: &str = "tone.wav";
/// A gentle master gain.
const GAIN: f32 = 0.5;

/// Native [`BufferSource`]: reads the asset from the crate's `assets/` directory.
#[cfg(not(target_arch = "wasm32"))]
struct FsSource;

#[cfg(not(target_arch = "wasm32"))]
impl BufferSource for FsSource {
    fn load<'a>(
        &'a self,
        key: &'a str,
        _region: ReadRegion,
    ) -> BufFuture<'a, Result<BufferData, LoadError>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join(key);
        let result = std::fs::read(&path)
            .map_err(|e| LoadError::Io(e.to_string()))
            .and_then(|bytes| decode_wav(&bytes));
        Box::pin(async move { result })
    }
}

/// Web [`BufferSource`]: fetches the asset over HTTP from the page's own origin.
#[cfg(target_arch = "wasm32")]
struct FetchSource;

#[cfg(target_arch = "wasm32")]
impl BufferSource for FetchSource {
    fn load<'a>(
        &'a self,
        key: &'a str,
        _region: ReadRegion,
    ) -> BufFuture<'a, Result<BufferData, LoadError>> {
        let url = key.to_string();
        Box::pin(async move {
            let response = gloo_net::http::Request::get(&url)
                .send()
                .await
                .map_err(|e| LoadError::Io(e.to_string()))?;
            let bytes = response
                .binary()
                .await
                .map_err(|e| LoadError::Io(e.to_string()))?;
            decode_wav(&bytes)
        })
    }
}

/// Decode WAV bytes (any bit depth, PCM or float) into interleaved `f32` samples, using `hound`.
fn decode_wav(bytes: &[u8]) -> Result<BufferData, LoadError> {
    let reader =
        hound::WavReader::new(Cursor::new(bytes)).map_err(|e| LoadError::Decode(e.to_string()))?;
    let spec = reader.spec();
    let decode = |e: hound::Error| LoadError::Decode(e.to_string());
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(decode)?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()
                .map_err(decode)?
        }
    };
    Ok(BufferData {
        samples,
        num_channels: spec.channels.max(1) as usize,
        sample_rate: spec.sample_rate as f64,
    })
}

/// Load the sample through `source`, install it, and start a looping `PlayBuf` for it.
async fn load_and_play(
    mut controller: Controller,
    source: impl BufferSource,
    engine_sample_rate: f32,
    channels: usize,
) {
    match source.load(SAMPLE, ReadRegion::all()).await {
        Ok(data) => {
            // Play at the sample's natural pitch on any device: scale the play rate by the ratio of
            // the buffer's sample rate to the engine's (PlayBuf advances in buffer frames per sample).
            let rate = (data.sample_rate / engine_sample_rate as f64) as f32;
            let _ = controller.buffer_set(0, Box::new(data.into()));
            controller.add_synthdef(player_def(channels, rate));
            let _ = controller.synth_new("player", ROOT_GROUP_ID, AddAction::Tail);
        }
        // On the web this prints to nowhere; a real app would log via the console.
        Err(err) => eprintln!("failed to load {SAMPLE}: {err}"),
    }
}

/// `PlayBuf.ar(1, bufnum = 0, rate, loop: 1) -> Out`, copied to every channel.
fn player_def(channels: usize, rate: f32) -> SynthDef {
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 0, output: 0 });
    }
    SynthDef {
        name: "player".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "PlayBuf",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),  // bufnum
                    InputRef::Constant(rate), // rate
                    InputRef::Constant(0.0),  // trigger
                    InputRef::Constant(0.0),  // startPos
                    InputRef::Constant(1.0),  // loop
                    InputRef::Constant(0.0),  // doneAction
                ],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    }
}

/// Build the engine with no synths yet (they start once the sample loads).
fn build_engine(sample_rate: f32, channels: usize) -> (Controller, World) {
    let (controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels.max(1),
        ..Options::default()
    });
    (controller, world)
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    // cpal's AudioWorklet backend re-instantiates this module on the audio thread, re-running
    // `main` there; only set up audio on the main browser thread.
    if example_audio::on_worklet_thread() {
        return;
    }

    // Build-then-load: start the (initially silent) stream, then load the sample off to the side and
    // begin playback when it arrives. `play_with` hands back the controller + resolved rate/channels.
    let (stream, (controller, sample_rate, channels)) =
        example_audio::play_with(GAIN, |sample_rate, channels| {
            let (controller, mut world) = build_engine(sample_rate as f32, channels);
            (
                move |out: &mut [f32], channels: usize| world.fill(out, channels),
                (controller, sample_rate as f32, channels),
            )
        });

    #[cfg(not(target_arch = "wasm32"))]
    {
        // The filesystem read resolves immediately, so the sample starts right away.
        block_on(load_and_play(controller, FsSource, sample_rate, channels));
        println!("playing a sample loaded from assets/{SAMPLE} for 10s...");
    }
    #[cfg(target_arch = "wasm32")]
    // The fetch genuinely awaits; playback begins when it completes.
    wasm_bindgen_futures::spawn_local(load_and_play(
        controller,
        FetchSource,
        sample_rate,
        channels,
    ));

    example_audio::keep_alive(stream, 10);
}

/// Drive a future to completion. Sufficient for sources that resolve synchronously (the filesystem
/// source's future is ready on first poll); the web build uses `spawn_local` instead.
#[cfg(not(target_arch = "wasm32"))]
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
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

    /// The checked-in asset should load from the filesystem through `FsSource` and play (440 Hz).
    #[test]
    fn loaded_sample_plays() {
        let (controller, mut world) = build_engine(SR, 1);
        block_on(load_and_play(controller, FsSource, SR, 1));
        let mut out = vec![0.0f32; SR as usize / 4];
        world.fill(&mut out, 1);
        assert!(
            out.iter().any(|s| s.abs() > 0.1),
            "the loaded sample was silent"
        );
        assert!(
            goertzel(&out, 440.0) > 5.0 * goertzel(&out, 880.0),
            "expected the loaded 440 Hz sample"
        );
    }
}
