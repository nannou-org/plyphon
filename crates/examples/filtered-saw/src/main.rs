//! A parameterised filtered saw, swept from the host - built with `plyphon_synthdef`.
//!
//! Where `example-filters` builds the same sound with its LFO *inside* the graph, this one exposes
//! the interesting values as **parameters** and lets the control plane drive them with
//! [`Controller::set_control`] - the shape most hosts want, since it puts the sequencing wherever
//! your application logic already lives.
//!
//! The def is authored with the [`SynthDefBuilder`]: constructors take the builder, return handles,
//! and `build_with` serializes the result. The builder guarantees the [`SynthDef`] ordering
//! contract (every unit input references an *earlier* unit) by construction - a handle can only
//! name a unit that already exists. Parameters are still **addressed by position** at runtime, so
//! the build closure returns a [`Params`] struct holding each parameter's index (read off its
//! `Channel` with `param_index()`), which `build_with` hands back alongside the def - no
//! hand-maintained constants that could drift from the declaration order. The `U_*` constants
//! document where each unit lands in the arena.
//!
//! The parameter flavours are worth choosing deliberately, because they compile to different
//! machinery:
//!
//! - `freq`/`rq` are plain [`Param::control`]s - one value per control block, set instantly.
//! - `cutoff`/`amp` are [`Param::lag`]s - a one-pole de-zippers each `/n_set` over `LAG` seconds, so
//!   sweeping the cutoff from the host glides instead of stepping once per control block. Without it
//!   a fast sweep zippers audibly, since a plain control parameter is a staircase at the block rate.
//!
//! Nothing is ever freed here (one permanent synth), so there is no NRT work and the `Nrt` handle is
//! dropped at build time; `example-motif` shows the full lifecycle.

use plyphon::{AddAction, Controller, Options, ROOT_GROUP_ID, SynthDef, World, engine};
use plyphon_synthdef::{Out, Rate, Saw, SynthDefBuilder, UGenBuilder, ugen};

// A custom UGen wrapper, declared here with the same exported `ugen!` macro the builder crate
// uses internally. RLPF is a builtin (`plyphon_synthdef::RLPF`, deliberately not imported above);
// redefining it locally proves downstream crates can wrap units the builder doesn't cover yet.
ugen!(
    /// Resonant two-pole low-pass (locally declared custom wrapper).
    "RLPF" => RLPF [ar: Rate::Audio, kr: Rate::Control](
        /// The signal to filter (default 0).
        input = 0.0,
        /// Cutoff frequency in Hz (default 440).
        freq = 440.0,
        /// Reciprocal of Q; smaller is more resonant (default 1).
        rq = 1.0,
    ) -> 1
);

/// The def's parameter indices, captured from the declared `Channel`s inside the build closure.
/// `params` is positional once compiled, so these *are* the ABI of the def - and because they
/// come out of the same closure that declares the parameters, they cannot drift from it.
#[derive(Clone, Copy)]
struct Params {
    /// Saw pitch in Hz (plain control).
    freq: usize,
    /// Filter cutoff in Hz (lagged, so host sweeps glide).
    cutoff: usize,
    /// Filter reciprocal-Q; smaller is more resonant (plain control).
    #[cfg_attr(not(test), allow(dead_code))] // documented ABI, asserted by the tests
    rq: usize,
    /// Output level (lagged, so level changes don't click).
    #[cfg_attr(not(test), allow(dead_code))]
    amp: usize,
}

// --- Unit indices, in calc order (fixed by construction order in `filtered_saw`,
// --- asserted by the tests). ---
#[cfg_attr(not(test), allow(dead_code))]
const U_SAW: u32 = 0;
#[cfg_attr(not(test), allow(dead_code))]
const U_RLPF: u32 = 1;
#[cfg_attr(not(test), allow(dead_code))]
const U_AMP: u32 = 2;

/// The two pitches the control plane alternates between (Hz) - a low bass fifth.
const FREQS: [f32; 2] = [55.0, 82.5];
/// Cutoff sweep bounds (Hz).
const CUTOFF_LOW: f32 = 180.0;
const CUTOFF_HIGH: f32 = 3200.0;
/// Reciprocal-Q of the filter (smaller = more resonant).
const RQ: f32 = 0.25;
/// The synth's own output level, before the master gain.
const AMP: f32 = 0.3;
/// De-zipper time for the lagged parameters (seconds).
const LAG: f32 = 0.02;
/// The control plane ticks this often (ms); the cutoff sweep advances one step per tick.
const TICK_MS: u32 = 40;
/// Ticks per full cutoff sweep (down and back up): 50 x 40 ms = 2 s.
const SWEEP_TICKS: u32 = 50;
/// A gentle master gain (the resonant peak boosts the level).
const GAIN: f32 = 0.15;

/// The def: `Saw -> RLPF -> * amp -> Out`, with `freq`/`cutoff`/`rq`/`amp` exposed as parameters.
///
/// Built with `plyphon_synthdef`: parameters are declared on the builder (their indices come back
/// as a [`Params`] next to the def), fluent builders wire the units, and `.repeat(n)` copies the mono
/// voice to `n` output channels via `Out`'s flat-spread. A builder emits its unit when it is
/// consumed as an input (or finalized with `.signal()`), so the units land exactly at the `U_*`
/// indices: Saw (consumed by `.input()`), RLPF (consumed by `.mul_add()`), MulAdd, Out.
///
/// Note on the cutoff: only RLPF's `input` is read at audio rate; `cutoff` and `rq` are taken
/// once per control block, which is why lagging the cutoff parameter (rather than making it
/// audio-rate) is what smooths a sweep.
fn filtered_saw(channels: usize) -> (SynthDef, Params) {
    SynthDefBuilder::build_with("filtered-saw", |g| {
        let freq = g.add_control_param("freq", FREQS[0]);
        let cutoff = g.add_lag_param("cutoff", CUTOFF_HIGH, LAG);
        let rq = g.add_control_param("rq", RQ);
        let amp = g.add_lag_param("amp", AMP, LAG);
        let sig = RLPF::ar(g)
            .input(Saw::ar(g).freq(freq))
            .freq(cutoff)
            .rq(rq)
            .mul_add(amp, 0.0);
        Out::ar(g).channels(sig.repeat(channels.max(1))).emit();
        // Plain data leaves the closure; the `Channel`s themselves cannot outlive the builder.
        Params {
            freq: freq.param_index().unwrap(),
            cutoff: cutoff.param_index().unwrap(),
            rq: rq.param_index().unwrap(),
            amp: amp.param_index().unwrap(),
        }
    })
}

/// The control plane: a triangle sweep of the cutoff, with the pitch stepping at each turn.
struct Controls {
    controller: Controller,
    node: i32,
    params: Params,
    tick: u32,
}

impl Controls {
    /// One control step. Runs off the audio thread, so a `QueueFull` here is a normal (if unlikely)
    /// outcome to ignore rather than a failure to handle.
    fn tick(&mut self) {
        let phase = self.tick % SWEEP_TICKS;
        // Triangle in [0, 1]: down for the first half of the sweep, back up for the second.
        let half = SWEEP_TICKS / 2;
        let t = if phase < half {
            1.0 - phase as f32 / half as f32
        } else {
            (phase - half) as f32 / half as f32
        };
        let cutoff = CUTOFF_LOW + t * (CUTOFF_HIGH - CUTOFF_LOW);
        let _ = self
            .controller
            .set_control(self.node, self.params.cutoff, cutoff);

        // At the bottom of each sweep, step to the other pitch. The oscillator's phase is
        // continuous across a retune, so this needs no envelope and makes no click.
        if phase == half {
            let freq = FREQS[(self.tick / SWEEP_TICKS) as usize % FREQS.len()];
            let _ = self
                .controller
                .set_control(self.node, self.params.freq, freq);
        }
        self.tick = self.tick.wrapping_add(1);
    }
}

/// Build the engine, install the def, spawn the one voice, and return the control plane and `World`.
fn build(sample_rate: f32, channels: usize) -> (Controls, World) {
    let out_channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: out_channels,
        ..Options::default()
    });

    let (def, params) = filtered_saw(out_channels);
    controller.add_synthdef(def);
    // `add_synthdef` only files the def away - this is where it is compiled and installed, and so
    // where an authoring mistake (an unknown unit name, a bad input reference) surfaces.
    let node = controller
        .synth_new("filtered-saw", ROOT_GROUP_ID, AddAction::Tail)
        .expect("failed to create the filtered-saw synth");

    (
        Controls {
            controller,
            node,
            params,
            tick: 0,
        },
        world,
    )
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
    println!("sweeping a host-driven filtered saw for 12s...");

    let (stream, mut controls) = example_audio::play_with(GAIN, |sample_rate, channels| {
        let (controls, mut world) = build(sample_rate as f32, channels);
        (
            move |out: &mut [f32], channels: usize| world.fill(out, channels),
            controls,
        )
    });
    example_audio::run_control(stream, 12_000, TICK_MS, move || controls.tick());
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt()
    }

    /// Magnitude of `samples` at `freq` (Hz) via the Goertzel algorithm.
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
        let mut out = vec![0.0f32; frames];
        world.fill(&mut out, 1);
        out
    }

    /// The indices captured in `Params` must agree with the def's by-name lookup, since parameters
    /// are addressed by position once compiled - a mismatch would silently set the wrong control.
    #[test]
    fn param_indices_match_declaration_order() {
        let (def, params) = filtered_saw(1);
        assert_eq!(def.param_index("freq"), Some(params.freq));
        assert_eq!(def.param_index("cutoff"), Some(params.cutoff));
        assert_eq!(def.param_index("rq"), Some(params.rq));
        assert_eq!(def.param_index("amp"), Some(params.amp));
        // Four units in calc order: Saw, RLPF, MulAdd, Out.
        assert_eq!(def.units.len(), 4);
        assert_eq!(def.units[U_SAW as usize].name, "Saw");
        assert_eq!(def.units[U_RLPF as usize].name, "RLPF");
        assert_eq!(def.units[U_AMP as usize].name, "MulAdd");
    }

    /// The voice should sound, stay bounded, and let the `cutoff` parameter actually gate the saw's
    /// upper harmonics: the 20th harmonic of the 55 Hz fundamental (1100 Hz) passes with the filter
    /// open and is suppressed with it closed.
    #[test]
    fn cutoff_parameter_gates_the_harmonics() {
        let (mut controls, mut world) = build(SR, 1);
        let quarter = (SR * 0.25) as usize;

        // Open (the def's default cutoff), measured after the first quarter-second settles.
        let _ = render(&mut world, quarter);
        let open_buf = render(&mut world, quarter);

        // Closed: one `/n_set`, then discard a quarter-second so the 20 ms lag has settled.
        controls
            .controller
            .set_control(controls.node, controls.params.cutoff, CUTOFF_LOW)
            .unwrap();
        let _ = render(&mut world, quarter);
        let closed_buf = render(&mut world, quarter);

        for buf in [&open_buf, &closed_buf] {
            assert!(buf.iter().all(|s| s.is_finite()), "output must stay finite");
            assert!(
                buf.iter().all(|&s| s.abs() < 4.0),
                "the resonant peak should stay bounded"
            );
        }
        assert!(rms(&open_buf) > 0.01, "the filtered saw should be audible");

        let open = goertzel(&open_buf, 1100.0);
        let closed = goertzel(&closed_buf, 1100.0);
        assert!(
            open > 4.0 * closed,
            "the 1100 Hz harmonic should pass when open, not when closed (open={open}, closed={closed})"
        );
    }

    /// A lagged `amp` should fade to silence rather than cut, and reach it.
    #[test]
    fn lagged_amp_fades_to_silence() {
        let (mut controls, mut world) = build(SR, 1);
        let quarter = (SR * 0.25) as usize;
        let _ = render(&mut world, quarter);

        controls
            .controller
            .set_control(controls.node, controls.params.amp, 0.0)
            .unwrap();
        // The block the `/n_set` lands in is still audible (the one-pole starts from AMP)...
        let fading = render(&mut world, (SR * 0.005) as usize);
        assert!(rms(&fading) > 0.0, "the fade should not be an instant cut");
        // ...and well past the lag time it is silent.
        let _ = render(&mut world, quarter);
        let silent = render(&mut world, quarter);
        assert!(
            rms(&silent) < 1e-6,
            "amp=0 should settle to silence (rms={})",
            rms(&silent)
        );
    }
}
