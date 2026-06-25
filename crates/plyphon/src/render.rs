//! Non-real-time (offline / score) rendering - plyphon's port of scsynth's `-N` mode.
//!
//! [`Render`] drives a [`World`] *offline*: faster than real time, deterministically, with no audio
//! device. It is the score-render counterpart to the real-time [`World::fill`] callback. The trick is
//! that nothing new is needed in the engine - [`World::fill`] already free-runs a *deterministic*
//! clock (it never resyncs the DLL the way [`World::fill_at`] does), and the scheduler already fires
//! time-tagged commands at exact within-block sample offsets. [`Render`] just steps that clock one
//! control block at a time, captures each block of output, and ticks the [`Nrt`] cleanup - so a
//! score of time-tagged commands renders to a buffer with the same sample-accurate timing it would
//! get live.
//!
//! Do not confuse this with the [`Nrt`] type: that is scsynth's NRT *cleanup* half (it drops freed
//! buffers and surfaces node events off the audio thread). [`Render`] is offline *rendering*, and it
//! owns and ticks an [`Nrt`] internally so freed memory is still reclaimed during a render.
//!
//! # Determinism
//!
//! [`Render::step`] drives [`World::fill_duplex`] (never the `_at` variant), so the engine clock
//! advances by the constant nominal increment `(block_size * 2^32 / sample_rate) as u64` per block
//! and the output is a pure function of `(Options, synthdefs, score, input)` - bit-identical across
//! runs and machines. The engine's per-synth RNG seeding is already deterministic, so randomness is
//! reproducible too.
//!
//! # Driving a render
//!
//! [`Render`] is intentionally low-level and command-agnostic, so it serves the typed [`Controller`](crate::Controller)
//! API and the OSC front-end alike. Each control block, feed every command whose time is *at or
//! before* [`Render::block_end`] (an inclusive cutoff, matching the World's scheduler), then call
//! [`Render::step`]:
//!
//! ```ignore
//! let mut render = Render::new(world, nrt, &options);
//! let mut out = Vec::new();
//! let end = until.end_time(score_max_time);
//! let mut i = 0;
//! while render.block_start() <= end {
//!     let cutoff = render.block_end();
//!     while i < score.len() && score[i].time <= cutoff {
//!         // apply score[i] to the Controller, scheduled at score[i].time
//!         i += 1;
//!     }
//!     out.extend_from_slice(render.step(&[]));
//! }
//! ```
//!
//! `plyphon-osc` builds the scsynth-compatible path (`parse_score` + `render_osc_score`) on top of
//! exactly this loop.

use alloc::vec;
use alloc::vec::Vec;
use core::time::Duration;

use plyphon_rt::{Event, Nrt, Options, World};

/// OSC/NTP fixed-point units in one second (OSC time is 32.32 fixed point). Independent of the
/// sample rate, so an OSC-time duration converts to units by multiplying by this alone.
pub const OSC_UNITS_PER_SEC: f64 = 4_294_967_296.0;

/// The deterministic OSC-units-per-control-block increment the offline clock advances by, identical
/// to the World's free-running `Clock`. Authoring helper for placing a command on an exact sample:
/// a command for global sample `s` (with `s` not block-aligned) is tagged
/// `(s / block_size) * nominal_increment + round((s % block_size) * 2^32 / sample_rate)`.
pub fn nominal_increment(sample_rate: f64, block_size: usize) -> u64 {
    (block_size as f64 * OSC_UNITS_PER_SEC / sample_rate) as u64
}

/// Convert a duration in seconds to OSC/NTP units.
fn secs_to_osc_units(secs: f64) -> u64 {
    (secs * OSC_UNITS_PER_SEC) as u64
}

/// How long an offline render runs.
pub enum RenderUntil {
    /// Until the last scheduled command's time, plus `tail` (so envelopes and reverbs ring out).
    /// Matches scsynth, which renders up to the final command's time tag.
    EndOfScore {
        /// Extra time rendered after the last command.
        tail: Duration,
    },
    /// An explicit total duration, regardless of the score's last command.
    Duration(Duration),
}

impl RenderUntil {
    /// The OSC/NTP time the offline clock must reach to finish, given the score's latest command
    /// time `score_max_time` (in OSC/NTP units; `0` for an empty or immediate-only score).
    pub fn end_time(&self, score_max_time: u64) -> u64 {
        match self {
            RenderUntil::EndOfScore { tail } => {
                score_max_time.saturating_add(secs_to_osc_units(tail.as_secs_f64()))
            }
            RenderUntil::Duration(d) => secs_to_osc_units(d.as_secs_f64()),
        }
    }
}

/// Drives a [`World`] offline, one control block at a time, on a deterministic free-running clock.
///
/// Owns the [`World`] and its [`Nrt`] cleanup for the render's duration. Build it from the same
/// [`Options`] used for [`engine`](crate::engine()), feed scheduled commands through the [`Controller`](crate::Controller)
/// in lockstep with [`Render::block_end`], and pull output with [`Render::step`].
pub struct Render {
    world: World,
    nrt: Nrt,
    /// Reused interleaved output scratch, exactly one control block wide (`block_size * out_channels`).
    out_block: Vec<f32>,
    /// Mirrors the World's free-running `Clock.buftime`: OSC/NTP time at the start of the block
    /// `step` will next compute. Starts at 0, advances by `increment` per `step`.
    buftime: u64,
    /// OSC/NTP units per control block - identical to the World's clock increment, so feeding stays
    /// in lockstep with the World's scheduler deadline.
    increment: u64,
    block_size: usize,
    out_channels: usize,
    in_channels: usize,
}

impl Render {
    /// Build an offline renderer around the [`World`] and [`Nrt`] from [`engine`](crate::engine()),
    /// using the same [`Options`].
    pub fn new(world: World, nrt: Nrt, options: &Options) -> Self {
        Render {
            world,
            nrt,
            out_block: vec![0.0; options.block_size * options.output_channels],
            buftime: 0,
            increment: nominal_increment(options.sample_rate, options.block_size),
            block_size: options.block_size,
            out_channels: options.output_channels,
            in_channels: options.input_channels,
        }
    }

    /// The control block size in samples.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// The length (in interleaved samples) of an input block [`Render::step`] expects when feeding
    /// input: `block_size * input_channels` (`0` when the engine has no input channels).
    pub fn input_block_len(&self) -> usize {
        self.block_size * self.in_channels
    }

    /// OSC/NTP time at the start of the block [`Render::step`] will next compute.
    pub fn block_start(&self) -> u64 {
        self.buftime
    }

    /// OSC/NTP time at the end of the block [`Render::step`] will next compute - the inclusive cutoff
    /// for feeding scheduled commands: feed every command whose time is `<= block_end()` before the
    /// next `step`, mirroring the World scheduler's `time <= deadline` dispatch.
    pub fn block_end(&self) -> u64 {
        self.buftime.wrapping_add(self.increment)
    }

    /// Compute exactly one control block and return the interleaved output (`block_size *
    /// output_channels` samples, valid until the next `step`).
    ///
    /// Pass `&[]` for no input, or a full input block of [`Render::input_block_len`] interleaved
    /// samples to feed the input buses (for `In.ar`). Ticks the [`Nrt`] cleanup so freed
    /// buffers/streams are reclaimed, and advances the offline clock by one block.
    pub fn step(&mut self, input: &[f32]) -> &[f32] {
        // An empty input block means "no input this block"; otherwise feed all input channels.
        let in_channels = if input.is_empty() {
            0
        } else {
            self.in_channels
        };
        self.world
            .fill_duplex(&mut self.out_block, self.out_channels, input, in_channels);
        self.nrt.process();
        self.buftime = self.buftime.wrapping_add(self.increment);
        &self.out_block
    }

    /// Drain the next node lifecycle [`Event`] (started/ended/paused/resumed), or `None`. Call in a
    /// loop after each [`Render::step`] to observe or forward node notifications.
    pub fn poll(&mut self) -> Option<Event> {
        self.nrt.poll()
    }

    /// Tick the [`Nrt`] cleanup to completion, dropping every freed `Box` still in flight. Call after
    /// the final [`Render::step`] (the offline analog of the real-time shutdown drain) before the
    /// `World` is dropped.
    pub fn finish(&mut self) {
        while self.nrt.process() > 0 {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AddAction, CommandTime, Controller, InputRef, Param, ROOT_GROUP_ID, Rate, SynthDef,
        UnitSpec, engine,
    };

    const SR: f64 = 48_000.0;
    const BLOCK: usize = 64;

    /// A click voice with a clean, sample-exact onset: a constant 0.5 held for 5 ms, freed by its
    /// `doneAction` - so the output is exactly `0.0` before the onset and `0.5` at it (ported from
    /// the `schedule` example).
    fn click_def() -> SynthDef {
        SynthDef {
            name: "click".into(),
            params: vec![],
            units: vec![
                // Line.ar(0.5, 0.5, 0.005, doneAction: 2): hold 0.5 for 5 ms, then free.
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

    /// A continuous white-noise voice (no done action), for the RNG-determinism test.
    fn noise_def() -> SynthDef {
        SynthDef {
            name: "noise".into(),
            params: vec![Param {
                name: "amp".into(),
                default: 1.0,
            }],
            units: vec![
                UnitSpec::new("WhiteNoise", Rate::Audio, vec![], 1),
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

    /// The nominal OSC-units-per-block increment at the test rate/block.
    fn inc() -> u64 {
        nominal_increment(SR, BLOCK)
    }

    /// The OSC/NTP tag that fires a command at exactly global sample `s` on the free-running clock
    /// (ported from the `schedule` example). `s` need not be block-aligned.
    fn time_for_sample(s: usize) -> u64 {
        let block = (s / BLOCK) as u64;
        let off = (s % BLOCK) as f64;
        block * inc() + (off * (OSC_UNITS_PER_SEC / SR)).round() as u64
    }

    /// The first sample index at or after `from` where the output departs from exact silence.
    fn first_onset(out: &[f32], from: usize) -> Option<usize> {
        (from..out.len()).find(|&i| out[i] != 0.0)
    }

    /// Assert each click onsets at its exact target sample, in time order.
    fn assert_onsets(out: &[f32], targets: &[usize]) {
        let mut from = 0;
        for (k, &s) in targets.iter().enumerate() {
            let onset = first_onset(out, from).unwrap_or_else(|| panic!("click {k} never sounded"));
            assert_eq!(
                onset, s,
                "click {k} should onset at sample {s}, got {onset}"
            );
            // Skip past this click's tone (the contiguous non-silent run) to the next onset.
            from = onset;
            while from < out.len() && out[from] != 0.0 {
                from += 1;
            }
        }
    }

    fn options() -> Options {
        Options {
            sample_rate: SR,
            output_channels: 1,
            ..Options::default()
        }
    }

    /// Schedule one `click` at `time` (OSC/NTP) via the controller's scheduling window.
    fn schedule_click(controller: &mut Controller, id: i32, time: u64) {
        let prev = controller.begin_scheduled(CommandTime::At(time));
        controller
            .synth_new_with_id(id, "click", ROOT_GROUP_ID, AddAction::Tail)
            .expect("schedule click");
        controller.begin_scheduled(prev);
    }

    /// Render `targets` (submitted in `order`) lazily through [`Render`]: each block, feed every
    /// click due by `block_end`, then step.
    fn render_lazy(targets: &[usize], order: &[usize], blocks: usize) -> Vec<f32> {
        let opts = options();
        let (mut controller, nrt, world) = engine(opts);
        controller.add_synthdef(click_def());
        let mut render = Render::new(world, nrt, &opts);

        // Build the time-sorted score (clicks may be submitted out of order).
        let mut score: Vec<(u64, i32)> = order
            .iter()
            .map(|&i| (time_for_sample(targets[i]), 1000 + i as i32))
            .collect();
        score.sort_by_key(|&(t, _)| t);

        let mut out = Vec::with_capacity(blocks * BLOCK);
        let mut i = 0;
        for _ in 0..blocks {
            let cutoff = render.block_end();
            while i < score.len() && score[i].0 <= cutoff {
                schedule_click(&mut controller, score[i].1, score[i].0);
                i += 1;
            }
            out.extend_from_slice(render.step(&[]));
        }
        out
    }

    /// Render `targets` the "all up front" way: schedule the whole score, then one big `World::fill`.
    /// The reference the lazy driver must match bit-for-bit.
    fn render_all_up_front(targets: &[usize], order: &[usize], blocks: usize) -> Vec<f32> {
        let opts = options();
        let (mut controller, _nrt, mut world) = engine(opts);
        controller.add_synthdef(click_def());
        for &i in order {
            schedule_click(
                &mut controller,
                1000 + i as i32,
                time_for_sample(targets[i]),
            );
        }
        let mut out = vec![0.0; blocks * BLOCK];
        world.fill(&mut out, 1);
        out
    }

    #[test]
    fn clicks_onset_at_their_exact_scheduled_sample() {
        // Non-block-aligned, well-spaced targets; submitted out of order on purpose.
        let targets = [600usize, 1503, 2305, 3100];
        let order = [2usize, 0, 3, 1];
        let out = render_lazy(&targets, &order, 4096 / BLOCK);
        assert_onsets(&out, &targets);
    }

    #[test]
    fn lazy_feed_matches_all_up_front() {
        let targets = [600usize, 1503, 2305, 3100];
        let order = [2usize, 0, 3, 1];
        let blocks = 4096 / BLOCK;
        assert_eq!(
            render_lazy(&targets, &order, blocks),
            render_all_up_front(&targets, &order, blocks),
            "lazy feed must be bit-identical to scheduling the whole score up front",
        );
    }

    #[test]
    fn block_aligned_target_matches_all_up_front() {
        // 2048 = 32 * 64 lands exactly on a block boundary. The inclusive `time <= block_end` feed
        // cutoff fires it in the block whose end equals its time (matching the World's `<=` deadline
        // and the all-up-front reference); a strict `<` cutoff would feed it a block late and diverge.
        let targets = [600usize, 2048, 3100];
        let order = [0usize, 1, 2];
        let blocks = 4096 / BLOCK;
        assert_eq!(
            render_lazy(&targets, &order, blocks),
            render_all_up_front(&targets, &order, blocks),
            "a block-aligned target must still match the all-up-front reference",
        );
    }

    #[test]
    fn render_is_deterministic_across_runs() {
        // White noise: identical only if per-synth RNG seeding is deterministic across engine builds.
        let run = || {
            let opts = options();
            let (mut controller, nrt, world) = engine(opts);
            controller.add_synthdef(noise_def());
            controller
                .synth_new("noise", ROOT_GROUP_ID, AddAction::Tail)
                .expect("noise");
            let mut render = Render::new(world, nrt, &opts);
            let mut out = Vec::new();
            for _ in 0..16 {
                out.extend_from_slice(render.step(&[]));
            }
            out
        };
        let a = run();
        let b = run();
        assert!(a.iter().any(|&s| s != 0.0), "noise should be audible");
        assert_eq!(a, b, "two offline renders must be bit-identical");
    }

    #[test]
    fn long_score_renders_without_overflow() {
        // Far more clicks than the scheduler (2048) or command ring (1024) could hold at once:
        // lazy feeding keeps in-flight commands to ~one block's worth, so nothing overflows.
        const N: usize = 3000;
        const SPACING: usize = 320; // > the 5 ms (240-sample) tone, so onsets stay isolated
        let targets: Vec<usize> = (0..N).map(|k| 200 + k * SPACING).collect();
        let order: Vec<usize> = (0..N).collect();
        let blocks = (targets[N - 1] + BLOCK).div_ceil(BLOCK) + 1;
        let out = render_lazy(&targets, &order, blocks);
        assert_onsets(&out, &targets);
    }

    #[test]
    fn increment_matches_the_world_clock() {
        let opts = options();
        let (_controller, nrt, world) = engine(opts);
        let render = Render::new(world, nrt, &opts);
        // At construction block_start is 0, so block_end is exactly one nominal increment - the same
        // value the World's free-running `Clock` advances by.
        assert_eq!(render.block_start(), 0);
        assert_eq!(render.block_end(), nominal_increment(SR, BLOCK));
    }
}
