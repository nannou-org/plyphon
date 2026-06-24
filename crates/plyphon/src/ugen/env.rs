//! `EnvGen` - a multi-segment envelope generator, plyphon's port of scsynth's `EnvGen`.
//!
//! The envelope is passed as a flat array of inputs, exactly as SuperCollider encodes an `Env`:
//! after the five control inputs (`gate`, `levelScale`, `levelBias`, `timeScale`, `doneAction`) come
//! `initialLevel`, `numSegments`, `releaseNode`, `loopNode`, then four inputs per segment
//! (`targetLevel`, `time`, `curveType`, `curveValue`). The generator walks the segments, shaping each
//! by its curve; with a release node it sustains there until `gate` falls, then plays the remaining
//! segments and fires its `doneAction`. Looping (`loopNode`) and gate retriggering are not yet
//! handled.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::ugen::registry::{BuildContext, UgenDef};
use crate::ugen::{BuiltUgen, DoneAction, Inputs, ProcessCtx, Ugen, ugen_spec};

/// Where the generator is in the envelope, stored as a `u32` so the state is [`Pod`].
mod phase {
    /// Playing the pre-release segments.
    pub const ATTACK: u32 = 0;
    /// Holding at the release node until the gate falls.
    pub const SUSTAIN: u32 = 1;
    /// Playing the post-release segments.
    pub const RELEASE: u32 = 2;
    /// Finished (holding the final level).
    pub const DONE: u32 = 3;
}

/// `EnvGen.ar/kr(env, gate, levelScale, levelBias, timeScale, doneAction)`.
///
/// `Pod` state for the rt-pool: `f64`s first, then the 4-byte fields (`prev_gate`, `seg_curve`, the
/// segment index, the [`phase`] tag, and two `0`/`1` flags) - six 4-byte fields after six `f64`s, so
/// `repr(C)` packs it with no implicit padding (72 bytes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct EnvGen {
    /// Current envelope level, before `levelScale`/`levelBias`.
    level: f64,
    pos: f64,
    seg_dur: f64,
    seg_start: f64,
    seg_end: f64,
    seg_curve_value: f64,
    prev_gate: f32,
    seg_curve: i32,
    /// Current segment index.
    seg: u32,
    /// Envelope position tag (see [`phase`]).
    phase: u32,
    /// `0`/`1`: whether the first-block setup has run.
    started: u32,
    /// `0`/`1`: whether the done action has already fired.
    fired: u32,
}

impl EnvGen {
    const GATE: usize = 0;
    const LEVEL_SCALE: usize = 1;
    const LEVEL_BIAS: usize = 2;
    const TIME_SCALE: usize = 3;
    const DONE_ACTION: usize = 4;
    /// First envelope input: `initialLevel`, `numSegments`, `releaseNode`, `loopNode`, then segments.
    const ENV: usize = 5;
    const SEGMENTS: usize = 9;

    /// Number of segments, clamped to the inputs actually supplied (so a malformed def cannot panic).
    fn num_segments(&self, ins: &Inputs<'_>) -> usize {
        let declared = get(ins, Self::ENV + 1).max(0.0) as usize;
        let available = ins.len().saturating_sub(Self::SEGMENTS) / 4;
        declared.min(available)
    }

    fn release_node(&self, ins: &Inputs<'_>) -> i32 {
        get(ins, Self::ENV + 2) as i32
    }

    /// `(targetLevel, time, curveType, curveValue)` for segment `i`.
    fn segment(&self, ins: &Inputs<'_>, i: usize) -> (f64, f64, i32, f64) {
        let base = Self::SEGMENTS + 4 * i;
        (
            get(ins, base) as f64,
            get(ins, base + 1) as f64,
            get(ins, base + 2) as i32,
            get(ins, base + 3) as f64,
        )
    }

    /// Begin segment `i`, ramping from the current level over its (scaled) duration.
    fn load_segment(&mut self, ins: &Inputs<'_>, i: usize, sample_rate: f64, time_scale: f64) {
        let (target, time, curve, curve_value) = self.segment(ins, i);
        self.seg = i as u32;
        self.seg_start = self.level;
        self.seg_end = target;
        self.seg_dur = (time * time_scale * sample_rate).max(1.0);
        self.seg_curve = curve;
        self.seg_curve_value = curve_value;
        self.pos = 0.0;
    }
}

impl Ugen for EnvGen {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let gate = ctx.ins.control(Self::GATE);
        let level_scale = ctx.ins.control(Self::LEVEL_SCALE) as f64;
        let level_bias = ctx.ins.control(Self::LEVEL_BIAS) as f64;
        let time_scale = (ctx.ins.control(Self::TIME_SCALE) as f64).max(0.0);
        let done_action = DoneAction::from_code(ctx.ins.control(Self::DONE_ACTION));
        let sample_rate = ctx.audio.sample_rate;
        let num_segments = self.num_segments(&ctx.ins);
        let release_node = self.release_node(&ctx.ins);

        if self.started == 0 {
            self.level = get(&ctx.ins, Self::ENV) as f64; // initialLevel
            if num_segments > 0 {
                self.load_segment(&ctx.ins, 0, sample_rate, time_scale);
                self.phase = phase::ATTACK;
            } else {
                self.phase = phase::DONE;
            }
            self.prev_gate = gate;
            self.started = 1;
        }

        // A falling gate begins the release phase: jump straight to the release segment (the segment
        // leaving the release node), ramping down from wherever the envelope currently sits. Matches
        // scsynth, where `check_gate` sets the stage to `releaseNode - 1` and the next step advances
        // it to `releaseNode`.
        if self.prev_gate >= 0.5
            && gate < 0.5
            && release_node >= 0
            && matches!(self.phase, phase::ATTACK | phase::SUSTAIN)
        {
            let release_seg = release_node as usize;
            if release_seg < num_segments {
                self.load_segment(&ctx.ins, release_seg, sample_rate, time_scale);
                self.phase = phase::RELEASE;
            } else {
                self.phase = phase::DONE;
            }
        }
        self.prev_gate = gate;

        let mut action = DoneAction::Nothing;
        for o in ctx.outs.audio(0).iter_mut() {
            match self.phase {
                phase::ATTACK | phase::RELEASE => {
                    let t = (self.pos / self.seg_dur).min(1.0);
                    self.level = shape(
                        self.seg_curve,
                        self.seg_curve_value,
                        self.seg_start,
                        self.seg_end,
                        t,
                    );
                    *o = (self.level * level_scale + level_bias) as f32;
                    self.pos += 1.0;
                    if self.pos >= self.seg_dur {
                        self.level = self.seg_end;
                        // Sustain once the segment *arriving* at the release node finishes, i.e. the
                        // just-completed segment is `releaseNode - 1` (scsynth's `m_stage + 1 ==
                        // releaseNode`). Hold there, still gated, until the gate falls.
                        let reached_release_node =
                            release_node >= 0 && self.seg as usize + 1 == release_node as usize;
                        if self.phase == phase::ATTACK && reached_release_node {
                            self.phase = phase::SUSTAIN;
                        } else if self.seg as usize + 1 < num_segments {
                            self.load_segment(
                                &ctx.ins,
                                self.seg as usize + 1,
                                sample_rate,
                                time_scale,
                            );
                        } else {
                            self.phase = phase::DONE;
                            if self.fired == 0 {
                                self.fired = 1;
                                action = action.max(done_action);
                            }
                        }
                    }
                }
                // Sustain, Done, or any unexpected tag: hold the current level.
                _ => {
                    *o = (self.level * level_scale + level_bias) as f32;
                }
            }
        }
        action
    }
}

/// Constructor for [`EnvGen`].
pub struct EnvGenCtor;

impl UgenDef for EnvGenCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUgen, BuildError> {
        Ok(ugen_spec(EnvGen {
            level: 0.0,
            pos: 0.0,
            seg_dur: 1.0,
            seg_start: 0.0,
            seg_end: 0.0,
            seg_curve_value: 0.0,
            prev_gate: 0.0,
            seg_curve: 1,
            seg: 0,
            phase: phase::ATTACK,
            started: 0,
            fired: 0,
        }))
    }
}

/// Read input `i` as a single value, or 0.0 if the UGen was built with fewer inputs.
fn get(ins: &Inputs<'_>, i: usize) -> f32 {
    if i < ins.len() { ins.control(i) } else { 0.0 }
}

/// Interpolate `start`..`end` at fraction `t` per a scsynth envelope curve type.
fn shape(curve: i32, curve_value: f64, start: f64, end: f64, t: f64) -> f64 {
    use std::f64::consts::PI;
    match curve {
        0 => {
            // Step: hold the start, jump to the target at the end.
            if t >= 1.0 { end } else { start }
        }
        2 => {
            // Exponential: a ratio sweep, with a small floor so a 0 endpoint stays finite.
            let s = if start.abs() < 1e-5 {
                1e-5_f64.copysign(if end == 0.0 { 1.0 } else { end })
            } else {
                start
            };
            let e = if end.abs() < 1e-5 {
                1e-5_f64.copysign(s)
            } else {
                end
            };
            s * (e / s).powf(t)
        }
        3 => {
            // Sine: an ease-in-out S-curve.
            start + (end - start) * (0.5 - 0.5 * (PI * t).cos())
        }
        5 => {
            // Custom curvature: `curve_value` 0 is linear, >0 eases out, <0 eases in.
            if curve_value.abs() < 0.001 {
                start + (end - start) * t
            } else {
                start + (end - start) * (1.0 - (t * curve_value).exp()) / (1.0 - curve_value.exp())
            }
        }
        // Linear (1) and anything unsupported.
        _ => start + (end - start) * t,
    }
}
