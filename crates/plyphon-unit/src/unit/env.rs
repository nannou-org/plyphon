//! `EnvGen` - a multi-segment envelope generator, plyphon's port of scsynth's `EnvGen`.
//!
//! The envelope is passed as a flat array of inputs, exactly as SuperCollider encodes an `Env`:
//! after the five control inputs (`gate`, `levelScale`, `levelBias`, `timeScale`, `doneAction`) come
//! `initialLevel`, `numSegments`, `releaseNode`, `loopNode`, then four inputs per segment
//! (`targetLevel`, `time`, `curveType`, `curveValue`). The generator walks the segments, shaping each
//! by its curve; with a release node it sustains there until `gate` falls, then plays the remaining
//! segments and fires its `doneAction`. The gate follows scsynth's `check_gate`: rising retriggers
//! from the current level, `<= 0` releases, `<= -1` force-releases over `|gate| - 1` seconds. The
//! gate is read once per block (audio-rate gates are block-quantised); looping (`loopNode`) is not
//! yet handled.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, Inputs, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;
use plyphon_dsp::rate::Rate;

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
/// segment index, the `phase` tag, and two `0`/`1` flags) - six 4-byte fields after six `f64`s, so
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

impl Unit for EnvGen {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let gate = ctx.ins.control(Self::GATE);
        let level_scale = ctx.ins.control(Self::LEVEL_SCALE) as f64;
        let level_bias = ctx.ins.control(Self::LEVEL_BIAS) as f64;
        let time_scale = (ctx.ins.control(Self::TIME_SCALE) as f64).max(0.0);
        let done_action = DoneAction::from_code(ctx.ins.control(Self::DONE_ACTION));
        let sample_rate = ctx.own.sample_rate;
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

        // scsynth's `check_gate`, evaluated once per block (the gate is read at control rate;
        // per-sample audio-rate gate checking is not modelled):
        //
        // 1. A rising gate (`<= 0` to `> 0`) *retriggers*: restart at segment 0 ramping from the
        //    current level (no jump), clearing done-ness so a fresh done action can fire.
        // 2. A gate falling to `<= -1` *force-releases*: a synthesized linear segment from the
        //    current level to the envelope's final level over `|gate| - 1` seconds (unscaled by
        //    `timeScale`, as in scsynth), regardless of the release node.
        // 3. A gate falling to `<= 0` begins the normal release: jump straight to the release
        //    segment (the segment leaving the release node), ramping down from wherever the
        //    envelope currently sits (scsynth sets the stage to `releaseNode - 1` and the next
        //    step advances it).
        if self.prev_gate <= 0.0 && gate > 0.0 {
            if num_segments > 0 {
                self.load_segment(&ctx.ins, 0, sample_rate, time_scale);
                self.phase = phase::ATTACK;
                self.fired = 0;
                ctx.done.clear_done();
            }
        } else if gate <= -1.0
            && self.prev_gate > -1.0
            && matches!(self.phase, phase::ATTACK | phase::SUSTAIN)
        {
            let final_level = if num_segments > 0 {
                self.segment(&ctx.ins, num_segments - 1).0
            } else {
                self.level
            };
            self.seg = num_segments as u32; // past the last segment: completion goes to DONE
            self.seg_start = self.level;
            self.seg_end = final_level;
            self.seg_dur = ((-gate - 1.0) as f64 * sample_rate).max(1.0);
            self.seg_curve = 1; // linear
            self.seg_curve_value = 0.0;
            self.pos = 0.0;
            self.phase = phase::RELEASE;
        } else if self.prev_gate > 0.0
            && gate <= 0.0
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
        // Mark the unit done (scsynth's `mDone`) once the envelope reaches its end, regardless of the
        // done *action* - so a `Done`/`FreeSelfWhenDone` watcher observes completion even at code 0.
        if self.phase == phase::DONE {
            ctx.done.mark_done();
        }
        action
    }
}

/// Constructor for [`EnvGen`].
pub struct EnvGenCtor;

impl UnitDef for EnvGenCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(EnvGen {
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

/// Read input `i` as a single value, or 0.0 if the unit was built with fewer inputs.
fn get(ins: &Inputs<'_>, i: usize) -> f32 {
    if i < ins.len() { ins.control(i) } else { 0.0 }
}

/// `IEnvGen.ar/kr(env, index)`: reads an envelope by position rather than playing it in time. The
/// `index` input (in seconds) is looked up in the flattened `Env`: input 0 is the index, then the
/// interpolation array `offset, startLevel, numSegments, totalDuration`, then four inputs per
/// segment (`duration, shapeCode, curveValue, endLevel`). The output is the envelope's value at
/// `index - offset`, held below `0` and above the total duration - scsynth's `IEnvGen`.
///
/// The segment values are read live from the inputs (they are constants in a baked def), so no
/// per-unit envelope copy is kept; the last computed level is cached and reused while the index is
/// unchanged, as scsynth does. Two deliberate divergences from scsynth: there the envelope is
/// copied once at ctor and never re-read, so wiring a *changing* signal into an envelope slot
/// diverges (constant-baked defs agree), and scsynth's Hold shape (8) outputs a stale cached
/// level and then stores the segment's end level - stateful in scan order - where plyphon
/// outputs the segment's start level throughout, the shape's evident intent.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct IEnvGen {
    /// The most recently computed output level.
    level: f32,
    /// The previous index value; a matching index reuses `level`. Starts `NaN` so the first sample
    /// always computes.
    prev_point: f32,
    /// `0`/`1`: control-rate output (one value) vs audio-rate (a full block from a per-sample index).
    audio: u32,
    _pad: u32,
}

impl IEnvGen {
    const INDEX: usize = 0;
    const OFFSET: usize = 1;
    const START_LEVEL: usize = 2;
    const NUM_SEGMENTS: usize = 3;
    const TOTAL_DUR: usize = 4;

    /// Map an index into scsynth's `m_envvals` array to the corresponding unit input: entry `0` is
    /// the start level (input 2), and every later entry `m` is input `4 + m` (the per-segment
    /// `duration, shapeCode, curveValue, endLevel` run, starting at input 5).
    fn envval(ins: &Inputs<'_>, m: usize) -> f32 {
        if m == 0 {
            get(ins, Self::START_LEVEL)
        } else {
            get(ins, 4 + m)
        }
    }

    /// The envelope's value at `point` seconds, by the same segment search and per-segment shaping
    /// scsynth's `IEnvGen` uses.
    fn level_at(ins: &Inputs<'_>, num_segments: usize, total_dur: f32, point: f32) -> f32 {
        if point >= total_dur {
            return Self::envval(ins, num_segments * 4);
        }
        if point <= 0.0 {
            return Self::envval(ins, 0);
        }
        // Walk the segments, subtracting each duration, until `point` falls inside one.
        let mut newtime = 0.0f32;
        let mut segpos = point;
        let mut seglen = 0.0f32;
        let mut stage = 0usize;
        let mut j = 0usize;
        while j < num_segments && point >= newtime {
            seglen = Self::envval(ins, j * 4 + 1);
            newtime += seglen;
            segpos -= seglen;
            stage = j;
            j += 1;
        }
        let stagemul = stage * 4;
        segpos += seglen;
        let beg_level = Self::envval(ins, stagemul) as f64;
        let shape_code = Self::envval(ins, stagemul + 2) as i32;
        // scsynth reads the curve value as an `int`, so a fractional curve is truncated.
        let curve_value = (Self::envval(ins, stagemul + 3) as i32) as f64;
        let end_level = Self::envval(ins, stagemul + 4) as f64;
        let pos = if seglen != 0.0 {
            (segpos / seglen) as f64
        } else {
            1.0
        };
        shape(shape_code, curve_value, beg_level, end_level, pos) as f32
    }
}

impl IEnvGen {
    /// Compute (and cache) the level for one raw index sample.
    fn eval(
        &mut self,
        ins: &Inputs<'_>,
        offset: f32,
        num_segments: usize,
        total_dur: f32,
        raw: f32,
    ) -> f32 {
        let point = (raw - offset).max(0.0);
        if point == self.prev_point {
            return self.level;
        }
        self.prev_point = point;
        self.level = Self::level_at(ins, num_segments, total_dur, point);
        self.level
    }
}

impl Unit for IEnvGen {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins;
        let offset = get(&ins, Self::OFFSET);
        // Clamp the declared stage count to the segments the inputs actually carry (four inputs
        // per segment after the header), so a malformed def cannot spin the segment walk beyond
        // the input list on the audio thread. scsynth walks its ctor-copied array unchecked.
        let num_segments =
            (get(&ins, Self::NUM_SEGMENTS) as usize).min(ins.len().saturating_sub(5) / 4);
        let total_dur = get(&ins, Self::TOTAL_DUR);

        if self.audio != 0 {
            let index = if ins.rate(Self::INDEX) == Rate::Audio {
                Some(ins.audio(Self::INDEX))
            } else {
                None
            };
            let broadcast = ins.control(Self::INDEX);
            for (i, o) in ctx.outs.audio(0).iter_mut().enumerate() {
                let raw = index.map_or(broadcast, |s| s[i]);
                *o = self.eval(&ins, offset, num_segments, total_dur, raw);
            }
        } else {
            let raw = ins.control(Self::INDEX);
            *ctx.outs.control(0) = self.eval(&ins, offset, num_segments, total_dur, raw);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`IEnvGen`].
pub struct IEnvGenCtor;

impl UnitDef for IEnvGenCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= IEnvGen::TOTAL_DUR {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(IEnvGen {
            level: 0.0,
            prev_point: f32::NAN,
            audio: (ctx.rate == Rate::Audio) as u32,
            _pad: 0,
        }))
    }
}

/// Interpolate `start`..`end` at fraction `t` per a scsynth envelope curve type.
fn shape(curve: i32, curve_value: f64, start: f64, end: f64, t: f64) -> f64 {
    use core::f64::consts::{FRAC_PI_2, PI};
    match curve {
        0 => {
            // Step: the whole segment sits at the target (the jump happens at segment start) -
            // scsynth's `shape_Step`.
            end
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
            s * math::powf(e / s, t)
        }
        3 => {
            // Sine: an ease-in-out S-curve.
            start + (end - start) * (0.5 - 0.5 * math::cos(PI * t))
        }
        4 => {
            // Welch: a quarter sine - convex rising, concave falling (scsynth's `shape_Welch`).
            if start <= end {
                start + (end - start) * math::sin(FRAC_PI_2 * t)
            } else {
                end - (end - start) * math::sin(FRAC_PI_2 - FRAC_PI_2 * t)
            }
        }
        5 => {
            // Custom curvature: `curve_value` 0 is linear, >0 eases out, <0 eases in.
            if curve_value.abs() < 0.001 {
                start + (end - start) * t
            } else {
                start
                    + (end - start) * (1.0 - math::exp(t * curve_value))
                        / (1.0 - math::exp(curve_value))
            }
        }
        6 => {
            // Squared: linear in sqrt space (scsynth's `shape_Squared`).
            let y1 = math::sqrt(start);
            let y2 = math::sqrt(end);
            let y = y1 + (y2 - y1) * t;
            y * y
        }
        7 => {
            // Cubed: linear in cube-root space (scsynth's `shape_Cubed`, which likewise takes
            // `pow(level, 1/3)` - NaN for a negative level, as in scsynth).
            let y1 = math::powf(start, 1.0 / 3.0);
            let y2 = math::powf(end, 1.0 / 3.0);
            let y = y1 + (y2 - y1) * t;
            y * y * y
        }
        8 => {
            // Hold: keep the start for the whole segment, jumping to the target at the end
            // (scsynth's `shape_Hold`).
            if t >= 1.0 { end } else { start }
        }
        // Linear (1) and anything unsupported.
        _ => start + (end - start) * t,
    }
}
