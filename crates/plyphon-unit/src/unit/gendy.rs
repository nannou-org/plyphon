//! `Gendy1`, `Gendy2` and `Gendy3` - plyphon's port of Xenakis's dynamic stochastic synthesis
//! (scsynth's `GendynUGens.cpp`).
//!
//! The oscillator walks a set of breakpoints (control points). Each has an amplitude and a
//! duration; both drift by a bounded random step drawn from one of seven distributions every time
//! the oscillator reaches the point, and the waveform is the linear interpolation between
//! successive points. The random steps come from the synth's random stream, one of the World's
//! streams, so two instances of the same def decorrelate and a `RandSeed` replays one exactly.
//!
//! The breakpoint arrays live in the unit's [`aux`](crate::unit::Aux) memory, allocated when the
//! synth starts from the first value of `initCPs` (scsynth `RTAlloc`s them at construction).

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{Aux, BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec_pool};
use plyphon_dsp::math;
use plyphon_dsp::rng::Rng;

/// Xenakis's random-walk step distributions, selected by the integer `ampdist`/`durdist` inputs.
/// `a` is the distribution parameter (clamped to `[0.0001, 1]`), `f` a uniform `[0, 1)` draw; the
/// result is a step in roughly `[-1, 1]`. A direct port of scsynth's `Gendyn_distribution`.
fn distribution(which: i32, a: f32, f: f32) -> f32 {
    let a = a.clamp(0.0001, 1.0);
    match which {
        // LINEAR: the uniform draw itself, mapped to bipolar.
        0 => 2.0 * f - 1.0,
        // CAUCHY.
        1 => {
            let c = math::atan(10.0 * a);
            (1.0 / a) * math::tan(c * (2.0 * f - 1.0)) * 0.1
        }
        // LOGIST.
        2 => {
            let c = 0.5 + (0.499 * a);
            let c = math::ln((1.0 - c) / c);
            let f = ((f - 0.5) * 0.998 * a) + 0.5;
            math::ln((1.0 - f) / f) / c
        }
        // HYPERBCOS.
        3 => {
            let c = math::tan(1.5692255 * a);
            let temp = math::tan(1.5692255 * a * f) / c;
            let temp = math::ln(temp * 0.999 + 0.001) * (-0.1447648);
            2.0 * temp - 1.0
        }
        // ARCSINE.
        4 => {
            // scsynth's literal `1.5707963f`, one ulp below `FRAC_PI_2` as an `f32`.
            #[allow(clippy::approx_constant)]
            let c = math::sin(1.570_796_3_f32 * a);
            // scsynth's `pi_f` is `std::acos(-1.f)`, evaluated by the platform's libm: one ulp
            // below `PI` on macOS.
            let pi_f = math::acos(-1.0f32);
            math::sin(pi_f * (f - 0.5) * a) / c
        }
        // EXPON.
        5 => {
            let c = math::ln(1.0 - (0.999 * a));
            let temp = math::ln(1.0 - (f * 0.999 * a)) / c;
            2.0 * temp - 1.0
        }
        // SINUS: the parameter alone (the driving oscillator is not modelled here).
        6 => 2.0 * a - 1.0,
        _ => 2.0 * f - 1.0,
    }
}

/// Fold `amp` back into `[-1, 1]` by reflection (scsynth's amplitude wrap).
fn fold_amp(mut amp: f32) -> f32 {
    if !(-1.0..=1.0).contains(&amp) {
        if amp < 0.0 {
            amp += 4.0;
        }
        amp %= 4.0;
        if (1.0..3.0).contains(&amp) {
            amp = 2.0 - amp;
        } else if amp > 1.0 {
            amp -= 4.0;
        }
    }
    amp
}

/// Fold `dur` back toward `[0, 1]` by reflection (scsynth's duration wrap).
fn fold_dur(mut dur: f32) -> f32 {
    if !(0.0..=1.0).contains(&dur) {
        if dur < 0.0 {
            dur += 2.0;
        }
        dur %= 2.0;
        dur = 2.0 - dur;
    }
    dur
}

/// `Gendy1(ampdist, durdist, adparam, ddparam, minfreq, maxfreq, ampscale, durscale, initCPs,
/// knum)`: an audio-rate dynamic-stochastic oscillator.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Gendy1 {
    /// Interpolation phase between the current and next breakpoint; wraps at `1.0`. Starts at `1.0`
    /// so the first sample immediately computes a breakpoint.
    phase: f64,
    /// Amplitude of the breakpoint the oscillator is leaving.
    amp: f32,
    /// Amplitude of the breakpoint it is heading toward.
    next_amp: f32,
    /// Phase increment per sample, from the current breakpoint's duration and the frequency range.
    speed: f32,
    /// Index of the current breakpoint within the memory arrays.
    index: u32,
    /// Number of breakpoints the memory arrays hold (the first value of `initCPs`).
    memory_size: u32,
    _pad: u32,
}

impl Gendy1 {
    const AMPDIST: usize = 0;
    const DURDIST: usize = 1;
    const ADPARAM: usize = 2;
    const DDPARAM: usize = 3;
    const MINFREQ: usize = 4;
    const MAXFREQ: usize = 5;
    const AMPSCALE: usize = 6;
    const DURSCALE: usize = 7;
    const INIT_CPS: usize = 8;
    const KNUM: usize = 9;

    /// Seed the breakpoint arrays: amplitudes uniform in `[-1, 1)`, durations in `[0, 1)`, drawn
    /// one breakpoint at a time (amplitude, then duration) as scsynth's constructor draws them.
    fn seed(rng: &mut Rng, amp_mem: &mut [f32], dur_mem: &mut [f32]) {
        for (a, d) in amp_mem.iter_mut().zip(dur_mem.iter_mut()) {
            *a = 2.0 * rng.next_unipolar() - 1.0;
            *d = rng.next_unipolar();
        }
    }
}

impl Unit for Gendy1 {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor fills the breakpoint arrays from the shared stream and writes 0, without
        // running the calc.
        let (amp_mem, dur_mem) = ctx.aux.f32_mut().split_at_mut(self.memory_size as usize);
        Gendy1::seed(ctx.rgen, amp_mem, dur_mem);
        DoneAction::Nothing
    }

    fn alloc(&mut self, ctx: &InitCtx<'_>, aux: &mut Aux<'_>) {
        // scsynth's `Gendy1_Ctor`: `mMemorySize = (int)ZIN0(8)`, at least 1, then one amplitude and
        // one duration array of that size.
        let memory_size = (ctx.ins.control(Self::INIT_CPS) as i32).max(1) as u32;
        let bytes = (memory_size as usize).saturating_mul(2 * core::mem::size_of::<f32>());
        if aux.alloc(bytes) {
            self.memory_size = memory_size;
        }
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let memory_size = self.memory_size as usize;
        let which_amp = ctx.ins.control(Self::AMPDIST) as i32;
        let which_dur = ctx.ins.control(Self::DURDIST) as i32;
        let aamp = ctx.ins.control(Self::ADPARAM);
        let adur = ctx.ins.control(Self::DDPARAM);
        let minfreq = ctx.ins.control(Self::MINFREQ);
        let maxfreq = ctx.ins.control(Self::MAXFREQ);
        let scaleamp = ctx.ins.control(Self::AMPSCALE);
        let scaledur = ctx.ins.control(Self::DURSCALE);
        // `knum` limits how many of the allocated breakpoints are active; out of range means all.
        let knum = ctx.ins.control(Self::KNUM) as i32;
        let num = if knum < 1 || knum as usize > memory_size {
            memory_size
        } else {
            knum as usize
        };
        let freq_mul = ctx.own.sample_dur as f32;

        let (amp_mem, dur_mem) = ctx.aux.f32_mut().split_at_mut(memory_size);

        let mut phase = self.phase;
        let mut amp = self.amp;
        let mut next_amp = self.next_amp;
        let mut speed = self.speed;
        let mut index = self.index as usize;
        for slot in ctx.outs.audio(0).iter_mut() {
            if phase >= 1.0 {
                phase -= 1.0;
                index = (index + 1) % num;
                amp = next_amp;
                next_amp = fold_amp(
                    amp_mem[index]
                        + scaleamp * distribution(which_amp, aamp, ctx.rgen.next_unipolar()),
                );
                amp_mem[index] = next_amp;
                let rate = fold_dur(
                    dur_mem[index]
                        + scaledur * distribution(which_dur, adur, ctx.rgen.next_unipolar()),
                );
                dur_mem[index] = rate;
                speed = (minfreq + (maxfreq - minfreq) * rate) * freq_mul * num as f32;
            }
            // Interpolated in double precision, as scsynth's `double phase` promotes it.
            *slot = ((1.0 - phase) * amp as f64 + phase * next_amp as f64) as f32;
            phase += speed as f64;
        }

        self.phase = phase;
        self.amp = amp;
        self.next_amp = next_amp;
        self.speed = speed;
        self.index = index as u32;
        DoneAction::Nothing
    }
}

/// Constructor for [`Gendy1`]. The breakpoint arrays are allocated when the synth starts, sized from
/// the first value of `initCPs`.
pub struct Gendy1Ctor;

impl UnitDef for Gendy1Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= Gendy1::KNUM {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec_pool(Gendy1 {
            phase: 1.0,
            amp: 0.0,
            next_amp: 0.0,
            speed: 100.0,
            index: 0,
            memory_size: 0,
            _pad: 0,
        }))
    }
}

/// Reflect `x` back into `[lower, upper]` (scsynth's `Gendyn_mirroring`), used by [`Gendy2`] and
/// [`Gendy3`] for both the random-walk steps and the breakpoints they move.
fn mirror(lower: f32, upper: f32, mut x: f32) -> f32 {
    if x > upper || x < lower {
        let range = upper - lower;
        if x < lower {
            x = (2.0 * upper - lower) - x;
        }
        x = (x - upper) % (2.0 * range);
        if x < range {
            x = upper - x;
        } else {
            x -= range;
        }
    }
    x
}

/// `Gendy2(ampdist, durdist, adparam, ddparam, minfreq, maxfreq, ampscale, durscale, initCPs, knum,
/// a, c)`: [`Gendy1`] with a second random walk. Each breakpoint's amplitude and duration move by a
/// *step* that itself random-walks (Hoffmann's primary and secondary walks). The amplitude step's
/// uniform value comes from a Lehmer-style map of the previous amplitude, `(amp * a + c) mod 1`,
/// rather than the random stream; the duration step's is drawn from the synth's stream.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Gendy2 {
    /// Interpolation phase between the current and next breakpoint; wraps at `1.0`. Starts at `1.0`
    /// so the first sample immediately computes a breakpoint.
    phase: f64,
    /// Amplitude of the breakpoint the oscillator is leaving.
    amp: f32,
    /// Amplitude of the breakpoint it is heading toward.
    next_amp: f32,
    /// Phase increment per sample, from the current breakpoint's duration and the frequency range.
    speed: f32,
    /// Index of the current breakpoint within the memory arrays.
    index: u32,
    /// Number of breakpoints each memory array holds (the first value of `initCPs`).
    memory_size: u32,
    _pad: u32,
}

impl Gendy2 {
    const AMPDIST: usize = 0;
    const DURDIST: usize = 1;
    const ADPARAM: usize = 2;
    const DDPARAM: usize = 3;
    const MINFREQ: usize = 4;
    const MAXFREQ: usize = 5;
    const AMPSCALE: usize = 6;
    const DURSCALE: usize = 7;
    const INIT_CPS: usize = 8;
    const KNUM: usize = 9;
    const A: usize = 10;
    const C: usize = 11;

    /// The four breakpoint arrays in the unit's memory: amplitudes, durations, amplitude steps and
    /// duration steps, `n` each (shorter only if the memory is missing).
    fn memory<'a>(aux: &'a mut Aux<'_>, n: usize) -> [&'a mut [f32]; 4] {
        let mem = aux.f32_mut();
        let (amp, rest) = mem.split_at_mut(n.min(mem.len()));
        let (dur, rest) = rest.split_at_mut(n.min(rest.len()));
        let (amp_step, rest) = rest.split_at_mut(n.min(rest.len()));
        let len = n.min(rest.len());
        let dur_step = &mut rest[..len];
        [amp, dur, amp_step, dur_step]
    }
}

impl Unit for Gendy2 {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // scsynth's `Gendy2_Ctor` fills the four arrays one breakpoint at a time (amplitude,
        // duration, amplitude step, duration step) from the synth's stream, and writes 0 without
        // running the calc.
        let [amp, dur, amp_step, dur_step] =
            Gendy2::memory(&mut ctx.aux, self.memory_size as usize);
        let points = amp
            .iter_mut()
            .zip(dur.iter_mut())
            .zip(amp_step.iter_mut())
            .zip(dur_step.iter_mut());
        for (((a, d), s), t) in points {
            *a = 2.0 * ctx.rgen.next_unipolar() - 1.0;
            *d = ctx.rgen.next_unipolar();
            *s = 2.0 * ctx.rgen.next_unipolar() - 1.0;
            *t = 2.0 * ctx.rgen.next_unipolar() - 1.0;
        }
        DoneAction::Nothing
    }

    fn alloc(&mut self, ctx: &InitCtx<'_>, aux: &mut Aux<'_>) {
        // `mMemorySize = (int)ZIN0(8)`, at least 1, then four arrays of that size.
        let memory_size = (ctx.ins.control(Self::INIT_CPS) as i32).max(1) as u32;
        let bytes = (memory_size as usize).saturating_mul(4 * core::mem::size_of::<f32>());
        if aux.alloc(bytes) {
            self.memory_size = memory_size;
        }
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let memory_size = self.memory_size as usize;
        let which_amp = ctx.ins.control(Self::AMPDIST) as i32;
        let which_dur = ctx.ins.control(Self::DURDIST) as i32;
        let aamp = ctx.ins.control(Self::ADPARAM);
        let adur = ctx.ins.control(Self::DDPARAM);
        let minfreq = ctx.ins.control(Self::MINFREQ);
        let maxfreq = ctx.ins.control(Self::MAXFREQ);
        let scaleamp = ctx.ins.control(Self::AMPSCALE);
        let scaledur = ctx.ins.control(Self::DURSCALE);
        let knum = ctx.ins.control(Self::KNUM) as i32;
        let lehmer_a = ctx.ins.control(Self::A);
        let lehmer_c = ctx.ins.control(Self::C);
        let freq_mul = ctx.own.sample_dur as f32;

        let [amp_mem, dur_mem, amp_step_mem, dur_step_mem] =
            Gendy2::memory(&mut ctx.aux, memory_size);
        if memory_size == 0 || dur_step_mem.len() < memory_size {
            return DoneAction::Nothing;
        }

        let mut phase = self.phase;
        let mut amp = self.amp;
        let mut next_amp = self.next_amp;
        let mut speed = self.speed;
        let mut index = self.index as usize;
        for slot in ctx.outs.audio(0).iter_mut() {
            if phase >= 1.0 {
                phase -= 1.0;
                // `knum` limits how many breakpoints are active; out of range means all.
                let num = if knum < 1 || knum as usize > memory_size {
                    memory_size
                } else {
                    knum as usize
                };
                index = (index + 1) % num;
                // The amplitude step's uniform value: a Lehmer map of the amplitude being left.
                let lehmer = (amp * lehmer_a + lehmer_c) % 1.0;
                amp = next_amp;

                let amp_step = mirror(
                    -1.0,
                    1.0,
                    amp_step_mem[index] + distribution(which_amp, aamp, lehmer.abs()),
                );
                amp_step_mem[index] = amp_step;
                next_amp = mirror(-1.0, 1.0, amp_mem[index] + scaleamp * amp_step);
                amp_mem[index] = next_amp;

                let dur_step = mirror(
                    -1.0,
                    1.0,
                    dur_step_mem[index] + distribution(which_dur, adur, ctx.rgen.next_unipolar()),
                );
                dur_step_mem[index] = dur_step;
                let rate = mirror(0.0, 1.0, dur_mem[index] + scaledur * dur_step);
                dur_mem[index] = rate;

                speed = (minfreq + (maxfreq - minfreq) * rate) * freq_mul;
                speed *= num as f32;
            }
            // Interpolated in double precision, as scsynth's `double phase` promotes it.
            *slot = ((1.0 - phase) * amp as f64 + phase * next_amp as f64) as f32;
            phase += speed as f64;
        }

        self.phase = phase;
        self.amp = amp;
        self.next_amp = next_amp;
        self.speed = speed;
        self.index = index as u32;
        DoneAction::Nothing
    }
}

/// Constructor for [`Gendy2`]. The breakpoint arrays are allocated when the synth starts, sized from
/// the first value of `initCPs`.
pub struct Gendy2Ctor;

impl UnitDef for Gendy2Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= Gendy2::C {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec_pool(Gendy2 {
            phase: 1.0,
            amp: 0.0,
            next_amp: 0.0,
            speed: 100.0,
            index: 0,
            memory_size: 0,
            _pad: 0,
        }))
    }
}

/// `Gendy3(ampdist, durdist, adparam, ddparam, freq, ampscale, durscale, initCPs, knum)`: the
/// breakpoints random-walk as in [`Gendy1`], but all of them move at the start of each period and
/// their durations are normalised so the period lasts exactly `1 / freq`. The first breakpoint's
/// amplitude stays at 0, and a breakpoint whose normalised duration is shorter than a sample is
/// dropped for the period.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Gendy3 {
    /// Phase through the period; wraps at `1.0`. Starts at `1.0` so the first sample computes the
    /// first period.
    phase: f64,
    /// Phase at which the current region (between two breakpoints) ends.
    next_phase: f64,
    /// Phase at which the current region began.
    last_phase: f64,
    /// Phase increment per sample (`freq * sampleDur`), set at each period's start.
    speed: f32,
    /// Amplitude of the breakpoint the region starts at.
    amp: f32,
    /// Amplitude of the breakpoint the region ends at.
    next_amp: f32,
    /// `1 / region length` truncated to an integer, as scsynth's `int interpmult` truncates it, and
    /// kept as a float between blocks as scsynth keeps it.
    interp_mult: f32,
    /// Index of the current region in the period's lists.
    index: i32,
    /// Number of breakpoints the memory arrays hold (the first value of `initCPs`).
    memory_size: u32,
}

impl Gendy3 {
    const AMPDIST: usize = 0;
    const DURDIST: usize = 1;
    const ADPARAM: usize = 2;
    const DDPARAM: usize = 3;
    const FREQ: usize = 4;
    const AMPSCALE: usize = 5;
    const DURSCALE: usize = 6;
    const INIT_CPS: usize = 7;
    const KNUM: usize = 8;

    /// Bytes of memory for `n` breakpoints: the period's phase list (`n + 1` doubles, with a guard),
    /// then the amplitude and duration memories (`n` floats each) and the period's amplitude list
    /// (`n + 1` floats, with a guard), rounded up to whole doubles.
    fn memory_bytes(n: usize) -> usize {
        let doubles = n.saturating_add(1);
        let floats = n.saturating_mul(3).saturating_add(1);
        doubles
            .saturating_add(floats.div_ceil(2))
            .saturating_mul(core::mem::size_of::<f64>())
    }

    /// The memory for `n` breakpoints, split as [`Gendy3::memory_bytes`] lays it out (shorter only if
    /// the memory is missing).
    fn memory<'a>(aux: &'a mut Aux<'_>, n: usize) -> Gendy3Memory<'a> {
        let mem = aux.cast_mut::<f64>();
        let (phase_list, rest) = mem.split_at_mut((n + 1).min(mem.len()));
        let floats: &mut [f32] = bytemuck::cast_slice_mut(rest);
        let (amp_mem, floats) = floats.split_at_mut(n.min(floats.len()));
        let (dur_mem, floats) = floats.split_at_mut(n.min(floats.len()));
        let len = (n + 1).min(floats.len());
        let amp_list = &mut floats[..len];
        Gendy3Memory {
            phase_list,
            amp_mem,
            dur_mem,
            amp_list,
        }
    }
}

/// [`Gendy3`]'s memory, split into its arrays.
struct Gendy3Memory<'a> {
    /// The period's region lengths, one per surviving breakpoint, then a guard (`mPhaseList`).
    phase_list: &'a mut [f64],
    /// Breakpoint amplitudes (`mMemoryAmp`).
    amp_mem: &'a mut [f32],
    /// Breakpoint durations (`mMemoryDur`).
    dur_mem: &'a mut [f32],
    /// The period's amplitudes, one per surviving breakpoint, then a guard (`mAmpList`).
    amp_list: &'a mut [f32],
}

impl Unit for Gendy3 {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // scsynth's `Gendy3_Ctor` fills the arrays one breakpoint at a time (amplitude, duration,
        // list amplitude) from the synth's stream, zeroes the first amplitude, and writes 0 without
        // running the calc. It leaves the lists' last (guard) elements uninitialised; they are set
        // here so nothing reads a previous tenant's bytes.
        let n = self.memory_size as usize;
        let mem = Gendy3::memory(&mut ctx.aux, n);
        if n == 0 || mem.amp_list.len() < n + 1 || mem.phase_list.len() < n + 1 {
            return DoneAction::Nothing;
        }
        let points = mem
            .amp_mem
            .iter_mut()
            .zip(mem.dur_mem.iter_mut())
            .zip(mem.amp_list.iter_mut())
            .zip(mem.phase_list.iter_mut());
        for (((a, d), l), p) in points {
            *a = 2.0 * ctx.rgen.next_unipolar() - 1.0;
            *d = ctx.rgen.next_unipolar();
            *l = 2.0 * ctx.rgen.next_unipolar() - 1.0;
            *p = 1.0;
        }
        mem.amp_list[n] = 0.0;
        mem.phase_list[n] = 1.0;
        mem.amp_mem[0] = 0.0;
        DoneAction::Nothing
    }

    fn alloc(&mut self, ctx: &InitCtx<'_>, aux: &mut Aux<'_>) {
        // `mMemorySize = (int)ZIN0(7)`, at least 1, then the two memories of that size and the two
        // period lists one longer.
        let memory_size = (ctx.ins.control(Self::INIT_CPS) as i32).max(1) as u32;
        if aux.alloc(Self::memory_bytes(memory_size as usize)) {
            self.memory_size = memory_size;
        }
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let memory_size = self.memory_size as usize;
        let which_amp = ctx.ins.control(Self::AMPDIST) as i32;
        let which_dur = ctx.ins.control(Self::DURDIST) as i32;
        let aamp = ctx.ins.control(Self::ADPARAM);
        let adur = ctx.ins.control(Self::DDPARAM);
        let freq = ctx.ins.control(Self::FREQ);
        let scaleamp = ctx.ins.control(Self::AMPSCALE);
        let scaledur = ctx.ins.control(Self::DURSCALE);
        let knum = ctx.ins.control(Self::KNUM) as i32;
        // The phase length of one sample (`mFreqMul`).
        let min_phase = ctx.own.sample_dur as f32;

        let mem = Gendy3::memory(&mut ctx.aux, memory_size);
        if memory_size == 0 || mem.amp_list.len() < memory_size + 1 {
            return DoneAction::Nothing;
        }

        let mut phase = self.phase;
        let mut amp = self.amp;
        let mut next_amp = self.next_amp;
        let mut speed = self.speed;
        let mut index = self.index;
        let mut interp_mult = self.interp_mult as i32;
        let mut last_phase = self.last_phase;
        let mut next_phase = self.next_phase;
        for slot in ctx.outs.audio(0).iter_mut() {
            if phase >= 1.0 {
                // A new period: move every active breakpoint, then normalise the durations.
                phase -= 1.0;
                let num = if knum < 1 || knum as usize > memory_size {
                    memory_size
                } else {
                    knum as usize
                };
                let mut dur_sum = 0.0f32;
                let points = mem.amp_mem.iter_mut().zip(mem.dur_mem.iter_mut());
                for (j, (a, d)) in points.take(num).enumerate() {
                    // The first breakpoint's amplitude always stays at 0.
                    if j > 0 {
                        let step = distribution(which_amp, aamp, ctx.rgen.next_unipolar());
                        *a = mirror(-1.0, 1.0, *a + scaleamp * step);
                    }
                    let step = distribution(which_dur, adur, ctx.rgen.next_unipolar());
                    // Normalised in a moment; no zero durations.
                    *d = mirror(0.01, 1.0, *d + scaledur * step);
                    dur_sum += *d;
                }
                let dur_sum = 1.0 / dur_sum;
                speed = freq * min_phase;
                // Keep the breakpoints whose normalised duration lasts at least a sample.
                let mut active = 0;
                let points = mem.amp_mem.iter().zip(mem.dur_mem.iter());
                for (&a, &d) in points.take(num) {
                    let d = d * dur_sum;
                    if d >= min_phase {
                        mem.amp_list[active] = a;
                        mem.phase_list[active] = d as f64;
                        active += 1;
                    }
                }
                mem.amp_list[active] = 0.0;
                mem.phase_list[active] = 2.0;
                next_phase = 0.0;
                next_amp = mem.amp_list[0];
                index = -1;
            }

            if phase >= next_phase {
                // Into the next region.
                index += 1;
                amp = next_amp;
                last_phase = next_phase;
                let i = index as usize;
                next_phase = last_phase + mem.phase_list.get(i).copied().unwrap_or(2.0);
                // Entering the guard region (the period's phases summing to just under 1), scsynth
                // reads the amplitude after the guard: past the list when every breakpoint
                // survived. That value is only ever multiplied by the guard region's zero
                // `interpmult` before the next period replaces it, so 0 stands in for it.
                next_amp = mem.amp_list.get(i + 1).copied().unwrap_or(0.0);
                interp_mult = (1.0 / (next_phase - last_phase)) as i32;
            }

            let interp = ((phase - last_phase) * interp_mult as f64) as f32;
            *slot = ((1.0 - interp) * amp) + (interp * next_amp);
            phase += speed as f64;
        }

        self.phase = phase;
        self.speed = speed;
        self.interp_mult = interp_mult as f32;
        self.amp = amp;
        self.next_amp = next_amp;
        self.last_phase = last_phase;
        self.next_phase = next_phase;
        self.index = index;
        DoneAction::Nothing
    }
}

/// Constructor for [`Gendy3`]. The breakpoint arrays are allocated when the synth starts, sized from
/// the first value of `initCPs`.
pub struct Gendy3Ctor;

impl UnitDef for Gendy3Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= Gendy3::KNUM {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec_pool(Gendy3 {
            phase: 1.0,
            next_phase: 0.0,
            last_phase: 0.0,
            speed: 100.0,
            amp: 0.0,
            next_amp: 0.0,
            interp_mult: 1.0,
            index: 0,
            memory_size: 0,
        }))
    }
}
