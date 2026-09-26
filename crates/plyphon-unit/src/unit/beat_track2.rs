//! `BeatTrack2` - plyphon's port of scsynth's template-matching beat tracker (`BeatTrack2.cpp`,
//! Nick Collins).
//!
//! The unit keeps a trail of the last few seconds of `numfeatures` consecutive control buses (onset
//! detection features, typically). Every half second it starts a search over 120 tempi (60-179 bpm)
//! and two groove templates (straight and swung sixteenths), amortised over 240 control blocks: for
//! each tempo and every phase at `phaseaccuracy` resolution it cross-correlates each feature's trail
//! with a click template, keeping each feature's best and second-best match. A tempo that other
//! features (now and in the previous search) agree on wins; it is adopted only when it confirms the
//! previous search's prediction of phase and period. Between searches a phasor at the current tempo
//! drives the beat outputs.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{
    self, Aux, BuiltUnit, DoneAction, InitCtx, LocalBufs, ProcessCtx, Unit, unit_spec_pool,
};
use plyphon_dsp::buffer::BufferTable;

/// The number of candidate tempi - scsynth's `g_numtempi`.
const NUM_TEMPI: usize = 120;

/// Convert a table of double literals to `f32`, as C converts a `float` array's initialisers.
const fn to_f32<const N: usize>(table: [f64; N]) -> [f32; N] {
    let mut out = [0.0f32; N];
    let mut i = 0;
    while i < N {
        out[i] = table[i] as f32;
        i += 1;
    }
    out
}

/// Beat period in seconds of each candidate tempo, 60 to 179 bpm - scsynth's `g_periods`.
#[rustfmt::skip]
static PERIODS: [f32; NUM_TEMPI] = to_f32([
    1.0, 0.98360655737705, 0.96774193548387, 0.95238095238095, 0.9375, 0.92307692307692,
    0.90909090909091, 0.8955223880597, 0.88235294117647, 0.8695652173913, 0.85714285714286,
    0.84507042253521, 0.83333333333333, 0.82191780821918, 0.81081081081081, 0.8,
    0.78947368421053, 0.77922077922078, 0.76923076923077, 0.75949367088608, 0.75,
    0.74074074074074, 0.73170731707317, 0.72289156626506, 0.71428571428571, 0.70588235294118,
    0.69767441860465, 0.68965517241379, 0.68181818181818, 0.67415730337079, 0.66666666666667,
    0.65934065934066, 0.65217391304348, 0.64516129032258, 0.63829787234043, 0.63157894736842,
    0.625, 0.61855670103093, 0.61224489795918, 0.60606060606061, 0.6, 0.59405940594059,
    0.58823529411765, 0.58252427184466, 0.57692307692308, 0.57142857142857, 0.56603773584906,
    0.5607476635514, 0.55555555555556, 0.55045871559633, 0.54545454545455, 0.54054054054054,
    0.53571428571429, 0.53097345132743, 0.52631578947368, 0.52173913043478, 0.51724137931034,
    0.51282051282051, 0.50847457627119, 0.50420168067227, 0.5, 0.49586776859504,
    0.49180327868852, 0.48780487804878, 0.48387096774194, 0.48, 0.47619047619048,
    0.47244094488189, 0.46875, 0.46511627906977, 0.46153846153846, 0.45801526717557,
    0.45454545454545, 0.45112781954887, 0.44776119402985, 0.44444444444444, 0.44117647058824,
    0.43795620437956, 0.43478260869565, 0.43165467625899, 0.42857142857143, 0.42553191489362,
    0.42253521126761, 0.41958041958042, 0.41666666666667, 0.41379310344828, 0.41095890410959,
    0.40816326530612, 0.40540540540541, 0.40268456375839, 0.4, 0.39735099337748,
    0.39473684210526, 0.3921568627451, 0.38961038961039, 0.38709677419355, 0.38461538461538,
    0.38216560509554, 0.37974683544304, 0.37735849056604, 0.375, 0.37267080745342,
    0.37037037037037, 0.3680981595092, 0.36585365853659, 0.36363636363636, 0.36144578313253,
    0.35928143712575, 0.35714285714286, 0.35502958579882, 0.35294117647059, 0.35087719298246,
    0.34883720930233, 0.34682080924855, 0.3448275862069, 0.34285714285714, 0.34090909090909,
    0.33898305084746, 0.33707865168539, 0.33519553072626,
]);

/// Click positions within a beat, as a fraction of the period, for the straight (first four) and
/// swung (last four) templates - scsynth's `g_sep`.
static SEP: [f32; 8] = to_f32([0.0, 0.25, 0.5, 0.75, 0.0, 0.32, 0.5, 0.82]);
/// Weight of each click position - scsynth's `g_weight`.
static WEIGHT: [f32; 4] = to_f32([1.0, 0.5, 0.9, 0.6]);
/// Blur of each click over the nine control blocks around it - scsynth's `g_weight2`.
static WEIGHT2: [f32; 9] = to_f32([0.05, 0.1, 0.3, 0.7, 1.0, 0.7, 0.3, 0.1, 0.05]);

/// `BeatTrack2.kr(busindex, numfeatures, windowsize = 2.0, phaseaccuracy = 0.02, lock = 0,
/// weightingscheme)`: six outputs - beat, quaver and semiquaver triggers, the tempo in beats per
/// second, the beat phase, and the groove (always 0).
///
/// `lock >= 0.5` freezes the outputs' tempo and keeps their phasor running while the model goes on
/// tracking. `weightingscheme` names a buffer whose first 120 samples weight the tempi, read at each
/// search step; a missing buffer weighs every tempo 1. A number past the World's buffers, or
/// negative (as scsynth compares it unsigned), names buffer 0.
///
/// `aux` holds, in 32-bit words: each tempo's phase count (`m_numphases`, 120 ints); the scores
/// scratch (`m_scores`, `2 * numfeatures` floats); each feature's trail (`m_pastfeatures`,
/// `numfeatures` rows of `buffersize` floats); and four ranks (best, second best, and the previous
/// search's two) of each feature's best score (float), phase, tempo and groove (ints), each
/// `4 * numfeatures` long.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BeatTrack2 {
    /// Phase resolution of the search in seconds - scsynth's `m_phaseaccuracy`.
    phaseaccuracy: f32,
    /// Feature buses tracked - scsynth's `m_numfeatures`; `0` if the unit could not size its memory.
    numfeatures: i32,
    /// Seconds of trail each template spans - scsynth's `m_temporalwindowsize`.
    temporalwindowsize: f32,
    /// Seconds of trail kept - scsynth's `m_fullwindowsize`.
    fullwindowsize: f32,
    /// Seconds per control block - scsynth's `m_krlength`.
    krlength: f32,
    /// Trail length in control blocks - scsynth's `m_buffersize`.
    buffersize: i32,
    /// Trail write position - scsynth's `m_counter`.
    counter: i32,
    /// Trail position when the current search began - scsynth's `m_startcounter`.
    startcounter: i32,
    /// Seconds since the last search began - scsynth's `m_calculationschedule`.
    calculationschedule: f32,
    /// Seconds between searches - scsynth's `m_calculationperiod`.
    calculationperiod: f32,
    /// The model's beat period in seconds - scsynth's `m_period`.
    period: f32,
    /// The model's groove - scsynth's `m_groove` (never changed from 0).
    groove: i32,
    /// The model's tempo in beats per second - scsynth's `m_currtempo`.
    currtempo: f32,
    /// The phase when the current search began - scsynth's `m_currphase`.
    currphase: f32,
    /// The model's beat phasor - scsynth's `m_phase`.
    phase: f32,
    /// Phase advance per control block - scsynth's `m_phaseperblock`.
    phaseperblock: f32,
    /// The output phasor, which runs on alone while locked - scsynth's `m_outputphase`.
    outputphase: f32,
    /// The output tempo - scsynth's `m_outputtempo`.
    outputtempo: f32,
    /// The output groove - scsynth's `m_outputgroove`.
    outputgroove: f32,
    /// The output phasor's advance per control block - scsynth's `m_outputphaseperblock`.
    outputphaseperblock: f32,
    /// Phase the previous search predicts for the next - scsynth's `m_predictphase`.
    predictphase: f32,
    /// Period the previous search predicts for the next - scsynth's `m_predictperiod`.
    predictperiod: f32,
    /// `0` idle, `1` searching, `2` deciding - scsynth's `m_amortisationstate`.
    amortisationstate: i32,
    /// Search steps done - scsynth's `m_amortcount`.
    amortcount: i32,
    /// Search steps in all (tempo x groove) - scsynth's `m_amortlength`.
    amortlength: i32,
    /// Whether the half-beat trigger has fired this beat - scsynth's `halftrig`.
    halftrig: i32,
    /// Whether the first quarter-beat trigger has fired this beat - scsynth's `q1trig`.
    q1trig: i32,
    /// Whether the third quarter-beat trigger has fired this beat - scsynth's `q2trig`.
    q2trig: i32,
    /// The tempo-weights buffer - scsynth's `m_tempoweights`.
    tempoweights: u32,
    /// `0` flat weighting, `2` weights from `tempoweights` - scsynth's `m_weightingscheme`.
    weightingscheme: i32,
}

/// The unit's `aux` regions, carved from its 32-bit words.
struct Mem<'a> {
    numphases: &'a mut [i32],
    scores: &'a mut [f32],
    pastfeatures: &'a mut [f32],
    bestscore: &'a mut [f32],
    bestphase: &'a mut [i32],
    besttempo: &'a mut [i32],
    bestgroove: &'a mut [i32],
}

/// `aux` words for `nf` features and a `buffersize`-block trail, or `None` on overflow.
fn mem_words(nf: usize, buffersize: usize) -> Option<usize> {
    let trail = nf.checked_mul(buffersize)?;
    NUM_TEMPI
        .checked_add(nf.checked_mul(2 + 16)?)?
        .checked_add(trail)
}

impl<'a> Mem<'a> {
    /// Split the unit's memory, or `None` if it holds less than `nf` features need.
    fn split(aux: &'a mut Aux<'_>, nf: usize, buffersize: usize) -> Option<Mem<'a>> {
        let words: &mut [i32] = aux.cast_mut();
        let words = words.get_mut(..mem_words(nf, buffersize)?)?;
        let (numphases, rest) = words.split_at_mut(NUM_TEMPI);
        let (scores, rest) = rest.split_at_mut(2 * nf);
        let (pastfeatures, rest) = rest.split_at_mut(nf * buffersize);
        let (bestscore, rest) = rest.split_at_mut(4 * nf);
        let (bestphase, rest) = rest.split_at_mut(4 * nf);
        let (besttempo, bestgroove) = rest.split_at_mut(4 * nf);
        Some(Mem {
            numphases,
            scores: bytemuck::cast_slice_mut(scores),
            pastfeatures: bytemuck::cast_slice_mut(pastfeatures),
            bestscore: bytemuck::cast_slice_mut(bestscore),
            bestphase,
            besttempo,
            bestgroove,
        })
    }
}

impl BeatTrack2 {
    const BUSINDEX: usize = 0;
    const NUMFEATURES: usize = 1;
    const WINDOWSIZE: usize = 2;
    const PHASEACCURACY: usize = 3;
    const LOCK: usize = 4;
    const WEIGHTINGSCHEME: usize = 5;

    /// `(numfeatures, fullwindowsize, buffersize)` from the constructor's inputs - scsynth's
    /// `(int)(ZIN0(1) + 0.001)`, `windowsize + 1.0 + 0.1` (in double) and
    /// `(int)(fullwindowsize / krlength)`.
    fn sizes(nf_in: f32, windowsize: f32, krlength: f32) -> (i32, f32, i32) {
        let nf = (nf_in as f64 + 0.001) as i32;
        let fullwindowsize = (windowsize as f64 + 1.0 + 0.1) as f32;
        let buffersize = (fullwindowsize / krlength) as i32;
        (nf, fullwindowsize, buffersize)
    }

    fn mem<'a>(&self, aux: &'a mut Aux<'_>) -> Option<Mem<'a>> {
        Mem::split(aux, self.numfeatures as usize, self.buffersize as usize)
    }

    /// Write the six outputs.
    fn write_outputs(ctx: &mut ProcessCtx<'_>, values: [f32; 6]) {
        for (i, v) in values.into_iter().enumerate() {
            *ctx.outs.control(i) = v;
        }
    }

    /// scsynth's `calculatetemplate`: score every phase of tempo `which` with groove template `j`
    /// against each feature's trail, and update each feature's best and second-best match.
    fn calculate_template(
        &self,
        buffers: &BufferTable,
        local: &LocalBufs<'_>,
        mem: &mut Mem<'_>,
        which: usize,
        j: usize,
    ) {
        let startcounter = self.startcounter;
        let numphases = mem.numphases[which];
        let period = PERIODS[which];
        let blockconvert = self.krlength;
        let windowsize = self.temporalwindowsize;
        let buffersize = self.buffersize;
        let nf = self.numfeatures as usize;

        // Complete beats that fit in the window.
        let beatsfit = (windowsize / period) as i32;

        let weight = match self.weightingscheme {
            0 => 1.0f32,
            1 => 1.0f32 / (beatsfit * 4) as f32,
            // A user buffer of per-tempo weights, read as it stands now; no data weighs 1. scsynth
            // reads past the end of a buffer shorter than the 120 tempi, which here also weighs 1.
            _ => unit::buffer_at(buffers, local, self.tempoweights as usize)
                .and_then(|b| b.data().get(which).copied())
                .unwrap_or(1.0),
        };

        for i in 0..numphases {
            for k in 0..nf {
                mem.scores[2 * k + j] = 0.0;
            }
            let phaseadd = i as f32 * self.phaseaccuracy;

            for h in 0..beatsfit {
                for (l, &w) in WEIGHT.iter().enumerate() {
                    let sep = phaseadd + (h as f32 * period) + (SEP[j * 4 + l] * period);
                    // Round to the nearest control block.
                    let blocks = ((sep / blockconvert) as f64 + 0.5) as i32;
                    let index = (startcounter + buffersize - blocks) % buffersize;
                    // Blur over four blocks either side.
                    for (m, &w2) in (-4i32..5).zip(&WEIGHT2) {
                        let actualindex = (index + buffersize + m) % buffersize;
                        for k in 0..nf {
                            // The index is always in the trail: `sep` stays within the kept window.
                            let past = usize::try_from(actualindex)
                                .ok()
                                .and_then(|a| mem.pastfeatures.get(k * buffersize as usize + a))
                                .copied()
                                .unwrap_or(0.0);
                            mem.scores[2 * k + j] += w * w2 * past;
                        }
                    }
                }
            }

            for k in 0..nf {
                let scorenow = mem.scores[2 * k + j] * weight;
                let second = nf + k;
                if scorenow > mem.bestscore[k] {
                    // The best so far moves down to second best.
                    mem.bestscore[second] = mem.bestscore[k];
                    mem.bestphase[second] = mem.bestphase[k];
                    mem.besttempo[second] = mem.besttempo[k];
                    mem.bestgroove[second] = mem.bestgroove[k];
                    mem.bestscore[k] = scorenow;
                    mem.bestphase[k] = i;
                    mem.besttempo[k] = which as i32;
                    mem.bestgroove[k] = j as i32;
                } else if scorenow > mem.bestscore[second] {
                    mem.bestscore[second] = scorenow;
                    mem.bestphase[second] = i;
                    mem.besttempo[second] = which as i32;
                    mem.bestgroove[second] = j as i32;
                }
            }
        }
    }

    /// scsynth's `finaldecision`: pick the feature whose best tempo the most other features (now
    /// and in the previous search) agree with, adopt it if it confirms the previous prediction, and
    /// predict the next search's phase and period from it.
    fn final_decision(&mut self, mem: &Mem<'_>) {
        let nf = self.numfeatures as usize;
        let mut bestcandidate = 0;
        let mut bestpreviousmatchsum = 0;

        for i in 0..nf {
            let mut matchsum = 0i32;
            let secondbest = mem.bestscore[nf + i];
            let excess = if secondbest != 0.0 {
                mem.bestscore[i] / secondbest
            } else {
                mem.bestscore[i]
            };
            let tempo = mem.besttempo[i];
            for j in 0..nf {
                if j != i && (mem.besttempo[j] - tempo).abs() < 5 {
                    matchsum += 1;
                }
                // The previous search's best tempi.
                if (mem.besttempo[2 * nf + j] - tempo).abs() < 5 {
                    matchsum += 1;
                }
            }
            if secondbest != 0.0 {
                matchsum = matchsum.wrapping_add(excess as i32);
            }
            if matchsum > bestpreviousmatchsum {
                bestcandidate = i;
                bestpreviousmatchsum = matchsum;
            }
        }

        // Seconds from the winner's phase to now (the search took `amortlength` blocks), in beats.
        let elapsed = (mem.bestphase[bestcandidate] as f32 * self.phaseaccuracy)
            + (self.krlength * self.amortlength as f32);
        let bestphase = (elapsed / self.period) % 1.0f32;

        let candidate_period = PERIODS[mem.besttempo[bestcandidate] as usize];
        if (bestphase - self.predictphase).abs() < ((2.0 * self.phaseaccuracy) / self.predictperiod)
            && ((candidate_period - self.predictperiod).abs() as f64) < 0.04
        {
            self.period = self.predictperiod;
            self.phase = bestphase;
            self.currtempo = 1.0 / self.period;
            self.phaseperblock = self.krlength / self.period;
        }

        self.predictperiod = candidate_period;
        self.predictphase = ((elapsed + self.calculationperiod) / self.period) % 1.0f32;
    }
}

impl Unit for BeatTrack2 {
    fn alloc(&mut self, ctx: &InitCtx<'_>, aux: &mut Aux<'_>) {
        // scsynth's `BeatTrack2_Ctor` allocations. A negative feature count or trail length makes
        // its `RTAlloc` size wrap and fail, clearing the unit; a zero one leaves it reading memory
        // it never allocated. Either way this unit allocates nothing and stays cleared.
        let ins = ctx.ins;
        let krlength = ctx.own.buf_dur as f32;
        let (nf, _, buffersize) = Self::sizes(
            ins.control(Self::NUMFEATURES),
            ins.control(Self::WINDOWSIZE),
            krlength,
        );
        if nf < 1 || buffersize < 1 {
            return;
        }
        let Some(words) = mem_words(nf as usize, buffersize as usize) else {
            return;
        };
        if aux.alloc(words.saturating_mul(4)) {
            self.numfeatures = nf;
            self.buffersize = buffersize;
        }
    }

    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // scsynth's `BeatTrack2_Ctor`.
        let ins = ctx.ins;
        if self.numfeatures == 0 {
            ctx.done.mark_done();
            return DoneAction::Nothing;
        }
        self.krlength = ctx.own.buf_dur as f32;
        self.phaseaccuracy = ins.control(Self::PHASEACCURACY);
        let (_, fullwindowsize, _) = Self::sizes(
            ins.control(Self::NUMFEATURES),
            ins.control(Self::WINDOWSIZE),
            self.krlength,
        );
        self.temporalwindowsize = ins.control(Self::WINDOWSIZE);
        self.fullwindowsize = fullwindowsize;

        let phaseaccuracy = self.phaseaccuracy;
        let Some(mem) = self.mem(&mut ctx.aux) else {
            return DoneAction::Nothing;
        };
        for (num, &period) in mem.numphases.iter_mut().zip(&PERIODS) {
            // At most 1 / 0.02 = 50 at the default accuracy. A non-finite count (a zero accuracy)
            // is undefined in C; it is taken as no phases, what x86 produces.
            let q = period / phaseaccuracy;
            *num = if q.is_finite() { q as i32 } else { 0 };
        }
        mem.pastfeatures.fill(0.0);
        mem.bestscore.fill(-9999.0);
        mem.bestphase.fill(0);
        mem.besttempo.fill(60);
        mem.bestgroove.fill(0);

        self.counter = 0;
        self.phase = 0.0;
        self.period = 0.5;
        self.groove = 0;
        self.currtempo = 2.0;
        self.phaseperblock = self.krlength / self.period;
        self.predictphase = 0.4;
        self.predictperiod = 0.3;
        self.outputphase = self.phase;
        self.outputtempo = self.currtempo;
        self.outputgroove = self.groove as f32;
        self.outputphaseperblock = self.phaseperblock;
        self.calculationperiod = 0.5;
        self.calculationschedule = 0.0;

        // A buffer number past the World's buffers means buffer 0. scsynth compares the signed
        // number with the unsigned buffer count, so a negative number (the language's default,
        // -2.1) is past them too, and the flat scheme is never selected.
        let mut bufnum = (ins.control(Self::WEIGHTINGSCHEME) + 0.001f32) as i32;
        if bufnum as u32 >= unit::num_buffers(ctx.buffers) as u32 {
            bufnum = 0;
        }
        if bufnum < 0 {
            self.weightingscheme = 0;
        } else {
            self.tempoweights = bufnum as u32;
            self.weightingscheme = 2;
        }

        self.halftrig = 0;
        self.q1trig = 0;
        self.q2trig = 0;
        Self::write_outputs(
            ctx,
            [
                0.0,
                0.0,
                0.0,
                self.outputtempo,
                self.outputphase,
                self.outputgroove,
            ],
        );
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if self.numfeatures == 0 {
            // Cleared, as scsynth's `ClearUnitOutputs`.
            Self::write_outputs(ctx, [0.0; 6]);
            ctx.done.mark_done();
            return DoneAction::Nothing;
        }
        let ins = ctx.ins;
        let nf = self.numfeatures as usize;
        let buffersize = self.buffersize;

        // Record this block's features: `numfeatures` consecutive control buses from `busindex`.
        self.counter = (self.counter + 1) % buffersize;
        let busnum = (ins.control(Self::BUSINDEX) + 0.001f32) as i32;
        let Some(mut mem) = Mem::split(&mut ctx.aux, nf, buffersize as usize) else {
            return DoneAction::Nothing;
        };
        for j in 0..nf {
            let value = busnum
                .checked_add(j as i32)
                .and_then(|bus| usize::try_from(bus).ok())
                .map_or(0.0, |bus| unit::control_in(ctx.buses, bus));
            mem.pastfeatures[j * buffersize as usize + self.counter as usize] = value;
        }

        self.calculationschedule += self.krlength;

        // A new search every `calculationperiod` seconds: move this search's best and second best
        // to the previous-search ranks and reset them (the groove ranks are not reset).
        if self.calculationschedule > self.calculationperiod {
            self.calculationschedule -= self.calculationperiod;
            for i in 0..2 {
                let pos1 = (2 + i) * nf;
                let pos2 = i * nf;
                for j in 0..nf {
                    mem.bestscore[pos1 + j] = mem.bestscore[pos2 + j];
                    mem.bestscore[pos2 + j] = -9999.0;
                    mem.bestphase[pos1 + j] = mem.bestphase[pos2 + j];
                    mem.bestphase[pos2 + j] = 0;
                    mem.besttempo[pos1 + j] = mem.besttempo[pos2 + j];
                    mem.besttempo[pos2 + j] = 60;
                }
            }
            self.amortisationstate = 1;
            self.amortcount = 0;
            self.amortlength = NUM_TEMPI as i32 * 2;
            self.startcounter = self.counter;
            self.currphase = self.phase;
        }

        match self.amortisationstate {
            1 => {
                // One tempo and groove per block.
                let step = self.amortcount;
                self.calculate_template(
                    ctx.buffers,
                    &ctx.local_bufs,
                    &mut mem,
                    (step >> 1) as usize,
                    (step % 2) as usize,
                );
                self.amortcount += 1;
                if self.amortcount == self.amortlength {
                    self.amortisationstate = 2;
                }
            }
            2 => {
                self.final_decision(&mem);
                self.amortisationstate = 0;
            }
            _ => {}
        }

        self.phase += self.phaseperblock;

        // Unlocked, the outputs follow the model; locked, their phasor runs on at its last tempo.
        if ins.control(Self::LOCK) < 0.5 {
            self.outputphase = self.phase;
            self.outputtempo = self.currtempo;
            self.outputgroove = self.groove as f32;
            self.outputphaseperblock = self.phaseperblock;
        } else {
            self.outputphase += self.outputphaseperblock;
        }

        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        // The phase output is the value before a beat wraps it.
        let mut out = [
            0.0,
            0.0,
            0.0,
            self.outputtempo,
            self.outputphase,
            self.outputgroove,
        ];

        if self.outputphase >= 1.0 {
            self.outputphase -= 1.0;
            out[0] = 1.0;
            out[1] = 1.0;
            out[2] = 1.0;
            self.halftrig = 0;
            self.q1trig = 0;
            self.q2trig = 0;
        }

        if self.outputphase as f64 >= 0.5 && self.halftrig == 0 {
            out[1] = 1.0;
            out[2] = 1.0;
            self.halftrig = 1;
        }

        let groove = (self.outputgroove as f64 * 0.07) as f32;

        if self.outputphase as f64 >= (0.25 + groove as f64) && self.q1trig == 0 {
            out[2] = 1.0;
            self.q1trig = 1;
        }

        if self.outputphase as f64 >= (0.75 + groove as f64) && self.q2trig == 0 {
            out[2] = 1.0;
            self.q2trig = 1;
        }

        Self::write_outputs(ctx, out);
        DoneAction::Nothing
    }
}

/// Constructor for [`BeatTrack2`]: the unit sizes its memory from its inputs when the synth starts.
pub struct BeatTrack2Ctor;

impl UnitDef for BeatTrack2Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= BeatTrack2::WEIGHTINGSCHEME {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec_pool(BeatTrack2::zeroed()))
    }
}
