//! `BeatTrack` - plyphon's port of scsynth's autocorrelation beat tracker (`BeatTrack.cpp`, Nick
//! Collins after Matthew Davies).
//!
//! Each FFT frame of the chain (1024 points, or the lower half of a 2048-point frame at 88.2/96
//! kHz) adds one value to a complex-domain onset detection function, smoothed and peak-picked over
//! 15 frames. Every 128 frames the unit starts an analysis amortised over the following control
//! blocks: the autocorrelation of the last 512 detection-function values, a tempo pick from it under
//! a general or (once a steady tempo is established) a context-weighted comb, a check for a change
//! of tempo, and a search for the beat phase. A phasor at the chosen tempo drives the outputs.
//!
//! Compiled only with the `fft` feature.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, Aux, BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec_aux};
use plyphon_dsp::math;
use plyphon_dsp::rate::RateInfo;

/// Half the 1024-point FFT the detection function assumes - scsynth's `NOVER2`.
const NOVER2: usize = 512;
/// Detection-function values autocorrelated per analysis - scsynth's `DFFRAMELENGTH`.
const DFFRAMELENGTH: i32 = 512;
/// Detection-function history - scsynth's `DFSTORE`.
const DFSTORE: i32 = 700;
/// Tempo lags searched - scsynth's `LAGS`.
const LAGS: usize = 128;
/// Frames between analyses - scsynth's `SKIP`.
const SKIP: i64 = 128;
/// Seconds per 1024-point hop at 44.1 kHz - scsynth's `FRAMEPERIOD`.
const FRAMEPERIOD: f64 = 0.01161;

/// The unit's `aux`: floats, in scsynth's order (the three `RTAlloc`'d arrays, then the arrays the
/// struct holds). `m_acf` is directly followed by `m_mg`, as in scsynth's struct, because the
/// tempo pick reads up to three values past the end of `m_acf`.
const PREVMAG: usize = 0;
const PREVPHASE: usize = PREVMAG + NOVER2;
const PREDICT: usize = PREVPHASE + NOVER2;
const DF: usize = PREDICT + NOVER2;
const ACF: usize = DF + DFSTORE as usize;
const MG: usize = ACF + DFFRAMELENGTH as usize;
const PHASEWEIGHTS: usize = MG + LAGS;
const AUX_FLOATS: usize = PHASEWEIGHTS + LAGS;

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

/// The unit's memory, split into scsynth's arrays.
struct Mem<'a> {
    prevmag: &'a mut [f32],
    prevphase: &'a mut [f32],
    predict: &'a mut [f32],
    df: &'a mut [f32],
    /// `m_acf` (512) followed by `m_mg` (128).
    acf_mg: &'a mut [f32],
    phaseweights: &'a mut [f32],
}

impl<'a> Mem<'a> {
    fn split(aux: &'a mut Aux<'_>) -> Option<Mem<'a>> {
        let floats = aux.f32_mut().get_mut(..AUX_FLOATS)?;
        let (prevmag, rest) = floats.split_at_mut(PREVPHASE - PREVMAG);
        let (prevphase, rest) = rest.split_at_mut(PREDICT - PREVPHASE);
        let (predict, rest) = rest.split_at_mut(DF - PREDICT);
        let (df, rest) = rest.split_at_mut(ACF - DF);
        let (acf_mg, phaseweights) = rest.split_at_mut(PHASEWEIGHTS - ACF);
        Some(Mem {
            prevmag,
            prevphase,
            predict,
            df,
            acf_mg,
            phaseweights,
        })
    }

    /// Detection-function value `i` of the ring (always in range: the indices are built from
    /// non-negative offsets modulo [`DFSTORE`]).
    fn df(&self, i: i32) -> f32 {
        usize::try_from(i)
            .ok()
            .and_then(|i| self.df.get(i))
            .copied()
            .unwrap_or(0.0)
    }
}

/// `BeatTrack.kr(chain, lock = 0)`: four outputs - beat, quaver and semiquaver triggers, and the
/// tempo in beats per second. `lock >= 0.5` freezes the outputs' tempo and keeps their phasor running
/// while the model goes on tracking.
///
/// The chain buffer's first 1024 floats are read as Cartesian bins whatever the buffer's coordinate
/// form, as scsynth does. `aux` holds, in floats: the previous frame's magnitudes and phases and the
/// predicted phases (`m_prevmag`, `m_prevphase`, `m_predict`, 512 each), the detection function
/// (`m_df`, 700), its autocorrelation (`m_acf`, 512) directly followed by the general model's
/// weights (`m_mg`, 128), and the phase weights (`m_phaseweights`, 128).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BeatTrack {
    /// Frames seen, from 1 - scsynth's `m_frame`.
    frame: i64,
    /// `m_srate / 44100` - scsynth's `m_srateconversion`.
    srateconversion: f32,
    /// Seconds per frame - scsynth's `m_frameperiod`.
    frameperiod: f32,
    /// Detection-function write position - scsynth's `m_dfcounter`.
    dfcounter: i32,
    /// Peak-picking ring position - scsynth's `m_dfmemorycounter`.
    dfmemorycounter: i32,
    /// The last 15 raw detection values - scsynth's `m_dfmemory`.
    dfmemory: [f32; 15],
    /// Best comb score of the tempo pick - scsynth's `m_besttorsum`.
    besttorsum: f32,
    /// Lag of the best comb score - scsynth's `m_bestcolumn`.
    bestcolumn: i32,
    /// Beat period in frames - scsynth's `m_tor`.
    tor: f32,
    /// `tor` rounded - scsynth's `m_torround`.
    torround: i32,
    /// Period under the general model - scsynth's `m_periodp`.
    periodp: f32,
    /// Period under the context model - scsynth's `m_periodg`.
    periodg: f32,
    /// Countdown confirming a tempo step - scsynth's `m_flagstep`.
    flagstep: i32,
    /// The general-model periods during the countdown - scsynth's `m_prevperiodp`.
    prevperiodp: [f32; 3],
    /// Best phase score - scsynth's `m_bestphasescore`.
    bestphasescore: f32,
    /// Phase (in frames) of the best phase score - scsynth's `m_bestphase`.
    bestphase: i32,
    /// The model's tempo in beats per second - scsynth's `m_currtempo`.
    currtempo: f32,
    /// The phase when the analysis began - scsynth's `m_currphase`.
    currphase: f32,
    /// The model's beat phasor - scsynth's `m_phase`.
    phase: f32,
    /// Phase advance per control block - scsynth's `m_phaseperblock`.
    phaseperblock: f32,
    /// The output phasor, which runs on alone while locked - scsynth's `m_outputphase`.
    outputphase: f32,
    /// The output tempo - scsynth's `m_outputtempo`.
    outputtempo: f32,
    /// The output phasor's advance per control block - scsynth's `m_outputphaseperblock`.
    outputphaseperblock: f32,
    /// Whether the half-beat trigger has fired this beat - scsynth's `halftrig`.
    halftrig: i32,
    /// Whether the first quarter-beat trigger has fired this beat - scsynth's `q1trig`.
    q1trig: i32,
    /// Whether the third quarter-beat trigger has fired this beat - scsynth's `q2trig`.
    q2trig: i32,
    /// The analysis step (`0` idle, `1`-`8` the stages) - scsynth's `m_amortisationstate`.
    amortisationstate: i32,
    /// Steps done in the current stage - scsynth's `m_amortcount`.
    amortcount: i32,
    /// Steps in the current stage - scsynth's `m_amortlength`.
    amortlength: i32,
    /// Control blocks since the analysis began - scsynth's `m_amortisationsteps`.
    amortisationsteps: i32,
    /// `1` once the context model is in use - scsynth's `m_stateflag`.
    stateflag: i32,
    /// Beats per bar (always 4) - scsynth's `m_timesig`.
    timesig: i32,
    /// Detection-function position the analysis autocorrelates from - scsynth's
    /// `m_storedfcounter`.
    storedfcounter: i32,
    /// Detection-function position the phase search counts back from - scsynth's
    /// `m_storedfcounterend`.
    storedfcounterend: i32,
    /// Explicit padding to the struct's 8-byte alignment.
    _pad: u32,
}

impl BeatTrack {
    const CHAIN: usize = 0;
    const LOCK: usize = 1;

    /// scsynth's `complexdf`: add one complex-domain onset value for this frame, then smooth and
    /// peak-pick the last 15 into the detection function.
    fn complexdf(&mut self, mem: &mut Mem<'_>, fftbuf: &[f32]) {
        let mut sum = 0.0f32;
        for k in 1..NOVER2 {
            let index = 2 * k;
            let real = fftbuf[index];
            let imag = fftbuf[index + 1];
            let mag = math::sqrt(real * real + imag * imag);
            let qmag = mem.prevmag[k];
            mem.prevmag[k] = mag;
            let phase = math::atan2(imag, real);
            let oldphase = mem.predict[k];
            mem.predict[k] = 2.0 * phase - mem.prevphase[k];
            mem.prevphase[k] = phase;
            let phasediff = phase - oldphase;
            let realpart = qmag - (mag * math::cos(phasediff));
            let imagpart = mag * math::sin(phasediff);
            sum += math::sqrt(realpart * realpart + imagpart * imagpart);
        }

        // Peak-pick against the 15 most recent values, centred 7 frames back.
        self.dfmemorycounter = (self.dfmemorycounter + 1) % 15;
        self.dfmemory[self.dfmemorycounter as usize] = sum;
        let refpos = self.dfmemorycounter + 15;
        let centreval = self.dfmemory[((refpos - 7) % 15) as usize];
        let mut rating = 0.0f32;
        for k in 0..15 {
            let mut nextval = centreval - self.dfmemory[((refpos - k) % 15) as usize];
            if nextval < 0.0 {
                nextval *= 10.0;
            }
            rating += nextval;
        }
        if rating < 0.0 {
            rating = 0.0;
        }
        self.dfcounter = (self.dfcounter + 1) % DFSTORE;
        mem.df[self.dfcounter as usize] = rating * 0.1f32;
    }

    /// scsynth's `BeatTrack_dofft`: fold in a frame, and every [`SKIP`] frames begin an analysis.
    fn dofft(&mut self, mem: &mut Mem<'_>, fftbuf: &[f32]) {
        self.complexdf(mem, fftbuf);
        if self.frame % SKIP == 0 {
            self.bestcolumn = 0;
            self.besttorsum = -1000.0;
            self.bestphasescore = -1000.0;
            self.bestphase = 0;
            self.amortisationstate = 1;
            self.amortcount = 0;
            self.amortlength = 128;
            self.amortisationsteps = 0;
            // Fix the time reference for the analysis: the start of the last 512 values, and the
            // newest value for the phase search.
            self.storedfcounter = self.dfcounter + DFSTORE - DFFRAMELENGTH;
            self.storedfcounterend = self.dfcounter;
            self.currphase = self.phase;
        }
    }

    /// scsynth's `autocorr`: autocorrelation lags `4j` to `4j + 3`, each scaled by
    /// `|lag - 512|`.
    fn autocorr(&self, mem: &mut Mem<'_>, j: i32) {
        let baseframe = self.storedfcounter + DFSTORE;
        for k in 0..4 {
            let lag = 4 * j + k;
            let correction = (lag - DFFRAMELENGTH).abs();
            let mut sum = 0.0f32;
            for i in lag..DFFRAMELENGTH {
                let val1 = mem.df((i + baseframe) % DFSTORE);
                let val2 = mem.df((i + baseframe - lag) % DFSTORE);
                sum += val1 * val2;
            }
            mem.acf_mg[lag as usize] = sum * correction as f32;
        }
    }

    /// scsynth's `findtor`: refine the best lag with the peaks near its multiples into a period in
    /// frames. Indices are one-based as in the MATLAB original; the largest reach three values past
    /// the end of `m_acf`, into `m_mg`, as in scsynth.
    fn findtor(&self, mem: &Mem<'_>) -> f32 {
        let ind = self.bestcolumn + 1;
        let acf = |i: i32| mem.acf_mg[(i - 1) as usize];

        let argmax = |lo: i32, hi: i32| {
            let mut best = 0;
            let mut maxval = -1000.0f32;
            for i in lo..=hi {
                let val = acf(i);
                if val > maxval {
                    maxval = val;
                    best = i - lo + 1;
                }
            }
            best
        };

        let ind2 = argmax(2 * ind - 1, 2 * ind + 1) + 2 * (ind + 1) - 2;
        let ind3 = argmax(3 * ind - 2, 3 * ind + 2) + 3 * ind - 4;
        let third = (ind3 as f32 / 3.0f32) as f64;
        if self.timesig == 4 {
            let ind4 = argmax(4 * ind - 3, 4 * ind + 3) + 4 * ind - 9;
            ((ind as f64 + ind2 as f64 * 0.5 + third + ind4 as f64 * 0.25) * 0.25) as f32
        } else {
            ((ind as f64 + ind2 as f64 * 0.5 + third) * 0.3333333) as f32
        }
    }

    /// scsynth's `beatperiod`: the comb score of lag `j` over `timesig` harmonics, weighted by the
    /// context model (`whichm != 0`, `g_m`) or the general model (`m_mg`).
    fn beatperiod(&mut self, mem: &Mem<'_>, j: i32, whichm: bool) {
        let mut sum = 0.0f32;
        for i in 1..=self.timesig {
            let num = 2 * i - 1;
            let wt = (1.0f64 / num as f32 as f64) as f32;
            for k in 0..num {
                let pos = k + i * j;
                if pos < 512 {
                    sum += mem.acf_mg[pos as usize] * wt;
                }
            }
        }
        let m = if whichm {
            G_M[j as usize]
        } else {
            mem.acf_mg[MG - ACF + j as usize]
        };
        sum *= m;
        if sum > self.besttorsum {
            self.besttorsum = sum;
            self.bestcolumn = j;
        }
    }

    /// scsynth's `findphase`: score phase `j` (in frames) by the detection function at each period
    /// back from the newest value, weighted `1 / k`, and optionally by the Gaussian around the
    /// predicted phase.
    fn findphase(&mut self, mem: &Mem<'_>, j: i32, gaussflag: bool, predicted: i32) {
        let period = self.torround;
        let baseframe = self.storedfcounterend + DFSTORE;
        let numfit = if period != 0 {
            DFFRAMELENGTH / period - 1
        } else {
            -1
        };
        let mut sum = 0.0f32;
        for k in 0..numfit {
            let location = (baseframe - (period * k) - j) % DFSTORE;
            sum += mem.df(location) / (k + 1) as f32;
        }
        if gaussflag {
            // Distance from the prediction within the period (always below the 128 weights).
            let diff = (predicted - j).abs().min((period - predicted + j).abs());
            sum *= usize::try_from(diff)
                .ok()
                .and_then(|d| mem.phaseweights.get(d))
                .copied()
                .unwrap_or(0.0);
        }
        if sum > self.bestphasescore {
            self.bestphasescore = sum;
            self.bestphase = j;
        }
    }

    /// scsynth's `setupphaseexpectation`: a Gaussian over phase distance, deviation a quarter
    /// period.
    fn setupphaseexpectation(&self, mem: &mut Mem<'_>) {
        let sigma = self.torround as f32 * 0.25f32;
        let mult = (1.0 / (2.5066283 * sigma as f64)) as f32;
        let mult2 = (1.0 / (2.0 * sigma as f64 * sigma as f64)) as f32;
        for (i, w) in mem.phaseweights.iter_mut().enumerate() {
            let i = i as i32;
            *w = mult * math::exp(-(i * i) as f32 * mult2);
        }
    }

    /// scsynth's `detectperiodchange`: a step between the models' periods must hold for three
    /// analyses before it counts.
    fn detectperiodchange(&mut self) -> bool {
        if self.flagstep == 0 {
            if (self.periodg - self.periodp).abs() > 3.9017f32 {
                self.flagstep = 3;
            }
        } else {
            self.flagstep -= 1;
        }
        if self.flagstep != 0 {
            self.prevperiodp[(self.flagstep - 1) as usize] = self.periodp;
        }
        if self.flagstep == 1 {
            self.flagstep = 0;
            let p = &self.prevperiodp;
            if (2.0 * p[0] - p[1] - p[2]).abs() < 7.8034f32 {
                return true;
            }
        }
        false
    }

    /// scsynth's `finaldecision`: adopt the period and phase, advancing the phase by the time the
    /// analysis took plus the detection function's seven-frame delay.
    fn finaldecision(&mut self, audio: &RateInfo) {
        let bl = audio.block_size as f32;
        let sr = audio.sample_rate as f32;
        self.currtempo = (1.0 / (self.tor * self.frameperiod) as f64) as f32;
        self.phaseperblock = (bl * self.currtempo) / sr;
        let mut timeelapsed = (self.amortisationsteps as f32 * bl) / sr;
        timeelapsed += 7.0 * self.frameperiod;
        let phaseelapsed = timeelapsed * self.currtempo;
        let phasebeforeamort = self.bestphase as f32 / self.torround as f32;
        self.phase = (phasebeforeamort + phaseelapsed) % 1.0f32;
        self.currphase = self.phase;
    }

    /// One step of the amortised analysis - the state switch at the top of scsynth's
    /// `BeatTrack_next`.
    fn amortise(&mut self, audio: &RateInfo, mem: &mut Mem<'_>) {
        match self.amortisationstate {
            // Autocorrelation, four lags per block.
            1 => {
                self.autocorr(mem, self.amortcount);
                self.amortcount += 1;
                if self.amortcount == self.amortlength {
                    self.amortisationstate = 2;
                    self.amortlength = 128;
                    self.amortcount = 0;
                    self.bestcolumn = 0;
                    self.besttorsum = -1000.0;
                }
            }
            // The general model's period, one lag per block.
            2 => {
                self.beatperiod(mem, self.amortcount, false);
                self.amortcount += 1;
                if self.amortcount == self.amortlength {
                    self.periodp = self.findtor(mem);
                    if self.stateflag == 1 {
                        self.amortisationstate = 3;
                        self.amortlength = 128;
                        self.amortcount = 0;
                        self.bestcolumn = 0;
                        self.besttorsum = -1000.0;
                    } else {
                        // Always a step the first time.
                        self.periodg = -1000.0;
                        self.amortisationstate = 4;
                    }
                }
            }
            // The context model's period.
            3 => {
                self.beatperiod(mem, self.amortcount, true);
                self.amortcount += 1;
                if self.amortcount == self.amortlength {
                    self.periodg = self.findtor(mem);
                    self.amortisationstate = 4;
                }
            }
            // A confirmed step moves the context to the new period; otherwise search the phase.
            4 => {
                if self.detectperiodchange() {
                    self.amortisationstate = 5;
                    self.amortlength = 128;
                    self.amortcount = 0;
                    self.bestcolumn = 0;
                    self.besttorsum = -1000.0;
                    self.stateflag = 1;
                    // `findmeter`: scsynth settles on 4/4.
                    self.timesig = 4;
                    // Centre the general model's Gaussian on the period.
                    let startindex = 128 - (self.periodp as f64 + 0.5) as i32;
                    for (ii, m) in mem.acf_mg[MG - ACF..].iter_mut().enumerate() {
                        *m = usize::try_from(startindex + ii as i32)
                            .ok()
                            .and_then(|i| G_MG.get(i))
                            .copied()
                            .unwrap_or(0.0);
                    }
                } else {
                    self.tor = if self.stateflag == 1 {
                        self.periodg
                    } else {
                        self.periodp
                    };
                    self.torround = (self.tor as f64 + 0.5) as i32;
                    self.amortisationstate = 7;
                    self.amortlength = self.torround;
                    self.amortcount = 0;
                }
            }
            // Redo the context model's period after a step, then a flat phase search.
            5 => {
                self.beatperiod(mem, self.amortcount, true);
                self.amortcount += 1;
                if self.amortcount == self.amortlength {
                    self.periodg = self.findtor(mem);
                    self.tor = self.periodg;
                    self.torround = (self.tor + 0.5f32) as i32;
                    self.amortisationstate = 6;
                    self.amortlength = self.torround;
                    self.amortcount = 0;
                    self.setupphaseexpectation(mem);
                }
            }
            // Flat phase search, one phase per block.
            6 => {
                self.findphase(mem, self.amortcount, false, 0);
                self.amortcount += 1;
                if self.amortcount == self.amortlength {
                    self.amortisationstate = 8;
                }
            }
            // Phase search, narrowed around the predicted phase under the context model.
            7 => {
                let predicted = (self.currphase * self.torround as f32 + 0.5f32) as i32;
                self.findphase(mem, self.amortcount, self.stateflag != 0, predicted);
                self.amortcount += 1;
                if self.amortcount == self.amortlength {
                    self.amortisationstate = 8;
                }
            }
            8 => {
                self.finaldecision(audio);
                self.amortisationstate = 0;
            }
            _ => {}
        }
    }
}

impl Unit for BeatTrack {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // scsynth's `BeatTrack_Ctor`: halve an 88.2/96 kHz rate (a double-size FFT is assumed) and
        // scale the frame period by it; a zeroed detection function; 2 beats per second.
        let mut srate = ctx.audio.sample_rate as f32;
        if srate as f64 > 44100.0 * 1.5 {
            srate = (srate as f64 * 0.5) as f32;
        }
        self.srateconversion = (srate as f64 / 44100.0) as f32;
        self.frameperiod = (FRAMEPERIOD / self.srateconversion as f64) as f32;

        if let Some(mem) = Mem::split(&mut ctx.aux) {
            // scsynth leaves the three `RTAlloc`'d arrays and `m_mg` uninitialised and clears
            // `m_df`; all start at zero here.
            mem.prevmag.fill(0.0);
            mem.prevphase.fill(0.0);
            mem.predict.fill(0.0);
            mem.df.fill(0.0);
            mem.acf_mg.fill(0.0);
            mem.phaseweights.fill(0.0);
        }

        self.frame = 1;
        self.dfcounter = DFSTORE - 1;
        self.dfmemorycounter = 14;
        self.dfmemory = [0.0; 15];
        self.currtempo = 2.0;
        self.currphase = 0.0;
        self.phase = 0.0;
        self.phaseperblock = (ctx.audio.block_size as f32 * 2.0) / ctx.audio.sample_rate as f32;
        self.outputphase = self.phase;
        self.outputtempo = self.currtempo;
        self.outputphaseperblock = self.phaseperblock;
        self.halftrig = 0;
        self.q1trig = 0;
        self.q2trig = 0;
        self.amortisationstate = 0;
        self.stateflag = 0;
        self.timesig = 4;
        self.flagstep = 0;

        *ctx.outs.control(0) = 0.0;
        *ctx.outs.control(1) = 0.0;
        *ctx.outs.control(2) = 0.0;
        *ctx.outs.control(3) = self.outputtempo;
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins;
        let Some(mut mem) = Mem::split(&mut ctx.aux) else {
            return DoneAction::Nothing;
        };

        // Reset by each analysis.
        self.amortisationsteps = self.amortisationsteps.wrapping_add(1);
        self.amortise(ctx.audio, &mut mem);

        // A ready frame (any `chain` not below 0) feeds the detection function. The first 1024
        // floats are read; a missing or smaller chain buffer, which scsynth reads past, adds none.
        let fbufnum = ins.control(Self::CHAIN);
        if fbufnum >= 0.0 || fbufnum.is_nan() {
            self.frame += 1;
            if let Some(buffer) =
                unit::buffer_at(ctx.buffers, &ctx.local_bufs, fbufnum as u32 as usize)
                && let Some(fftbuf) = buffer.data().get(..2 * NOVER2)
            {
                self.dofft(&mut mem, fftbuf);
            }
        }

        self.phase += self.phaseperblock;

        // Unlocked, the outputs follow the model; locked, their phasor runs on at its last tempo.
        if ins.control(Self::LOCK) < 0.5f32 {
            self.outputphase = self.phase;
            self.outputtempo = self.currtempo;
            self.outputphaseperblock = self.phaseperblock;
        } else {
            self.outputphase += self.outputphaseperblock;
        }

        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        let mut out = [0.0f32, 0.0, 0.0, self.outputtempo];
        if self.outputphase >= 1.0 {
            self.outputphase -= 1.0;
            out[0] = 1.0;
            out[1] = 1.0;
            out[2] = 1.0;
            self.halftrig = 0;
            self.q1trig = 0;
            self.q2trig = 0;
        }
        if self.outputphase >= 0.5 && self.halftrig == 0 {
            out[1] = 1.0;
            out[2] = 1.0;
            self.halftrig = 1;
        }
        if self.outputphase >= 0.25 && self.q1trig == 0 {
            out[2] = 1.0;
            self.q1trig = 1;
        }
        if self.outputphase >= 0.75 && self.q2trig == 0 {
            out[2] = 1.0;
            self.q2trig = 1;
        }
        for (i, v) in out.into_iter().enumerate() {
            *ctx.outs.control(i) = v;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`BeatTrack`]: its memory is a fixed size, reserved at build.
pub struct BeatTrackCtor;

impl UnitDef for BeatTrackCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= BeatTrack::LOCK {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec_aux(
            BeatTrack::zeroed(),
            AUX_FLOATS * core::mem::size_of::<f32>(),
            core::mem::align_of::<f32>(),
        ))
    }
}

/// The context-dependent (Gaussian) tempo weighting over the 128 lags - scsynth's `g_m`.
#[rustfmt::skip]
static G_M: [f32; 128] = to_f32([
    0.00054069, 0.00108050, 0.00161855, 0.00215399, 0.00268594, 0.00321356, 0.00373600, 0.00425243, 0.00476204,
    0.00526404, 0.00575765, 0.00624213, 0.00671675, 0.00718080, 0.00763362, 0.00807455, 0.00850299, 0.00891836,
    0.00932010, 0.00970771, 0.01008071, 0.01043866, 0.01078115, 0.01110782, 0.01141834, 0.01171242, 0.01198982,
    0.01225033, 0.01249378, 0.01272003, 0.01292899, 0.01312061, 0.01329488, 0.01345182, 0.01359148, 0.01371396,
    0.01381939, 0.01390794, 0.01397980, 0.01403520, 0.01407439, 0.01409768, 0.01410536, 0.01409780, 0.01407534,
    0.01403838, 0.01398734, 0.01392264, 0.01384474, 0.01375410, 0.01365120, 0.01353654, 0.01341062, 0.01327397,
    0.01312710, 0.01297054, 0.01280484, 0.01263053, 0.01244816, 0.01225827, 0.01206139, 0.01185807, 0.01164884,
    0.01143424, 0.01121478, 0.01099099, 0.01076337, 0.01053241, 0.01029861, 0.01006244, 0.00982437, 0.00958484,
    0.00934429, 0.00910314, 0.00886181, 0.00862067, 0.00838011, 0.00814049, 0.00790214, 0.00766540, 0.00743057,
    0.00719793, 0.00696778, 0.00674036, 0.00651591, 0.00629466, 0.00607682, 0.00586256, 0.00565208, 0.00544551,
    0.00524301, 0.00504470, 0.00485070, 0.00466109, 0.00447597, 0.00429540, 0.00411944, 0.00394813, 0.00378151,
    0.00361959, 0.00346238, 0.00330989, 0.00316210, 0.00301899, 0.00288053, 0.00274669, 0.00261741, 0.00249266,
    0.00237236, 0.00225646, 0.00214488, 0.00203755, 0.00193440, 0.00183532, 0.00174025, 0.00164909, 0.00156174,
    0.00147811, 0.00139810, 0.00132161, 0.00124854, 0.00117880, 0.00111228, 0.00104887, 0.00098848, 0.00093100,
    0.00087634, 0.00082438,
]);

/// A Gaussian over 257 lags, centred on lag 128, from which a 128-lag window centred on the
/// current period is taken - scsynth's `g_mg`.
#[rustfmt::skip]
static G_MG: [f32; 257] = to_f32([
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000004, 0.00000055, 0.00000627, 0.00005539, 0.00037863, 0.00200318, 0.00820201, 0.02599027, 0.06373712,
    0.12096648, 0.17767593, 0.20196826, 0.17767593, 0.12096648, 0.06373712, 0.02599027, 0.00820201, 0.00200318,
    0.00037863, 0.00005539, 0.00000627, 0.00000055, 0.00000004, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
]);
