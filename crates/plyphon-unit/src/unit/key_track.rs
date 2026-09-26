//! `KeyTrack` - plyphon's port of scsynth's key tracker (`KeyTrack.cpp`, Nick Collins).
//!
//! Each FFT frame of a 4096-point chain is reduced to a 12-bin chroma vector (60 notes from A1 to
//! G#6, each a weighted sum of the powers of the bins around its first six partials), which leaks by
//! `chromaleak` per frame. The chroma is matched against diatonic major and minor templates for all
//! 24 keys, the matches are integrated into a leaky key histogram whose decay reaches -40 dB after
//! `keydecay` seconds, and the output is the histogram's best key: `0`-`11` are C to B major,
//! `12`-`23` C to B minor.
//!
//! Compiled only with the `fft` feature.

mod tables;

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, pv, unit_spec_aux};
use plyphon_dsp::math;
use tables::{BINS_44100, BINS_48000, WEIGHTS_44100, WEIGHTS_48000};

/// Half the 4096-point FFT size the weighting tables assume - scsynth's `NOVER2`. The unit reads
/// the chain buffer's first `NOVER2` floats (the bins below half Nyquist) and allocates `NOVER2`
/// floats of scratch for their powers.
const NOVER2: usize = 2048;

/// Izmirli's diatonic major template (scsynth's `g_diatonicmajor`, a `double` table).
const DIATONIC_MAJOR: [f64; 12] = [5.0, 0.0, 3.5, 0.0, 4.5, 4.0, 0.0, 4.5, 0.0, 3.5, 0.0, 4.0];
/// Izmirli's diatonic minor template (scsynth's `g_diatonicminor`, a `double` table).
const DIATONIC_MINOR: [f64; 12] = [5.0, 0.0, 3.5, 4.5, 0.0, 4.0, 0.0, 4.5, 3.5, 0.0, 0.0, 4.0];
/// The major scale's pitch classes (scsynth's `g_major`).
const MAJOR: [usize; 7] = [0, 2, 4, 5, 7, 9, 11];
/// The natural and harmonic minor scale's pitch classes (scsynth's `g_minor`).
const MINOR: [usize; 7] = [0, 2, 3, 5, 7, 8, 11];

/// `KeyTrack.kr(chain, keydecay = 2.0, chromaleak = 0.5)`: the best-matching key of the chain's
/// spectrum, `0`-`23`. The key is re-evaluated on every ready frame and held between frames.
///
/// The chain is assumed to be a 4096-point FFT (2048 at 88.2/96 kHz, whose lower half stands in for
/// it); a sample rate other than 44100 (or 88200) uses scsynth's 48000 Hz tables. `aux` holds the
/// `NOVER2`-float power scratch (scsynth's `m_FFTBuf`), of which the first half is used.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct KeyTrack {
    /// Leaky pitch-class energy - scsynth's `m_chroma`.
    chroma: [f32; 12],
    /// The current frame's match against each key's template - scsynth's `m_key`.
    key: [f32; 24],
    /// Leaky integration of `key` - scsynth's `m_histogram`.
    histogram: [f32; 24],
    /// Seconds per 4096-point hop at the table's sample rate - scsynth's `m_frameperiod`.
    frameperiod: f32,
    /// The histogram's current best key - scsynth's `m_currentKey`.
    current_key: i32,
    /// `1` when the 48000 Hz tables are in use, `0` for the 44100 Hz ones (scsynth's `m_weights`
    /// and `m_bins` pointers).
    tables48000: u32,
}

impl KeyTrack {
    const CHAIN: usize = 0;
    const KEYDECAY: usize = 1;
    const CHROMALEAK: usize = 2;

    /// scsynth's `KeyTrack_calculatekey`: fold one ready frame of chain buffer `bufnum` into the
    /// chroma and key histogram, and pick the new best key.
    fn calculate_key(&mut self, ctx: &mut ProcessCtx<'_>, bufnum: usize) {
        let ins = ctx.ins;
        let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum) else {
            return;
        };
        // The powers are computed from Cartesian bins: a polar frame is converted in place with
        // scsynth's lookup-table `ToComplexApx` (KeyTrack.cpp:1093).
        pv::to_complex(&mut buffer, ctx.fft.complex());
        let data = buffer.data();
        // scsynth reads the first `NOVER2` floats whatever the buffer's size; a smaller chain buffer
        // has no such data, and the frame is skipped.
        if data.len() < NOVER2 {
            return;
        }
        let fftbuf = ctx.aux.f32_mut();
        if fftbuf.len() < NOVER2 {
            return;
        }

        // Powers of the bins below half Nyquist; entry 0 holds `dc^2 + nyq^2`.
        for i in (0..NOVER2).step_by(2) {
            fftbuf[i >> 1] = (data[i] * data[i]) + (data[i + 1] * data[i + 1]);
        }

        let (weights, bins) = if self.tables48000 != 0 {
            (&WEIGHTS_48000, &BINS_48000)
        } else {
            (&WEIGHTS_44100, &BINS_44100)
        };

        let chromaleak = ins.control(Self::CHROMALEAK);
        for c in self.chroma.iter_mut() {
            *c *= chromaleak;
        }
        for i in 0..60 {
            // Note 0 is A1.
            let chromaindex = (i + 9) % 12;
            let indexbase = 12 * i;
            let mut sum = 0.0f32;
            for j in 0..12 {
                let index = indexbase + j;
                sum += weights[index] * fftbuf[bins[index] as usize];
            }
            self.chroma[chromaindex] += sum;
        }

        // The templates are `double`, so each term is summed in double and stored back to float.
        for (i, key) in self.key[..12].iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for &degree in &MAJOR {
                let index = (i + degree) % 12;
                sum = (sum as f64 + self.chroma[index] as f64 * DIATONIC_MAJOR[degree]) as f32;
            }
            *key = sum;
        }
        for (i, key) in self.key[12..].iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for &degree in &MINOR {
                let index = (i + degree) % 12;
                sum = (sum as f64 + self.chroma[index] as f64 * DIATONIC_MINOR[degree]) as f32;
            }
            *key = sum;
        }

        // `keydecay` seconds to fall 40 dB, as a per-frame leak: `0.01 ^ (1 / frames)`, with the
        // frame count floored at 0.001 (scsynth's `sc_max(0.001f, keyleak / m_frameperiod)`).
        let frames = ins.control(Self::KEYDECAY) / self.frameperiod;
        let frames = if 0.001f32 > frames { 0.001f32 } else { frames };
        let keyleak = math::powf(0.01f32, 1.0f32 / frames);

        let mut bestkey = 0;
        let mut bestscore = 0.0f32;
        for (i, (h, &k)) in self.histogram.iter_mut().zip(&self.key).enumerate() {
            *h = (keyleak * *h) + k;
            if *h > bestscore {
                bestscore = *h;
                bestkey = i as i32;
            }
        }
        self.current_key = bestkey;
    }
}

impl Unit for KeyTrack {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // scsynth's `KeyTrack_Ctor`: halve an 88.2/96 kHz rate (a double-size FFT is assumed), then
        // pick the tables by rate - 44100, or 48000 for anything else. The chroma, key and histogram
        // start at zero (the zeroed state), and the output is 0.
        let mut srate = ctx.audio.sample_rate as f32;
        if srate as f64 > 44100.0 * 1.5 {
            srate = (srate as f64 * 0.5) as f32;
        }
        if (srate as f64 + 0.01) as i32 == 44100 {
            self.tables48000 = 0;
            self.frameperiod = 0.046439909297052f64 as f32;
        } else {
            self.tables48000 = 1;
            self.frameperiod = 0.042666666666667f64 as f32;
        }
        self.current_key = 0;
        *ctx.outs.control(0) = 0.0;
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // scsynth's `KeyTrack_next`: a frame is ready when `chain + 0.001 > -0.01`.
        let fbufnum = (ctx.ins.control(Self::CHAIN) as f64 + 0.001) as f32;
        if fbufnum > -0.01f32 {
            self.calculate_key(ctx, fbufnum as u32 as usize);
        }
        *ctx.outs.control(0) = self.current_key as f32;
        DoneAction::Nothing
    }
}

/// Constructor for [`KeyTrack`]: the power scratch is a fixed `NOVER2` floats, reserved at build.
pub struct KeyTrackCtor;

impl UnitDef for KeyTrackCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= KeyTrack::CHROMALEAK {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec_aux(
            KeyTrack::zeroed(),
            NOVER2 * core::mem::size_of::<f32>(),
            core::mem::align_of::<f32>(),
        ))
    }
}
