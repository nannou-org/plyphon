//! Spectral onset detectors - plyphon's ports of scsynth's `PV_JensenAndersen` and
//! `PV_HainsworthFoote` (`FeatureDetection.cpp`).
//!
//! Each reads the FFT chain (input 0) and, on a frame, compares the frame's magnitudes with the
//! previous frame's, combines a few spectral features into one weighted sum, and fires when the sum
//! exceeds `threshold`: the whole output block is `1` on the block that fires and `0` otherwise.
//! After firing, the detector ignores further onsets until `waittime` seconds have passed.
//!
//! As in scsynth, the detector sizes itself from the chain buffer when the synth starts
//! (`PV_OnsetDetectionBase_Ctor`), keeping one magnitude per bin from the previous frame, and it
//! converts the chain frame to polar form in place with scsynth's lookup-table `ToPolarApx`
//! (`FeatureDetection.cpp`:169, 283). Compiled only with the `fft` feature.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{
    self, Aux, BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, pv, unit_spec_pool,
};
use plyphon_dsp::math;

/// The state every onset detector shares - scsynth's `PV_OnsetDetectionBase`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct OnsetBase {
    /// The chain buffer's bin count when the synth started (scsynth's `m_numbins`): `-1` when there
    /// was no buffer, as for scsynth's empty (zero-sample) buffer.
    numbins: i32,
    /// Magnitudes the previous-frame table holds, one per bin; `0` when the synth started without a
    /// chain buffer, so there is no table.
    prev_len: u32,
    /// `1` while the detector ignores onsets after firing.
    waiting: i32,
    /// Samples counted since firing.
    wait_samp: i32,
    /// Samples to ignore onsets for after firing.
    wait_len: i32,
    /// `1` when the synth started with a buffer too short to hold a frame, whose table size is
    /// negative: scsynth's allocation fails and the unit reports done, though it keeps running.
    alloc_failed: u32,
}

impl OnsetBase {
    /// scsynth's `PV_OnsetDetectionBase_Ctor`: read the chain buffer the synth starts with and, if
    /// there is one, allocate a zeroed previous-frame table of one magnitude per bin.
    fn alloc(&mut self, ctx: &InitCtx<'_>, aux: &mut Aux<'_>) {
        let fbufnum = ctx.ins.control(0);
        let buffer = (fbufnum >= 0.0)
            .then(|| unit::buffer_at(ctx.buffers, &ctx.local_bufs, fbufnum as usize))
            .flatten();
        let Some(buffer) = buffer else {
            self.numbins = -1;
            return;
        };
        let samples = i32::try_from(buffer.data().len()).unwrap_or(i32::MAX);
        self.numbins = (samples - 2) >> 1;
        let Ok(numbins) = usize::try_from(self.numbins) else {
            self.alloc_failed = 1;
            return;
        };
        if aux.alloc(numbins * core::mem::size_of::<f32>()) {
            aux.f32_mut().fill(0.0);
            self.prev_len = numbins as u32;
        }
    }

    /// The constructor's output: none, apart from the done flag a failed allocation raises.
    fn init(&self, ctx: &mut ProcessCtx<'_>) {
        if self.alloc_failed != 0 {
            ctx.done.mark_done();
        }
    }

    /// Count this block's samples towards the end of a wait.
    fn tick(&mut self, samples: i32) {
        if self.waiting == 1 {
            self.wait_samp = self.wait_samp.wrapping_add(samples);
            if self.wait_samp >= self.wait_len {
                self.waiting = 0;
            }
        }
    }

    /// Fire if `sum` exceeds `threshold` and no wait is running, starting a wait of `wait_len`
    /// samples. Returns the block's output value.
    fn fire(&mut self, sum: f32, threshold: f32, samples: i32, wait_len: f64) -> f32 {
        if sum > threshold && self.waiting == 0 {
            self.waiting = 1;
            self.wait_samp = samples;
            self.wait_len = wait_len as i32;
            1.0
        } else {
            0.0
        }
    }
}

/// Run `detect` over the frame on the chain, if one is ready, and fill the output block with the
/// value it returns (`0` between frames). The block's sample count (scsynth's `inNumSamples`) is
/// counted towards a running wait first.
///
/// `detect` receives the frame's polar bins (converted in place), its bin count, and the previous
/// frame's magnitudes (the table's first `bins.len()` entries), which it updates. A missing chain
/// buffer is a frame with no bins and a bin count of `-1`, as scsynth reads an empty buffer's: the
/// features are still computed, over nothing. A frame with more bins than the table holds is
/// skipped: the reference reads and writes past its table (or through a null one) there.
fn onset_block(
    ctx: &mut ProcessCtx<'_>,
    base: &mut OnsetBase,
    detect: impl FnOnce(&mut OnsetBase, &[pv::Bin], i32, &mut [f32], i32) -> f32,
) {
    let samples = ctx.outs.audio(0).len() as i32;
    base.tick(samples);
    let fbufnum = ctx.ins.control(0);
    let mut outval = 0.0;
    // The reference tests `!(fbufnum < 0)`, so a NaN chain counts as a frame too.
    if fbufnum >= 0.0 || fbufnum.is_nan() {
        let tables = ctx.fft.complex();
        let mut buffer = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, fbufnum as usize);
        // scsynth's `ToPolarApx` (`FeatureDetection.cpp`:169, 283).
        let (bins, numbins): (&[pv::Bin], i32) =
            match buffer.as_mut().and_then(|b| pv::to_polar(b, tables)) {
                Some(spectrum) => {
                    let len = spectrum.bins.len() as i32;
                    (spectrum.bins, len)
                }
                None => (&[], -1),
            };
        if bins.len() <= base.prev_len as usize {
            let prev = &mut ctx.aux.f32_mut()[..bins.len()];
            outval = detect(base, bins, numbins, prev, samples);
        }
    }
    ctx.outs.audio(0).fill(outval);
}

/// `PV_JensenAndersen(buffer, propsc, prophfe, prophfc, propsf, threshold, waittime)`: an onset
/// detector over four spectral features from Jensen and Andersen (2003).
///
/// Each frame computes, over the bins `k = 1..=numbins` (DC and Nyquist excluded) and normalised by
/// the bin count: the spectral centroid `sum(k * mag) / sum(mag)`, the high-frequency energy (the
/// magnitude sum above bin `(int)(4000 / sr) * numbins`, which is bin 0 at any rate above 4 kHz),
/// the high-frequency content `sum(k^2 * mag)`, and the spectral flux `sum(|mag - prevmag|)`. It
/// fires when `propsc`, `prophfe`, `prophfc` and `propsf` weight the four features' changes since
/// the previous frame into a sum above `threshold` (scsynth's `PV_JensenAndersen_next`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvJensenAndersen {
    base: OnsetBase,
    /// The previous frame's high-frequency content.
    hfc: f32,
    /// The previous frame's high-frequency energy.
    hfe: f32,
    /// The previous frame's spectral centroid.
    sc: f32,
    /// The previous frame's spectral flux.
    sf: f32,
    /// Bins above this index count towards the high-frequency energy.
    fourk_index: i32,
}

impl PvJensenAndersen {
    const PROPSC: usize = 1;
    const PROPHFE: usize = 2;
    const PROPHFC: usize = 3;
    const PROPSF: usize = 4;
    const THRESHOLD: usize = 5;
    const WAITTIME: usize = 6;
}

impl Unit for PvJensenAndersen {
    fn alloc(&mut self, ctx: &InitCtx<'_>, aux: &mut Aux<'_>) {
        self.base.alloc(ctx, aux);
    }

    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // scsynth's `PV_JensenAndersen_Ctor`: the index truncates `4000 / FULLRATE` before scaling
        // by the bin count. The output starts at zero (`ClearUnitOutputs`).
        self.base.init(ctx);
        let scale = (4000.0 / ctx.audio.sample_rate) as i32;
        self.fourk_index = scale.wrapping_mul(self.base.numbins);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins;
        let world_rate = world_sample_rate(ctx);
        let Self {
            base,
            hfc: prev_hfc,
            hfe: prev_hfe,
            sc: prev_sc,
            sf: prev_sf,
            fourk_index,
        } = self;
        let k4 = *fourk_index;
        onset_block(ctx, base, |base, bins, numbins, prev, samples| {
            let (mut magsum, mut magsumk, mut magsumkk, mut sfsum, mut hfesum) =
                (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for (i, (bin, &qmag)) in bins.iter().zip(prev.iter()).enumerate() {
                let mag = bin.x;
                let k = i as i32 + 1;
                magsum += mag;
                magsumk += k as f32 * mag;
                magsumkk += k.wrapping_mul(k) as f32 * mag;
                sfsum += (mag - qmag).abs();
                if i as i32 > k4 {
                    hfesum += mag;
                }
            }
            let binmult = 1.0f32 / numbins as f32;
            let sc = (magsumk / magsum) * binmult;
            let hfe = hfesum * binmult;
            let hfc = magsumkk * binmult * binmult * binmult;
            let sf = sfsum * binmult;
            let scdiff = sc - *prev_sc;
            let hfediff = hfe - *prev_hfe;
            let hfcdiff = hfc - *prev_hfc;
            let sfdiff = sf - *prev_sf;
            *prev_sc = sc;
            *prev_hfe = hfe;
            *prev_hfc = hfc;
            *prev_sf = sf;
            let sum = (ins.control(Self::PROPSC) * scdiff)
                + (ins.control(Self::PROPHFE) * hfediff)
                + (ins.control(Self::PROPHFC) * hfcdiff)
                + (ins.control(Self::PROPSF) * sfdiff);
            let wait_len = f64::from(ins.control(Self::WAITTIME)) * world_rate;
            let outval = base.fire(sum, ins.control(Self::THRESHOLD), samples, wait_len);
            for (q, bin) in prev.iter_mut().zip(bins) {
                *q = bin.x;
            }
            outval
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`PvJensenAndersen`]: the unit sizes its table from the chain buffer when the
/// synth starts.
pub struct PvJensenAndersenCtor;

impl UnitDef for PvJensenAndersenCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= PvJensenAndersen::WAITTIME {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec_pool(PvJensenAndersen::zeroed()))
    }
}

/// `1 / ln(2)`, the factor turning a natural log into a base-2 one: the reference's own literal,
/// rounded to `f32`.
#[allow(clippy::approx_constant)]
const LMULT: f32 = 1.442_695_040_889_f64 as f32;

/// `PV_HainsworthFoote(buffer, proph, propf, threshold, waittime)`: an onset detector combining
/// Hainsworth's modified Kullback-Leibler distance and Foote's spectral dissimilarity (Hainsworth
/// 2003).
///
/// Each frame computes the mean positive base-2 log-ratio of each magnitude to the previous frame's
/// (floored at `0.0001`) over the bins from 30 Hz to 5 kHz, and one minus the normalised correlation
/// of the two frames' magnitudes. It fires when `proph` and `propf` weight the two into a sum above
/// `threshold` (scsynth's `PV_HainsworthFoote_next`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvHainsworthFoote {
    base: OnsetBase,
    /// The previous frame's squared magnitude sum (`1` before the first frame).
    prev_norm: f32,
    /// The first bin past the Kullback-Leibler band, `(int)(5000 / sr * numbins)`.
    k5_index: i32,
    /// The first bin of the Kullback-Leibler band, `(int)(30 / sr * numbins)`.
    h30_index: i32,
    /// Padding for a whole number of words.
    _pad: u32,
}

impl PvHainsworthFoote {
    const PROPH: usize = 1;
    const PROPF: usize = 2;
    const THRESHOLD: usize = 3;
    const WAITTIME: usize = 4;
}

impl Unit for PvHainsworthFoote {
    fn alloc(&mut self, ctx: &InitCtx<'_>, aux: &mut Aux<'_>) {
        self.base.alloc(ctx, aux);
    }

    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // scsynth's `PV_HainsworthFoote_Ctor`, at the World's sample rate. The output starts at
        // zero (`ClearUnitOutputs`).
        self.base.init(ctx);
        let world_rate = world_sample_rate(ctx);
        let numbins = f64::from(self.base.numbins);
        self.k5_index = ((5000.0 / world_rate) * numbins) as i32;
        self.h30_index = ((30.0 / world_rate) * numbins) as i32;
        self.prev_norm = 1.0;
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins;
        let full_rate = ctx.audio.sample_rate;
        let Self {
            base,
            prev_norm,
            k5_index,
            h30_index,
            ..
        } = self;
        let (k5, h30) = (*k5_index, *h30_index);
        onset_block(ctx, base, |base, bins, _numbins, prev, samples| {
            let (mut mkl, mut footesum, mut norm) = (0.0f32, 0.0f32, 0.0f32);
            for (i, (bin, &qmag)) in bins.iter().zip(prev.iter()).enumerate() {
                let mag = bin.x;
                let i = i as i32;
                if i >= h30 && i < k5 {
                    // The floor is a `double` compare and store, as the reference writes it.
                    let prevmag = if f64::from(qmag) < 0.0001 {
                        0.0001f64 as f32
                    } else {
                        qmag
                    };
                    let dnk = math::ln(mag / prevmag) * LMULT;
                    if dnk > 0.0 {
                        mkl += dnk;
                    }
                }
                norm += mag * mag;
                footesum += mag * qmag;
            }
            mkl /= k5.wrapping_sub(h30) as f32;
            let footediv = math::sqrt(norm) * math::sqrt(*prev_norm);
            let footediv = if footediv < 0.0001 { 0.0001 } else { footediv };
            // `1.0 - x` is a `double` subtraction, rounded to `f32` once.
            let foote = (1.0 - f64::from(footesum / footediv)) as f32;
            *prev_norm = norm;
            let sum = (ins.control(Self::PROPH) * mkl) + (ins.control(Self::PROPF) * foote);
            let wait_len = f64::from(ins.control(Self::WAITTIME)) * full_rate;
            let outval = base.fire(sum, ins.control(Self::THRESHOLD), samples, wait_len);
            for (q, bin) in prev.iter_mut().zip(bins) {
                *q = bin.x;
            }
            outval
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`PvHainsworthFoote`]: the unit sizes its table from the chain buffer when the
/// synth starts.
pub struct PvHainsworthFooteCtor;

impl UnitDef for PvHainsworthFooteCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= PvHainsworthFoote::WAITTIME {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec_pool(PvHainsworthFoote::zeroed()))
    }
}

/// The World's sample rate (scsynth's `world->mSampleRate`), which a resampled graph does not
/// share: its own audio rate (`FULLRATE`) over the oversample factor.
fn world_sample_rate(ctx: &ProcessCtx<'_>) -> f64 {
    ctx.audio.sample_rate / ctx.resample_factor as f64
}
