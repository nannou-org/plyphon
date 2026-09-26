//! `Onsets` - plyphon's port of scsynth's onset detector (`Onsets.cpp`, built on Dan Stowell's
//! OnsetsDS library, `onsetsds.c`). Compiled only with the `fft` feature.
//!
//! Each frame is loaded in polar form, optionally whitened, reduced to one onset detection function
//! (ODF) value, and compared with a running median of recent values: an onset is reported when the
//! median-removed value crosses the threshold upwards.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::spec_stats::{analysis_frame, check_rate};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, pv, unit_spec_pool};
use plyphon_dsp::math;

/// `ODS_ODF_POWER`: the frame's power.
const ODF_POWER: i32 = 0;
/// `ODS_ODF_MAGSUM`: the sum of magnitudes.
const ODF_MAGSUM: i32 = 1;
/// `ODS_ODF_COMPLEX`: complex-domain deviation from the predicted frame.
const ODF_COMPLEX: i32 = 2;
/// `ODS_ODF_RCOMPLEX`: complex-domain deviation, counting only bins that did not decrease.
const ODF_RCOMPLEX: i32 = 3;
/// `ODS_ODF_PHASE`: phase deviation.
const ODF_PHASE: i32 = 4;
/// `ODS_ODF_WPHASE`: magnitude-weighted phase deviation.
const ODF_WPHASE: i32 = 5;
/// `ODS_ODF_MKL`: modified Kullback-Leibler deviation.
const ODF_MKL: i32 = 6;

/// OnsetsDS's `PI` float constant, `3.1415926535898f` (`onsetsds.h:40`): the float nearest pi.
const PI: f32 = core::f32::consts::PI;
/// OnsetsDS's `MINUSPI`, `-3.1415926535898f` (`onsetsds.h:41`).
const MINUSPI: f32 = -core::f32::consts::PI;
/// OnsetsDS's `TWOPI`, `6.28318530717952646f` (`onsetsds.h:42`): the float nearest two pi.
const TWOPI: f32 = core::f32::consts::TAU;
/// OnsetsDS's `INV_TWOPI`, `0.1591549430919f` (`onsetsds.h:43`).
const INV_TWOPI: f32 = 0.159_154_94;
/// `ods_log1`, `log(0.1)` as OnsetsDS writes it, `-2.30258509` (`onsetsds.h:38`) - a truncation of
/// `-ln(10)`, not the exact double.
#[allow(clippy::approx_constant)]
const ODS_LOG1: f64 = -2.302_585_09;

/// `onsetsds_phase_rewrap` (`onsetsds.c:31`): a phase outside `(-pi, pi)` is wrapped back into it.
fn phase_rewrap(phase: f32) -> f32 {
    if phase > MINUSPI && phase < PI {
        phase
    } else {
        phase + TWOPI * (1.0 + math::floor((MINUSPI - phase) * INV_TWOPI))
    }
}

/// `onsetsds_memneeded` (`onsetsds.c:36`): the floats an OnsetsDS needs for `odftype`, an FFT of
/// `fftsize` samples and a median span of `medspan` frames - the frame, the peak profile, the ODF
/// history, the median scratch, and the per-bin history the ODF keeps. `None` for an unknown
/// `odftype`.
fn floats_needed(odftype: i32, fftsize: usize, medspan: usize) -> Option<usize> {
    let numbins = (fftsize >> 1) - 1;
    let history = match odftype {
        ODF_POWER | ODF_MAGSUM => 0,
        ODF_COMPLEX | ODF_RCOMPLEX => 3,
        ODF_PHASE | ODF_WPHASE => 2,
        ODF_MKL => 1,
        _ => return None,
    };
    Some(2 * medspan + fftsize + numbins + 2 + history * numbins)
}

/// `Onsets.kr(chain, threshold = 0.5, odftype = 'rcomplex', relaxtime = 1, floor = 0.1, mingap = 10,
/// medianspan = 11, whtype = 1, rawodf = 0)`: `1` on each frame where an onset is detected, `0`
/// otherwise, held between frames. `odftype` selects the detection function, `0` to `6`: power,
/// magnitude sum, complex, rectified complex, phase, weighted phase, modified Kullback-Leibler.
///
/// Per frame:
///
/// - Whitening (unless `whtype` is `0`): each magnitude, the DC and Nyquist terms included, is
///   divided by `max(floor, peak)`, where `peak` tracks the bin's recent peak magnitude and relaxes
///   towards the current one by a coefficient that falls to `0.1` after `relaxtime` seconds.
/// - The ODF value is computed from the (whitened) frame and the per-bin history the ODF keeps, and
///   scaled by a factor of the FFT size.
/// - Its median over the last `medianspan` frames is subtracted, and an onset is detected when the
///   result rises above `threshold` from at or below it, at most once in every `mingap + 1` frames.
///
/// With `rawodf > 0` (read when the synth starts) the unit outputs the scaled ODF value instead.
///
/// The chain buffer is converted to polar form in place with `ToPolarApx`; the unit works on its
/// own copy of the frame. On the first frame the unit sizes its memory from the frame's size,
/// `odftype` and `medianspan`, allocates it from the engine's pool, and latches all three with
/// `relaxtime`; `threshold`, `floor`, `mingap` and `whtype` are read every frame.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Onsets {
    /// The last output, held between frames (scsynth's `outval`).
    outval: f32,
    /// `1` once the first frame has sized and initialised the detector (scsynth's `!m_needsinit`).
    initialised: u32,
    /// `1` to output the ODF value rather than detections, from `rawodf > 0` at synth start.
    rawodf: u32,
    // The OnsetsDS fields the detector reads (`onsetsds.h:103`).
    /// The sample rate (`srate`).
    srate: f32,
    /// The whitening relaxation time in seconds (`relaxtime`).
    relaxtime: f32,
    /// The per-frame whitening relaxation coefficient (`relaxcoef`).
    relaxcoef: f32,
    /// The lowest peak magnitude whitening divides by (`floor`).
    floor: f32,
    /// The ODF's magnitude threshold, or the MKL epsilon (`odfparam`).
    odfparam: f32,
    /// The ODF's scale for the FFT size (`normfactor`).
    normfactor: f32,
    /// The ODF value after median removal (`odfvalpost`).
    odfvalpost: f32,
    /// The previous frame's [`odfvalpost`](Self::odfvalpost) (`odfvalpostprev`).
    odfvalpostprev: f32,
    /// The detection threshold (`thresh`).
    thresh: f32,
    /// The detection function (`odftype`).
    odftype: i32,
    /// The whitening type; `0` disables whitening (`whtype`).
    whtype: i32,
    /// `1` when the current frame is an onset (`detected`).
    detected: u32,
    /// `1` when the median span is odd (`med_odd`).
    med_odd: u32,
    /// Frames in the median (`medspan`).
    medspan: u32,
    /// Frames to suppress detection after an onset (`mingap`).
    mingap: u32,
    /// Frames of suppression left (`gapleft`).
    gapleft: u32,
    /// The frame size the detector was set up for (`fftsize`).
    fftsize: u32,
    /// Bins between DC and Nyquist (`numbins`).
    numbins: u32,
}

impl Onsets {
    const THRESHOLD: usize = 1;
    const ODFTYPE: usize = 2;
    const RELAXTIME: usize = 3;
    const FLOOR: usize = 4;
    const MINGAP: usize = 5;
    const MEDIANSPAN: usize = 6;
    const WHTYPE: usize = 7;
    const RAWODF: usize = 8;

    /// `onsetsds_init` (`onsetsds.c:88`), for scsynth's polar frames; the caller zeroes the detector
    /// memory. `odftype` is one of the seven known types.
    fn ods_init(&mut self, odftype: i32, fftsize: usize, medspan: usize, srate: f32) {
        self.srate = srate;
        let numbins = (fftsize >> 1) - 1;
        let realnumbins = numbins + 2;
        self.setrelax(1.0, fftsize >> 1);
        self.floor = 0.1_f64 as f32;
        // `odfparam` is assigned from a double literal. The float literals in the scale factors are
        // promoted to double wherever the other operand is.
        let (odfparam, normfactor): (f64, f32) = match odftype {
            ODF_POWER => (0.01, 2560.0 / (realnumbins as u64 * fftsize as u64) as f32),
            ODF_MAGSUM => (
                0.01,
                (f64::from(113.137085_f32) / (realnumbins as f64 * math::sqrt(fftsize as f64)))
                    as f32,
            ),
            ODF_COMPLEX | ODF_RCOMPLEX => (
                0.01,
                (f64::from(231.70475_f32) / math::powf(fftsize as f64, 1.5)) as f32,
            ),
            ODF_PHASE => (0.01, 5.12 / fftsize as f32),
            ODF_WPHASE => (
                0.0001,
                (f64::from(115.852_37_f32) / math::powf(fftsize as f64, 1.5)) as f32,
            ),
            _ => (0.01, 7.68 * 0.25 / fftsize as f32),
        };
        self.odfparam = odfparam as f32;
        self.normfactor = normfactor;
        self.odfvalpost = 0.0;
        self.odfvalpostprev = 0.0;
        self.thresh = 0.5;
        self.odftype = odftype;
        self.whtype = 1;
        self.detected = 0;
        self.med_odd = u32::from(medspan & 1 != 0);
        self.medspan = medspan as u32;
        self.mingap = 0;
        self.gapleft = 0;
        self.fftsize = fftsize as u32;
        self.numbins = numbins as u32;
    }

    /// `onsetsds_setrelax` (`onsetsds.c:182`): the relaxation coefficient for `time` seconds at
    /// `hopsize` samples per frame.
    fn setrelax(&mut self, time: f32, hopsize: usize) {
        self.relaxtime = time;
        self.relaxcoef = if time == 0.0 {
            0.0
        } else {
            math::exp((ODS_LOG1 * hopsize as f64) / f64::from(time * self.srate)) as f32
        };
    }

    /// `onsetsds_process` (`onsetsds.c:171`) over the detector memory `mem`, the frame already
    /// loaded into its first `fftsize` floats.
    fn ods_process(&mut self, mem: &mut [f32]) {
        let fftsize = self.fftsize as usize;
        let numbins = self.numbins as usize;
        let medspan = self.medspan as usize;
        let (curr, rest) = mem.split_at_mut(fftsize);
        let (psp, rest) = rest.split_at_mut(numbins + 2);
        let (odfvals, rest) = rest.split_at_mut(medspan);
        let (sortbuf, other) = rest.split_at_mut(medspan);
        self.whiten(curr, psp);
        self.odf(curr, odfvals, other);
        self.detect(odfvals, sortbuf);
    }

    /// `onsetsds_whiten` (`onsetsds.c:261`). `psp` holds the peak profile DC first, then the bins,
    /// then Nyquist; the frame `curr` is `[dc, nyq, mag0, phase0, ...]`.
    fn whiten(&self, curr: &mut [f32], psp: &mut [f32]) {
        if self.whtype == 0 {
            return;
        }
        let relaxcoef = self.relaxcoef;
        let numbins = self.numbins as usize;
        let floor = self.floor;
        let track = |val: f32, oldval: f32| {
            if val < oldval {
                val + (oldval - val) * relaxcoef
            } else {
                val
            }
        };
        let ods_max = |a: f32, b: f32| if a > b { a } else { b };

        psp[0] = track(curr[0].abs(), psp[0]);
        psp[numbins + 1] = track(curr[1].abs(), psp[numbins + 1]);
        for i in 0..numbins {
            psp[i + 1] = track(curr[2 + 2 * i].abs(), psp[i + 1]);
        }

        curr[0] /= ods_max(floor, psp[0]);
        curr[1] /= ods_max(floor, psp[numbins + 1]);
        for i in 0..numbins {
            curr[2 + 2 * i] /= ods_max(floor, psp[i + 1]);
        }
    }

    /// `onsetsds_odf` (`onsetsds.c:319`): shift the ODF history and compute the new value into
    /// `odfvals[0]`, updating the per-bin history `other`.
    fn odf(&self, curr: &[f32], odfvals: &mut [f32], other: &mut [f32]) {
        let numbins = self.numbins as usize;
        let mag = |i: usize| curr[2 + 2 * i];
        let phase = |i: usize| curr[3 + 2 * i];
        // The history moves down one place. scsynth does this with an overlapping `memcpy`, which
        // the platforms' `memcpy` performs as a `memmove`.
        let medspan = odfvals.len();
        odfvals.copy_within(..medspan - 1, 1);

        let val = match self.odftype {
            ODF_POWER => {
                let mut val = curr[1] * curr[1] + curr[0] * curr[0];
                for i in 0..numbins {
                    val += mag(i) * mag(i);
                }
                val
            }
            ODF_MAGSUM => {
                let mut val = curr[1].abs() + curr[0].abs();
                for i in 0..numbins {
                    val += mag(i).abs();
                }
                val
            }
            ODF_COMPLEX | ODF_RCOMPLEX => {
                // `other` holds mag, phase and phase difference per bin.
                let rectify = self.odftype == ODF_RCOMPLEX;
                let mut totdev = 0.0f64;
                for i in 0..numbins {
                    let curmag = mag(i).abs();
                    let predmag = other[3 * i];
                    let yesterphase = other[3 * i + 1];
                    let yesterphasediff = other[3 * i + 2];
                    // Rectifying skips bins that decreased: `!(curmag < predmag)`.
                    let grew = curmag.partial_cmp(&predmag) != Some(core::cmp::Ordering::Less);
                    if curmag > self.odfparam && (!rectify || grew) {
                        let predphase = yesterphase + yesterphasediff;
                        let deviation = predphase - phase(i);
                        let deviation = math::sqrt(
                            predmag * predmag + curmag * curmag
                                - predmag * curmag * math::cos(phase_rewrap(deviation)),
                        );
                        totdev += f64::from(deviation);
                    }
                }
                for i in 0..numbins {
                    other[3 * i] = mag(i).abs();
                    let diff = phase(i) - other[3 * i + 1];
                    other[3 * i + 1] = phase(i);
                    other[3 * i + 2] = phase_rewrap(diff);
                }
                totdev as f32
            }
            ODF_PHASE | ODF_WPHASE => {
                // `other` holds phase and phase difference per bin. The read position advances only
                // past bins above the threshold, so later bins read earlier bins' history.
                let weighted = self.odftype == ODF_WPHASE;
                let mut totdev = 0.0f64;
                let mut tbpointer = 0;
                for i in 0..numbins {
                    if mag(i).abs() > self.odfparam {
                        let deviation = phase(i) - other[tbpointer] - other[tbpointer + 1];
                        tbpointer += 2;
                        let deviation = phase_rewrap(deviation);
                        totdev += f64::from(if weighted {
                            (deviation * mag(i).abs()).abs()
                        } else {
                            deviation.abs()
                        });
                    }
                }
                for i in 0..numbins {
                    let diff = phase(i) - other[2 * i];
                    other[2 * i] = phase(i);
                    other[2 * i + 1] = phase_rewrap(diff);
                }
                totdev as f32
            }
            _ => {
                // `ODF_MKL`: `other` holds each bin's previous magnitude.
                let mut totdev = 0.0f64;
                for (i, yestermag) in other[..numbins].iter_mut().enumerate() {
                    let curmag = mag(i).abs();
                    let deviation = curmag.abs() / (yestermag.abs() + self.odfparam);
                    totdev += math::ln(f64::from(1.0 + deviation));
                    *yestermag = curmag;
                }
                totdev as f32
            }
        };
        odfvals[0] = val * self.normfactor;
    }

    /// `onsetsds_detect` (`onsetsds.c:498`): remove the median of the ODF history and test the
    /// threshold crossing, honouring the gap after a detection.
    fn detect(&mut self, odfvals: &[f32], sortbuf: &mut [f32]) {
        let medspan = self.medspan as usize;
        self.odfvalpostprev = self.odfvalpost;
        sortbuf.copy_from_slice(odfvals);
        selection_sort(sortbuf);
        self.odfvalpost = if self.med_odd != 0 {
            odfvals[0] - sortbuf[(medspan - 1) >> 1]
        } else {
            odfvals[0] - ((sortbuf[medspan >> 1] + sortbuf[(medspan >> 1) - 1]) * 0.5)
        };
        if self.gapleft != 0 {
            self.gapleft -= 1;
            self.detected = 0;
        } else {
            let detected = self.odfvalpost > self.thresh && self.odfvalpostprev <= self.thresh;
            self.detected = u32::from(detected);
            if detected {
                self.gapleft = self.mingap;
            }
        }
    }
}

/// OnsetsDS's `SelectionSort` (`onsetsds.c:481`): ascending, moving the largest remaining value to
/// the end with `>` comparisons.
fn selection_sort(array: &mut [f32]) {
    let mut length = array.len();
    while length > 0 {
        let mut max = 0;
        for i in 1..length {
            if array[i] > array[max] {
                max = i;
            }
        }
        array.swap(length - 1, max);
        length -= 1;
    }
}

impl Unit for Onsets {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `Onsets_Ctor` (`Onsets.cpp:108`): choose the calc from `rawodf`, zero the output, and run
        // no calc. The detector itself is set up on the first frame.
        self.rawodf = u32::from(ctx.ins.control(Self::RAWODF) > 0.0);
        *ctx.outs.control(0) = 0.0;
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `Onsets_next` / `Onsets_next_rawodf` (`Onsets.cpp:122`, `159`).
        let Some(bufnum) = analysis_frame(ctx, self.outval) else {
            return DoneAction::Nothing;
        };
        let ins = ctx.ins;
        let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum) else {
            *ctx.outs.control(0) = self.outval;
            return DoneAction::Nothing;
        };
        let samples = buffer.data().len();
        // `ToPolarApx(buf)` (`Onsets.cpp:127`, `164`).
        if pv::to_polar(&mut buffer, ctx.fft.complex()).is_none() {
            *ctx.outs.control(0) = self.outval;
            return DoneAction::Nothing;
        }

        if self.initialised == 0 {
            // The first frame fixes the size: `onsetsds_memneeded`, `onsetsds_init` and
            // `onsetsds_setrelax` with this frame's `odftype`, `medianspan` and `relaxtime`.
            let odftype = ins.control(Self::ODFTYPE) as i32;
            let relaxtime = ins.control(Self::RELAXTIME);
            let medspan = ins.control(Self::MEDIANSPAN) as i32;
            // An unknown `odftype` or a span under one frame would make scsynth write through an
            // unusable allocation; the frame is skipped instead.
            let floats = match usize::try_from(medspan) {
                Ok(medspan) if medspan >= 1 => floats_needed(odftype, samples, medspan),
                _ => None,
            };
            let Some(floats) = floats else {
                *ctx.outs.control(0) = self.outval;
                return DoneAction::Nothing;
            };
            // A size no pool can hold fails the allocation, which silences the unit for good.
            let bytes = floats
                .checked_mul(core::mem::size_of::<f32>())
                .unwrap_or(isize::MAX as usize);
            if !ctx.aux.alloc(bytes) {
                return DoneAction::Nothing;
            }
            ctx.aux.f32_mut()[..floats].fill(0.0);
            self.ods_init(
                odftype,
                samples,
                medspan as usize,
                ctx.audio.sample_rate as f32,
            );
            self.setrelax(relaxtime, samples >> 1);
            self.initialised = 1;
        }

        // The "painless" parameters, read every frame.
        self.thresh = ins.control(Self::THRESHOLD);
        self.floor = ins.control(Self::FLOOR);
        self.mingap = ins.control(Self::MINGAP) as i32 as u32;
        self.whtype = ins.control(Self::WHTYPE) as i32;

        // `onsetsds_loadframe` for scsynth's polar frames: copy the frame as it stands. A frame
        // smaller than the detector's would be read past its end; it is skipped.
        let fftsize = self.fftsize as usize;
        let data = buffer.data();
        if data.len() < fftsize {
            *ctx.outs.control(0) = self.outval;
            return DoneAction::Nothing;
        }
        let mem = ctx.aux.f32_mut();
        mem[..fftsize].copy_from_slice(&data[..fftsize]);
        self.ods_process(mem);

        self.outval = if self.rawodf != 0 {
            mem[fftsize + self.numbins as usize + 2]
        } else if self.detected != 0 {
            1.0
        } else {
            0.0
        };
        *ctx.outs.control(0) = self.outval;
        DoneAction::Nothing
    }
}

/// Constructor for [`Onsets`]: the unit allocates its detector memory on the first frame.
pub struct OnsetsCtor;

impl UnitDef for OnsetsCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        check_rate(ctx, Onsets::RAWODF + 1)?;
        Ok(unit_spec_pool(Onsets::zeroed()))
    }
}
