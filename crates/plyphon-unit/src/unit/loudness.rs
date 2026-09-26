//! `Loudness` - plyphon's port of scsynth's perceptual loudness model (`Loudness.cpp`, Nick
//! Collins). Compiled only with the `fft` feature.
//!
//! Each frame's power spectrum is summed into 42 bands, each band converted to phons through an
//! equal-loudness contour, the bands summed as intensities, and the total converted to sones. The
//! band layout assumes a 1024-sample FFT at 44.1 or 48 kHz.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::spec_stats::check_rate;
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec_aux};
use plyphon_dsp::math;

/// Number of bands (scsynth's `m_numbands`).
const NUM_BANDS: usize = 42;

/// The first FFT bin of each band, numbered from 1 (bin `h` is packed slots `2h`, `2h + 1`);
/// scsynth's `eqlbandbins` (`Loudness.cpp:29`). The 43rd entry is never read.
const EQLBANDBINS: [usize; 43] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 19, 22, 25, 28, 32, 36, 41, 46, 52, 58, 65, 73, 82,
    92, 103, 116, 129, 144, 161, 180, 201, 225, 251, 280, 312, 348, 388, 433, 483, 513,
];

/// The number of bins in each band (scsynth's `eqlbandsizes`, `Loudness.cpp:32`).
const EQLBANDSIZES: [usize; NUM_BANDS] = [
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 13, 15,
    17, 19, 21, 24, 26, 29, 32, 36, 40, 45, 50, 29,
];

/// The packed samples a frame needs: the last band's last bin is 511, read at slots 1022 and 1023.
const MIN_SAMPLES: usize = 2 * (EQLBANDBINS[NUM_BANDS - 1] + EQLBANDSIZES[NUM_BANDS - 1]);

/// The equal-loudness contours: each band's level in dB at each of the [`PHONS`] levels (scsynth's
/// `contours`, `Loudness.cpp:35`).
const CONTOURS: [[f32; 11]; NUM_BANDS] = [
    [
        47.88, 59.68, 68.55, 75.48, 81.71, 87.54, 93.24, 98.84, 104.44, 109.94, 115.31,
    ],
    [
        29.04, 41.78, 51.98, 60.18, 67.51, 74.54, 81.34, 87.97, 94.61, 101.21, 107.74,
    ],
    [
        20.72, 32.83, 43.44, 52.18, 60.24, 67.89, 75.34, 82.7, 89.97, 97.23, 104.49,
    ],
    [
        15.87, 27.14, 37.84, 46.94, 55.44, 63.57, 71.51, 79.34, 87.14, 94.97, 102.37,
    ],
    [
        12.64, 23.24, 33.91, 43.27, 52.07, 60.57, 68.87, 77.1, 85.24, 93.44, 100.9,
    ],
    [
        10.31, 20.43, 31.03, 40.54, 49.59, 58.33, 66.89, 75.43, 83.89, 92.34, 100.8,
    ],
    [
        8.51, 18.23, 28.83, 38.41, 47.65, 56.59, 65.42, 74.16, 82.89, 91.61, 100.33,
    ],
    [
        7.14, 16.55, 27.11, 36.79, 46.16, 55.27, 64.29, 73.24, 82.15, 91.06, 99.97,
    ],
    [
        5.52, 14.58, 25.07, 34.88, 44.4, 53.73, 62.95, 72.18, 81.31, 90.44, 99.57,
    ],
    [
        3.98, 12.69, 23.1, 32.99, 42.69, 52.27, 61.66, 71.15, 80.54, 89.93, 99.31,
    ],
    [
        2.99, 11.43, 21.76, 31.73, 41.49, 51.22, 60.88, 70.51, 80.11, 89.7, 99.3,
    ],
    [
        2.35, 10.58, 20.83, 30.86, 40.68, 50.51, 60.33, 70.08, 79.83, 89.58, 99.32,
    ],
    [
        2.05, 10.12, 20.27, 30.35, 40.22, 50.1, 59.97, 69.82, 79.67, 89.52, 99.38,
    ],
    [
        2.0, 9.93, 20.0, 30.07, 40.0, 49.93, 59.87, 69.8, 79.73, 89.67, 99.6,
    ],
    [
        2.19, 10.0, 20.0, 30.0, 40.0, 50.0, 59.99, 69.99, 79.98, 89.98, 99.97,
    ],
    [
        2.71, 10.56, 20.61, 30.71, 40.76, 50.81, 60.86, 70.96, 81.01, 91.06, 101.17,
    ],
    [
        3.11, 11.05, 21.19, 31.41, 41.53, 51.64, 61.75, 71.95, 82.05, 92.15, 102.33,
    ],
    [
        2.39, 10.69, 21.14, 31.52, 41.73, 51.95, 62.11, 72.31, 82.46, 92.56, 102.59,
    ],
    [
        1.5, 10.11, 20.82, 31.32, 41.62, 51.92, 62.12, 72.32, 82.52, 92.63, 102.56,
    ],
    [
        -0.17, 8.5, 19.27, 29.77, 40.07, 50.37, 60.57, 70.77, 80.97, 91.13, 101.23,
    ],
    [
        -1.8, 6.96, 17.77, 28.29, 38.61, 48.91, 59.13, 69.33, 79.53, 89.71, 99.86,
    ],
    [
        -3.42, 5.49, 16.36, 26.94, 37.31, 47.61, 57.88, 68.08, 78.28, 88.41, 98.39,
    ],
    [
        -4.73, 4.38, 15.34, 25.99, 36.39, 46.71, 57.01, 67.21, 77.41, 87.51, 97.41,
    ],
    [
        -5.73, 3.63, 14.74, 25.48, 35.88, 46.26, 56.56, 66.76, 76.96, 87.06, 96.96,
    ],
    [
        -6.24, 3.33, 14.59, 25.39, 35.84, 46.22, 56.52, 66.72, 76.92, 87.04, 97.0,
    ],
    [
        -6.09, 3.62, 15.03, 25.83, 36.37, 46.7, 57.0, 67.2, 77.4, 87.57, 97.68,
    ],
    [
        -5.32, 4.44, 15.9, 26.7, 37.28, 47.6, 57.9, 68.1, 78.3, 88.52, 98.78,
    ],
    [
        -3.49, 6.17, 17.52, 28.32, 38.85, 49.22, 59.52, 69.72, 79.92, 90.2, 100.61,
    ],
    [
        -0.81, 8.58, 19.73, 30.44, 40.9, 51.24, 61.52, 71.69, 81.87, 92.15, 102.63,
    ],
    [
        2.91, 11.82, 22.64, 33.17, 43.53, 53.73, 63.96, 74.09, 84.22, 94.45, 104.89,
    ],
    [
        6.68, 15.19, 25.71, 36.03, 46.25, 56.31, 66.45, 76.49, 86.54, 96.72, 107.15,
    ],
    [
        10.43, 18.65, 28.94, 39.02, 49.01, 58.98, 68.93, 78.78, 88.69, 98.83, 109.36,
    ],
    [
        13.56, 21.65, 31.78, 41.68, 51.45, 61.31, 71.07, 80.73, 90.48, 100.51, 111.01,
    ],
    [
        14.36, 22.91, 33.19, 43.09, 52.71, 62.37, 71.92, 81.38, 90.88, 100.56, 110.56,
    ],
    [
        15.06, 23.9, 34.23, 44.05, 53.48, 62.9, 72.21, 81.43, 90.65, 99.93, 109.34,
    ],
    [
        15.36, 23.9, 33.89, 43.31, 52.4, 61.42, 70.29, 79.18, 88.0, 96.69, 105.17,
    ],
    [
        15.6, 23.9, 33.6, 42.7, 51.5, 60.2, 68.7, 77.3, 85.8, 94.0, 101.7,
    ],
    [
        15.6, 23.9, 33.6, 42.7, 51.5, 60.2, 68.7, 77.3, 85.8, 94.0, 101.7,
    ],
    [
        15.6, 23.9, 33.6, 42.7, 51.5, 60.2, 68.7, 77.3, 85.8, 94.0, 101.7,
    ],
    [
        15.6, 23.9, 33.6, 42.7, 51.5, 60.2, 68.7, 77.3, 85.8, 94.0, 101.7,
    ],
    [
        15.6, 23.9, 33.6, 42.7, 51.5, 60.2, 68.7, 77.3, 85.8, 94.0, 101.7,
    ],
    [
        15.6, 23.9, 33.6, 42.7, 51.5, 60.2, 68.7, 77.3, 85.8, 94.0, 101.7,
    ],
];

/// The loudness levels, in phons, the [`CONTOURS`] columns correspond to (scsynth's `phons`,
/// `Loudness.cpp:77`, a `double` table).
const PHONS: [f64; 11] = [
    2.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
];

/// `Loudness.kr(chain, smask = 0.25, tmask = 1)`: the perceived loudness of each frame, in sones.
///
/// Each band sums its bins' power `re^2 + im^2`, with spectral masking inside the band: a bin's power
/// is at least `smask` times the previous bin's (leaky, so masking decays across the band). The
/// band's level `10 * log10(sum * 76032.936 + 0.001)` dB becomes phons by interpolating the band's
/// equal-loudness contour (`0` below it, `100` above it). Temporal masking holds each band's phons
/// against their decay: a band never drops by more than `tmask` phons per frame. The bands are
/// summed as intensities `10^(phons / 10) - 0.001`, the total converted back to phons, and output
/// as sones, `2^((phons - 40) / 10)`, held between frames.
///
/// The chain buffer is read as it stands, without a coordinate conversion, and must hold at least
/// 1024 samples; a smaller frame is skipped. The per-band phons live in `aux` (42 floats), cleared
/// when the synth starts. The unit calculates in its constructor, so a chain that already carries a
/// frame then is analysed straight away.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Loudness {
    /// The last loudness in sones, held between frames (scsynth's `m_sones`).
    sones: f32,
}

impl Loudness {
    const SMASK: usize = 1;
    const TMASK: usize = 2;

    /// Analyse the frame in buffer `bufnum` - scsynth's `Loudness_dofft` (`Loudness.cpp:131`).
    fn dofft(&mut self, ctx: &mut ProcessCtx<'_>, bufnum: usize) {
        let Some(buffer) = unit::buffer_at(ctx.buffers, &ctx.local_bufs, bufnum) else {
            return;
        };
        // No coordinate conversion: scsynth reads `buf->data` as it stands (`Loudness.cpp:151`).
        let data = buffer.data();
        if data.len() < MIN_SAMPLES {
            return;
        }
        let smask = ctx.ins.control(Self::SMASK);
        let tmask = ctx.ins.control(Self::TMASK);
        let bands = &mut ctx.aux.f32_mut()[..NUM_BANDS];

        let mut loudsum = 0.0f32;
        for (k, erb) in bands.iter_mut().enumerate() {
            let bandstart = EQLBANDBINS[k];
            let bandend = bandstart + EQLBANDSIZES[k];
            let mut bsum = 0.0f32;
            let mut lastpower = 0.0f32;
            for h in bandstart..bandend {
                let real = data[2 * h];
                let imag = data[2 * h + 1];
                let power = real * real + imag * imag;
                // `sc_max(lastpower * smask, power)`.
                let masked = lastpower * smask;
                let power = if masked > power { masked } else { power };
                lastpower = power;
                bsum += power;
            }

            // `log10` of a float is the single-precision overload.
            // scsynth's `76032.936f`, which rounds to the same float.
            let mut db = 10.0 * math::log10(bsum * 76_032.94 + 0.001);
            let contour = &CONTOURS[k];
            if db < contour[0] {
                db = 0.0;
            } else if db > contour[10] {
                db = PHONS[10] as f32;
            } else {
                let mut prop = 0.0f32;
                let mut j = 1;
                while j < 11 {
                    if db < contour[j] {
                        prop = (db - contour[j - 1]) / (contour[j] - contour[j - 1]);
                        break;
                    }
                    if j == 10 {
                        prop = 1.0;
                        break;
                    }
                    j += 1;
                }
                // `(1.f - prop) * phons[j - 1] + prop * phons[j]`, in double against the table.
                db = (f64::from(1.0 - prop) * PHONS[j - 1] + f64::from(prop) * PHONS[j]) as f32;
            }

            // `sc_max(db, m_ERBbands[k] - tmask)`.
            let decayed = *erb - tmask;
            *erb = if db > decayed { db } else { decayed };
            // `loudsum += pow(10, 0.1 * band) - 0.001`: the addend is double, the sum a float.
            loudsum =
                (f64::from(loudsum) + (math::powf(10.0f64, 0.1 * f64::from(*erb)) - 0.001)) as f32;
        }

        let phontotal = (10.0 * math::log10(f64::from(loudsum) + 0.001)) as f32;
        // `pow(2.f, float)` is the single-precision overload.
        self.sones = math::powf(2.0f32, (phontotal - 40.0) / 10.0);
    }
}

impl Unit for Loudness {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `Loudness_Ctor` (`Loudness.cpp:83`): clear the bands, then calculate one sample.
        ctx.aux.f32_mut()[..NUM_BANDS].fill(0.0);
        self.sones = 0.0;
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `Loudness_next` (`Loudness.cpp:102`): a chain above `-0.01` is a frame, the buffer number
        // truncated as `(uint32)`.
        let fbufnum = ctx.ins.control(0);
        if fbufnum > -0.01 {
            self.dofft(ctx, fbufnum as u32 as usize);
        }
        *ctx.outs.control(0) = self.sones;
        DoneAction::Nothing
    }
}

/// Constructor for [`Loudness`]: 42 floats of band state, fixed in size.
pub struct LoudnessCtor;

impl UnitDef for LoudnessCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        check_rate(ctx, Loudness::TMASK + 1)?;
        Ok(unit_spec_aux(
            Loudness::zeroed(),
            NUM_BANDS * core::mem::size_of::<f32>(),
            core::mem::align_of::<f32>(),
        ))
    }
}
