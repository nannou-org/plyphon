//! `Pitch` - plyphon's port of scsynth's autocorrelation pitch tracker (`DelayUGens.cpp`).
//!
//! The input, optionally downsampled, fills a buffer ([aux memory](crate::unit::Aux) allocated when
//! the synth starts, as scsynth's `Pitch_Ctor` does). Each time the buffer is full the tracker
//! checks the amplitude, then searches the autocorrelation for its first peak after the zero-lag
//! lobe, stepping coarsely through long lags (`maxBinsPerOctave`), refines the peak by parabolic
//! interpolation, and, if the frequency is in range, passes it through a running median. It then
//! shifts the buffer left by one execution period and keeps filling.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{Aux, BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec_pool};
use plyphon_dsp::rate::Rate;

const IN: usize = 0;
const INIT_FREQ: usize = 1;
const MIN_FREQ: usize = 2;
const MAX_FREQ: usize = 3;
const EXEC_FREQ: usize = 4;
const MAX_BINS: usize = 5;
const MEDIAN: usize = 6;
const AMP_THRESHOLD: usize = 7;
const PEAK_THRESHOLD: usize = 8;
const DOWNSAMP: usize = 9;
const GET_CLARITY: usize = 10;

/// The longest running median (scsynth's `kMAXMEDIANSIZE`).
const MAX_MEDIAN_SIZE: usize = 32;

/// `ceil(log2(x))` as scsynth's `LOG2CEIL` computes it: `32 - clz(x - 1)`, so `0` gives 32 and
/// `1` gives 0.
fn log2_ceil(x: i32) -> i32 {
    32 - x.wrapping_sub(1).leading_zeros() as i32
}

/// `sc_clip` on `f32`: `std::max(std::min(x, hi), lo)`.
fn clip_f32(x: f32, lo: f32, hi: f32) -> f32 {
    let x = if hi < x { hi } else { x };
    if x < lo { lo } else { x }
}

/// `Pitch.kr(in, initFreq, minFreq, maxFreq, execFreq, maxBinsPerOctave, median, ampThreshold,
/// peakThreshold, downSample, clar)`: outputs the tracked frequency and whether (or, with `clar`,
/// how clearly) the last analysis found one.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Pitch {
    /// The running median's values, kept sorted (`m_values`).
    values: [f32; MAX_MEDIAN_SIZE],
    /// Each median slot's age in analyses; the oldest is replaced next (`m_ages`).
    ages: [i32; MAX_MEDIAN_SIZE],
    /// The last frequency output.
    freq: f32,
    /// `minFreq` at construction.
    min_freq: f32,
    /// `maxFreq` at construction.
    max_freq: f32,
    /// The last "has frequency" output (1/0, or the clarity).
    has_freq: f32,
    /// The analysis sample rate: the full rate over the downsampling (and, for a control-rate input,
    /// over the block size).
    srate: f32,
    /// `ampThreshold` at construction.
    amp_thresh: f32,
    /// `peakThreshold` at construction.
    peak_thresh: f32,
    /// Shortest period searched, in analysis samples.
    min_period: i32,
    /// Longest period searched, and the autocorrelation window length.
    max_period: i32,
    /// Analysis samples between analyses (at least one block).
    exec_period: i32,
    /// Next write position in the buffer.
    index: u32,
    /// Read position within the input block (audio input), or the downsampling counter (control).
    readp: i32,
    /// Buffer length in analysis samples.
    size: u32,
    /// Keep one input sample in every `downsamp`.
    downsamp: i32,
    /// `ceil(log2(maxBinsPerOctave))`: lags beyond this octave are stepped through coarsely.
    max_log2_bins: i32,
    /// Running median length, `0..=32`.
    median_size: i32,
    /// Whether the second output is the clarity rather than 1.
    get_clarity: u32,
    /// Whether the input is audio-rate (`Pitch_next_a`) rather than control-rate (`Pitch_next_k`).
    audio_in: u32,
}

impl Pitch {
    /// Keep a sorted window of the last `median_size` frequencies, replacing the oldest with `value`,
    /// and return the median (scsynth's `insertMedian`).
    fn insert_median(&mut self, value: f32) -> f32 {
        let size = self.median_size as usize;
        let values = &mut self.values[..size];
        let ages = &mut self.ages[..size];
        let last = size as i32 - 1;
        // Find the oldest slot and age the others.
        let mut pos = 0;
        for (i, age) in ages.iter_mut().enumerate() {
            if *age == last {
                pos = i;
            } else {
                *age += 1;
            }
        }
        // Slide the gap down while the new value is smaller than its neighbour, then up while it is
        // larger.
        while pos != 0 && value < values[pos - 1] {
            values[pos] = values[pos - 1];
            ages[pos] = ages[pos - 1];
            pos -= 1;
        }
        while pos as i32 != last && value > values[pos + 1] {
            values[pos] = values[pos + 1];
            ages[pos] = ages[pos + 1];
            pos += 1;
        }
        values[pos] = value;
        ages[pos] = 0;
        values[size >> 1]
    }

    /// Append one analysis sample and, when the buffer is full, analyse it (the body shared by
    /// `Pitch_next_a` and `Pitch_next_k`), updating `freq` and `has_freq`. `k_rate` selects
    /// `Pitch_next_k`'s double-precision parabolic refinement.
    fn push(&mut self, buf: &mut [f32], z: f32, freq: &mut f32, has_freq: &mut f32, k_rate: bool) {
        let size = self.size as usize;
        let mut index = self.index as usize;
        buf[index] = z;
        index += 1;
        if index >= size {
            *has_freq = 0.0;
            self.analyse(buf, freq, has_freq, k_rate);
            // Shift the buffer left by one execution period for the next fill.
            let exec_period = self.exec_period as usize;
            let interval = size - exec_period;
            buf.copy_within(exec_period..size, 0);
            index = interval;
        }
        self.index = index as u32;
    }

    /// One autocorrelation analysis of the full buffer.
    fn analyse(&mut self, buf: &[f32], freq: &mut f32, has_freq: &mut f32, k_rate: bool) {
        let min_period = self.min_period;
        let max_period = self.max_period;
        // The window never exceeds half the buffer unless `maxperiod << 1` overflowed, where scsynth
        // reads past its buffer; it is kept to the buffer here.
        let window = (max_period.max(0) as usize).min(buf.len());

        // Only look for a pitch if some sample in the window reaches the amplitude threshold.
        let amp_ok = buf[..window].iter().any(|s| s.abs() >= self.amp_thresh);
        if !amp_ok {
            return;
        }

        let max_log2_bins = self.max_log2_bins;
        let lag = |i: i32| lag_sum(buf, i, window);
        let bin_step = |i: i32| {
            let octave = log2_ceil(i);
            if octave <= max_log2_bins {
                1
            } else {
                // `1L << (octave - maxlog2bins)`, stored to an `int`.
                (1i64 << (octave - max_log2_bins)) as i32
            }
        };

        // The zero-lag value sets the peak threshold.
        let zero_lag = buf[..window].iter().fold(0.0f32, |acc, &s| acc + s * s);
        let threshold = zero_lag * self.peak_thresh;

        // Skip the zero-lag lobe: step until a lag's sum drops below the threshold.
        let mut i = 1;
        while i <= max_period {
            if lag(i) < threshold {
                break;
            }
            i = i.wrapping_add(bin_step(i));
        }
        let start_period = i;
        let mut period = start_period;

        // Find the first peak above the threshold.
        let mut max_sum = threshold;
        let mut found_peak = false;
        i = start_period;
        while i <= max_period {
            if i >= min_period {
                let sum = lag(i);
                if sum > threshold {
                    if sum > max_sum {
                        found_peak = true;
                        max_sum = sum;
                        period = i;
                    }
                } else if found_peak {
                    break;
                }
            }
            i = i.wrapping_add(bin_step(i));
        }
        if !found_peak {
            return;
        }

        // The sums either side of the peak; with a coarse step the peak may not be a local maximum
        // yet, so slide to one.
        let mut prev_sum = if period > 0 { lag(period - 1) } else { 0.0 };
        let mut next_sum = if period < max_period {
            lag(period + 1)
        } else {
            0.0
        };
        while prev_sum > max_sum && period > 0 {
            next_sum = max_sum;
            max_sum = prev_sum;
            period -= 1;
            prev_sum = lag(period - 1);
        }
        while next_sum > max_sum && period < max_period {
            prev_sum = max_sum;
            max_sum = next_sum;
            period += 1;
            next_sum = lag(period + 1);
        }

        // Parabolic interpolation for a fractional period. `Pitch_next_k` writes the constants as
        // doubles, so its arithmetic runs in double; `Pitch_next_a`'s runs in float.
        let (beta, gamma) = if k_rate {
            (
                (0.5 * (next_sum - prev_sum) as f64) as f32,
                (2.0 * max_sum as f64 - next_sum as f64 - prev_sum as f64) as f32,
            )
        } else {
            (
                0.5 * (next_sum - prev_sum),
                2.0 * max_sum - next_sum - prev_sum,
            )
        };
        let fperiod = period as f32 + beta / gamma;
        let temp_freq = self.srate / fperiod;
        if temp_freq >= self.min_freq && temp_freq <= self.max_freq {
            *freq = temp_freq;
            if self.median_size > 1 {
                *freq = self.insert_median(*freq);
            }
            *has_freq = if self.get_clarity != 0 {
                // The clarity: the peak's size normalised by the zero-lag value.
                max_sum / zero_lag
            } else {
                1.0
            };
        }
    }
}

/// The autocorrelation at lag `i` over `window` samples: `sum(buf[i + j] * buf[j])`, accumulated in
/// order. The peak refinement can slide to lag `-1` or to lag `max_period + 1`, which makes scsynth
/// read one element before or past its buffer; those reads yield 0 here.
fn lag_sum(buf: &[f32], i: i32, window: usize) -> f32 {
    if i >= 0 && (i as usize).saturating_add(window) <= buf.len() {
        let lagged = &buf[i as usize..i as usize + window];
        lagged
            .iter()
            .zip(&buf[..window])
            .fold(0.0f32, |acc, (&a, &b)| acc + a * b)
    } else {
        let mut sum = 0.0f32;
        for (j, &b) in buf[..window].iter().enumerate() {
            let a = usize::try_from(i as i64 + j as i64)
                .ok()
                .and_then(|k| buf.get(k))
                .copied()
                .unwrap_or(0.0);
            sum += a * b;
        }
        sum
    }
}

impl Unit for Pitch {
    fn init(&mut self, _ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `Pitch_Ctor` writes 0 to both outputs without a calc; its set-up is in `alloc`.
        DoneAction::Nothing
    }

    fn alloc(&mut self, ctx: &InitCtx<'_>, aux: &mut Aux<'_>) {
        // `Pitch_Ctor`'s parameters, each read once from its input's first value.
        let ins = ctx.ins;
        self.freq = ins.control(INIT_FREQ);
        self.min_freq = ins.control(MIN_FREQ);
        self.max_freq = ins.control(MAX_FREQ);
        let exec_freq = clip_f32(ins.control(EXEC_FREQ), self.min_freq, self.max_freq);
        self.max_log2_bins = log2_ceil(ins.control(MAX_BINS) as i32);
        self.median_size = (ins.control(MEDIAN) as i32).clamp(0, MAX_MEDIAN_SIZE as i32);
        self.amp_thresh = ins.control(AMP_THRESHOLD);
        self.peak_thresh = ins.control(PEAK_THRESHOLD);
        let downsamp = ins.control(DOWNSAMP) as i32;

        let full_rate = ctx.audio.sample_rate;
        let full_block = ctx.audio.block_size as i32;
        if self.audio_in != 0 {
            self.downsamp = downsamp.min(full_block).max(1);
            self.srate = (full_rate / self.downsamp as f32 as f64) as f32;
        } else {
            self.downsamp = downsamp.max(1);
            self.srate = (full_rate / full_block.wrapping_mul(self.downsamp) as f32 as f64) as f32;
        }
        self.min_period = (self.srate / self.max_freq) as i64 as i32;
        self.max_period = (self.srate / self.min_freq) as i64 as i32;
        self.exec_period = ((self.srate / exec_freq) as i32).max(full_block);
        let size = self.max_period.wrapping_shl(1).max(self.exec_period);
        self.size = size as u32;

        // The median starts full of `initFreq`, aged `0..size` (`initMedian`).
        let median_size = self.median_size as usize;
        self.values[..median_size].fill(self.freq);
        for (i, age) in self.ages[..median_size].iter_mut().enumerate() {
            *age = i as i32;
        }
        self.get_clarity = (ins.control(GET_CLARITY) > 0.0) as u32;

        let bytes = (size as u32 as usize).saturating_mul(core::mem::size_of::<f32>());
        aux.alloc(bytes);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let buf = ctx.aux.f32_mut();
        if buf.len() < self.size as usize {
            return DoneAction::Nothing;
        }
        let buf = &mut buf[..self.size as usize];
        let mut freq = self.freq;
        let mut has_freq = self.has_freq;
        if self.audio_in != 0 {
            // `Pitch_next_a`: every `downsamp`-th sample of the block, carrying the remainder over.
            let input = ctx.ins.audio(IN);
            let ksamps = ctx.audio.block_size as i32;
            let mut readp = self.readp;
            loop {
                // scsynth reads `ZIN(kPitchIn)[readp]`, and `ZIN` points one sample before the
                // block (`ZOFF` is 1), so the analysis takes sample `readp - 1`. At `readp == 0` that
                // is outside the input's buffer (the end of whichever wire buffer precedes it in
                // scsynth's wire space); it reads 0 here.
                let z = usize::try_from(readp - 1)
                    .ok()
                    .and_then(|k| input.get(k))
                    .copied()
                    .unwrap_or(0.0);
                self.push(buf, z, &mut freq, &mut has_freq, false);
                readp += self.downsamp;
                if readp >= ksamps {
                    break;
                }
            }
            self.readp = readp - ksamps;
        } else {
            // `Pitch_next_k`: one input value in every `downsamp` blocks.
            self.readp += 1;
            if self.readp == self.downsamp {
                self.readp = 0;
                let z = ctx.ins.control(IN);
                self.push(buf, z, &mut freq, &mut has_freq, true);
            }
        }
        *ctx.outs.control(0) = freq;
        *ctx.outs.control(1) = has_freq;
        self.freq = freq;
        self.has_freq = has_freq;
        DoneAction::Nothing
    }
}

/// Constructor for [`Pitch`]: control rate only, as the language class defines it. The calc follows
/// the input's rate, and the buffer is allocated when the synth starts from the frequency inputs.
pub struct PitchCtor;

impl UnitDef for PitchCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= GET_CLARITY {
            return Err(BuildError::WrongInputCount);
        }
        if ctx.rate != Rate::Control {
            return Err(BuildError::UnsupportedRate(ctx.rate));
        }
        Ok(unit_spec_pool(Pitch {
            audio_in: (ctx.input_rates[IN] == Rate::Audio) as u32,
            ..Pitch::zeroed()
        }))
    }
}
