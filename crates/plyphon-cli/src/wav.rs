//! WAV decode/encode helpers shared by the `render` subcommand.
//!
//! Lifted from the `example-render` demo (kept self-contained there for teaching). The CLI is the
//! production renderer, so the helpers live here; if a third consumer appears, factor a `plyphon-wav`
//! crate.

use std::io::{Cursor, Seek, Write};

use hound::{SampleFormat as HoundFormat, WavReader, WavSpec, WavWriter};

use crate::cli::SampleFormat;

/// Interleaved-`f32` WAV contents.
pub struct Wav {
    pub samples: Vec<f32>,
    pub channels: usize,
    pub sample_rate: f64,
}

/// The hound [`WavSpec`] for `format` at `channels`/`sample_rate`.
pub fn spec(format: SampleFormat, channels: usize, sample_rate: f64) -> WavSpec {
    let (bits, sample_format) = match format {
        SampleFormat::F32 => (32, HoundFormat::Float),
        SampleFormat::I16 => (16, HoundFormat::Int),
        SampleFormat::I24 => (24, HoundFormat::Int),
    };
    WavSpec {
        channels: channels as u16,
        sample_rate: sample_rate as u32,
        bits_per_sample: bits,
        sample_format,
    }
}

/// Write one sample in `format`, converting and clamping to the integer formats (the inverse of
/// [`decode`]).
pub fn write_sample<W: Write + Seek>(format: SampleFormat, writer: &mut WavWriter<W>, s: f32) {
    let result = match format {
        SampleFormat::F32 => writer.write_sample(s),
        SampleFormat::I16 => writer.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16),
        SampleFormat::I24 => writer.write_sample((s.clamp(-1.0, 1.0) * 8_388_607.0) as i32),
    };
    result.expect("write sample");
}

/// Decode WAV bytes (any bit depth, PCM or float) into interleaved `f32`.
pub fn decode(bytes: &[u8]) -> Result<Wav, String> {
    let reader = WavReader::new(Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        HoundFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?,
        HoundFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()
                .map_err(|e| e.to_string())?
        }
    };
    Ok(Wav {
        samples,
        channels: spec.channels.max(1) as usize,
        sample_rate: spec.sample_rate as f64,
    })
}

/// Feeds a decoded input WAV into a render/playback one control block at a time, zero-padded past
/// its end.
pub struct WavInput {
    wav: Option<Wav>,
    /// Next input frame to emit.
    frame: usize,
}

impl WavInput {
    pub fn new(wav: Option<Wav>) -> Self {
        WavInput { wav, frame: 0 }
    }

    /// Fill one interleaved input block (already zeroed by the caller) from the WAV at the cursor.
    pub fn fill(&mut self, block: &mut [f32]) {
        let Some(wav) = &self.wav else { return };
        if wav.channels == 0 {
            return;
        }
        let frames = block.len() / wav.channels;
        let total = wav.samples.len() / wav.channels;
        let avail = total.saturating_sub(self.frame).min(frames);
        let src = &wav.samples[self.frame * wav.channels..(self.frame + avail) * wav.channels];
        block[..src.len()].copy_from_slice(src);
        self.frame += avail;
    }
}
