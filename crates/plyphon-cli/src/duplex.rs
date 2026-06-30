//! Bridging a live capture stream into the engine for `In.ar` (server only).
//!
//! cpal has no duplex stream, so input and output are two streams on independent clocks. The capture
//! callback pushes interleaved `f32` into an `rtrb` ring (see [`crate::audio::play_input`]); this
//! drains it on the output callback and feeds the engine. To keep input sample-faithful, the engine is
//! driven one exact control block at a time - `World::fill_duplex` only deposits input at block starts,
//! so a host buffer that straddles a block would drop input. [`Duplex::fill`] therefore renders whole
//! blocks and serves cpal's arbitrary-sized buffer from the just-rendered block, carrying any leftover
//! frames to the next callback (a carry-FIFO). This makes faithful input independent of the host buffer
//! size, on any backend - the portable alternative to forcing a block-aligned hardware buffer.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use plyphon::World;
use rtrb::Consumer;

/// Drains the capture ring and drives the engine in whole control blocks, serving cpal's output buffer
/// from a one-block carry. Owns the `World` (lives only on the audio thread, like the output-only path).
pub struct Duplex {
    world: World,
    input: Consumer<f32>,
    out_channels: usize,
    in_channels: usize,
    block_size: usize,
    /// Reused interleaved input block (`block_size * in_channels`).
    block_in: Vec<f32>,
    /// Reused interleaved output block (`block_size * out_channels`); doubles as the carry buffer.
    block_out: Vec<f32>,
    /// Frames of `block_out` already emitted; `== block_size` means "render a fresh block".
    carry: usize,
    /// Input samples zero-filled because the ring underran (cumulative).
    underflow: Arc<AtomicU64>,
}

impl Duplex {
    /// Wrap `world` and the capture-ring `consumer`. `carry` starts full so the first [`fill`](
    /// Duplex::fill) renders immediately.
    pub fn new(
        world: World,
        input: Consumer<f32>,
        in_channels: usize,
        block_size: usize,
        out_channels: usize,
    ) -> Self {
        Duplex {
            world,
            input,
            out_channels,
            in_channels,
            block_size,
            block_in: vec![0.0; block_size * in_channels],
            block_out: vec![0.0; block_size * out_channels],
            carry: block_size,
            underflow: Arc::new(AtomicU64::new(0)),
        }
    }

    /// A handle to the underrun counter (read off the audio thread for the shutdown xrun report).
    pub fn underflow(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.underflow)
    }

    /// Fill `out` (interleaved, the resolved output channel count wide), rendering whole engine blocks
    /// and carrying any leftover across calls. Allocation-free in steady state; RT-safe.
    pub fn fill(&mut self, out: &mut [f32]) {
        let Duplex {
            world,
            input,
            out_channels,
            in_channels,
            block_size,
            block_in,
            block_out,
            carry,
            underflow,
        } = self;
        let (in_channels, out_channels, block_size) = (*in_channels, *out_channels, *block_size);
        reblock(out, out_channels, block_out, carry, block_size, |block| {
            let short = drain_input(input, block_in);
            if short > 0 {
                underflow.fetch_add(short as u64, Ordering::Relaxed);
            }
            world.fill_duplex(block, out_channels, block_in, in_channels);
        });
    }
}

/// Serve `out` (`out_channels`-wide interleaved) from fixed-size blocks produced by `render`, carrying
/// the unconsumed tail of the last block across calls. `*carry` is the number of frames of `block_out`
/// already emitted (start at `block_size` to force a render on the first call); `render` fills all
/// `block_size * out_channels` of its argument with the next block. `render` is called exactly once per
/// whole block regardless of `out.len()`, so a block-quantized producer stays sample-faithful for any
/// host buffer size.
fn reblock(
    out: &mut [f32],
    out_channels: usize,
    block_out: &mut [f32],
    carry: &mut usize,
    block_size: usize,
    mut render: impl FnMut(&mut [f32]),
) {
    if out_channels == 0 {
        return;
    }
    let frames = out.len() / out_channels;
    let mut done = 0;
    while done < frames {
        if *carry >= block_size {
            render(block_out);
            *carry = 0;
        }
        let n = (block_size - *carry).min(frames - done);
        let src = *carry * out_channels;
        let dst = done * out_channels;
        out[dst..dst + n * out_channels].copy_from_slice(&block_out[src..src + n * out_channels]);
        *carry += n;
        done += n;
    }
}

/// Pop one block of interleaved input from `consumer` into `block`, zero-filling any shortfall (the
/// ring underran). Returns the number of zero-filled samples.
fn drain_input(consumer: &mut Consumer<f32>, block: &mut [f32]) -> usize {
    block.fill(0.0);
    // The returned remainder is the unfilled tail - exactly the samples left as silence.
    let (_popped, zeros) = consumer.pop_partial_slice(block);
    zeros.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};
    use rtrb::RingBuffer;

    const SR: f32 = 48_000.0;
    const BLOCK: usize = 64;

    fn goertzel(samples: &[f32], freq: f32) -> f32 {
        let n = samples.len();
        let k = (0.5 + n as f32 * freq / SR).floor();
        let w = 2.0 * std::f32::consts::PI * k / n as f32;
        let coeff = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        for &x in samples {
            let s = x + coeff * s1 - s2;
            s2 = s1;
            s1 = s;
        }
        (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0).sqrt() / n as f32
    }

    #[test]
    fn drain_input_zero_fills_a_short_ring() {
        let (mut prod, mut cons) = RingBuffer::<f32>::new(16);
        assert!(prod.push_partial_slice(&[1.0, 2.0, 3.0]).1.is_empty());
        let mut block = [9.0f32; 8];
        let short = drain_input(&mut cons, &mut block);
        assert_eq!(short, 5);
        assert_eq!(&block[..3], &[1.0, 2.0, 3.0]);
        assert!(block[3..].iter().all(|&s| s == 0.0), "tail must be silence");
    }

    #[test]
    fn drain_input_full_block_has_no_shortfall() {
        let (mut prod, mut cons) = RingBuffer::<f32>::new(16);
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        assert!(prod.push_partial_slice(&data).1.is_empty());
        let mut block = [0.0f32; 8];
        assert_eq!(drain_input(&mut cons, &mut block), 0);
        assert_eq!(&block, data.as_slice());
    }

    #[test]
    fn reblock_is_continuous_across_odd_buffer_sizes() {
        // A fake render emitting an ever-increasing per-frame ramp; out_channels = 2 to exercise the
        // interleaved copy. Pulling in non-block-multiple sizes must reconstruct one unbroken ramp.
        let mut next = 0.0f32;
        let mut block_out = vec![0.0f32; BLOCK * 2];
        let mut carry = BLOCK;
        let mut render = |blk: &mut [f32]| {
            for f in 0..BLOCK {
                blk[f * 2] = next;
                blk[f * 2 + 1] = next;
                next += 1.0;
            }
        };
        let mut collected = Vec::new();
        for &frames in &[100usize, 7, 257, 64, 1, 191] {
            let mut out = vec![0.0f32; frames * 2];
            reblock(&mut out, 2, &mut block_out, &mut carry, BLOCK, &mut render);
            collected.extend_from_slice(&out);
        }
        // Every frame's two channels equal its global index, contiguously.
        for (i, chunk) in collected.chunks_exact(2).enumerate() {
            assert_eq!(chunk[0], i as f32, "frame {i} discontinuous");
            assert_eq!(chunk[1], i as f32);
        }
    }

    #[test]
    fn duplex_passes_input_through_in_ar_on_odd_buffers() {
        // engine: 1 out, 1 in -> In.ar(1) reads the first hardware input channel.
        let (mut controller, _nrt, world) = engine(Options {
            sample_rate: SR as f64,
            block_size: BLOCK,
            output_channels: 1,
            input_channels: 1,
            ..Options::default()
        });
        let thru = SynthDef {
            name: "thru".to_string(),
            params: vec![],
            units: vec![
                UnitSpec::new("In", Rate::Audio, vec![InputRef::Constant(1.0)], 1),
                UnitSpec::new(
                    "Out",
                    Rate::Audio,
                    vec![
                        InputRef::Constant(0.0),
                        InputRef::Unit { unit: 0, output: 0 },
                    ],
                    0,
                ),
            ],
        };
        controller.add_synthdef(thru);
        controller
            .synth_new("thru", ROOT_GROUP_ID, AddAction::Tail)
            .unwrap();

        // Push a 440 Hz tone into the capture ring as "hardware input".
        let frames = BLOCK * 200;
        let tone: Vec<f32> = (0..frames)
            .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / SR).sin() * 0.5)
            .collect();
        let (mut prod, cons) = RingBuffer::<f32>::new(frames + BLOCK);
        assert!(prod.push_partial_slice(&tone).1.is_empty());

        let mut duplex = Duplex::new(world, cons, 1, BLOCK, 1);

        // Drive the output in a deliberately non-block-multiple buffer (100 frames).
        let mut out = Vec::new();
        let mut blk = [0.0f32; 100];
        while out.len() < frames - 100 {
            duplex.fill(&mut blk);
            out.extend_from_slice(&blk);
        }

        assert!(
            out.iter().any(|s| s.abs() > 0.1),
            "input was not passed through"
        );
        let m440 = goertzel(&out, 440.0);
        let m880 = goertzel(&out, 880.0);
        assert!(
            m440 > 5.0 * m880,
            "expected the 440 Hz input at the output: m440={m440}, m880={m880}"
        );
        assert_eq!(
            duplex.underflow().load(Ordering::Relaxed),
            0,
            "the ring was pre-filled, so there should be no underrun"
        );
    }
}
