//! `In`/`InFeedback`/`InTrig`/`LagIn` - read signals from audio or control bus channels,
//! plyphon's ports of scsynth's `In`, `InFeedback`, `InTrig` and `LagIn` (`IOUGens.cpp`).

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::decay::LOG001;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::bus::Buses;
use plyphon_dsp::rate::Rate;
use plyphon_dsp::{math, ops};

/// `In.ar(bus, numChannels)` / `In.kr(bus, numChannels)`: reads `numChannels` consecutive bus
/// channels starting at `bus`, one per output. `In.ar` reads the audio bus bank, `In.kr` the
/// control bus bank, chosen by the unit's rate. The number of channels is fixed at build time (it
/// determines how many outputs the unit has). Channels past the end of the bus read as zero.
///
/// `In.ar` reads a channel only if it was written *this* block (scsynth's `In_next_a` touched
/// check), outputting zero otherwise - so a reader ordered before its writer, or whose writer was
/// freed, hears silence rather than the last-written block frozen. `InFeedback` is the same unit
/// with `feedback` set: it reads the channel unconditionally, picking up the previous block's
/// signal for deliberate one-block-delay feedback. `In.kr` reads the control bus unconditionally,
/// as scsynth's `In_next_k` does.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct In {
    num_channels: u32,
    /// `0`/`1`: whether this reads the audio (`In.ar`) or control (`In.kr`) bus bank.
    audio: u32,
    /// `0`/`1`: whether an untouched channel is still read (`InFeedback`) or zeroed (`In`).
    feedback: u32,
}

impl In {
    const BUS: usize = 0;
}

impl Unit for In {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let base = ctx.ins.control(Self::BUS) as usize;
        let num_channels = self.num_channels as usize;
        if self.audio != 0 {
            let factor = ctx.resample_factor;
            for o in 0..num_channels {
                let dst = ctx.outs.audio(o);
                // This sub-block tick reads its `dst.len() / factor` World-rate samples of the
                // World-block-wide bus channel and zero-order-holds them up to the wire's full length.
                // For an ordinary graph (`tick` 0, `factor` 1) this is a straight copy of the channel.
                let world_samples = dst.len() / factor;
                let offset = ctx.tick * world_samples;
                let live = self.feedback != 0
                    || unit::audio_in_touched(ctx.buses, base + o, ctx.buf_counter);
                let chan = unit::audio_in(ctx.buses, base + o);
                if live && chan.len() >= offset + world_samples {
                    if factor == 1 {
                        // The common (non-oversampled) case: a straight copy, with no per-sample
                        // division for the compiler to grind through.
                        dst.copy_from_slice(&chan[offset..offset + world_samples]);
                    } else {
                        for (j, slot) in dst.iter_mut().enumerate() {
                            *slot = chan[offset + j / factor];
                        }
                    }
                } else {
                    dst.fill(0.0);
                }
            }
        } else {
            for o in 0..num_channels {
                *ctx.outs.control(o) = unit::control_in(ctx.buses, base + o);
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`In`]: `feedback` selects `In` (`false`) or `InFeedback` (`true`) semantics.
pub struct InCtor {
    /// Whether the built unit reads untouched channels (`InFeedback`) or zeroes them (`In`).
    pub feedback: bool,
}

impl UnitDef for InCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(In {
            num_channels: ctx.num_outputs as u32,
            audio: (ctx.rate == Rate::Audio) as u32,
            feedback: self.feedback as u32,
        }))
    }
}

/// The control-bus channel window of scsynth's `IOUnit` (`m_fbusChannel`/`m_bus`), as
/// `IO_k_update_channels` maintains it: the window moves only when the bus number changes *and*
/// all `num_channels` channels from it lie on the control bus. Otherwise it stays where it was -
/// on channel 0, where the constructor points it, if no valid bus has been seen.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ControlBusWindow {
    /// The bus number last seen (scsynth's `m_fbusChannel`, `-1` before the first calc).
    fbus_channel: f32,
    /// The control bus channel the window starts at (scsynth's `m_bus - mControlBus`).
    first: u32,
}

impl ControlBusWindow {
    /// A window at channel 0, with no bus number seen yet.
    const NEW: ControlBusWindow = ControlBusWindow {
        fbus_channel: -1.0,
        first: 0,
    };

    /// `IO_k_update_channels`: move to bus `fbus_channel` if it changed and fits.
    fn update(&mut self, fbus_channel: f32, num_channels: usize, num_control_buses: usize) {
        if fbus_channel != self.fbus_channel {
            self.fbus_channel = fbus_channel;
            let bus_channel = fbus_channel as i32;
            let last_channel = bus_channel as i64 + num_channels as i64;
            if !(bus_channel < 0 || last_channel > num_control_buses as i64) {
                self.first = bus_channel as u32;
            }
        }
    }

    /// `readControlBus(bus + i, firstOutputChannel + i, maxChannel)`: channel `i` of the window,
    /// or zero when the channel the bus number asks for (`(int)fbus + i`) is past the control bus.
    fn read(&self, buses: &Buses, fbus_channel: f32, i: usize) -> f32 {
        let wanted = fbus_channel as i32 as i64 + i as i64;
        if wanted < unit::num_control_buses(buses) as i64 {
            unit::control_in(buses, self.first as usize + i)
        } else {
            0.0
        }
    }
}

/// `InTrig.kr(bus, numChannels)`: reads `numChannels` control bus channels from `bus`, but outputs
/// a channel's value only on a block in which it was written - by `Out.kr` and its relatives - and
/// zero otherwise, so a bus write reads as a one-block trigger.
///
/// A direct port of scsynth's `InTrig_next_k`, with its bus window (`IO_k_update_channels`). In a
/// reblocked or resampled graph only the first tick of each World block reads the bus and the rest
/// output zero (`InTrig_next_k_reblock`). An audio-rate `InTrig` outputs zero, as scsynth's
/// `InTrig_Ctor` switches it to `ClearUnitOutputs`.
///
/// scsynth's `/c_set` also marks the channels it sets as written; plyphon's does not, so here
/// `InTrig` sees writes from units but not from `/c_set`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct InTrig {
    window: ControlBusWindow,
    num_channels: u32,
    /// `0`/`1`: whether this is an audio-rate (always silent) `InTrig`.
    audio: u32,
}

impl InTrig {
    const BUS: usize = 0;

    /// `InTrig_next_k`.
    fn next_k(&mut self, ctx: &mut ProcessCtx<'_>) {
        let num_channels = self.num_channels as usize;
        let fbus_channel = ctx.ins.control(Self::BUS);
        self.window.update(
            fbus_channel,
            num_channels,
            unit::num_control_buses(ctx.buses),
        );
        for i in 0..num_channels {
            let touched = unit::control_in_touched(
                ctx.buses,
                self.window.first as usize + i,
                ctx.buf_counter,
            );
            *ctx.outs.control(i) = if touched {
                self.window.read(ctx.buses, fbus_channel, i)
            } else {
                0.0
            };
        }
    }

    /// Every output to zero.
    fn clear(&self, ctx: &mut ProcessCtx<'_>) {
        for i in 0..self.num_channels as usize {
            ctx.outs.audio(i).fill(0.0);
        }
    }
}

impl Unit for InTrig {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if self.audio != 0 {
            self.clear(ctx);
        } else {
            // The constructor runs `InTrig_next_k` even in a reblocked graph.
            self.next_k(ctx);
        }
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if self.audio != 0 || ctx.tick != 0 {
            self.clear(ctx);
        } else {
            self.next_k(ctx);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`InTrig`].
pub struct InTrigCtor;

impl UnitDef for InTrigCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.is_empty() {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(InTrig {
            window: ControlBusWindow::NEW,
            num_channels: ctx.num_outputs as u32,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// How many channels a [`LagIn`] smooths (scsynth's `kMaxLags`); channels past it output zero.
const MAX_LAGS: usize = 16;

/// `LagIn.kr(bus, numChannels, lag)`: reads `numChannels` control bus channels from `bus`, each
/// smoothed by a one-pole lag that reaches -60 dB in `lag` seconds.
///
/// A direct port of scsynth's `LagIn`: the constructor reads the bus unsmoothed
/// (`LagIn_next_0`), then each block steps `y = z + b1 * (y - z)` once per channel, in single
/// precision (`LagIn_next_k`). The coefficient `b1 = exp(log001 / (lag * sr))` is computed once,
/// from the World's control rate. Only the first 16 channels (`kMaxLags`) are read; the rest output
/// zero. In a reblocked or resampled graph only the first tick of each World block steps the lag
/// and the others repeat its value (`LagIn_next_k_reblock`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LagIn {
    window: ControlBusWindow,
    num_channels: u32,
    /// The lag coefficient (scsynth's `m_b1`).
    b1: f32,
    /// Each channel's smoothed value (scsynth's `m_y1`).
    y1: [f32; MAX_LAGS],
}

impl LagIn {
    const BUS: usize = 0;
    const LAG: usize = 1;

    /// `LagIn_next_k` (`smooth`) or `LagIn_next_0` (not `smooth`).
    fn next(&mut self, ctx: &mut ProcessCtx<'_>, smooth: bool) {
        let num_channels = self.num_channels as usize;
        let fbus_channel = ctx.ins.control(Self::BUS);
        self.window.update(
            fbus_channel,
            num_channels,
            unit::num_control_buses(ctx.buses),
        );
        for i in 0..num_channels {
            *ctx.outs.control(i) = if i < MAX_LAGS {
                let z = self.window.read(ctx.buses, fbus_channel, i);
                let y = if smooth {
                    ops::zapgremlins(z + self.b1 * (self.y1[i] - z))
                } else {
                    z
                };
                self.y1[i] = y;
                y
            } else {
                0.0
            };
        }
    }
}

impl Unit for LagIn {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let lag = ctx.ins.control(Self::LAG);
        // The World's control rate (`world->mBufRate.mSampleRate`), which a reblocked or resampled
        // graph does not share: the World's sample rate over its block size.
        let world_rate = ctx.audio.sample_rate
            / ctx.resample_factor as f64
            / ctx.buses.audio().block_size() as f64;
        self.b1 = if lag == 0.0 {
            0.0
        } else {
            math::exp(LOG001 / (lag as f64 * world_rate)) as f32
        };
        self.next(ctx, false);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if ctx.tick == 0 {
            self.next(ctx, true);
        } else {
            for i in 0..self.num_channels as usize {
                *ctx.outs.control(i) = self.y1.get(i).copied().unwrap_or(0.0);
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`LagIn`].
pub struct LagInCtor;

impl UnitDef for LagInCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(LagIn {
            window: ControlBusWindow::NEW,
            num_channels: ctx.num_outputs as u32,
            b1: 0.0,
            y1: [0.0; MAX_LAGS],
        }))
    }
}
