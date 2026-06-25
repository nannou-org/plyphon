//! The World's time-tag scheduler and drift-correcting clock - plyphon's port of scsynth's
//! `mScheduler` and the `mOSCbuftime`/DLL machinery in `SC_CoreAudio.cpp`.
//!
//! A time-tagged OSC bundle reaches the World as one or more [`Command`]s tagged with an absolute
//! OSC/NTP time (see [`CommandTime`](crate::command::CommandTime)). The [`Scheduler`] holds those
//! until their time arrives; the [`Clock`] tracks where "now" is on the OSC timeline and converts a
//! scheduled time into a within-block sample offset.
//!
//! Faithful to scsynth, the *schedule is keyed in absolute OSC/NTP time and resolved on the audio
//! thread* against a continuously drift-corrected clock - not a fixed origin. Each host buffer the
//! clock resyncs to the buffer's OSC time and a DLL (delay-locked loop) refines the OSC-units-per-
//! block increment from the true hardware sample rate, so timing stays accurate as the audio device
//! clock drifts against the host clock.

use alloc::collections::BinaryHeap;
use core::cmp::{Ordering, Reverse};

use plyphon_dsp::math;

use crate::command::Command;

/// 2^32, the number of OSC/NTP fixed-point units in one second (OSC time is 32.32 fixed point).
const OSC_UNITS_PER_SEC: f64 = 4_294_967_296.0;

/// The DLL smoothing coefficient (scsynth's `0.002` in `SC_CoreAudio.cpp`): a first-order low-pass
/// on the instantaneous sample-rate estimate.
const DLL_SMOOTH: f64 = 0.002;

/// A [`Command`] waiting in the [`Scheduler`] for its OSC/NTP time.
struct Scheduled {
    time: u64,
    /// A monotonic tie-breaker so equal-time commands (e.g. a bundle's messages) keep submission
    /// order - scsynth's `stabilityCount`.
    seq: u64,
    command: Command,
}

impl Ord for Scheduled {
    fn cmp(&self, other: &Self) -> Ordering {
        self.time.cmp(&other.time).then(self.seq.cmp(&other.seq))
    }
}

impl PartialOrd for Scheduled {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Scheduled {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.seq == other.seq
    }
}

impl Eq for Scheduled {}

/// A fixed-capacity min-heap of time-tagged commands - plyphon's port of scsynth's
/// `PriorityQueueT<SC_ScheduledEvent, 2048>`.
///
/// Capacity is reserved up front and never exceeded (a push at capacity is refused), so the heap
/// never allocates or frees on the audio thread.
pub(crate) struct Scheduler {
    heap: BinaryHeap<Reverse<Scheduled>>,
    capacity: usize,
    seq: u64,
}

impl Scheduler {
    /// Create a scheduler holding up to `capacity` pending commands.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Scheduler {
            heap: BinaryHeap::with_capacity(capacity),
            capacity,
            seq: 0,
        }
    }

    /// Schedule `command` for OSC/NTP `time`. Returns `Err(command)` (the command handed back) if
    /// the scheduler is at capacity.
    pub fn push(&mut self, time: u64, command: Command) -> Result<(), Command> {
        if self.heap.len() >= self.capacity {
            return Err(command);
        }
        let seq = self.seq;
        self.seq = self.seq.wrapping_add(1);
        self.heap.push(Reverse(Scheduled { time, seq, command }));
        Ok(())
    }

    /// The OSC/NTP time of the earliest pending command (scsynth's `NextTime`), or `None` if empty.
    pub fn next_time(&self) -> Option<u64> {
        self.heap.peek().map(|Reverse(s)| s.time)
    }

    /// Remove and return the earliest pending command and its time (scsynth's `Remove`).
    pub fn pop(&mut self) -> Option<(u64, Command)> {
        self.heap.pop().map(|Reverse(s)| (s.time, s.command))
    }
}

/// The World's drift-correcting OSC/NTP clock - plyphon's port of scsynth's `mOSCbuftime` plus the
/// `SC_CoreAudio.cpp` DLL.
pub(crate) struct Clock {
    /// OSC/NTP time at the start of the control block about to run (scsynth's `mOSCbuftime`).
    buftime: u64,
    /// OSC/NTP units the clock advances per control block (scsynth's `mOSCincrement`),
    /// drift-corrected by the DLL.
    increment: u64,
    /// `sample_rate / 2^32` - converts an OSC-time delta to samples (scsynth's `mOSCtoSamples`).
    to_samples: f64,
    /// `block_size * 2^32` - the numerator of `increment = numerator / smooth_rate` (scsynth's
    /// `mOSCincrementNumerator`).
    numerator: f64,
    /// The DLL's smoothed estimate of the true hardware sample rate (scsynth's `mSmoothSampleRate`).
    smooth_rate: f64,
    /// The previous resync's `(buffer_time, emitted_frames)`, for the DLL's rate estimate; `None`
    /// before the first resync.
    prev: Option<(u64, u64)>,
}

impl Clock {
    /// A clock for `sample_rate` Hz and `block_size`-sample control blocks, starting at OSC time 0
    /// with the nominal increment.
    pub fn new(sample_rate: f64, block_size: usize) -> Self {
        let numerator = block_size as f64 * OSC_UNITS_PER_SEC;
        Clock {
            buftime: 0,
            increment: (numerator / sample_rate) as u64,
            to_samples: sample_rate / OSC_UNITS_PER_SEC,
            numerator,
            smooth_rate: sample_rate,
            prev: None,
        }
    }

    /// Resync to a host-supplied buffer OSC time, refining the DLL from the frames emitted since the
    /// last resync (scsynth's per-callback `mOSCbuftime` set and `mOSCincrement` update). `emitted`
    /// is the total frames emitted before this buffer.
    pub fn resync(&mut self, buffer_time: u64, emitted: u64) {
        if let Some((prev_time, prev_emitted)) = self.prev {
            let dsec = buffer_time.wrapping_sub(prev_time) as f64 / OSC_UNITS_PER_SEC;
            let dsample = emitted.wrapping_sub(prev_emitted) as f64;
            if dsec > 0.0 && dsample > 0.0 {
                let inst = dsample / dsec;
                self.smooth_rate += DLL_SMOOTH * (inst - self.smooth_rate);
                self.increment = (self.numerator / self.smooth_rate) as u64;
            }
        }
        self.buftime = buffer_time;
        self.prev = Some((buffer_time, emitted));
    }

    /// The OSC/NTP time at the end of the current block (scsynth's `nextTime = oscTime + oscInc`).
    pub fn block_end(&self) -> u64 {
        self.buftime.wrapping_add(self.increment)
    }

    /// The within-block sample offset for a command scheduled at OSC/NTP `time` (scsynth's
    /// `mSampleOffset`), clamped to `[0, block_size - 1]`. A late time (before the block start)
    /// clamps to 0.
    pub fn sample_offset(&self, time: u64, block_size: usize) -> usize {
        let diff = (time.wrapping_sub(self.buftime) as i64) as f64 * self.to_samples + 0.5;
        math::floor(diff).clamp(0.0, (block_size - 1) as f64) as usize
    }

    /// Advance to the next control block (scsynth's `oscTime = mOSCbuftime = nextTime`).
    pub fn advance(&mut self) {
        self.buftime = self.block_end();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// One OSC/NTP unit-per-sample at 48 kHz.
    fn units_per_sample(sr: f64) -> f64 {
        OSC_UNITS_PER_SEC / sr
    }

    fn drain(scheduler: &mut Scheduler) -> Vec<i32> {
        let mut out = Vec::new();
        while let Some((_, command)) = scheduler.pop() {
            match command {
                Command::FreeNode { node } => out.push(node),
                _ => unreachable!(),
            }
        }
        out
    }

    #[test]
    fn scheduler_pops_in_time_then_submission_order() {
        let mut s = Scheduler::new(8);
        // Same time (300) for nodes 3 and 30: node 3 is pushed first, so it must pop first.
        // (`Command` has no `Debug`, so assert on `is_ok` rather than `unwrap`.)
        assert!(s.push(300, Command::FreeNode { node: 3 }).is_ok());
        assert!(s.push(100, Command::FreeNode { node: 1 }).is_ok());
        assert!(s.push(300, Command::FreeNode { node: 30 }).is_ok());
        assert!(s.push(200, Command::FreeNode { node: 2 }).is_ok());
        assert_eq!(drain(&mut s), [1, 2, 3, 30]);
    }

    #[test]
    fn scheduler_refuses_at_capacity_and_hands_the_command_back() {
        let mut s = Scheduler::new(2);
        assert!(s.push(1, Command::FreeNode { node: 1 }).is_ok());
        assert!(s.push(2, Command::FreeNode { node: 2 }).is_ok());
        match s.push(3, Command::FreeNode { node: 3 }) {
            Err(Command::FreeNode { node: 3 }) => {}
            _ => panic!("expected the command handed back on overflow"),
        }
    }

    #[test]
    fn clock_nominal_increment_is_one_block() {
        let clock = Clock::new(48_000.0, 64);
        // From OSC time 0 the block end is exactly the nominal increment: block_size / sr seconds.
        let nominal = (64.0 * OSC_UNITS_PER_SEC / 48_000.0) as u64;
        assert_eq!(clock.block_end(), nominal);
    }

    #[test]
    fn clock_sample_offset_maps_time_to_samples_and_clamps() {
        let mut clock = Clock::new(48_000.0, 64);
        // Resync (no DLL update on the first call) to move the block start off zero.
        let base = 1_000_000_000_000;
        clock.resync(base, 0);
        let ups = units_per_sample(48_000.0);

        // 10 samples into the block -> offset 10.
        assert_eq!(clock.sample_offset(base + (10.0 * ups) as u64, 64), 10);
        // A time before the block start (late) clamps to 0.
        assert_eq!(clock.sample_offset(base - 5_000, 64), 0);
        // Beyond the block clamps to block_size - 1.
        assert_eq!(clock.sample_offset(base + (100.0 * ups) as u64, 64), 63);
    }

    #[test]
    fn clock_dll_tracks_a_faster_device_rate() {
        let mut clock = Clock::new(48_000.0, 64);
        let nominal_inc = clock.block_end(); // increment from OSC time 0 at the nominal 48 kHz
        // Feed buffers spaced as if the device runs fast (48.1 kHz): more samples elapse per second,
        // so the DLL-smoothed increment (OSC units per block) should settle below the nominal one.
        let step = (64.0 * OSC_UNITS_PER_SEC / 48_100.0) as u64;
        let start = 1_000_000_000_000u64;
        let mut last = start;
        for k in 0..3000u64 {
            last = start + k * step;
            clock.resync(last, k * 64);
        }
        // After the last resync the block start is `last`, so `block_end - last` is the increment.
        let inc = clock.block_end() - last;
        assert!(
            inc < nominal_inc,
            "smoothed increment {inc} should fall below nominal {nominal_inc} for a faster device"
        );
    }
}
