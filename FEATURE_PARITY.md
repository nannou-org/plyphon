# Feature parity with scsynth

A living checklist of where plyphon stands against [`scsynth`][scsynth], to gauge what works today
and what is left to port. plyphon is an early-stage research engine, so most of the surface is still
ahead of it. A checked box (`[x]`) is on par with scsynth; an unchecked box (`[ ]`) is not done yet -
partial items stay unchecked and spell out what is missing.

## Engine & architecture

- [x] Lock-free real-time / control split (`World` / `Controller` / `Nrt`); no locks, allocation, or blocking on the audio thread
- [x] No `unsafe` (`#![forbid(unsafe_code)]`) and no global mutable state - instances are passed by argument, so the engine is multi-instance and headless-testable (`World::fill`)
- [x] One engine across native and `wasm32-unknown-unknown` (runs in the browser)
- [x] Reblocks any host buffer size to the internal control block
- [x] Per-UGen RNG (Taus88) and engine-owned wavetables
- [x] Multi-channel output buses and audio input buses (duplex via `World::fill_duplex`)
- [x] Calc rates: scalar, control, audio
- [ ] Demand rate
- [ ] Non-real-time (score render) mode - real-time only so far
- [ ] OSC bundle time-tag scheduling - bundles are applied immediately, the time tag is ignored

Dynamic binary plugin loading (`.scx`) is intentionally out of scope: UGens are compiled into the
engine (pure Rust, no FFI), so there is nothing to load at runtime.

## UGens (21 of scsynth's ~250, grouped by category)

- [ ] **I/O** - have Out, In; missing ReplaceOut, OffsetOut, XOut, LocalIn/LocalOut, InFeedback, SoundIn
- [ ] **Oscillators** - have SinOsc, Saw, Pulse, LFSaw, LFPulse, Impulse; missing Blip, VarSaw, SyncSaw, LFTri/LFPar/LFCub, Osc/OscN, COsc, FSinOsc, Klang, Klank
- [ ] **Noise** - have WhiteNoise; missing PinkNoise, BrownNoise, GrayNoise, ClipNoise, Dust/Dust2, LFNoise0/1/2, LFDNoise*, Crackle
- [ ] **Filters** - have LPF, HPF, Lag; missing BPF, BRF, RLPF, RHPF, Resonz, Ringz, OnePole/OneZero, TwoPole/TwoZero, Integrator, LeakDC, Slew, Decay/Decay2, Formlet, MoogFF, MidEQ
- [ ] **Envelopes** - have EnvGen, Line; missing XLine, Linen, IEnvGen, DemandEnvGen
- [ ] **Panning** - have Pan2; missing LinPan2, Pan4, Balance2, Rotate2, XFade2, LinXFade2, PanAz, Splay
- [ ] **Dynamics** - have Amplitude; missing Compander, Limiter, Normalizer, DetectSilence
- [ ] **Math / multichannel** - have BinaryOpUGen, UnaryOpUGen, MulAdd; missing Sum3/Sum4, Select, Index, Clip/Wrap/Fold, LinLin/LinExp
- [ ] **Buffer playback** - have PlayBuf, DiskIn; missing BufRd, BufWr, RecordBuf, DiskOut, VDiskIn, TGrains, GrainBuf
- [ ] **Triggers / timing** - none yet: Trig/Trig1, TDelay, Latch, Gate, Phasor, Sweep, Timer, PulseCount, PulseDivider, Stepper, ToggleFF, SendTrig, SendReply, Done, FreeSelf, Pause
- [ ] **Info** - none yet: SampleRate, SampleDur, ControlRate, BufFrames, BufDur, NumChannels, RadiansPerSample
- [ ] **Delays / reverb** - none yet: DelayN/L/C, CombN/L/C, AllpassN/L/C, FreeVerb, GVerb, Pluck, PitchShift
- [ ] **Demand-rate** - none yet: Demand, Duty/TDuty, Dseq, Dser, Drand, Dwhite, Dseries, Dgeom (needs demand rate)
- [ ] **FFT / spectral** - none yet: FFT/IFFT, the `PV_*` set, Pitch, Onsets, BeatTrack
- [ ] **Chaos / rate conversion** - none yet: Lorenz, LinCong, Henon, ... and A2K/K2A/T2A/DC

## OSC server commands (22 of ~65)

**Server / top-level** (0/10)

- [ ] /notify
- [ ] /status
- [ ] /quit
- [ ] /cmd
- [ ] /dumpOSC
- [ ] /clearSched
- [ ] /sync
- [ ] /error
- [ ] /version
- [ ] /rtMemoryStatus

**SynthDef** (1/5)

- [x] /d_recv
- [ ] /d_load
- [ ] /d_loadDir
- [ ] /d_free
- [ ] /d_freeAll

**Synth** (1/4)

- [x] /s_new
- [ ] /s_get
- [ ] /s_getn
- [ ] /s_noid

**Node** (7/15)

- [x] /n_set
- [x] /n_free
- [x] /n_map
- [x] /n_mapn
- [x] /n_before
- [x] /n_after
- [x] /n_order
- [ ] /n_setn
- [ ] /n_fill
- [ ] /n_run - the engine already pauses/resumes nodes (`Controller::node_run`); just not wired to OSC
- [ ] /n_query
- [ ] /n_trace
- [ ] /n_mapa
- [ ] /n_mapan
- [ ] /n_cmd

**Group** (5/8)

- [x] /g_new
- [x] /g_head
- [x] /g_tail
- [x] /g_freeAll
- [x] /g_deepFree
- [ ] /p_new
- [ ] /g_dumpTree
- [ ] /g_queryTree

**Unit** (0/1)

- [ ] /u_cmd

**Control bus** (2/5)

- [x] /c_set
- [x] /c_setn
- [ ] /c_fill
- [ ] /c_get
- [ ] /c_getn

**Buffer** (6/17)

- [x] /b_alloc
- [x] /b_allocRead
- [x] /b_read
- [x] /b_free
- [x] /b_zero
- [x] /b_query
- [ ] /b_write
- [ ] /b_close
- [ ] /b_set
- [ ] /b_setn
- [ ] /b_fill
- [ ] /b_gen
- [ ] /b_get
- [ ] /b_getn
- [ ] /b_allocReadChannel
- [ ] /b_readChannel
- [ ] /b_setSampleRate

## Replies, notifications & done actions

- [x] Replies: /done, /fail, /b_info, and node notifications /n_go /n_end /n_off /n_on
- [ ] /status.reply, /synced (the `/sync` barrier), /tr (SendTrig), /n_info (`/n_query`), /g_queryTree.reply, and the `/c_get` / `/b_get` getters
- [ ] Done actions beyond 0 (none), 1 (pause self), 2 (free self): codes 3-14, the free/pause variants that also touch neighbours or the enclosing group

## SynthDefs & buffers

- [x] SCgf binary SynthDefs load via `/d_recv` (and the [`scgf`](crates/scgf) crate also encodes them); named parameters are folded from SC's `Control` UGens
- [ ] Control family beyond plain `Control`: `TrigControl`/`LagControl` parse but behave as plain controls; SynthDef variants
- [x] Buffers: allocate, free, zero, query, and asynchronous loading through an app-provided [`BufferSource`](crates/plyphon-buffers) (the I/O seam), plus chunk-streaming playback with `DiskIn`
- [ ] Writing/recording buffers to disk, `b_gen` wavetable fills, and `b_get`/`b_set` element access

[scsynth]: https://github.com/supercollider/supercollider
