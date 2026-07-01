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
- [x] Per-instance UGen auxiliary memory - a UGen can reserve a build-time-sized buffer (a delay line, sized from a scalar input like `maxdelaytime`) folded into the synth's single rt-pool block, so the whole delay/comb/allpass family is a pure UGen addition. Unlike scsynth's per-unit `RTAlloc` there is still one allocation and one free per synth; the arena is left uninitialised (no `/s_new` memset) and units guard their own cold start, as scsynth's `_z` calc variants do
- [x] Multi-channel output buses and audio input buses (duplex via `World::fill_duplex`)
- [x] Calc rates: scalar, control, audio
- [x] Demand rate - pull-based, like scsynth: a consumer (`Demand`/`Duty`) pulls a source's next value (or resets it) on the audio thread, recursing through nested sources; sources are single-output and emit `NaN` at exhaustion. Demand units are split out of the per-block calc list into a separate demand plan with their own block span, and the recursion is allocation-free (a bounded stack copy of `Pod` state); off-RT compilation rejects over-large state or over-deep nesting so the audio thread stays bounded
- [x] Non-real-time (score render) mode - `Render` drives the engine offline on a deterministic free-running clock (no DLL resync), feeding a time-tagged score lazily; the `plyphon-osc` `parse_score`/`render_osc_score` pair renders scsynth's binary OSC command file to audio (see `example-render`)
- [x] OSC bundle time-tag scheduling - bundle time tags schedule sample-accurately on the audio thread, against a drift-corrected clock (a DLL tracking the device rate); `OffsetOut` delays a scheduled synth's whole output by its creation offset for the synth's life (scsynth's `OffsetOut_next` delay-and-carry, the leading samples silent and each block's tail carried into the next via per-channel `aux`), so the onset lands on the exact sample and no input sample is dropped. The one divergence: plyphon units have no destructor, so the final `offset` samples in the carry are not flushed on free (scsynth's `OffsetOut_Dtor`) - ~0 for a voice ending in silence

Dynamic binary plugin loading (`.scx`) is intentionally out of scope: UGens are compiled into the
engine (pure Rust, no FFI), so there is nothing to load at runtime.

The **shared-memory / internal-server** surface is also intentionally out of scope, for the same
reason plyphon shares no memory with its clients: scsynth's **shm scope ring** (the transport behind
`ScopeOut2`, mapped and polled by the GUI), **shared controls** (the mmap'd shared control-bus region),
the `/late` lateness reply (a late bundle is resolved to the soonest sample, never reported), and the
legacy **binary integer OSC command form** (`inData[0] == 0` selecting `gCmdArray[int]`; plyphon
dispatches string addresses only). A host that wants scope/metering uses plyphon's own **`ScopeOut`**
instead (implemented - see the I/O category below), which streams every input sample over the RT-safe
chunk ring rather than shared memory. `LocalBuf` (SynthDef-side local buffers, e.g. for an FFT chain)
is **deferred**, not excluded - a chain buffer is `/b_alloc`'d instead, so it does not block the `PV_*`
family.

Intra-graph **reblock and resample** (scsynth's `kGraph_Reblock` / `kGraph_Resample`, per-SynthDef
`Reblock(n)` / `Resample(n)`) are **done**: a def can run its graph at a smaller power-of-two control
block than the World's (tighter envelope/trigger timing, lower-latency `LocalIn`/`LocalOut` feedback -
the local bus shrinks to the graph block) and/or oversampled by a power-of-two factor (anti-aliasing
for nonlinear units). The `GraphDef` carries its own `RateInfo` pair (smaller block and/or higher
sample rate), `Graph::process` runs the calc list `num_ticks = (world_block / block) * factor` times
per World block (collapsing to one pass, zero-cost, for an ordinary def), and the boundary
`In`/`Out`/`OffsetOut`/`AudioControl` cross the World-block-wide bus per tick: `In` reads its tick
slice and zero-order-holds it up to the graph rate, `Out` decimates the oversampled interior down
(`audio_out_decimated`, which clears the channel on the first writer then sums each tick's slice).
Interior DSP units are unaware, packing their wires at the graph block. Authored with
`Controller::add_synthdef_reblocked(def, block)` / `add_synthdef_resampled(def, factor)`; a reblocked
linear chain is byte-identical to the non-reblocked one and an oversampled band-limited chain matches
to within oscillator phase drift. (`LocalIn`/`LocalOut` need no decimate/ZOH - their bus is per-synth
and already graph-rate - and `OffsetOut`'s onset offset, World-block-relative, coarsens to per-tick, a
documented edge.)

Reblock/resample also load from a **scsynth version-3 binary def**: `scgf` parses the v3 framing (a
per-def `int32` size prefix and the trailing `blockSize`/`resampleFactor` fields), and
`plyphon::synthdef::read::parse` hands each def's setting to `Controller::add_synthdef_rate`, so a
`.scsyndef` compiled with `Reblock`/`Resample` is honoured on `/d_recv` and `/d_load` - not only via
the programmatic API. The **control-driven** forms (`blockSize -1` / `resampleFactor -1.0`, where the
value comes from a synth control at instantiation) are unsupported - plyphon bakes the graph block into
the per-synth layout at compile - and fall back to no reblock/resample.

## UGens (250 of scsynth's ~382 standard DSP UGens, grouped by category)

The remaining gap to full scsynth compatibility is UGen breadth; the engine seams these need (aux
memory, per-unit RNG, the demand vtable, the FFT/PV bin seam, trailing-input spec arrays) are all in
place, so most are pure per-unit ports. Note the two operator *shells* below (`BinaryOpUGen` /
`UnaryOpUGen`) each cover dozens of `special_index` operators, so their breadth far exceeds one UGen.

- [ ] **I/O** - have Out, ReplaceOut (overwrites the bus channel instead of summing, via a `write_replace` bus method), OffsetOut, In, LocalIn, LocalOut, InFeedback (a per-synth feedback bus with a one-block delay; `InFeedback` aliases `In`), and ScopeOut - a live monitoring/analysis tap that streams every sample of its (multichannel) input off the audio thread to the app, the shared-memory-free equivalent of scsynth's `ScopeOut2`. It reuses `DiskOut`'s bounded lock-free chunk-ring transport (`Controller::cue_scope` returns the `StreamConsumer` the app drains); scsynth's `ScopeOut` (an interleaved plain-`SndBuf` read only by the in-process internal server) and `ScopeOut2` (a shm ring) are both replaced by this one no-shm streaming path. Several `ScopeOut` units on distinct bufnums tap several graph points at once. Missing XOut (crossfade write - the touched-based mix under reblock is deferred), SoundIn
- [ ] **Oscillators** - have SinOsc, Saw, Pulse, LFSaw, LFPulse, Impulse, plus LFTri, LFPar, LFCub, VarSaw, SyncSaw (scsynth's own `f64` phase accumulators) and FSinOsc (a resonator sine), plus the additive/modal resonator banks Klang (a fixed additive sum of self-oscillating `2·cos(w)` sine partials) and Klank (a bank of decaying `Ringz`-style resonators driven by a shared excitation input - modal synthesis). Both take a trailing `[freq, amp, phase|ringtime]` spec array (interleaved triples, scsynth's `.flop.flat`), compute the per-partial coefficients + running state once into aux memory on the first block (a `warmed` guard over the un-zeroed arena), then the block loop only advances the bank. The wavetable oscillators read a user buffer as a single-cycle table with the same normalised phase accumulator `SinOsc` uses: Osc (interpolating) reads scsynth's `(a, b)` interpolation-format wavetable via a new `plyphon_dsp::wavetable` reader/packer (`to_wavetable`/`lookup_wavetable`), fed by `/b_gen … wavetable`; OscN truncates a plain buffer (nearest-lower, any size, since the phase is a float not a masked fixed-point index); COsc sums two copies detuned by `beats` Hz (a chorusing oscillator); and the wavetable-crossfade oscillators VOsc and VOsc3 morph across a bank of consecutively-numbered equal-size wavetables by the fractional part of `bufpos` (ramped across the block), VOsc3 summing three detuned voices. SinOscFB is a sine that phase-modulates itself by its own last output (feedback FM), and LFGauss is a Gaussian-shaped grain/LFO whose ramp either loops or fires a done action. Blip is a band-limited impulse train evaluated directly from the Dirichlet-kernel closed form (a normalised sum of `numharm` cosine harmonics, clamped to Nyquist) rather than scsynth's cosecant lookup table, and Formant is a pitch-synchronous windowed-grain formant oscillator, completing the classic oscillator set. (`DynKlank`/`DynKlang` are *not* server UGens - sclang expands them at SynthDef-build time into a summed `Ringz`/`SinOsc` bank, supported here via plyphon's dynamic `Ringz` (which re-derives its coefficients on any freq/decay change) and the new `Sum3`/`Sum4` summers; the buffer-lookup units Select/Index/Shaper/DegreeToKey are done - see Math / multichannel.)
- [x] **Granular** - GrainSin (windowed sine grains spawned on a rising trigger and panned across the output channels by an equal-power `cos`/`sin` law). Grains live in a fixed 64-slot in-struct array (no allocation), removed by swapping the last active grain into the vacated slot; the default window is scsynth's inline `sin²` (Hann) oscillator recurrence, and `envbufnum >= 0` reads a user buffer as the window. GrainFM makes each grain an FM carrier/modulator pair (peak deviation `index * modfreq`); GrainIn windows the live input signal; GrainBuf plays a mono buffer per grain (start `pos`, pitch `rate`, `interp` none/linear/cubic, wrapped read); TGrains is the same buffer grain always using the default window, centred on `centerPos` seconds with an explicit `amp` folded into the pan gains. Warp1 is the self-triggering granular time-stretcher: no trigger input, each output channel an independent grain cloud (a decorrelated read of the same buffer at pitch `freqScale`, spawning every `windowSize/overlaps` seconds with a `windowRandRatio`-randomised window from a per-unit `Rng`); its per-channel grain banks live in build-time-sized auxiliary memory, capped at scsynth's 16 channels
- [x] **Noise** - have WhiteNoise, ClipNoise, GrayNoise, PinkNoise, BrownNoise, Dust, Dust2 (each embeds the per-unit Taus88 `Rng`, reproducing scsynth's `SC_RGen.h` bit-tricks with safe `f32::from_bits`), plus the low-frequency/dynamic noise family LFNoise0/1/2, LFClipNoise (a whole-sample counter between random values, so transitions quantise to the sample rate) and the dynamic LFDNoise0/1/3, LFDClipNoise (a floating phase decremented by `freq * sampleDur`, so `freq` can be modulated at audio rate and transitions land off-grid; LFDNoise3's cubic reconstruction reuses the new `plyphon_dsp::interp::cubicinterp`). The chaotic/deterministic set completes it: Crackle (the map `y0 = |y1*param - y2 - 0.05|`, seeded from the RGen), Logistic (the logistic map `y = param*y*(1-y)` iterated at `freq`, held between iterations, from an `init` seed), Hasher (Thomas Wang's integer hash of the input's bits packed into `[-1, 1)`, so equal inputs give equal outputs), and MantissaMask (keep only the top `bits` mantissa bits - a cheap bit-crush)
- [x] **Filters** - have LPF, HPF, Lag, plus the primitives OnePole, OneZero, Integrator, LeakDC, TwoPole, TwoZero, Decay, Decay2; the resonant biquads RLPF, RHPF, BPF, BRF, Resonz, Ringz; and the fixed-coefficient/delay set LPZ1/HPZ1/LPZ2/HPZ2/BPZ2/BRZ2, Delay1/Delay2, Slope, Slew, APF (all `f64` state flushed with a shared `zap`, coefficients derived per block as `Butter` does), plus the explicit-coefficient sections FOS and SOS (direct-form-II first-/second-order sections that take the raw difference-equation coefficients as inputs). scsynth's whole `B*` EQ suite (BLowPass/BHiPass/BBandPass/BBandStop/BPeakEQ/BAllPass/BLowShelf/BHiShelf) is a *language-side* macro that computes RBJ biquad coefficients and feeds `SOS`, so those defs load as `SOS` + coefficient math - porting `SOS` covers the set. The lag family is rounded out with Lag2/Lag3 (two/three `Lag`s in series, progressively smoother) and the asymmetric LagUD/Lag2UD/Lag3UD (separate rise/fall smoothing times), plus the linear-ramp smoothers Ramp (resamples and interpolates every `lagTime`) and VarLag (ramps from `start` toward the input, rescaling in-flight when the time changes). The resonant EQ/formant filters Formlet (a formant resonator - an attack `Ringz` subtracted from a decay `Ringz` at the same frequency) and MidEQ (a parametric peak/notch that boosts or cuts `db` around `freq`) share the `Ringz` coefficient math. MoogFF is the Moog-ladder resonant low-pass (Fontana feedback form, `gain` 0-4), and Median is a running-median spike-rejecter over a fixed odd window (in-struct array, capped at 32, no allocation). Hilbert splits the input into an analytic `[real, 90-degree]` pair via a 12-stage IIR all-pass phase-difference network (no FFT), and FreqShift ring-modulates that pair with a quadrature oscillator (the shared sine table) for single-sideband frequency shifting - completing the filter set. (The look-ahead dynamics `Normalizer`/`Limiter` are tracked under Dynamics; `Flip` is a `Normalizer` helper, not a filter.)
- [ ] **Envelopes** - have EnvGen, Line, XLine (exponential line, latch + done-action like Line); missing Linen, IEnvGen, DemandEnvGen
- [ ] **Panning** - have Pan2, LinPan2, Balance2, XFade2, LinXFade2, Rotate2 (equal-power units share `Pan2`'s cos/sin law); missing Pan4, PanAz, PanB/PanB2/BiPanB2, DecodeB2 (client-side Splay is out of scope)
- [x] **Dynamics** - have Amplitude, Compander (side-chain compressor/expander with a smoothed gain), DetectSilence (done-action on silence), and the look-ahead peak processors Limiter (caps the peak *at* `level`) and Normalizer (drives the peak *to* `level`) - both delay the signal by `dur` through a rotating triple-buffer sized in aux memory from the constant `dur`, so the gain glides in before each peak arrives at the output (they emit silence for the first `2*dur` of look-ahead latency); the mode is the only difference in the shared kernel
- [ ] **Math / multichannel** - BinaryOpUGen and UnaryOpUGen now implement scsynth's **full audio-rate operator set** - every pure binary op (add/sub/mul/div/idiv/mod, the comparisons, min/max, the bit/shift ops, lcm/gcd, round/roundUp/trunc, atan2/hypot/hypotx, pow, the ring/sqr/dif family, thresh/amclip/scaleneg/clip2/excess/fold2/wrap2/firstArg) and every pure unary op (neg/not/bitNot/abs, ceil/floor/frac/sign, squared/cubed, signed-sqrt, exp/recip, the midicps/cpsmidi/octcps/dbamp/... conversions, log/log2/log10, the full trig/hyperbolic set, distort/softclip, the rect/han/welch/tri windows, ramp/scurve). Kernels live in `plyphon_dsp::ops` (mirroring scsynth's `SC_Inline*Op.h`) and are shared with the range shapers. Deferred: the graph-RNG ops (rand/rand2/linrand/coin/rrand/exprand - they ride the noise UGens' per-unit RNG) and non-signal ops (isNil/asFloat/...). Have MulAdd, Sum3/Sum4 (the optimised 3-/4-input summers scsynth's SynthDef optimiser rewrites addition chains into, so a `.sum`/`Mix`/`DynKlank` def loads), the range shapers Clip, Wrap, Fold, ModDif, InRange, InRect, LinExp, Unwrap, and the selection/lookup family Select (pass one of several signal inputs by index) plus Index, IndexL, WrapIndex, FoldIndex (read a `/b_alloc`'d buffer as a lookup table by index - clip / linear-interpolate / wrap / fold the index into range respectively; the integer wrap/fold reuse the new `plyphon_dsp::ops::iwrap`/`ifold`, shared with `Dibrown`), plus Shaper (waveshape a signal through a `(a, b)`-wavetable transfer function, e.g. a `/b_gen cheby … wavetable` Chebyshev table, via `plyphon_dsp::wavetable::shape_wavetable`) and DegreeToKey (map a floored scale-degree through a scale buffer to a key, wrapping whole octaves by Euclidean modulo). Missing the helper UGens LinLin, AmpComp/AmpCompA
- [ ] **Buffer playback / recording** - have PlayBuf, DiskIn, RecordBuf (record into a buffer with overdub/run/loop/doneAction), BufWr (write channels at a phase index), DiskOut (stream channels out to a cued recording buffer, drained off the audio thread to a sink); missing BufRd, VDiskIn
- [ ] **Triggers / timing** - have SendTrig (fires `/tr` on a rising edge, at control or audio rate), SendReply (emits a custom OSC path with a bounded number of values, over a dedicated node-message ring), FreeSelf, PauseSelf, Done, FreeSelfWhenDone, PauseSelfWhenDone, Free, Pause, plus the in-graph trigger/hold/flip-flop set Trig, Trig1, TDelay, Latch, Gate, ToggleFF, SetResetFF, Schmidt and the counting/timing set PulseCount, PulseDivider, Stepper, ZeroCrossing, Timer, Sweep, Phasor (a shared `Sig` helper reads each input at its declared rate and edges are detected across block boundaries), and the signal-measurement set Peak, RunningMin, RunningMax (running |max|/min/max, reset on a trigger), PeakFollower (an amplitude envelope follower - instant attack, exponential release), MostChange/LeastChange (pass whichever of two inputs moved most/least) and LastValue (a hysteresis sample-and-hold); missing Poll
- [ ] **Info** - have SampleRate, SampleDur, RadiansPerSample, ControlRate, ControlDur, NumOutputBuses, NumInputBuses, NumAudioBuses, NumControlBuses, NumRunningSynths, NumBuffers, BufFrames, BufChannels, BufSamples, BufSampleRate, BufRateScale, BufDur, and SubsampleOffset. `SubsampleOffset` needed a real engine feature: the scheduler now retains the *fractional* (sub-sample) part of a scheduled event's within-block position (scsynth's `mSubsampleOffset`) instead of flooring it away. `Clock::block_offset` returns the integer sample offset (clamped, round-to-nearest as before) alongside the unclamped fractional remainder in `[0, 1)`; it is threaded World -> Graph -> `ProcessCtx::subsample_offset` exactly parallel to the existing integer `sample_offset`, and the `SubsampleOffset` UGen snapshots it once on its first block (like `OffsetOut` captures the integer offset). `OffsetOut` is unchanged - scsynth's `OffsetOut` honours only the integer offset
- [ ] **Diagnostics** - have CheckBadValues (classify each sample as `0` ok / `1` NaN / `2` infinite / `3` subnormal via `f32::classify`) and Sanitize (replace any NaN/infinite/subnormal sample with a replacement signal); scsynth's `post` console diagnostics are omitted (plyphon does no printing on the audio thread, so the `id`/`post` inputs are accepted but ignored)
- [x] **Delays / reverb** - have the full core delay-line family DelayN/L/C, CombN/L/C (recirculating comb) and AllpassN/L/C, all on per-instance aux memory. They share one power-of-two circular buffer and one read kernel that splits three ways: interpolation of the fractional tap (none/linear/cubic, via `plyphon_dsp::interp`), feedback (a plain delay vs a comb/allpass recirculating the delayed value with `sc_CalcFeedback`'s -60 dB-over-decaytime coefficient), and allpass-vs-comb (the allpass subtracts the feed-forward path). The aux arena is not zeroed on instantiation, so the cold-start (`_z`) guard reproduces scsynth's signed-index checks (a tap before the start of writing reads 0, never recycled memory). The buffer-backed twins BufDelayN/L/C, BufCombN/L/C and BufAllpassN/L/C reuse that whole read kernel, but the line is a `/b_alloc`'d buffer at `bufnum` (resolved each block via `buffer_at_mut`) instead of aux memory - so the line is shared, resizable and outlives the synth; only the buffer's largest power-of-two prefix is used (scsynth's `BUFMASK`), so a requested delay past that prefix clamps to it, and `dsamp` is seeded once from the buffer at init. DelTapWr/DelTapRd split one line held in a mono buffer into a writer and readers: DelTapWr advances a wrapping write head and outputs it each sample (an integer carried through the audio wire via its `f32` bits, `from_bits`/`to_bits` - scsynth's reinterpret trick), zeroing the buffer on its first block; one or more DelTapRd read the head off that wire (its first sample) and tap the buffer `delTime` behind it, wrapping and interpolating (none/linear/cubic) - so several taps share one line (a multi-tap delay). Pluck is the Karplus-Strong plucked string: a cubic comb on aux memory (reusing the delay read kernel, `sc_CalcFeedback` and cold-start guard) whose feedback runs through a one-zero damping lowpass (`(1 - |coef|)*value + coef*lastsamp`), with the excitation `in` gated into the line for one delay period on each rising `trig`. PitchShift is a time-domain granular pitch shifter: the input is written into an aux delay line and read back by four overlapping triangular-windowed grains 90 degrees apart, each grain's read head drifting against the write head at `1 - pitchRatio` (so it replays the recent past transposed), a fresh grain spawned round-robin every quarter-window and crossfaded over the one it replaces, with per-grain pitch/time jitter from a per-unit `Rng` (`pitchDispersion`/`timeDispersion`). FreeVerb/FreeVerb2 are the classic freeverb: eight parallel damped combs (each a delay line with a one-pole lowpass in its feedback, the `damp` control) summed into four series Schroeder allpasses (feedback 0.5), mixed with the dry signal by `mix`; `room` sets the comb feedback. FreeVerb2 is true-stereo - two banks whose line lengths differ by a 23-sample spread, sharing one `process_bank` kernel and the `0.015*(in+in2)` excitation, cross-mixed to two outputs. The fixed 44.1 kHz-tuned lines live in aux memory, zeroed on the first block. GVerb is the large Griesinger-style FDN reverb: the input is band-limited and diffused into four recirculating delay lines (damped, decay-gained, mixed by a Hadamard matrix), with four early-reflection taps into a long tap delay and a diffused stereo tail; all 13 delay/diffuser buffers live in aux memory, sized at build. The one divergence: scsynth sizes the diffusers from the *initial* `roomsize`/`spread` and only rescales the FDN lengths on modulation, whereas plyphon requires `roomsize`/`spread`/`maxroomsize` to be compile-time constants (so the whole aux layout is fixed), leaving `revtime`/`damping`/the levels freely modulatable
- [ ] **Demand-rate** - have Demand, Duty, Dseq, Dseries, Dgeom (geometric), Dwhite, Diwhite (integer white), Dbrown/Dibrown (float/integer bounded random walk, folding back into `[lo, hi]`), Dbufrd/Dbufwr (demand-rate buffer read/write, via the buffer reach threaded into the demand pull), Dpoll (post a demanded value to the host), plus the list-selection sources Dser (serial, but length-counted - it yields a fixed number of values rather than full passes), Drand (a random pick from the list) and Dxrand (a random pick that never immediately repeats). Dser/Drand/Dxrand share Dseq's list-cycling/child-reset machinery (a nested demand item is pulled to exhaustion), and the random pair carry a per-instance RNG using `next_irand`. The generator sources reuse the RNG's `next_irand`/`next_irand2` (scsynth's `RGen::irand`/`irand2`). Missing: TDuty, Dwrand (weighted list), Dshuf (shuffled list - needs persistent shuffle state), Dswitch/Dswitch1, Dstutter, Ddup, Dconst, Dreset
- [ ] **FFT / spectral** (behind the default-on `fft` feature; realfft, std-only) - have FFT, IFFT (short-time analysis/resynthesis over a packed-spectrum chain buffer, via the shared `FftTables`), PV_MagMul, PV_MagSquared, plus the single-buffer ops PV_MagAbove, PV_MagBelow, PV_MagClip, PV_LocalMax, PV_PhaseShift90, PV_PhaseShift270, PV_BrickWall, PV_Conj (`pv_ops.rs`; `pv::spectrum` added for coord-independent bin edits). The bin-level seam is complete: buffers track a `coord` (Complex/Polar, scsynth's `SndBuf::coord`), `pv::to_polar`/`to_complex` convert in place idempotently (the `ToPolarApx`/`ToComplexApx` analogue, using exact `hypot`/`atan2`), `pv::pv_frame` is the `PV_GET_BUF` preamble, and `pv::Spectrum`/`Bin` are the typed `SCPolarBuf`/`SCComplexBuf` views - so the rest of the Cartesian and polar `PV_*` family is a per-unit port. The two-buffer ops PV_Add, PV_Mul, PV_Div, PV_Max, PV_Min, PV_CopyPhase and PV_Copy (`pv_combine.rs`) are also done - they read the read-only second buffer via `pv::bin_as_complex`/`bin_as_polar` (leaving it untouched, as `PV_MagMul` does). Missing: PV_BinWipe/BinShift/MagShift, the buffer-owning ops (PV_MagSmear/RectComb/Diffuser/MagFreeze/BinScramble/RandComb), FFTTrigger, Unpack1FFT/PackFFT, Pitch, Onsets, BeatTrack, and `LocalBuf` (SynthDef-side local FFT buffers; not needed - a chain buffer is `/b_alloc`'d)
- [ ] **Chaos / rate conversion** - have A2K, K2A, T2A, T2K (audio trigger -> control by taking the block maximum, so an edge anywhere in the block survives), DC, plus the sample-and-hold chaotic maps CuspN, QuadN, LinCongN, GbmanN, StandardN, LatoocarfianN (each iterates its map at a `freq` rate in `f64`; shared 1D/2D drivers). Missing: the L/C (linear/cubic-interpolating) variants, HenonN (3-state stability logic), FBSine, Lorenz, Gendy
- [ ] **Physical modeling** - have Spring (a driven damped mass-spring), Ball, TBall (a ball bouncing on a moving floor, TBall emitting the collision velocity; both embed a per-unit `Rng` for their jitter dither); missing MdaPiano and the other bundled models

## OSC server commands (65 of ~65)

The *getters* (`/status`, `/sync`, `/rtMemoryStatus`, `/n_query`, `/c_get`/`/c_getn`, `/s_get`/`/s_getn`,
`/b_get`/`/b_getn`, `/g_queryTree`) read live engine state over a third RT→NRT ring - a fixed-size
`Copy` `Reply` enum mirroring the events ring. They are *asynchronous*: the dispatcher issues one
query per element, the engine answers a block later, and the dispatcher reassembles the answers (in
the FIFO order the queries were issued) into one OSC reply message. The RT side returns only numeric
indices and values; all names are resolved control-side. `/g_dumpTree` reuses the tree walk but
formats to an optional host text sink. `/b_gen` fills buffers control-side (sine1/2/3, cheby, with
normalize) or via an engine-side copy.

A second group are **server/transport commands**: they concern the connection or the host process,
not the synthesis engine, so they live in the host/transport layer rather than the engine OSC
front-end. As in scsynth - where the audio thread always writes node notifications to FIFOs and the
comm layer alone decides delivery (iterating the registered reply-addresses, `mUsers`) - plyphon's
`OscDispatcher` always *emits* the node notifications, and who receives them is the host's call. The
`plyphon-cli` server implements these: `/notify` (a client's per-connection notification subscription,
its `notified` set mirroring `mUsers`), `/quit`, `/dumpOSC`, and `/version`. (The engine-state
queries `/status`/`/sync`/`/rtMemoryStatus` are *not* server commands - the server forwards them to
the dispatcher and routes each async answer back to the requester, alongside the other getters.)

The **host actions** need a plugin registry or filesystem the engine does not model, so they are
surfaced upward as typed capabilities on a `Host` trait the dispatcher drives in `run_pending` -
exactly as `/b_allocRead` already defers I/O to an app-provided `BufferSource`: `/cmd`/`/u_cmd` (a
`CommandHost`), `/d_load`/`/d_loadDir` (a `DefSource`), and `/b_write` + `/b_close` (a `BufferSink`,
the engine streaming the buffer out race-free - a whole-buffer snapshot, or `leaveOpen=1` left open
for `DiskOut`). The OSC surface is now complete; the one command that touches the audio thread,
`/n_trace`, streams each calc unit's I/O over the reply ring to a host text sink. (scsynth's `/n_cmd`
is unimplemented - commented out in its command table - so plyphon omits it too.) `/b_read` splices a
file region into an existing buffer at `bufStartFrame` (scsynth's `BufReadCmd`), and with `leaveOpen=1`
keeps the file open and streams it off disk into a `DiskIn` (the read counterpart to `/b_write
leaveOpen=1`); `/b_readChannel leaveOpen=1` streams only the selected channels through a deinterleaving
stream wrapper. The **buffer surface is now complete with no deferrals**. The CLI server now also
captures **live hardware input** for `In.ar` (`--input-channels` > 0): cpal has no duplex stream, so a
separate capture stream feeds an `rtrb` jitter/drift ring that the output callback drains, reblocking the
engine to exact control blocks (a carry-FIFO) so input stays sample-faithful on any host buffer size.
The remaining gap is non-OSC: **UGen breadth**.

**Server / top-level** (10/10)

- [x] /notify - server-owned (plyphon-cli): per-connection subscription to node notifications
- [x] /status - engine query: real ugen/synth/group/synthdef counts (avg/peak CPU reported as `0.0`)
- [x] /quit - server-owned (plyphon-cli)
- [x] /cmd - routes a plugin command to an app-provided `CommandHost` (plyphon ships none, so this is a seam for embedders); the host owns any reply, as scsynth's `PlugIn_DoCmd` does
- [x] /dumpOSC - server-owned (plyphon-cli)
- [x] /clearSched - engine front-end (clears the World scheduler)
- [x] /sync - engine query: a command-stream barrier answered with `/synced`
- [x] /error - engine front-end: permanent (`0`/`1`) and bundle-local (`-1`/`-2`) modes gate `/fail`
- [x] /version - server-owned (plyphon-cli)
- [x] /rtMemoryStatus - engine query: rt-pool free/largest-chunk bytes

**SynthDef** (5/5)

- [x] /d_recv
- [x] /d_load - loads a SynthDef file through an app-provided `DefSource`, registers each def, replies `/done /d_load` (plyphon reads one file; scsynth globs the path)
- [x] /d_loadDir - loads every def file under a directory through the `DefSource`
- [x] /d_free
- [x] /d_freeAll

**Synth** (4/4)

- [x] /s_new - all five add actions, including `addReplace` (4): the new synth takes the target
  node's exact slot and the target (with its subtree) is freed, the replaced node's `/n_end` firing
  before the new node's `/n_go` (scsynth's `Node_Replace`)
- [x] /s_get - getter; reply echoes the as-given control token (index or name)
- [x] /s_getn
- [x] /s_noid - partial: detaches control-name resolution; the node keeps running and stays reachable
  by control index (plyphon does not reassign a hidden negative id)

**Node** (14/15)

- [x] /n_set
- [x] /n_free
- [x] /n_map
- [x] /n_mapn
- [x] /n_before
- [x] /n_after
- [x] /n_order
- [x] /n_setn
- [x] /n_fill
- [x] /n_run
- [x] /n_query - getter; one `/n_info` per node (parent/prev/next/isGroup, head/tail for a group)
- [x] /n_trace - dumps each calc unit's inputs/outputs (first sample, scsynth's `ZIN0`/`ZOUT0`) for one block, streamed over the reply ring to a host text sink (headless, no OSC reply); unit names resolved control-side by calc index
- [x] /n_mapa - maps an `AudioControl` parameter to an audio bus (its audio wire takes the bus each block); a no-op on a control-rate param
- [x] /n_mapan - the range form of `/n_mapa`
- [ ] /n_cmd - not applicable: unimplemented in scsynth itself (commented out in its command table), so plyphon omits it

**Group** (8/8)

- [x] /g_new - all five add actions, including `addReplace` (4): a fresh empty group takes the target
  node's slot and the target subtree is freed; the root group cannot be replaced (a no-op, as scsynth
  guards with `kSCErr_ReplaceRootGroup`)
- [x] /g_head
- [x] /g_tail
- [x] /g_freeAll
- [x] /g_deepFree
- [x] /p_new - emulated by an ordinary group, as scsynth does
- [x] /g_dumpTree - getter; formats an indented tree to an optional host text sink (no OSC reply)
- [x] /g_queryTree - getter; pre-order tree stream with optional control values

**Unit** (1/1)

- [x] /u_cmd - routes a unit command (node id + unit index + name) to the `CommandHost`, mirroring scsynth's `Unit_DoCmd`

**Control bus** (5/5)

- [x] /c_set
- [x] /c_setn
- [x] /c_fill
- [x] /c_get - getter
- [x] /c_getn - getter

**Buffer** (17/17)

- [x] /b_alloc
- [x] /b_allocRead
- [x] /b_read - `leaveOpen=0` reads a file region into the already-allocated buffer at `bufStartFrame`, keeping its dimensions (scsynth's `BufReadCmd` - a region splice via a `WriteBufferRegion` command, not a whole-buffer replace; the file's channel count must match). `leaveOpen=1` keeps the file open and streams it off disk into a `DiskIn` (a `CueStream` slot fed each `run_pending` tick from a host `BufferStream`; the CLI streams a WAV from disk), ended by `/b_close`
- [x] /b_write - `leaveOpen=0` writes a whole-buffer snapshot (the engine streams the buffer's samples out race-free to an app-provided [`BufferSink`] - no shared buffer memory - driven across `run_pending` ticks); `leaveOpen=1` installs a `DiskOut` recording slot and leaves the sink open for streaming. Replies `/done /b_write <bufnum>`. Partial `numFrames`/`startFrame` ranges are deferred; header/sample formats are the sink's choice (the path)
- [x] /b_close - closes a `leaveOpen=1` stream: the engine flushes `DiskOut`'s final partial chunk (mirroring scsynth's `DiskOut_Dtor`, so every frame is written) and frees the recording slot, then the sink is closed and `/done /b_close <bufnum>` replied. The slot is freed afterwards (it holds no flat data, unlike scsynth's `SndBuf`)
- [x] /b_free
- [x] /b_zero
- [x] /b_query
- [x] /b_set
- [x] /b_setn
- [x] /b_fill
- [x] /b_gen - partial: sine1/2/3, cheby (+ normalize, + `wavetable` packing for `Osc`/`COsc`/`VOsc`),
  copy; mono only; always generates fresh (no accumulate-onto-existing)
- [x] /b_get - getter
- [x] /b_getn - getter
- [x] /b_allocReadChannel - reads only the selected file channels into a fresh buffer (control-side `CopyChannels` deinterleave; out-of-range channel reads as silence)
- [x] /b_readChannel - the channel-subset form of `/b_read`: deinterleaves the selected channels (to a width that must match the buffer) and splices them into the existing buffer at `bufStartFrame`. `leaveOpen=1` streams only the selected channels off disk into a `DiskIn`, via a `ChannelSelectStream` wrapper that deinterleaves the selection per chunk (the streaming analogue of the in-memory splice)
- [x] /b_setSampleRate

## Replies, notifications & done actions

- [x] Replies: /done, /fail, /b_info, and node notifications /n_go /n_end /n_off /n_on - each
  carries the full `/n_info`-shaped position (node, parent, prev, next, isGroup, plus head/tail for a
  group), captured at the moment of the event as scsynth's `Node_StateMsg` does (for `/n_end`, before
  the node leaves the tree, so a deep free reports each descendant with an already-removed predecessor
  reading back as `-1`)
- [x] Getter replies: /status.reply, /synced (the `/sync` barrier), /rtMemoryStatus.reply, /n_info
  (`/n_query`), /g_queryTree.reply, /c_set·/c_setn (`/c_get`·`/c_getn`), /n_set·/n_setn
  (`/s_get`·`/s_getn`), /b_set·/b_setn (`/b_get`·`/b_getn`)
- [x] /tr (SendTrig, over a dedicated best-effort trigger ring) and SendReply's custom `/<path>
  [nodeID, replyID, values...]` (over a parallel best-effort node-message ring; path + values are
  carried inline in a bounded `Copy` carrier, no audio-thread allocation) and /n_move from
  out-of-band node moves (broadcast in `/n_info`'s format when a move command relinks the tree)
- [x] Done actions beyond 0 (none), 1 (pause self), 2 (free self): codes 3-14, the free/pause variants that also touch neighbours or the enclosing group
- [x] Per-unit done flag (scsynth's `mDone`, kept in the RT-pool block): producers (`EnvGen`/`Line`/`PlayBuf`) mark completion independently of the done action, so the done-watching units (`Done`/`FreeSelfWhenDone`/`PauseSelfWhenDone`) can observe a source unit finishing

## SynthDefs & buffers

- [x] SCgf binary SynthDefs load via `/d_recv` (and the [`scgf`](crates/scgf) crate also encodes them); named parameters are folded from SC's `Control` UGens
- [x] Control family beyond plain `Control`: `AudioControl` (an audio-rate parameter, lifted to an audio wire each block and mappable with `/n_mapa`), `TrigControl` (a `/n_set` is seen for one block then resets to 0), and `LagControl` (a control-rate one-pole de-zipper, lag times from the folded UGen's inputs)
- [x] Buffers: allocate, free, zero, query, `b_gen` (sine/cheby/copy) fills, `b_get`/`b_set` element access, and asynchronous loading through an app-provided [`BufferSource`](crates/plyphon-buffers) (the I/O seam), plus chunk-streaming playback with `DiskIn` and chunk-streaming recording with `DiskOut` (drained off the audio thread through the mirror [`BufferSink`] write seam)
- [x] `/b_write` to disk through an app-provided [`BufferSink`]: the engine streams a buffer's samples out race-free (it shares no buffer memory with the host, unlike scsynth reading `SndBuf::data` from its NRT thread), driven across `run_pending` ticks; the CLI writes a float WAV. `leaveOpen=0` snapshots the whole buffer; `leaveOpen=1` leaves the sink open for `DiskOut` streaming, ended by `/b_close`. Channel-subset reads (`/b_allocReadChannel`/`/b_readChannel`) and `/n_trace` (a per-unit I/O dump to a text sink) complete the OSC surface. `b_gen` wavetable fills are now packed into scsynth's `(a, b)` format and consumed by `Osc`; partial frame ranges remain

[scsynth]: https://github.com/supercollider/supercollider
