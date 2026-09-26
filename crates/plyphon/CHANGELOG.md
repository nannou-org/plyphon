# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/nannou-org/plyphon/compare/plyphon-v0.1.2...plyphon-v0.2.0) - 2026-09-26

### Added

- *(unit)* size LocalBuf storage from its inputs when the synth starts
- *(unit)* allocate input-sized unit memory when the synth starts
- *(controller)* reuse a freed def's id
- *(unit)* PV_Diffuser, Gendy1, IEnvGen, TDuty, and a Duty input-order fix
- *(unit)* graph-owned local buffers (LocalBuf, MaxLocalBufs, ClearBuf, SetBuf)
- *(unit)* BEQSuite biquad filters (BLowPass, BHiPass, BPeakEQ, BLowShelf, BHiShelf, BBandPass)
- *(unit)* shared graph random stream with Rand-family, RandSeed, and random operators
- *(controller)* batchable synth creation for /s_new-with-controls

### Fixed

- *(unit)* guard short frames in the two-buffer spectral ops
- *(unit)* output zero from SetBuf and ClearBuf like scsynth
- *(unit)* pass asInteger through unchanged like scsynth
- *(unit)* use scsynth's bipolar draw for audio-rate rrand
- *(rt)* decollide the graph random stream's seed from the unit ladder
- *(unit)* clamp IEnvGen's stage count to its actual inputs
- *(unit)* reject a non-constant Gendy1 initCPs input
- *(unit)* fire Duty's doneAction when the level stream ends
- *(unit)* freeze Duty and TDuty when the duration stream ends
- *(unit)* scale PV_Diffuser's shifted-bin count by its trig input
- *(controller)* stop automatic node-id allocation at i32::MAX
- *(rt)* reject duplicate node ids and surface tree-add failures
- *(rt)* seed lag param state on the first tick, not at build

### Other

- Pull RandSeed's seed through the demand protocol
- Read Duty and TDuty's reset by its rate, as scsynth does
- Port PV_JensenAndersen, PV_HainsworthFoote and RunningSum
- Port PV_MagNoise, PV_RandComb, PV_RandWipe and PV_BinScramble
- Port FFTTrigger
- Port PV_MagFreeze, PV_MagShift, PV_PhaseShift, PV_MagDiv, PV_BinWipe, PV_RectComb2 and PV_ConformalMap
- Port Onsets
- Port MFCC
- Port Loudness
- Port SpecCentroid, SpecFlatness and SpecPcile
- Merge pull request #53 from nannou-org/port-misc-units
- Merge pull request #52 from nannou-org/port-convolution-family
- Port BeatTrack
- Port BeatTrack2
- Port KeyTrack
- Match scsynth's two-buffer PV preamble
- Conjugate PV_Conj's bins by subtracting from zero, as scsynth does
- Convert spectra with scsynth's lookup tables
- Pin the sin-based chaos renders per platform
- Compute the sample-and-hold hold length as scsynth does
- Take LinCongN's modulo through scsynth's sc_mod
- Wrap StandardN with scsynth's mod2pi
- Re-seed CuspN, QuadN, LatoocarfianN and StandardN on an init input change
- Pin the earlier chaos ports against scsynth
- Port the GbmanL and QuadC chaos generators
- Port the LinCongL and LinCongC chaos generators
- Port the LatoocarfianL and LatoocarfianC chaos generators
- Port the HenonN and HenonC chaos generators
- Port the FBSine chaos generators
- Merge pull request #47 from nannou-org/port-small-units
- Port the BEQSuite's calcs exactly
- Mark control buses set by /c_set as written, as scsynth does
- Port XOut exactly
- Copy on the first write to an untouched audio bus, as scsynth's Out does
- Port PSinGrain
- Port IndexInBetween and DetectIndex
- Port SendPeakRMS
- Port InTrig and LagIn
- Port BlockSize and NodeID
- Port Vibrato
- Port Linen
- Port Flip
- Port BAllPass and BBandStop
- Merge pull request #46 from nannou-org/port-demand-units
- Port DemandEnvGen
- Port Dwrand
- Port Dswitch1 and Dswitch
- Port Dconst and Dreset
- Port Ddup and Dstutter
- Support the random operators at calc and demand rate
- Draw every random unit from the synth's stream, as scsynth does
- Draw from the World's random streams, as scsynth does
- Construct every unit before the first calc, as Graph_FirstCalc does
- Move Convolution's out-of-range cases to the full FFT range
- Merge pull request #39 from nannou-org/fft-full-range
- Merge pull request #37 from nannou-org/convolution
- Draw with scsynth's formulas at every existing RNG call site
- Run BinaryOpUGen and UnaryOpUGen at demand rate
- Port PackFFT and Unpack1FFT
- Port PV_BinShift, PV_MagSmear and PV_RectComb
- Port the interpolating chaos generators
- LocalOut writes nothing when its width differs from LocalIn
- Size FFT, IFFT and PV_Diffuser from the chain buffer
- Remove the SynthDef init specializer
- *(plyphon)* measure running, creating and freeing synths
- Merge pull request #25 from nannou-org/free-def-ids

## [0.1.2](https://github.com/nannou-org/plyphon/compare/plyphon-v0.1.1...plyphon-v0.1.2) - 2026-07-07

### Other

- Merge pull request #9 from nannou-org/controller-try-send-batch
- Add Controller::try_send_batch for all-or-none command submission

## [0.1.1](https://github.com/nannou-org/plyphon/compare/plyphon-v0.1.0...plyphon-v0.1.1) - 2026-07-02

### Other

- Expose host registry and precompile APIs
