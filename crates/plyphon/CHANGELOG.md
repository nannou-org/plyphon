# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/nannou-org/plyphon/compare/plyphon-v0.1.2...plyphon-v0.2.0) - 2026-07-23

### Added

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

## [0.1.2](https://github.com/nannou-org/plyphon/compare/plyphon-v0.1.1...plyphon-v0.1.2) - 2026-07-07

### Other

- Merge pull request #9 from nannou-org/controller-try-send-batch
- Add Controller::try_send_batch for all-or-none command submission

## [0.1.1](https://github.com/nannou-org/plyphon/compare/plyphon-v0.1.0...plyphon-v0.1.1) - 2026-07-02

### Other

- Expose host registry and precompile APIs
