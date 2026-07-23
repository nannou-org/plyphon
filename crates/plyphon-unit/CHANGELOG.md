# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/nannou-org/plyphon/compare/plyphon-unit-v0.1.1...plyphon-unit-v0.2.0) - 2026-07-23

### Added

- *(unit)* PV_Diffuser, Gendy1, IEnvGen, TDuty, and a Duty input-order fix
- *(unit)* graph-owned local buffers (LocalBuf, MaxLocalBufs, ClearBuf, SetBuf)
- *(unit)* BEQSuite biquad filters (BLowPass, BHiPass, BPeakEQ, BLowShelf, BHiShelf, BBandPass)
- *(unit)* shared graph random stream with Rand-family, RandSeed, and random operators

### Fixed

- *(unit)* guard short frames in the two-buffer spectral ops
- *(unit)* output zero from SetBuf and ClearBuf like scsynth
- *(unit)* pass asInteger through unchanged like scsynth
- *(unit)* use scsynth's bipolar draw for audio-rate rrand
- *(unit)* clamp IEnvGen's stage count to its actual inputs
- *(unit)* reject a non-constant Gendy1 initCPs input
- *(unit)* fire Duty's doneAction when the level stream ends
- *(unit)* freeze Duty and TDuty when the duration stream ends
- *(unit)* pass differently-sized PV_Diffuser frames through untouched
- *(unit)* scale PV_Diffuser's shifted-bin count by its trig input

### Other

- *(unit)* correct the BEQ, IEnvGen and rand divergence notes
- *(unit)* drop now-redundant Rng link targets
- *(unit)* document TrigRand::new
- *(unit)* fix private intra-doc link in fft module docs

## [0.1.1](https://github.com/nannou-org/plyphon/compare/plyphon-unit-v0.1.0...plyphon-unit-v0.1.1) - 2026-07-02

### Other

- Expose host registry and precompile APIs
