# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/nannou-org/plyphon/compare/plyphon-dsp-v0.1.0...plyphon-dsp-v0.2.0) - 2026-09-26

### Added

- *(unit)* graph-owned local buffers (LocalBuf, MaxLocalBufs, ClearBuf, SetBuf)

### Fixed

- *(dsp)* clamp cycle lookup at float boundary

### Other

- Convert spectra with scsynth's lookup tables
- Mark control buses set by /c_set as written, as scsynth does
- Port XOut exactly
- Copy on the first write to an untouched audio bus, as scsynth's Out does
- Port InTrig and LagIn
- Support the random operators at calc and demand rate
- Draw every random unit from the synth's stream, as scsynth does
- Merge pull request #39 from nannou-org/fft-full-range
- Seed and draw the RNG as scsynth's RGen does
