# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/nannou-org/plyphon/compare/plyphon-rt-v0.1.1...plyphon-rt-v0.2.0) - 2026-09-26

### Added

- *(unit)* size LocalBuf storage from its inputs when the synth starts
- *(unit)* allocate input-sized unit memory when the synth starts
- *(unit)* graph-owned local buffers (LocalBuf, MaxLocalBufs, ClearBuf, SetBuf)
- *(unit)* shared graph random stream with Rand-family, RandSeed, and random operators

### Fixed

- *(rt)* decollide the graph random stream's seed from the unit ladder
- *(rt)* reject duplicate node ids and surface tree-add failures
- *(rt)* seed lag param state on the first tick, not at build

### Other

- Convert spectra with scsynth's lookup tables
- Mark control buses set by /c_set as written, as scsynth does
- Draw every random unit from the synth's stream, as scsynth does
- Draw from the World's random streams, as scsynth does
- Construct every unit before the first calc, as Graph_FirstCalc does
- Size FFT, IFFT and PV_Diffuser from the chain buffer

## [0.1.1](https://github.com/nannou-org/plyphon/compare/plyphon-rt-v0.1.0...plyphon-rt-v0.1.1) - 2026-07-02

### Other

- updated the following local packages: plyphon-unit
