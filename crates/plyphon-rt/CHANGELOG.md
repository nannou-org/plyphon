# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2](https://github.com/nannou-org/plyphon/compare/plyphon-rt-v0.1.1...plyphon-rt-v0.1.2) - 2026-07-23

### Added

- *(unit)* graph-owned local buffers (LocalBuf, MaxLocalBufs, ClearBuf, SetBuf)
- *(unit)* shared graph random stream with Rand-family, RandSeed, and random operators

### Fixed

- *(rt)* decollide the graph random stream's seed from the unit ladder
- *(rt)* reject duplicate node ids and surface tree-add failures
- *(rt)* seed lag param state on the first tick, not at build

## [0.1.1](https://github.com/nannou-org/plyphon/compare/plyphon-rt-v0.1.0...plyphon-rt-v0.1.1) - 2026-07-02

### Other

- updated the following local packages: plyphon-unit
