# SC3 Processing Clean-Room Boundary

This directory is the machine-generated conformance pack for RFC 0005 Phase 6
Spec 100. It contains black-box vectors for the oracle-only units and captured
reference vectors for the compatible-source translated units. Implementers and
clean-room reviewers may use the checked-in `.scd` inputs, binary vectors,
manifest, public class/help contracts, and the family-specific public
materials listed below.

The capture operator was allowed to inspect the pinned server sources. Those
server sources are not copied, summarized, or transcribed into this pack.

The compatible-source extension captured `EnvDetect`, `DFM1`, `MoogLadder`,
`MoogVCF`, `BlitB3*`, `DNoiseRing`, `PV_MagSmooth`, and `PV_Morph`. That
extension inspected only their spec-permitted class/server material and build
metadata. It loaded the pre-existing oracle-only plugin binaries during
capture but did not inspect, transcribe, summarize, decompile, or receive
details from any prohibited server file listed below.

## Permitted implementation materials

| Family | Permitted material |
| --- | --- |
| `Decimator` | Pinned class/help contract, standard sample-and-hold and PCM quantization literature, and this black-box pack |
| `BMoog` | Pinned class/help contract, DFL's “Stilson's Moog filter code” functional equations/data at <https://www.musicdsp.org/en/latest/Filters/145-stilson-s-moog-filter-code.html> (site revision `43f15628`), and this black-box pack |
| `Perlin3` | Pinned class contract, Ken Perlin's published improved-noise equations/permutation, and this black-box pack |
| `RosslerL` | Pinned class-file Rössler equations/defaults, standard fourth-order Runge–Kutta and linear interpolation references, and this black-box pack |
| `PV_Freeze` | Pinned class/help behavior, existing Plyphon spectral primitives, and this black-box pack |

## Prohibited server files

Clean-room implementers and reviewers must not inspect, transcribe, summarize,
or receive implementation details from:

- `source/DistortionUGens/DistortionUGens.cpp` for `Decimator`;
- `source/BlackrainUGens/BlackrainUGens.cpp` for `BMoog`;
- `source/MCLDUGens/MCLDChaosUGens.cpp` for `Perlin3` or `RosslerL`;
- `source/JoshUGens/JoshPVUGens.cpp` for `PV_Freeze`.

Do not use derived notes, generated decompilations, or another person's
description of those prohibited implementations as a substitute.

## Clean-room implementer attestation

- Implementer:
- Date:
- Plyphon revision:
- Families implemented:
- Permitted inputs used:
- I attest that I did not inspect, transcribe, summarize, or receive details
  from the prohibited server files listed above:

## Independent reviewer attestation

- Reviewer:
- Date:
- Reviewed Plyphon revision:
- Families reviewed:
- Permitted inputs used:
- I compared the implementation and tests only against the permitted
  materials and this conformance pack:
- I found no evidence that prohibited server implementation material entered
  the clean-room implementation:

## Implementer attestation — 2026-07-28

- Implementer: clean-room implementation contributor
- Date: 2026-07-28
- Plyphon revision: `86b5d040592ace4dfbb6a4b3c2b2cd05a629b669`
- Families implemented: `Decimator`, `BMoog`, `Perlin3`, `RosslerL`, `PV_Freeze`
- Permitted inputs used: the reviewed Spec 100 contract; pinned class/help interfaces; this
  verified black-box conformance pack; the public BMoog functional table named above; Ken
  Perlin's published improved-noise equations and permutation; standard Rössler equations,
  fourth-order Runge–Kutta, and linear interpolation; existing Plyphon unit, buffer, and
  phase-vocoder primitives
- I attest that I did not inspect, transcribe, summarize, or receive details from the prohibited
  server files listed above.

## Independent reviewer attestation — 2026-07-28

- Reviewer: independent clean-room reviewer
- Date: 2026-07-28
- Reviewed Plyphon revision: `86b5d040592ace4dfbb6a4b3c2b2cd05a629b669`
- Families reviewed: `Decimator`, `BMoog`, `Perlin3`, `RosslerL`, `PV_Freeze`
- Permitted inputs used: the reviewed Spec 100 contract; the verified black-box conformance pack
  and its manifest, capture scripts, verifier, and retained vectors; the public BMoog functional
  table retained in the pack; Ken Perlin's published permutation and improved-noise equations as
  represented by the permitted contract; standard Rössler equations, fourth-order Runge–Kutta,
  and linear interpolation; existing Plyphon unit, buffer, and phase-vocoder primitives
- I compared the implementation and tests only against the permitted materials and this
  conformance pack.
- I found no evidence that prohibited server implementation material entered the clean-room
  implementation.

## Independent delta-review attestation — 2026-07-29

- Reviewers: fresh independent clean-room delta reviewers
- Date: 2026-07-29
- Reviewed Plyphon range:
  `86b5d040592ace4dfbb6a4b3c2b2cd05a629b669..65fb4e2e9eb796945d571f8e22b34ce9290e8d99`
- Families reviewed: `Decimator`, `BMoog`, `Perlin3`, `RosslerL`, `PV_Freeze`
- Permitted inputs used: the source-exact Spec 100 contract; the verified primary and boundary
  black-box conformance packs and their manifests, capture scripts, verifiers, and retained
  vectors; the public BMoog functional table retained in the pack; Ken Perlin's published
  permutation and improved-noise equations; standard Rössler equations, fourth-order Runge–Kutta,
  and linear interpolation; existing Plyphon unit, buffer, and phase-vocoder primitives
- The reviewers compared the aggregate implementation and tests only against the permitted
  materials above. Direct-source families were reviewed separately against their permitted pinned
  sources.
- The reviewers did not inspect, transcribe, summarize, or receive implementation details from any
  prohibited server file listed above and found no evidence that such material entered the
  clean-room implementation.
