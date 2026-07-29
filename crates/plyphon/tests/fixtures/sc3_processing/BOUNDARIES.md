# SC3 Processing Boundary Oracle Extension

This boundary-only extension retains source-exact observations that are
deliberately separate from the original `manifest.json` capture. It uses the
same pinned SuperCollider and sc3-plugins revisions and the same temporary
class/plugin isolation. Read `CLEANROOM.md` before using any vector.

Run:

```sh
./capture_boundaries.sh
python3 verify_boundaries.py
```

`boundary_manifest.json` records the exact binaries, commands, scripts, vector
shapes, hashes, and unsafe omissions. `verify_boundaries.py` checks behavior,
not just shape: Decimator cadence and non-finite controls; isolated BMoog
cutoff/Q boundaries and safe non-finite controls; Perlin3 negative-cell and
256-cell periodicity; RosslerL low-frequency cadence, finite destabilization,
and every non-finite control; PV_Freeze early-freeze/non-finite behavior; and
black-box positive/negative-zero pairs for every input of the oracle-only
Decimator, BMoog, Perlin3, RosslerL, and PV_Freeze units. Reciprocal witness
lanes prove that the two IEEE-754 zero signs reached the server distinctly.

The live PV_Freeze 128-to-256-to-128 size-change probe is not retained as a
vector or run automatically. An isolated NRT black-box attempt terminated the
pinned scsynth before it produced a render, so no source behavior from that
case is safe to canonicalize. BMoog non-finite cutoff is also omitted because
it can reach an undefined non-finite index or table access. Required finite
out-of-range BMoog cutoffs run in separate NRT processes.
