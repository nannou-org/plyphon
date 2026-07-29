# SC3 Processing Oracle Pack

This fixture pack captures every Spec 100 family against:

- SuperCollider 3.14.1, commit `426edf6d8742e1cc3bd85b51ca0c4e595d37a903`;
- sc3-plugins commit `66047341f83e25cbaf3b106f35bd1174a3bbee7c`;
- 48 kHz sample rate and a 64-sample control block.

`manifest.json` is the source of truth for every capture command, loaded plugin
hash, vector layout, control schedule, and retained file hash. It contains the
five oracle-only families plus compatible-source vectors for `EnvDetect`,
`DFM1` with `noiselevel=0`, audio- and control-rate `MoogLadder`, `MoogVCF`,
all four `BlitB3` variants, deterministic and seeded `DNoiseRing` schedules,
`PV_MagSmooth`, and two-chain `PV_Morph`. Vectors are raw little-endian
IEEE-754 `f32` values in sample-frame-major, channel-interleaved order.

DFM1's source forces a private minimum noise floor even when `noiselevel=0`;
the capture is therefore not described as noise-free. Its two type outputs
read the same checked-in, changing binary-rational input with a strong positive
DC bias and use moderate changing frequency, resonance, and input gain. Three
independently seeded stability captures and the final canonical capture are
retained. `verify.py` compares every pair with the unchanged DSP tolerance and
requires all four hashes to differ.

The control-rate `MoogLadder` capture drives both lanes from scheduled named
control inputs, avoiding native control-oscillator compatibility as a second
test surface. `K2A.ar` transports each result to the audio capture. The
manifest labels that transport explicitly, and differential replay compares
at the final K2A sample of every live block after decoding the current
control-rate result as
`block_start + (block_end - block_start) * 64 / 63`. The decode removes only
K2A's known 63-of-64 interpolation residue. Those nine declared blocks retain
constructor, state evolution, and all changing source/cutoff/resonance events
without treating the full interpolated audio vector as raw control-rate output.
Every scheduled cutoff stays within the 375 Hz Nyquist limit of the 750 Hz
control rate, so the oracle exercises the valid source-parity domain rather
than Plyphon's explicit out-of-range safety clamp.

The capture script builds the required sc3-plugins modules and every required
SuperCollider core plugin from the two pinned source checkouts into separate
temporary build directories. It copies the pinned SuperCollider class library
into a temporary class path, starts `sclang` with default class paths disabled,
and passes only the temporary plugin directory to `scsynth`. No installed
user/system plugin binary or class is loaded. The installed `sclang` and
`scsynth` executables are used only after their pinned-commit versions and
binary hashes are recorded.

The three PV captures use FFT size 128, hop 1, and block size 64 so every ready
callback is separated by a negative-token callback. Their first channels retain
IFFT audio, K2A-interpolated chain-token/ready/control lanes, the audio-rate
positive-edge pulse, and all decoded magnitude/phase pairs. The manifest keeps
the full positive-edge pulse set distinct from the exact negative-token and
ready callback sample sets. The constructor pulse is sample 0; subsequent
positive edges precede their ready callback and decoded snapshot by 63 samples,
while negative-token and ready callbacks alternate every 64 samples.
`verify.py` requires an exact 1:1 `decoded[i]` to `ready_callback[i]` mapping,
the complete control schedule including its tail hold, at least four distinct
decoded spectra, and four corresponding distinct non-zero IFFT windows.
The morph capture has a compact companion vector containing the pinned
ordinary-bin phases for source A and source B. This preserves both source
phase branches so the raw interpolation regression can use decoded reference
evidence rather than reconstructing phase through the implementation under
test.

All three PV captures read the committed `pv_source_a.f32` and
`pv_source_b.f32` buffers directly into NRT buffers; no native oscillator
output is used as FFT input. Each 128-sample FFT window contains one positive,
exactly representable binary-rational impulse. Source A uses offsets
`[64,63,62,65,61,66,60,67,59,68,58,69]`; source B uses
`[57,70,56,71,55,72,54,73,53,74,52,75]`. These near-center positions retain
high analysis-window gain while making phase change from window to window and
keeping the two inputs distinct at every ordinary bin. Amplitudes are at most
one half, change by window, and differ between the two source buffers. This
keeps every spectral bin non-zero and yields changing spectra while keeping
IFFT residue below the unchanged oracle tolerance.

Decoded ordinary-bin phases are polar coordinates: `+pi` and `-pi` name the
same direction. Their error is therefore the shortest angular distance
`abs(atan2(sin(actual - reference), cos(actual - reference)))`, evaluated
against `max(0.0016, 1e-4 + 1e-3 * abs(reference))` for the three units that
use SuperCollider's approximate polar grid. The 0.0016-radian floor covers two
adjacent 1/1024-slope lookup cells after stateful phase accumulation across
different FFT backends. Magnitudes, the explicit zero-phase DC/Nyquist endpoint
fields, and IFFT samples use direct absolute error with the unchanged
`1e-4 + 1e-3 * abs(reference)` tolerance. The manifest pins the permitted
SuperCollider core helper path and content hash.
The demand capture retains a same-seed three-draw baseline and proves the
subsequent probe position after one, two, and scheduled source draws. Those
draw-order lanes reseed at each pull so each position is directly observable;
the separate seeded stochastic lane seeds once and retains its transitions.

Run:

```sh
./capture.sh
python3 verify.py
```

Use `SPEC100_CAPTURE_SCOPE=pv ./capture.sh` to regenerate only the deterministic
PV source buffers, the three PV captures, and the morph source-phase companion
while still rebuilding and recording the complete pinned plugin set.

The default source locations are the pinned review checkouts used for the
original capture. Override `SC3_PLUGINS_SOURCE`, `SUPERCOLLIDER_SOURCE`,
`SCLANG`, or `SCSYNTH` to point at equivalent exact revisions.

Read `CLEANROOM.md` before using the pack to implement or review an
oracle-only unit.
