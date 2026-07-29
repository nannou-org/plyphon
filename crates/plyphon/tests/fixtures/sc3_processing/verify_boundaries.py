#!/usr/bin/env python3
"""Verify provenance and exact semantics in the clean-room boundary pack."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import struct
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent
EXPECTED_SC = "426edf6d8742e1cc3bd85b51ca0c4e595d37a903"
EXPECTED_SC3 = "66047341f83e25cbaf3b106f35bd1174a3bbee7c"
REQUIRED_VECTORS = {
    "decimator_boundaries",
    "decimator_controls",
    *{f"bmoog_boundaries_{index}" for index in range(9)},
    "bmoog_controls",
    "perlin3_boundaries",
    "rossler_l_boundaries",
    "rossler_l_controls",
    "pv_freeze_early",
    "pv_freeze_controls",
    "signed_zero_boundaries",
}
RENDERED_FRAMES = 448


def sha256(path: pathlib.Path) -> str:
    """Return one file's lowercase SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f32_bits(value: float) -> bytes:
    """Return the exact retained representation of one value."""
    return struct.pack("<f", value)


def read_rows(vector: dict[str, Any]) -> list[tuple[float, ...]]:
    """Read one manifest vector into frame-major rows."""
    values = [
        value[0]
        for value in struct.iter_unpack("<f", (ROOT / vector["path"]).read_bytes())
    ]
    channels = vector["channels"]
    return [
        tuple(values[offset : offset + channels])
        for offset in range(0, len(values), channels)
    ]


def require(condition: bool, message: str, failures: list[str]) -> None:
    """Append one failure when a semantic condition is false."""
    if not condition:
        failures.append(message)


def exact_columns(
    rows: list[tuple[float, ...]],
    left: int,
    right: int,
    frames: range,
) -> bool:
    """Return whether two columns are bit-identical over selected frames."""
    return all(f32_bits(rows[frame][left]) == f32_bits(rows[frame][right]) for frame in frames)


def verify_decimator(
    vectors: dict[str, dict[str, Any]],
    failures: list[str],
    observations: list[str],
) -> None:
    """Pin finite and non-finite Decimator rate/bit behavior."""
    rows = read_rows(vectors["decimator_boundaries"])
    live = rows[:RENDERED_FRAMES]
    require(
        all(math.isfinite(value) for row in live for value in row),
        "Decimator boundary vector must stay finite",
        failures,
    )
    require(
        all(row[1] == 0.0 and row[2] == 0.0 for row in live),
        "Decimator frames 0..447 channels 1/2: negative and zero rate must remain at the zero constructor hold",
        failures,
    )
    require(
        exact_columns(rows, 3, 4, range(RENDERED_FRAMES)),
        "Decimator frames 0..447 channels 3/4: 96000 Hz must be bit-identical to 48000 Hz",
        failures,
    )
    require(
        all(
            f32_bits(rows[frame][3]) != f32_bits(rows[frame - 1][3])
            for frame in range(1, RENDERED_FRAMES)
        ),
        "Decimator channel 3: sample-rate cadence must update on every retained frame",
        failures,
    )
    require(
        max(abs(row[0] - row[3]) for row in live) <= 2.0e-7,
        "Decimator channels 0/3: sample-rate output must follow every ramp sample within quantizer residue",
        failures,
    )
    observations.append(
        "Decimator frames 0..447: rate -12000 and 0 both hold constructor zero; "
        "rate 96000 is bit-identical to rate 48000 and updates every sample."
    )

    controls = read_rows(vectors["decimator_controls"])
    nan_frames = [
        frame for frame in range(RENDERED_FRAMES) if math.isnan(controls[frame][3])
    ]
    require(
        nan_frames == list(range(66, 194)),
        "Decimator bits=NaN channel 3 must emit NaN exactly at frames 66..193",
        failures,
    )
    require(
        exact_columns(controls, 0, 2, range(RENDERED_FRAMES)),
        "Decimator rate NaN and -infinity channels must be bit-identical",
        failures,
    )
    require(
        exact_columns(controls, 0, 6, range(193))
        and not exact_columns(controls, 0, 6, range(193, RENDERED_FRAMES)),
        "Decimator rate NaN/-infinity must match rate zero through frame 192 but remain cadence-poisoned after finite recovery",
        failures,
    )
    require(
        exact_columns(controls, 1, 7, range(RENDERED_FRAMES)),
        "Decimator rate +infinity must match sample-rate reference through invalid and recovery schedules",
        failures,
    )
    require(
        exact_columns(controls, 4, 5, range(RENDERED_FRAMES))
        and exact_columns(controls, 4, 8, range(RENDERED_FRAMES)),
        "Decimator bits +/-infinity must match bits=31 pass-through",
        failures,
    )
    require(
        exact_columns(controls, 3, 8, range(194, RENDERED_FRAMES)),
        "Decimator bits=NaN must recover to the finite bits=4 schedule at frame 194",
        failures,
    )
    observations.append(
        "Decimator controls: rate NaN/-inf behave as rate 0 until recovery but poison "
        "cadence thereafter; rate +inf matches sample rate; bits NaN emits NaN at "
        "frames 66..193 then recovers at 194; bits +/-inf match bits=31."
    )


def verify_bmoog(
    vectors: dict[str, dict[str, Any]],
    failures: list[str],
    observations: list[str],
) -> None:
    """Pin out-of-range and non-finite BMoog behavior."""
    cases = [read_rows(vectors[f"bmoog_boundaries_{index}"]) for index in range(9)]
    for index, rows in enumerate(cases):
        live = rows[:RENDERED_FRAMES]
        require(
            all(math.isfinite(value) for row in live for value in row),
            f"BMoog isolated case {index} must stay finite through frame 447",
            failures,
        )
        require(
            exact_columns(rows, 0, 3, range(RENDERED_FRAMES)),
            f"BMoog isolated case {index}: mode 3 must select the same low-pass output as mode 0",
            failures,
        )

    comparisons = [
        (0, 5, "cutoff 19 versus 20"),
        (1, 5, "cutoff -100 versus 20"),
        (2, 6, "cutoff 24001 versus 24000"),
        (3, 7, "q -0.5 versus 0"),
        (4, 8, "q 1.5 versus 1"),
    ]
    for actual, boundary, label in comparisons:
        require(
            any(
                f32_bits(cases[actual][frame][0])
                != f32_bits(cases[boundary][frame][0])
                for frame in range(RENDERED_FRAMES)
            ),
            f"BMoog channel 0: {label} must remain observably distinct",
            failures,
        )
    observations.append(
        "BMoog frames 0..447: cutoff 19, -100, and 24001 and q -0.5/1.5 all "
        "remain finite, are not clamped to 20/20/24000/0/1 respectively, and mode 3 "
        "is bit-identical to mode 0. Each risky cutoff case ran in its own NRT process."
    )

    controls = read_rows(vectors["bmoog_controls"])
    for channel in range(3):
        nan_frames = [
            frame
            for frame in range(RENDERED_FRAMES)
            if math.isnan(controls[frame][channel])
        ]
        require(
            nan_frames == list(range(129, RENDERED_FRAMES)),
            f"BMoog q non-finite channel {channel} must become permanently NaN at frame 129",
            failures,
        )
    require(
        exact_columns(controls, 0, 1, range(RENDERED_FRAMES))
        and exact_columns(controls, 0, 2, range(RENDERED_FRAMES)),
        "BMoog q NaN/+infinity/-infinity lanes must be bit-identical",
        failures,
    )
    for channel in (3, 4, 5):
        require(
            exact_columns(controls, channel, 6, range(RENDERED_FRAMES)),
            f"BMoog mode non-finite channel {channel} must select mode-zero low-pass",
            failures,
        )
    observations.append(
        "BMoog controls: q NaN/+inf/-inf become NaN at frame 129 and do not recover "
        "after the finite control event; mode NaN/+inf/-inf stay finite and are "
        "bit-identical to mode 0. Non-finite cutoff was omitted as index/table UB."
    )


def verify_perlin3(
    vector: dict[str, Any],
    failures: list[str],
    observations: list[str],
) -> None:
    """Pin negative-cell boundaries and independent +256 periodicity."""
    rows = read_rows(vector)
    live = rows[:64]
    require(
        all(math.isfinite(value) for row in live for value in row),
        "Perlin3 boundary vector must stay finite",
        failures,
    )
    for channel in range(14):
        require(
            all(
                f32_bits(row[channel]) == f32_bits(live[0][channel])
                for row in live
            ),
            f"Perlin3 channel {channel} must be a stable audio-rate constant",
            failures,
        )
    for group in ((0, 3, 6), (1, 4, 7), (2, 5, 8), (9, 10, 11, 12, 13)):
        reference = f32_bits(live[0][group[0]])
        require(
            all(f32_bits(live[0][channel]) == reference for channel in group[1:]),
            f"Perlin3 +256 periodicity group {group} changed",
            failures,
        )
    require(
        len({f32_bits(live[0][channel]) for channel in (0, 1, 2)}) == 3,
        "Perlin3 values immediately below/at/above the negative lattice boundary must be distinct",
        failures,
    )
    observations.append(
        "Perlin3 frame 0: x=-256-epsilon/-256/-256+epsilon equals the matching "
        "x=-epsilon/0/+epsilon and x=256-epsilon/256/256+epsilon values exactly; "
        "adding 256 independently to x, y, z, or all three preserves the base bits."
    )


def verify_rossler(
    vectors: dict[str, dict[str, Any]],
    failures: list[str],
    observations: list[str],
) -> None:
    """Pin low-frequency cadence, destabilization, and non-finite controls."""
    rows = read_rows(vectors["rossler_l_boundaries"])
    for left, right in ((0, 3), (0, 6), (1, 4), (1, 7), (2, 5), (2, 8)):
        require(
            exact_columns(rows, left, right, range(RENDERED_FRAMES)),
            "RosslerL freq 0/-1/0.0005 lanes must be bit-identical",
            failures,
        )
    require(
        all(
            f32_bits(rows[frame][0]) == f32_bits(0.05)
            and rows[frame][1] == 0.0
            and rows[frame][2] == 0.0
            for frame in range(RENDERED_FRAMES)
        ),
        "RosslerL sub-0.001 frequencies must retain the initial scaled coordinates over 448 frames",
        failures,
    )
    require(
        all(math.isfinite(rows[frame][channel]) for frame in range(3) for channel in range(9, 12)),
        "RosslerL h=5 must emit three finite frames before destabilizing",
        failures,
    )
    require(
        all(
            math.isnan(rows[frame][channel])
            for frame in range(3, RENDERED_FRAMES)
            for channel in range(9, 12)
        ),
        "RosslerL h=5 must emit NaN on all coordinates from frame 3 without recovery",
        failures,
    )
    require(
        abs(rows[1][9] - -93.56978607177734) <= 1.0e-6
        and abs(rows[1][10] - 10.84375) <= 1.0e-6
        and abs(rows[1][11] - 5164.42529296875) <= 1.0e-6,
        "RosslerL h=5 frame 1 finite destabilization witness changed",
        failures,
    )
    observations.append(
        "RosslerL frames 0..447: freq 0, -1, and 0.0005 are bit-identical and "
        "remain (0.05,0,0); finite h=5 at 48 kHz emits frame 1 "
        "(-93.569786,10.84375,5164.4253), a finite frame 2, then all-NaN from frame 3."
    )

    controls = read_rows(vectors["rossler_l_controls"])
    high_reference = 24 * 3
    floor_reference = 25 * 3
    for kind, reference in ((0, high_reference), (1, high_reference), (2, floor_reference)):
        lane = (kind * 8) * 3
        for coordinate in range(3):
            require(
                exact_columns(
                    controls,
                    lane + coordinate,
                    reference + coordinate,
                    range(RENDERED_FRAMES),
                ),
                "RosslerL non-finite frequency branch differs from its finite reference",
                failures,
            )

    for kind in range(3):
        for control in range(1, 5):
            lane = (kind * 8 + control) * 3
            require(
                all(
                    math.isfinite(controls[frame][lane + coordinate])
                    for frame in range(71)
                    for coordinate in range(3)
                )
                and all(
                    not math.isfinite(controls[frame][lane + coordinate])
                    for frame in range(71, RENDERED_FRAMES)
                    for coordinate in range(3)
                ),
                f"RosslerL kind {kind} control {control}: a/b/c/h must become permanently non-finite at integration frame 71",
                failures,
            )
            non_finite = [
                controls[frame][lane + coordinate]
                for frame in range(71, RENDERED_FRAMES)
                for coordinate in range(3)
            ]
            negative_infinities = sum(value == -math.inf for value in non_finite)
            require(
                negative_infinities == (7 if kind == 2 and control == 3 else 0)
                and sum(math.isnan(value) for value in non_finite)
                == len(non_finite) - negative_infinities,
                f"RosslerL kind {kind} control {control}: non-finite classification changed",
                failures,
            )

    invalid_expected = (math.nan, math.inf, -math.inf)
    for kind in range(3):
        for control in range(5, 8):
            lane = (kind * 8 + control) * 3
            invalid_frames = [
                frame
                for frame in range(RENDERED_FRAMES)
                if any(
                    not math.isfinite(controls[frame][lane + coordinate])
                    for coordinate in range(3)
                )
            ]
            require(
                invalid_frames == list(range(64, 199)),
                f"RosslerL kind {kind} initial-coordinate control {control}: non-finite interval must be frames 64..198",
                failures,
            )
            value = controls[64][lane + (control - 5)]
            expected = invalid_expected[kind]
            require(
                (math.isnan(value) if math.isnan(expected) else value == expected),
                f"RosslerL kind {kind} initial-coordinate control {control}: frame 64 must expose the control's non-finite value",
                failures,
            )
    observations.append(
        "RosslerL controls: freq NaN/+inf follow the 48 kHz branch and -inf follows "
        "the 0.001 Hz floor; NaN/+inf/-inf a,b,c,h become non-finite at integration "
        "frame 71 and never recover (c=-inf includes seven -inf values, otherwise NaN); "
        "xi/yi/zi expose the invalid coordinate at frame "
        "64, are non-finite through 198, and recover finite at 199."
    )


def verify_pv_freeze(
    vectors: dict[str, dict[str, Any]],
    failures: list[str],
    observations: list[str],
) -> None:
    """Pin early-freeze staging, non-finite comparison, and live size changes."""
    early = read_rows(vectors["pv_freeze_early"])
    pulse_frames = [frame for frame, row in enumerate(early) if row[4] >= 0.5]
    require(
        pulse_frames == [0, 65, 193, 321, 449, 577],
        "PV_Freeze early capture callback pulses changed",
        failures,
    )
    decoded_frames = [128, 256, 384, 512, 640]
    for frame in decoded_frames:
        require(
            early[frame][1] >= 0.0 and early[frame][2] == 1.0 and early[frame][3] == 1.0,
            f"PV_Freeze early frame {frame}: token/ready/freeze lanes changed",
            failures,
        )
    require(
        any(
            abs(early[128][5 + 2 * bin] - early[256][5 + 2 * bin]) > 1.0e-3
            for bin in range(65)
        ),
        "PV_Freeze early stages 1/2 must pass two distinct magnitudes",
        failures,
    )
    for frame in (384, 512, 640):
        require(
            max(
                abs(early[frame][5 + 2 * bin] - early[256][5 + 2 * bin])
                for bin in range(65)
            )
            <= 1.0e-6,
            f"PV_Freeze early frame {frame}: magnitudes must remain frozen from frame 256",
            failures,
        )
    recurrence_errors = []
    for bin_index in range(1, 64):
        channel = 6 + 2 * bin_index
        delta = math.atan2(
            math.sin(early[256][channel] - early[128][channel]),
            math.cos(early[256][channel] - early[128][channel]),
        )
        previous = early[256][channel]
        for frame in (384, 512, 640):
            wanted = math.atan2(math.sin(previous + delta), math.cos(previous + delta))
            actual = early[frame][channel]
            recurrence_errors.append(
                abs(math.atan2(math.sin(actual - wanted), math.cos(actual - wanted)))
            )
            previous = actual
    require(
        max(recurrence_errors) <= 0.0032,
        "PV_Freeze early phase advance no longer repeats the first learned difference",
        failures,
    )
    observations.append(
        "PV_Freeze freeze=1 from construction: decoded frames 128 and 256 remain "
        "distinct and nonzero; frames 384/512/640 retain frame-256 magnitudes within 1e-6 "
        "while advancing the learned phase difference (max circular error 0.0032 rad)."
    )

    controls = read_rows(vectors["pv_freeze_controls"])
    for left, right in ((0, 15), (10, 15), (5, 20)):
        require(
            exact_columns(controls, left, right, range(1472)),
            "PV_Freeze non-finite control lane differs from its comparison reference",
            failures,
        )
    require(
        all(
            f32_bits(controls[frame][lane])
            == f32_bits(controls[frame][0])
            for frame in range(320)
            for lane in (5, 10, 15, 20)
        )
        and all(
            f32_bits(controls[frame][lane])
            == f32_bits(controls[frame][0])
            for frame in range(704, 1472)
            for lane in (5, 10, 15, 20)
        ),
        "PV_Freeze comparison lanes must agree before invalid control and after recovery",
        failures,
    )
    require(
        any(
            f32_bits(controls[frame][5]) != f32_bits(controls[frame][15])
            for frame in range(320, 704)
        ),
        "PV_Freeze +infinity/one branch must differ from zero during freeze",
        failures,
    )
    require(
        math.isnan(controls[384][3])
        and controls[384][8] == math.inf
        and controls[384][13] == -math.inf,
        "PV_Freeze frame 384 must expose NaN/+infinity/-infinity controls",
        failures,
    )
    observations.append(
        "PV_Freeze freeze controls: NaN and -inf are bit-identical to freeze=0; "
        "+inf is bit-identical to freeze=1; all lanes converge again at frame 704 "
        "after the finite-zero recovery."
    )

    observations.append(
        "PV_Freeze live 128->256->128 size change is omitted: an isolated NRT "
        "black-box probe terminated scsynth before producing a render, so no source "
        "behavior is safe to canonicalize."
    )


def verify_signed_zero(
    vector: dict[str, Any],
    failures: list[str],
    observations: list[str],
) -> None:
    """Pin black-box signed-zero behavior for every oracle-only unit input."""
    rows = read_rows(vector)
    live = range(RENDERED_FRAMES)
    require(
        all(rows[frame][0] == math.inf for frame in live)
        and all(rows[frame][1] == -math.inf for frame in live),
        "signed-zero reciprocal witnesses must remain +infinity and -infinity",
        failures,
    )

    pairs = [
        (2, 3, "Decimator source"),
        (4, 5, "Decimator source reciprocal"),
        (6, 7, "Decimator rate"),
        (8, 9, "Decimator bits"),
        (10, 11, "BMoog source"),
        (12, 13, "BMoog source reciprocal"),
        (14, 15, "BMoog cutoff"),
        (16, 17, "BMoog q"),
        (18, 19, "BMoog mode"),
        (20, 21, "Perlin3 x"),
        (22, 23, "Perlin3 y"),
        (24, 25, "Perlin3 z"),
    ]
    for control_index, control in enumerate(
        ("freq", "a", "b", "c", "h", "xi", "yi", "zi")
    ):
        first = 26 + control_index * 6
        for coordinate_index, coordinate in enumerate(("x", "y", "z")):
            if control != "xi" or coordinate != "x":
                pairs.append(
                    (
                        first + coordinate_index,
                        first + 3 + coordinate_index,
                        f"RosslerL {control} {coordinate}",
                    )
                )
    pairs.extend(
        [
            (74, 78, "PV_Freeze IFFT"),
            (76, 80, "PV_Freeze ready"),
            (77, 81, "PV_Freeze ready pulse"),
        ]
    )
    for left, right, label in pairs:
        require(
            exact_columns(rows, left, right, live),
            f"{label}: +0 and -0 lanes must be bit-identical",
            failures,
        )
    require(
        f32_bits(rows[0][56]) == f32_bits(0.0)
        and f32_bits(rows[0][59]) == f32_bits(-0.0)
        and exact_columns(rows, 56, 59, range(1, RENDERED_FRAMES)),
        "RosslerL xi must retain the zero sign at x frame 0, then converge",
        failures,
    )
    require(
        all(rows[frame][75] >= 0.0 and rows[frame][79] >= 0.0 for frame in (128, 256, 384)),
        "PV_Freeze signed-zero chains must both publish valid FFT tokens at decoded frames",
        failures,
    )
    observations.append(
        "Signed zero: reciprocal witnesses retain +inf/-inf while +0/-0 are "
        "bit-identical for every Decimator, BMoog, Perlin3, RosslerL, and "
        "PV_Freeze input probe over frames 0..447, except RosslerL xi preserves "
        "the x-output zero sign at frame 0 before converging."
    )


def main() -> None:
    """Verify the boundary-only manifest, assets, and behavioral conclusions."""
    manifest = json.loads((ROOT / "boundary_manifest.json").read_text())
    failures: list[str] = []
    observations: list[str] = []
    require(manifest.get("schema") == 1, "boundary manifest schema must be 1", failures)
    require(
        manifest["provenance"]["supercollider"]["revision"] == EXPECTED_SC,
        "unexpected SuperCollider revision",
        failures,
    )
    require(
        manifest["provenance"]["sc3_plugins"]["revision"] == EXPECTED_SC3,
        "unexpected sc3-plugins revision",
        failures,
    )
    vectors = manifest["vectors"]
    require(set(vectors) == REQUIRED_VECTORS, "boundary manifest vector set changed", failures)

    assets = manifest["assets"]
    asset_paths = {entry["path"] for entry in assets}
    for entry in assets:
        path = ROOT / entry["path"]
        if not path.is_file():
            failures.append(f"missing asset: {entry['path']}")
        elif path.stat().st_size != entry["bytes"]:
            failures.append(f"size mismatch: {entry['path']}")
        elif sha256(path) != entry["sha256"]:
            failures.append(f"sha256 mismatch: {entry['path']}")

    for name, vector in vectors.items():
        path = ROOT / vector["path"]
        require(
            vector["path"] in asset_paths and vector["script"] in asset_paths,
            f"{name}: vector or script is not hashed",
            failures,
        )
        expected_bytes = vector["frames"] * vector["channels"] * 4
        if not path.is_file():
            failures.append(f"{name}: missing vector")
        elif path.stat().st_size != expected_bytes:
            failures.append(
                f"{name}: expected {expected_bytes} bytes, got {path.stat().st_size}"
            )

    if not failures:
        verify_decimator(vectors, failures, observations)
        verify_bmoog(vectors, failures, observations)
        verify_perlin3(vectors["perlin3_boundaries"], failures, observations)
        verify_rossler(vectors, failures, observations)
        verify_pv_freeze(vectors, failures, observations)
        verify_signed_zero(vectors["signed_zero_boundaries"], failures, observations)

    if failures:
        raise SystemExit("\n".join(failures))
    print(
        f"verified {len(assets)} assets and {len(vectors)} semantic boundary vectors"
    )
    for observation in observations:
        print(f"- {observation}")


if __name__ == "__main__":
    main()
