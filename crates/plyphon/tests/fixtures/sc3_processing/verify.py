#!/usr/bin/env python3
"""Verify retained provenance, vector shape, and non-vacuous oracle content."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import struct
from itertools import combinations
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent
EXPECTED_SC = "426edf6d8742e1cc3bd85b51ca0c4e595d37a903"
EXPECTED_SC3 = "66047341f83e25cbaf3b106f35bd1174a3bbee7c"
EXPECTED_POLAR_HELPER_SHA256 = (
    "7791dc85d8026eb6a665bd686ff21a91cf6beaaa3949ad54f444007893837a2f"
)
REQUIRED_VECTORS = {
    "decimator",
    "bmoog",
    "perlin3_ar",
    "perlin3_kr",
    "rossler_l",
    "env_detect",
    "dfm1",
    "moog_ladder_ar",
    "moog_ladder_kr",
    "moog_vcf",
    "blit_b3",
    "dnoise_ring",
    "pv_freeze",
    "pv_mag_smooth",
    "pv_morph",
}
REQUIRED_SC3_PLUGINS = {
    "AntiAliasingOscillators.scx",
    "BhobFFT.scx",
    "BhobFilt.scx",
    "BlackrainUGens.scx",
    "DistortionUGens.scx",
    "JoshPVUGens.scx",
    "JoshUGens.scx",
    "MCLDChaosUGens.scx",
    "MCLDFFTUGens.scx",
    "NoiseRing.scx",
    "SLUGens.scx",
    "TJUGens.scx",
}
PV_SOURCE_AMPLITUDES = {
    "pv_source_a.f32": [
        1 / 16,
        1 / 8,
        1 / 4,
        3 / 8,
        1 / 2,
        5 / 16,
        3 / 16,
        7 / 16,
        1 / 32,
        9 / 32,
        3 / 32,
        11 / 32,
    ],
    "pv_source_b.f32": [
        3 / 8,
        1 / 16,
        7 / 16,
        1 / 8,
        5 / 16,
        3 / 32,
        1 / 2,
        5 / 32,
        13 / 32,
        1 / 4,
        1 / 32,
        11 / 32,
    ],
}
PV_SOURCE_OFFSETS = {
    "pv_source_a.f32": [64, 63, 62, 65, 61, 66, 60, 67, 59, 68, 58, 69],
    "pv_source_b.f32": [57, 70, 56, 71, 55, 72, 54, 73, 53, 74, 52, 75],
}
DFM1_STABILITY_CAPTURES = [
    "dfm1_stability_1.f32",
    "dfm1_stability_2.f32",
    "dfm1_stability_3.f32",
    "dfm1.f32",
]


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def read_rows(path: pathlib.Path, frames: int, channels: int) -> list[tuple[float, ...]]:
    values = [
        value[0] for value in struct.iter_unpack("<f", path.read_bytes())
    ]
    return [
        tuple(values[offset : offset + channels])
        for offset in range(0, frames * channels, channels)
    ]


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def verify_dfm1(vector: dict[str, Any], failures: list[str]) -> None:
    source_path = ROOT / "dfm1_source.f32"
    source_values = [
        value[0] for value in struct.iter_unpack("<f", source_path.read_bytes())
    ]
    expected_source = [
        5 / 8
        + (1 / 4 if index % 2 == 0 else -1 / 4)
        + ((index % 16) - 8) / 256
        for index in range(640)
    ]
    require(
        len(source_values) == 640
        and [f32_bits(value) for value in source_values]
        == [f32_bits(value) for value in expected_source],
        "dfm1: deterministic DC-biased source buffer changed",
        failures,
    )
    require(
        vector.get("source_assets") == ["dfm1_source.f32"]
        and vector.get("source_range") == [0.34765625, 0.8984375],
        "dfm1: deterministic source metadata changed",
        failures,
    )
    require(
        vector.get("noiselevel") == 0
        and "minimum noise floor" in vector.get("reference_dither", ""),
        "dfm1: private reference dither is not documented",
        failures,
    )
    require(
        vector.get("stability_captures") == DFM1_STABILITY_CAPTURES,
        "dfm1: stability capture set changed",
        failures,
    )

    captures = {
        path: [
            value[0]
            for value in struct.iter_unpack("<f", (ROOT / path).read_bytes())
        ]
        for path in DFM1_STABILITY_CAPTURES
    }
    require(
        all(len(values) == 640 * 2 for values in captures.values()),
        "dfm1: stability capture shape changed",
        failures,
    )
    require(
        len({sha256(ROOT / path) for path in DFM1_STABILITY_CAPTURES})
        == len(DFM1_STABILITY_CAPTURES),
        "dfm1: captures are not independently seeded",
        failures,
    )

    expected_comparisons = []
    for reference_path, actual_path in combinations(DFM1_STABILITY_CAPTURES, 2):
        reference = captures[reference_path]
        actual = captures[actual_path]
        worst_index = max(
            range(len(reference)),
            key=lambda index: abs(actual[index] - reference[index])
            / (1e-4 + 1e-3 * abs(reference[index])),
        )
        reference_value = reference[worst_index]
        actual_value = actual[worst_index]
        absolute_error = abs(actual_value - reference_value)
        tolerance = 1e-4 + 1e-3 * abs(reference_value)
        ratio = absolute_error / tolerance
        require(
            ratio <= 1,
            (
                f"dfm1: {reference_path} vs {actual_path} sample {worst_index} "
                f"exceeds fixed DSP tolerance with ratio {ratio}"
            ),
            failures,
        )
        expected_comparisons.append(
            {
                "reference": reference_path,
                "actual": actual_path,
                "worst_sample": worst_index,
                "reference_value": reference_value,
                "actual_value": actual_value,
                "absolute_error": absolute_error,
                "tolerance": tolerance,
                "tolerance_ratio": ratio,
                "within_tolerance": ratio <= 1,
            }
        )
    require(
        vector.get("stability_pairwise") == expected_comparisons,
        "dfm1: retained pairwise stability evidence does not match the captures",
        failures,
    )


def verify_moog_ladder_kr(vector: dict[str, Any], failures: list[str]) -> None:
    comparison_samples = list(range(63, 576, 64))
    require(
        vector.get("comparison_samples") == comparison_samples,
        "moog_ladder_kr: K2A block-end comparison sample set changed",
        failures,
    )
    require(
        vector.get("audio_transport")
        == "K2A.ar interpolation of each control-rate output"
        and vector.get("comparison_decode")
        == "current = block_start + (block_end - block_start) * 64 / 63"
        and "decode each live block" in vector.get("comparison_semantics", ""),
        "moog_ladder_kr: K2A transport/comparison semantics changed",
        failures,
    )
    schedule = vector["control_event_schedule"]
    require(
        [entry["frame"] for entry in schedule] == [0, 128, 256, 384, 512]
        and all(
            set(entry)
            == {
                "frame",
                "sourceA",
                "cutoffA",
                "resA",
                "sourceB",
                "cutoffB",
                "resB",
            }
            for entry in schedule
        ),
        "moog_ladder_kr: named source/cutoff/res schedule changed",
        failures,
    )
    require(
        vector.get("valid_cutoff_range_hz") == [0, 375]
        and all(
            0 <= entry["cutoffA"] <= 375 and 0 <= entry["cutoffB"] <= 375
            for entry in schedule
        ),
        "moog_ladder_kr: cutoff schedule escaped control-rate Nyquist",
        failures,
    )
    rows = read_rows(ROOT / vector["path"], vector["frames"], vector["channels"])
    compared = [
        tuple(
            rows[frame - 63][channel]
            + (rows[frame][channel] - rows[frame - 63][channel]) * 64 / 63
            for channel in range(vector["channels"])
        )
        for frame in comparison_samples
    ]
    require(
        all(all(math.isfinite(value) and value != 0 for value in row) for row in compared)
        and all(
            len({f32_bits(row[channel]) for row in compared}) >= 4
            for channel in range(vector["channels"])
        ),
        "moog_ladder_kr: block-end lane coverage is zero, non-finite, or vacuous",
        failures,
    )


def verify_dnoise(vector: dict[str, Any], failures: list[str]) -> None:
    rows = read_rows(ROOT / vector["path"], vector["frames"], vector["channels"])
    pull_frames = vector["pull_frames"]
    pulls = [rows[frame] for frame in pull_frames]
    for pull_index, pull in enumerate(pulls):
        require(
            f32_bits(pull[4]) == f32_bits(pull[1]),
            f"dnoise_ring: one-draw probe mismatch at pull {pull_index}",
            failures,
        )
        require(
            f32_bits(pull[6]) == f32_bits(pull[2]),
            f"dnoise_ring: two-draw probe mismatch at pull {pull_index}",
            failures,
        )

    schedule = vector["deterministic_schedule"]["change"]
    for pull_index, pull in enumerate(pulls):
        expected_channel = 2 if schedule[pull_index % len(schedule)] == 1 else 1
        require(
            f32_bits(pull[8]) == f32_bits(pull[expected_channel]),
            f"dnoise_ring: scheduled draw-order mismatch at pull {pull_index}",
            failures,
        )

    def rotate(state: int, shift: int, bits: int = 4) -> int:
        mask = (1 << bits) - 1
        shift %= bits
        if shift == 0:
            return state & mask
        return ((state >> shift) | (state << (bits - shift))) & mask

    change_zero_state = 11
    change_one_state = 11
    scheduled_state = 13
    chance_one = [0, 1]
    chance_schedule = vector["deterministic_schedule"]["chance"]
    shift_schedule = vector["deterministic_schedule"]["shift"]
    for pull_index, pull in enumerate(pulls):
        change_zero_state = rotate(change_zero_state, 1)
        require(
            pull[3] == float(change_zero_state),
            f"dnoise_ring: change=0 transition mismatch at pull {pull_index}",
            failures,
        )

        change_one_state = rotate(change_one_state, 1)
        if chance_one[pull_index % len(chance_one)] == 1:
            change_one_state |= 1
        else:
            change_one_state &= ~1
        require(
            pull[5] == float(change_one_state),
            f"dnoise_ring: change=1 transition mismatch at pull {pull_index}",
            failures,
        )

        schedule_index = pull_index % len(schedule)
        scheduled_state = rotate(scheduled_state, shift_schedule[schedule_index])
        if schedule[schedule_index] == 1:
            if chance_schedule[schedule_index] == 1:
                scheduled_state |= 1
            else:
                scheduled_state &= ~1
        require(
            pull[7] == float(scheduled_state),
            f"dnoise_ring: scheduled transition mismatch at pull {pull_index}",
            failures,
        )

    for channel in (3, 5, 7, 9):
        values = [pull[channel] for pull in pulls]
        require(
            all(value == math.trunc(value) and 0 <= value <= 15 for value in values),
            f"dnoise_ring: ring channel {channel} escaped its four-bit state",
            failures,
        )
        require(
            len({f32_bits(value) for value in values}) >= 2,
            f"dnoise_ring: ring channel {channel} is vacuous",
            failures,
        )


def verify_pv(
    name: str,
    vector: dict[str, Any],
    failures: list[str],
) -> None:
    source_schedule = vector["control_event_schedule"]
    require(
        len(source_schedule) == 1
        and (
            (
                name != "pv_morph"
                and source_schedule[0].get("impulse_offsets")
                == PV_SOURCE_OFFSETS["pv_source_a.f32"]
            )
            or (
                name == "pv_morph"
                and source_schedule[0].get("source_a_impulse_offsets")
                == PV_SOURCE_OFFSETS["pv_source_a.f32"]
                and source_schedule[0].get("source_b_impulse_offsets")
                == PV_SOURCE_OFFSETS["pv_source_b.f32"]
            )
        ),
        f"{name}: manifest does not pin the exact moving-impulse schedule",
        failures,
    )
    rows = read_rows(ROOT / vector["path"], vector["frames"], vector["channels"])
    block_frames = vector["control_block_frames"]
    expected_negative_callbacks = list(
        range(block_frames, vector["frames"] - block_frames, block_frames * 2)
    )
    expected_ready_callbacks = list(
        range(block_frames * 2, vector["frames"] - block_frames, block_frames * 2)
    )
    expected_edge_pulses = [0] + [
        sample - (block_frames - 1) for sample in expected_ready_callbacks
    ]
    negative_callbacks = vector["negative_token_callback_samples"]
    ready_callbacks = vector["ready_callback_samples"]
    edge_pulses = vector["ready_positive_edge_pulse_samples"]
    decoded_frames = vector["decoded_frame_samples"]
    require(
        negative_callbacks == expected_negative_callbacks,
        f"{name}: negative-token callback sample set is not exact",
        failures,
    )
    require(
        ready_callbacks == expected_ready_callbacks,
        f"{name}: ready callback sample set is not exact",
        failures,
    )
    require(
        edge_pulses == expected_edge_pulses,
        f"{name}: ready positive-edge pulse sample set is not exact",
        failures,
    )

    pulse_channel = vector["ready_positive_edge_pulse_channel"]
    actual_edge_pulses = [
        frame
        for frame, row in enumerate(rows)
        if f32_bits(row[pulse_channel]) != f32_bits(0.0)
    ]
    require(
        actual_edge_pulses == expected_edge_pulses
        and all(
            f32_bits(rows[frame][pulse_channel]) == f32_bits(1.0)
            for frame in actual_edge_pulses
        ),
        f"{name}: captured ready positive-edge pulses are not exact",
        failures,
    )

    token_channel = vector["interpolated_chain_token_channel"]
    ready_channel = vector["interpolated_ready_flag_channel"]
    require(
        all(
            f32_bits(rows[frame][token_channel]) == f32_bits(-1.0)
            and f32_bits(rows[frame][ready_channel]) == f32_bits(0.0)
            for frame in negative_callbacks
        ),
        f"{name}: negative-token callback values are not exact",
        failures,
    )
    require(
        all(
            rows[frame][token_channel] >= 0.0
            and f32_bits(rows[frame][ready_channel]) == f32_bits(1.0)
            for frame in ready_callbacks
        ),
        f"{name}: ready callback values are not exact",
        failures,
    )
    callback_samples = sorted(negative_callbacks + ready_callbacks)
    require(
        callback_samples
        == list(range(block_frames, vector["frames"] - block_frames, block_frames))
        and all(
            later - earlier == block_frames
            for earlier, later in zip(callback_samples, callback_samples[1:])
        ),
        f"{name}: negative and ready callbacks do not alternate every 64 samples",
        failures,
    )
    require(
        edge_pulses[1:]
        == [
            sample - vector["positive_edge_to_ready_callback_frames"]
            for sample in ready_callbacks
        ],
        f"{name}: positive edges do not map exactly to ready callbacks",
        failures,
    )

    expected_mapping = [
        {
            "ready_callback": index,
            "ready_callback_sample": sample,
            "decoded_sample": sample,
        }
        for index, sample in enumerate(ready_callbacks, start=1)
    ]
    require(
        decoded_frames == ready_callbacks
        and vector["decoded_ready_mapping"] == expected_mapping,
        f"{name}: decoded[i] is not mapped 1:1 to ready callback[i]",
        failures,
    )
    phase_channels = vector["decoded_phase_channels"]
    require(
        phase_channels
        == {
            "ordinary_complex_bins": [8, 132, 2],
            "dc_nyquist_zero_phase": [6, 134],
        },
        f"{name}: decoded phase-channel contract changed",
        failures,
    )
    ordinary_start, ordinary_end, ordinary_step = phase_channels[
        "ordinary_complex_bins"
    ]
    ordinary_phase_channels = range(
        ordinary_start,
        ordinary_end + 1,
        ordinary_step,
    )
    require(
        all(
            -math.pi <= rows[frame][channel] <= math.pi
            for frame in ready_callbacks
            for channel in ordinary_phase_channels
        ),
        f"{name}: ordinary decoded phase escaped [-pi, pi]",
        failures,
    )
    require(
        all(
            f32_bits(rows[frame][channel]) == f32_bits(0.0)
            for frame in ready_callbacks
            for channel in phase_channels["dc_nyquist_zero_phase"]
        ),
        f"{name}: DC/Nyquist endpoint phase is not exactly zero",
        failures,
    )

    decoded_start, decoded_end = vector["decoded_channels"]
    decoded = [
        tuple(f32_bits(value) for value in rows[frame][decoded_start : decoded_end + 1])
        for frame in decoded_frames
    ]
    require(
        len(set(decoded)) >= 4,
        f"{name}: decoded spectra do not change across four ready frames",
        failures,
    )
    require(
        any(any(bits != 0 for bits in spectrum) for spectrum in decoded),
        f"{name}: decoded spectra are all zero",
        failures,
    )

    window_frames = vector["ifft_window_frames"]
    windows = [
        tuple(f32_bits(rows[index][0]) for index in range(frame, frame + window_frames))
        for frame in ready_callbacks
        if frame + window_frames <= len(rows)
    ]
    nonzero_windows = [
        window for window in windows if any(bits not in (0, 0x80000000) for bits in window)
    ]
    require(
        len(set(nonzero_windows)) >= 4,
        f"{name}: fewer than four distinct non-zero IFFT windows at ready events",
        failures,
    )

    control_channel = vector["interpolated_control_channel"]
    controls = [rows[frame][control_channel] for frame in ready_callbacks]
    expected = {
        "pv_freeze": [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
        "pv_mag_smooth": [
            0.1,
            0.1,
            0.25,
            0.75,
            1,
            0,
            0.5,
            0.2,
            0.2,
            0.2,
            0.2,
        ],
        "pv_morph": [0, 0.25, 0.5, 0.75, 1, 0.6, 0.2, 0.9, 0.9, 0.9, 0.9],
    }[name]
    require(
        len(controls) == len(expected)
        and all(
            math.isclose(actual, wanted, rel_tol=0, abs_tol=1e-6)
            for actual, wanted in zip(controls, expected)
        ),
        f"{name}: ready-callback control schedule does not match the manifest contract",
        failures,
    )
    control_name = {
        "pv_freeze": "freeze",
        "pv_mag_smooth": "factor",
        "pv_morph": "morph",
    }[name]
    schedule = vector["ready_callback_control_schedule"]
    require(
        len(schedule) == len(expected)
        and [entry["ready_callback"] for entry in schedule]
        == list(range(1, len(expected) + 1))
        and [entry["sample"] for entry in schedule] == ready_callbacks
        and all(
            math.isclose(entry[control_name], wanted, rel_tol=0, abs_tol=1e-6)
            for entry, wanted in zip(schedule, expected)
        ),
        f"{name}: manifest control schedule does not cover every ready callback",
        failures,
    )
    tail_start = 10 if name == "pv_freeze" else 9
    require(
        all(
            entry.get("tail_hold") is True
            for entry in schedule[tail_start - 1 :]
        ),
        f"{name}: final ready callbacks are not marked as control tail hold",
        failures,
    )


def verify_pv_morph_source_phases(
    vector: dict[str, Any],
    failures: list[str],
) -> None:
    """Prove the companion phases are distinct and reproduce raw PV_Morph output."""
    capture = vector.get("source_phase_capture")
    expected_frames = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408]
    require(
        capture
        == {
            "script": "pv_morph_source_phases.scd",
            "path": "pv_morph_source_phases.f32",
            "frames": 1536,
            "channels": 126,
            "decoded_frame_samples": expected_frames,
            "channels_order": [
                "source_a_bin_1_phase",
                "... source A ordinary-bin phases through bin 63 ...",
                "source_a_bin_63_phase",
                "source_b_bin_1_phase",
                "... source B ordinary-bin phases through bin 63 ...",
                "source_b_bin_63_phase",
            ],
            "source_a_channels": [0, 62],
            "source_b_channels": [63, 125],
        },
        "pv_morph: source-phase capture layout changed",
        failures,
    )
    if not isinstance(capture, dict):
        return

    path = ROOT / capture["path"]
    if not path.is_file():
        failures.append("pv_morph: source-phase capture is missing")
        return
    expected_bytes = capture["frames"] * capture["channels"] * 4
    require(
        path.stat().st_size == expected_bytes,
        (
            "pv_morph: source-phase capture expected "
            f"{expected_bytes} bytes, got {path.stat().st_size}"
        ),
        failures,
    )
    if path.stat().st_size != expected_bytes:
        return

    source_rows = read_rows(path, capture["frames"], capture["channels"])
    morph_rows = read_rows(ROOT / vector["path"], vector["frames"], vector["channels"])
    inspected = [(128, 0.0), (384, 0.5), (640, 1.0)]
    for frame, morph in inspected:
        for bin_index in range(1, 64):
            source_a = source_rows[frame][bin_index - 1]
            source_b = source_rows[frame][63 + bin_index - 1]
            source_distance = abs(
                math.atan2(
                    math.sin(source_a - source_b),
                    math.cos(source_a - source_b),
                )
            )
            require(
                source_distance > 0.0016,
                (
                    "pv_morph: source phases are not distinct at "
                    f"frame {frame}, bin {bin_index}"
                ),
                failures,
            )

            expected = (1.0 - morph) * source_a + morph * source_b
            actual = morph_rows[frame][6 + 2 * bin_index]
            error = abs(
                math.atan2(
                    math.sin(actual - expected),
                    math.cos(actual - expected),
                )
            )
            require(
                error <= 0.0016,
                (
                    "pv_morph: raw source-phase interpolation mismatch at "
                    f"frame {frame}, bin {bin_index}: expected {expected}, "
                    f"got {actual}, circular error {error}"
                ),
                failures,
            )


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    failures: list[str] = []

    require(manifest.get("schema") == 2, "manifest schema must be 2", failures)
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
    require(
        manifest["provenance"]["supercollider"]["approximate_polar_helper"]
        == {
            "path": "include/plugin_interface/SC_Complex.h",
            "sha256": EXPECTED_POLAR_HELPER_SHA256,
        },
        "approximate polar helper provenance changed",
        failures,
    )
    require(
        manifest["settings"]["dsp_tolerance"]
        == {
            "absolute": 0.0001,
            "relative": 0.001,
            "formula": "abs_error <= 1e-4 + 1e-3 * abs(reference)",
        },
        "DSP tolerance changed",
        failures,
    )
    require(
        manifest["settings"]["spectral_comparison"]
        == {
            "ordinary_phase_distance": (
                "abs(atan2(sin(actual - reference), cos(actual - reference)))"
            ),
            "ordinary_phase_tolerance": {
                "default": "dsp_tolerance",
                "approximate_path_units": [
                    "pv_freeze",
                    "pv_mag_smooth",
                    "pv_morph",
                ],
                "approximate_path_absolute_floor_radians": 0.0016,
                "formula": "max(0.0016, 1e-4 + 1e-3 * abs(reference))",
            },
            "magnitude_dc_nyquist_ifft_distance": "abs(actual - reference)",
            "magnitude_dc_nyquist_ifft_tolerance": "dsp_tolerance",
        },
        "spectral comparison contract changed",
        failures,
    )

    plugin_entries = manifest["provenance"]["sc3_plugins"][
        "loaded_plugin_binaries"
    ]
    plugin_names = {pathlib.Path(entry["loaded_path"]).name for entry in plugin_entries}
    require(
        REQUIRED_SC3_PLUGINS <= plugin_names,
        "manifest does not prove every required sc3 plugin binary",
        failures,
    )
    for entry in plugin_entries:
        require(
            entry["bytes"] > 0
            and len(entry["sha256"]) == 64
            and all(character in "0123456789abcdef" for character in entry["sha256"]),
            f"invalid loaded-plugin metadata: {entry['loaded_path']}",
            failures,
        )
        require(
            entry["source_revision"] in {"supercollider", "sc3_plugins"},
            f"plugin source revision is not pinned: {entry['loaded_path']}",
            failures,
        )
        require(
            not entry["source_path"].startswith("/Applications/"),
            f"plugin came from an installed application: {entry['loaded_path']}",
            failures,
        )

    assets = manifest["assets"]
    asset_paths = {entry["path"] for entry in assets}
    for entry in assets:
        path = ROOT / entry["path"]
        if not path.is_file():
            failures.append(f"missing asset: {entry['path']}")
            continue
        if path.stat().st_size != entry["bytes"]:
            failures.append(f"size mismatch: {entry['path']}")
        if sha256(path) != entry["sha256"]:
            failures.append(f"sha256 mismatch: {entry['path']}")

    vectors = manifest["vectors"]
    require(
        set(vectors) == REQUIRED_VECTORS,
        "manifest vector set does not exactly cover the required families",
        failures,
    )
    for name, vector in vectors.items():
        path = ROOT / vector["path"]
        expected_bytes = vector["frames"] * vector["channels"] * 4
        require(vector["path"] in asset_paths, f"{name}: vector is not hashed", failures)
        require(vector["script"] in asset_paths, f"{name}: script is not hashed", failures)
        if not path.is_file():
            failures.append(f"{name}: missing vector")
            continue
        if path.stat().st_size != expected_bytes:
            failures.append(
                f"{name}: expected {expected_bytes} bytes, got {path.stat().st_size}"
            )
            continue
        rows = read_rows(path, vector["frames"], vector["channels"])
        values = [value for row in rows for value in row]
        require(
            all(math.isfinite(value) for value in values),
            f"{name}: vector contains a non-finite f32",
            failures,
        )
        require(
            any(value != 0 for value in values),
            f"{name}: vector is entirely zero",
            failures,
        )
        require(
            len({f32_bits(value) for value in values}) >= 4,
            f"{name}: vector has fewer than four distinct f32 values",
            failures,
        )
        for source_asset in vector.get("source_assets", []):
            require(
                source_asset in asset_paths,
                f"{name}: deterministic source buffer is not hashed",
                failures,
            )
        source_phase_capture = vector.get("source_phase_capture")
        if source_phase_capture is not None:
            require(
                source_phase_capture["path"] in asset_paths,
                f"{name}: source-phase vector is not hashed",
                failures,
            )
            require(
                source_phase_capture["script"] in asset_paths,
                f"{name}: source-phase script is not hashed",
                failures,
            )

    for source_name, amplitudes in PV_SOURCE_AMPLITUDES.items():
        source_path = ROOT / source_name
        require(
            source_path.stat().st_size == 1536 * 4,
            f"{source_name}: expected exactly 1536 f32 samples",
            failures,
        )
        source_values = [
            value[0] for value in struct.iter_unpack("<f", source_path.read_bytes())
        ]
        expected_source = [0.0] * 1536
        for window, (offset, amplitude) in enumerate(
            zip(PV_SOURCE_OFFSETS[source_name], amplitudes)
        ):
            expected_source[(window * 128) + offset] = amplitude
        require(
            all(math.isfinite(value) for value in source_values)
            and [f32_bits(value) for value in source_values]
            == [f32_bits(value) for value in expected_source],
            f"{source_name}: deterministic sparse FFT-window impulses changed",
            failures,
        )

    if "dnoise_ring" in vectors:
        verify_dnoise(vectors["dnoise_ring"], failures)
    if "dfm1" in vectors:
        verify_dfm1(vectors["dfm1"], failures)
    if "moog_ladder_kr" in vectors:
        verify_moog_ladder_kr(vectors["moog_ladder_kr"], failures)
    for name in ("pv_freeze", "pv_mag_smooth", "pv_morph"):
        if name in vectors:
            verify_pv(name, vectors[name], failures)
    if "pv_morph" in vectors:
        verify_pv_morph_source_phases(vectors["pv_morph"], failures)

    table_entry = manifest["bmoog_public_functional_table"]
    table = json.loads((ROOT / table_entry["json_path"]).read_text())
    require(len(table) == 199, f"BMoog table has {len(table)} values", failures)
    require(
        sha256(ROOT / table_entry["json_path"]) == table_entry["json_sha256"],
        "BMoog JSON table hash mismatch",
        failures,
    )
    require(
        sha256(ROOT / table_entry["f32_path"]) == table_entry["f32_sha256"],
        "BMoog f32 table hash mismatch",
        failures,
    )

    if failures:
        raise SystemExit("\n".join(failures))
    print(
        f"verified {len(assets)} retained assets, {len(vectors)} vectors, "
        "DFM1 pairwise stability, demand draw order, three changing PV spectra/IFFT captures, "
        "and the 199-value BMoog table"
    )


if __name__ == "__main__":
    main()
