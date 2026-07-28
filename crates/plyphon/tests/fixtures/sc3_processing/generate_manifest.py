#!/usr/bin/env python3
"""Generate hashes and capture provenance for the SC3 processing and spectral oracle pack."""

from __future__ import annotations

import hashlib
import json
import pathlib
import struct
import subprocess
import sys
from itertools import combinations
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent


def sha256(path: pathlib.Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(*args: str) -> str:
    """Run a command and return its stripped standard output."""
    return subprocess.check_output(args, text=True).strip()


def asset(path: str) -> dict[str, object]:
    """Describe one retained asset by relative path, byte size, and digest."""
    full_path = ROOT / path
    return {
        "path": path,
        "bytes": full_path.stat().st_size,
        "sha256": sha256(full_path),
    }


def command(
    sclang: pathlib.Path,
    runtime_dir: pathlib.Path,
    plugin_dir: pathlib.Path,
    scsynth: pathlib.Path,
    script: str,
    vector: str,
    script_args: list[str] | None = None,
) -> str:
    """Build the reproducible command recorded for one oracle capture."""
    middle = "".join(f" {argument}" for argument in (script_args or []))
    return (
        f"{sclang} -a --include-path {runtime_dir / 'classes'} "
        f"{ROOT / script} {ROOT / vector}{middle} {plugin_dir} {scsynth}"
    )


def vector(
    script: str,
    path: str,
    frames: int,
    channels: int,
    channels_order: list[str],
    control_event_schedule: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    """Build the common manifest fields for one retained vector."""
    return {
        "script": script,
        "path": path,
        "frames": frames,
        "channels": channels,
        "captured_frame_range": [0, frames - 1],
        "warm_up_frames": 0,
        "channels_order": channels_order,
        "control_event_schedule": control_event_schedule,
        **extra,
    }


def dfm1_stability(paths: list[str]) -> list[dict[str, Any]]:
    """Summarize every pair of independent DFM1 captures at the DSP tolerance."""
    captures = {
        path: [
            value[0]
            for value in struct.iter_unpack("<f", (ROOT / path).read_bytes())
        ]
        for path in paths
    }
    comparisons = []
    for reference_path, actual_path in combinations(paths, 2):
        reference = captures[reference_path]
        actual = captures[actual_path]
        if len(reference) != len(actual):
            raise SystemExit(
                f"DFM1 stability length mismatch: {reference_path}, {actual_path}"
            )
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
        comparisons.append(
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
    if not all(comparison["within_tolerance"] for comparison in comparisons):
        worst = max(comparisons, key=lambda comparison: comparison["tolerance_ratio"])
        raise SystemExit(
            "DFM1 independent captures exceed the fixed DSP tolerance: "
            f"{worst['reference']} vs {worst['actual']} sample "
            f"{worst['worst_sample']} ratio {worst['tolerance_ratio']}"
        )
    return comparisons


def pv_capture_samples(
    path: str,
    frames: int,
    channels: int,
    block_frames: int,
) -> tuple[list[int], list[int], list[int]]:
    """Classify PV edge pulses, negative callbacks, and ready callbacks."""
    values = [
        value[0] for value in struct.iter_unpack("<f", (ROOT / path).read_bytes())
    ]
    if len(values) != frames * channels:
        raise SystemExit(f"{path}: size does not match {frames}x{channels} f32 values")
    rows = [
        values[offset : offset + channels]
        for offset in range(0, len(values), channels)
    ]
    edge_pulses = [
        frame
        for frame in range(frames)
        if rows[frame][4] >= 0.5
    ]
    negative_callbacks: list[int] = []
    ready_callbacks: list[int] = []
    for frame in range(block_frames, frames - block_frames, block_frames):
        token = rows[frame][1]
        ready = rows[frame][2]
        if token == -1.0 and ready == 0.0:
            negative_callbacks.append(frame)
        elif token >= 0.0 and ready == 1.0:
            ready_callbacks.append(frame)
        else:
            raise SystemExit(
                f"{path}: block boundary {frame} is neither a negative-token "
                "nor ready callback sample"
            )
    return edge_pulses, negative_callbacks, ready_callbacks


def pv_vector(
    script: str,
    path: str,
    control_name: str,
    control_schedule: list[dict[str, Any]],
    source_schedule: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    """Build a PV manifest entry with exhaustive callback and decode metadata."""
    frames = 1536
    channels = 135
    block_frames = 64
    edge_pulses, negative_callbacks, ready_callbacks = pv_capture_samples(
        path,
        frames,
        channels,
        block_frames,
    )
    if len(control_schedule) != len(ready_callbacks):
        raise SystemExit(
            f"{path}: {len(control_schedule)} control entries do not cover "
            f"{len(ready_callbacks)} ready callbacks"
        )
    complete_control_schedule = [
        {
            "ready_callback": index,
            "sample": sample,
            **entry,
        }
        for index, (sample, entry) in enumerate(
            zip(ready_callbacks, control_schedule),
            start=1,
        )
    ]
    return vector(
        script,
        path,
        frames,
        channels,
        [
            "ifft_audio",
            "k2a_interpolated_chain_token",
            "k2a_interpolated_ready_flag",
            f"k2a_interpolated_{control_name}",
            "ready_positive_edge_pulse",
            "bin_0_magnitude",
            "bin_0_phase",
            "... alternating magnitude/phase through bin 64 ...",
            "bin_64_magnitude",
            "bin_64_phase",
        ],
        source_schedule,
        fft_size=128,
        fft_hop=1.0,
        control_block_frames=block_frames,
        interpolated_chain_token_channel=1,
        interpolated_ready_flag_channel=2,
        interpolated_control_channel=3,
        ready_positive_edge_pulse_channel=4,
        decoded_channels=[5, 134],
        decoded_order="alternating magnitude, phase for bins 0 through 64",
        decoded_phase_channels={
            "ordinary_complex_bins": [8, 132, 2],
            "dc_nyquist_zero_phase": [6, 134],
        },
        ready_positive_edge_pulse_samples=edge_pulses,
        constructor_edge_pulse_sample=edge_pulses[0],
        negative_token_callback_samples=negative_callbacks,
        ready_callback_samples=ready_callbacks,
        callback_spacing_frames=block_frames,
        positive_edge_to_ready_callback_frames=block_frames - 1,
        decoded_frame_samples=ready_callbacks,
        decoded_ready_mapping=[
            {
                "ready_callback": index,
                "ready_callback_sample": sample,
                "decoded_sample": sample,
            }
            for index, sample in enumerate(ready_callbacks, start=1)
        ],
        ready_callback_control_schedule=complete_control_schedule,
        ifft_window_frames=block_frames,
        **extra,
    )


def main() -> None:
    """Validate capture inputs and write the complete oracle manifest."""
    if len(sys.argv) != 9:
        raise SystemExit(
            "usage: generate_manifest.py PLUGINS SC3_BUILD CORE_BUILD RUNTIME "
            "SC3_SOURCE SC_SOURCE SCLANG SCSYNTH"
        )

    plugin_dir = pathlib.Path(sys.argv[1]).resolve()
    build_dir = pathlib.Path(sys.argv[2]).resolve()
    core_build_dir = pathlib.Path(sys.argv[3]).resolve()
    runtime_dir = pathlib.Path(sys.argv[4]).resolve()
    sc3_source = pathlib.Path(sys.argv[5]).resolve()
    sc_source = pathlib.Path(sys.argv[6]).resolve()
    sclang = pathlib.Path(sys.argv[7]).resolve()
    scsynth = pathlib.Path(sys.argv[8]).resolve()

    table = json.loads((ROOT / "bmoog_gain_table.json").read_text())
    if len(table) != 199 or not all(isinstance(value, (int, float)) for value in table):
        raise SystemExit("bmoog_gain_table.json must contain exactly 199 numbers")
    table_bytes = b"".join(struct.pack("<f", float(value)) for value in table)
    (ROOT / "bmoog_gain_table.f32").write_bytes(table_bytes)

    plugin_entries = []
    for path in sorted(plugin_dir.glob("*.scx")):
        built_source = build_dir / "source" / path.name
        core_source = core_build_dir / "server" / "plugins" / path.name
        if built_source.exists():
            source = str(built_source)
            source_revision = "sc3_plugins"
        elif core_source.exists():
            source = str(core_source)
            source_revision = "supercollider"
        else:
            raise SystemExit(f"loaded plugin has no pinned build output: {path.name}")
        plugin_entries.append(
            {
                "loaded_path": str(path),
                "source_path": source,
                "source_revision": source_revision,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    extension_class_entries = [
        {
            "loaded_path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(
            (runtime_dir / "classes" / "Spec100Extensions").glob("*.sc")
        )
    ]

    capture_commands = [
        command(sclang, runtime_dir, plugin_dir, scsynth, "decimator.scd", "decimator.f32"),
        command(sclang, runtime_dir, plugin_dir, scsynth, "bmoog.scd", "bmoog.f32"),
        command(
            sclang,
            runtime_dir,
            plugin_dir,
            scsynth,
            "perlin3.scd",
            "perlin3_ar.f32",
            ["ar"],
        ),
        command(
            sclang,
            runtime_dir,
            plugin_dir,
            scsynth,
            "perlin3.scd",
            "perlin3_kr.f32",
            ["kr"],
        ),
        command(sclang, runtime_dir, plugin_dir, scsynth, "rossler_l.scd", "rossler_l.f32"),
        command(sclang, runtime_dir, plugin_dir, scsynth, "env_detect.scd", "env_detect.f32"),
        *[
            command(
                sclang,
                runtime_dir,
                plugin_dir,
                scsynth,
                "dfm1.scd",
                path,
                [str(ROOT / "dfm1_source.f32")],
            )
            for path in (
                "dfm1_stability_1.f32",
                "dfm1_stability_2.f32",
                "dfm1_stability_3.f32",
                "dfm1.f32",
            )
        ],
        command(
            sclang,
            runtime_dir,
            plugin_dir,
            scsynth,
            "moog_ladder.scd",
            "moog_ladder_ar.f32",
            ["ar"],
        ),
        command(
            sclang,
            runtime_dir,
            plugin_dir,
            scsynth,
            "moog_ladder.scd",
            "moog_ladder_kr.f32",
            ["kr"],
        ),
        command(sclang, runtime_dir, plugin_dir, scsynth, "moog_vcf.scd", "moog_vcf.f32"),
        command(sclang, runtime_dir, plugin_dir, scsynth, "blit_b3.scd", "blit_b3.f32"),
        command(
            sclang,
            runtime_dir,
            plugin_dir,
            scsynth,
            "dnoise_ring.scd",
            "dnoise_ring.f32",
        ),
        command(
            sclang,
            runtime_dir,
            plugin_dir,
            scsynth,
            "pv_freeze.scd",
            "pv_freeze.f32",
            [str(ROOT / "pv_source_a.f32")],
        ),
        command(
            sclang,
            runtime_dir,
            plugin_dir,
            scsynth,
            "pv_mag_smooth.scd",
            "pv_mag_smooth.f32",
            [str(ROOT / "pv_source_a.f32")],
        ),
        command(
            sclang,
            runtime_dir,
            plugin_dir,
            scsynth,
            "pv_morph.scd",
            "pv_morph.f32",
            [str(ROOT / "pv_source_a.f32"), str(ROOT / "pv_source_b.f32")],
        ),
        command(
            sclang,
            runtime_dir,
            plugin_dir,
            scsynth,
            "pv_morph_source_phases.scd",
            "pv_morph_source_phases.f32",
            [str(ROOT / "pv_source_a.f32"), str(ROOT / "pv_source_b.f32")],
        ),
    ]

    retained_paths = [
        "CLEANROOM.md",
        "README.md",
        "capture.sh",
        "generate_dfm1_input.py",
        "generate_pv_inputs.py",
        "generate_manifest.py",
        "verify.py",
        "decimator.scd",
        "decimator.f32",
        "bmoog.scd",
        "bmoog.f32",
        "perlin3.scd",
        "perlin3_ar.f32",
        "perlin3_kr.f32",
        "rossler_l.scd",
        "rossler_l.f32",
        "env_detect.scd",
        "env_detect.f32",
        "dfm1.scd",
        "dfm1_source.f32",
        "dfm1_stability_1.f32",
        "dfm1_stability_2.f32",
        "dfm1_stability_3.f32",
        "dfm1.f32",
        "moog_ladder.scd",
        "moog_ladder_ar.f32",
        "moog_ladder_kr.f32",
        "moog_vcf.scd",
        "moog_vcf.f32",
        "blit_b3.scd",
        "blit_b3.f32",
        "dnoise_ring.scd",
        "dnoise_ring.f32",
        "pv_freeze.scd",
        "pv_freeze.f32",
        "pv_mag_smooth.scd",
        "pv_mag_smooth.f32",
        "pv_morph.scd",
        "pv_morph.f32",
        "pv_morph_source_phases.scd",
        "pv_morph_source_phases.f32",
        "pv_source_a.f32",
        "pv_source_b.f32",
        "bmoog_gain_table.json",
        "bmoog_gain_table.f32",
    ]

    vectors = {
        "decimator": vector(
            "decimator.scd",
            "decimator.f32",
            512,
            10,
            [
                "changing",
                "rate_0_bits_8",
                "rate_24000_bits_8_5",
                "rate_48000_bits_1",
                "rate_12000_bits_0_999",
                "rate_12000_bits_1_001",
                "rate_12000_bits_30_999",
                "rate_12000_bits_31",
                "negative_input_rate_16000_bits_4",
                "zero_input_rate_48000_bits_12",
            ],
            [
                {"frame": 0, "changingRate": 44100, "changingBits": 24},
                {"frame": 256, "changingRate": 12000, "changingBits": 8.5},
                {"frame": 384, "changingRate": 48000, "changingBits": 31},
            ],
        ),
        "bmoog": vector(
            "bmoog.scd",
            "bmoog.f32",
            512,
            4,
            ["mode_0", "mode_1", "mode_2", "mode_3"],
            [
                {"frame": 0, "cutoff": 440, "q": 0.2},
                {"frame": 128, "cutoff": 1200, "q": 0.65},
                {"frame": 256, "cutoff": 6000, "q": 0.9},
                {"frame": 384, "cutoff": 220, "q": 0.1},
            ],
        ),
        "perlin3_ar": vector(
            "perlin3.scd",
            "perlin3_ar.f32",
            512,
            3,
            ["base", "translated", "scaled_reflected"],
            [],
            node_rate="audio",
        ),
        "perlin3_kr": vector(
            "perlin3.scd",
            "perlin3_kr.f32",
            512,
            3,
            ["base", "translated", "scaled_reflected"],
            [],
            node_rate="control",
        ),
        "rossler_l": vector(
            "rossler_l.scd",
            "rossler_l.f32",
            512,
            3,
            ["x", "y", "z"],
            [
                {
                    "frame": 0,
                    "freq": 6000,
                    "a": 0.2,
                    "b": 0.2,
                    "c": 5.7,
                    "h": 0.05,
                    "xi": 0.1,
                    "yi": 0,
                    "zi": 0,
                },
                {"frame": 192, "xi": 0.35, "yi": -0.2, "zi": 0.15},
                {
                    "frame": 320,
                    "freq": 12000,
                    "a": 0.36,
                    "b": 0.35,
                    "c": 4.5,
                    "h": 0.03,
                },
                {"frame": 448, "freq": 48000},
            ],
            interleaving="x, y, z for each frame",
        ),
        "env_detect": vector(
            "env_detect.scd",
            "env_detect.f32",
            640,
            1,
            ["envelope"],
            [
                {"frame": 0, "attack": 100, "release": 0},
                {"frame": 128, "attack": 8, "release": 2},
                {"frame": 256, "attack": 0, "release": 12},
                {"frame": 384, "attack": 250, "release": 50},
                {"frame": 512, "attack": 32, "release": 4},
            ],
        ),
        "dfm1": vector(
            "dfm1.scd",
            "dfm1.f32",
            640,
            2,
            ["type_0", "type_1"],
            [
                {"frame": 0, "freq": 1000, "res": 0.05, "inputGain": 1.5},
                {"frame": 128, "freq": 1800, "res": 0.1, "inputGain": 1.75},
                {"frame": 256, "freq": 3200, "res": 0.15, "inputGain": 1.25},
                {"frame": 384, "freq": 700, "res": 0.08, "inputGain": 2},
                {"frame": 512, "freq": 4400, "res": 0.2, "inputGain": 1},
            ],
            noiselevel=0,
            reference_dither=(
                "DFM1 retains its private minimum noise floor when noiselevel is zero; "
                "the source RNG is intentionally not cross-engine deterministic"
            ),
            source_assets=["dfm1_source.f32"],
            source_pattern=(
                "positive binary-rational Nyquist alternation and ramp with a 5/8 DC bias"
            ),
            source_range=[0.34765625, 0.8984375],
            stability_captures=[
                "dfm1_stability_1.f32",
                "dfm1_stability_2.f32",
                "dfm1_stability_3.f32",
                "dfm1.f32",
            ],
            stability_pairwise=dfm1_stability(
                [
                    "dfm1_stability_1.f32",
                    "dfm1_stability_2.f32",
                    "dfm1_stability_3.f32",
                    "dfm1.f32",
                ]
            ),
        ),
        "moog_ladder_ar": vector(
            "moog_ladder.scd",
            "moog_ladder_ar.f32",
            640,
            2,
            ["audio_rate_cutoff_and_resonance", "control_rate_cutoff_and_resonance"],
            [
                {"frame": 0, "cutoff": 440, "res": 0.2},
                {"frame": 128, "cutoff": 1600, "res": 0.55},
                {"frame": 256, "cutoff": 7200, "res": 0.85},
                {"frame": 384, "cutoff": 220, "res": 0.1},
                {"frame": 512, "cutoff": 3500, "res": 0.7},
            ],
            node_rate="audio",
        ),
        "moog_ladder_kr": vector(
            "moog_ladder.scd",
            "moog_ladder_kr.f32",
            640,
            2,
            ["scheduled_control_lane_a", "scheduled_control_lane_b"],
            [
                {
                    "frame": 0,
                    "sourceA": 0.45,
                    "cutoffA": 120,
                    "resA": 0.1,
                    "sourceB": 0.3,
                    "cutoffB": 180,
                    "resB": 0.2,
                },
                {
                    "frame": 128,
                    "sourceA": 0.48,
                    "cutoffA": 160,
                    "resA": 0.15,
                    "sourceB": 0.27,
                    "cutoffB": 220,
                    "resB": 0.24,
                },
                {
                    "frame": 256,
                    "sourceA": 0.44,
                    "cutoffA": 240,
                    "resA": 0.2,
                    "sourceB": 0.33,
                    "cutoffB": 300,
                    "resB": 0.28,
                },
                {
                    "frame": 384,
                    "sourceA": 0.5,
                    "cutoffA": 80,
                    "resA": 0.12,
                    "sourceB": 0.29,
                    "cutoffB": 140,
                    "resB": 0.14,
                },
                {
                    "frame": 512,
                    "sourceA": 0.46,
                    "cutoffA": 280,
                    "resA": 0.25,
                    "sourceB": 0.35,
                    "cutoffB": 340,
                    "resB": 0.32,
                },
            ],
            node_rate="control",
            valid_cutoff_range_hz=[0, 375],
            audio_transport="K2A.ar interpolation of each control-rate output",
            comparison_samples=list(range(63, 576, 64)),
            comparison_decode=(
                "current = block_start + (block_end - block_start) * 64 / 63"
            ),
            comparison_semantics=(
                "decode each live block's current control-rate result from its first and final "
                "K2A samples, then compare at the declared block-end sample; the full retained "
                "vector documents transport but is not a raw control-rate oracle"
            ),
        ),
        "moog_vcf": vector(
            "moog_vcf.scd",
            "moog_vcf.f32",
            640,
            4,
            [
                "control_cutoff_control_resonance",
                "audio_cutoff_control_resonance",
                "control_cutoff_audio_resonance",
                "audio_cutoff_audio_resonance",
            ],
            [
                {"frame": 0, "cutoff": 440, "res": 0.2},
                {"frame": 128, "cutoff": 1400, "res": 0.5},
                {"frame": 256, "cutoff": 6800, "res": 0.85},
                {"frame": 384, "cutoff": 180, "res": 0.1},
                {"frame": 512, "cutoff": 3200, "res": 0.7},
            ],
        ),
        "blit_b3": vector(
            "blit_b3.scd",
            "blit_b3.f32",
            640,
            4,
            ["BlitB3", "BlitB3Saw", "BlitB3Square", "BlitB3Tri"],
            [
                {"frame": 0, "freq": 440, "leak": 0.99, "leak2": 0.97},
                {"frame": 128, "freq": 0},
                {"frame": 256, "freq": -220},
                {"frame": 384, "freq": 24000},
                {"frame": 512, "freq": 880, "leak": 0.93, "leak2": 0.89},
            ],
            voice_recreated=False,
        ),
        "dnoise_ring": vector(
            "dnoise_ring.scd",
            "dnoise_ring.f32",
            1024,
            10,
            [
                "baseline_draw_0",
                "baseline_draw_1",
                "baseline_draw_2",
                "change_0_ring",
                "probe_after_one_draw",
                "change_1_ring_chance_0_1",
                "probe_after_two_draws",
                "scheduled_ring",
                "probe_after_scheduled_draw_count",
                "seeded_stochastic_transition_ring",
            ],
            [],
            seed=1956,
            pull_period_frames=64,
            pull_frames=list(range(0, 1024, 64)),
            deterministic_schedule={
                "change": [0, 1, 1, 0],
                "chance": [1, 0, 1, 0],
                "shift": [1, 2, 3, 0],
                "numBits": 4,
                "resetval": 13,
            },
            draw_order_contract={
                "baseline_channels": [0, 1, 2],
                "one_draw_probe_channel": 4,
                "two_draw_probe_channel": 6,
                "scheduled_probe_channel": 8,
                "seed_reset_each_pull": True,
            },
        ),
        "pv_freeze": pv_vector(
            "pv_freeze.scd",
            "pv_freeze.f32",
            "freeze",
            [
                {"stage": 0, "freeze": 0},
                {"stage": 1, "freeze": 0},
                {"stage": 2, "freeze": 0},
                {"stage": 3, "freeze": 1},
                {"stage": 4, "freeze": 1},
                {"stage": 5, "freeze": 0},
                {"stage": 6, "freeze": 0},
                {"stage": 7, "freeze": 1},
                {"stage": 8, "freeze": 1},
                {"stage": 9, "freeze": 1, "tail_hold": True},
                {"stage": 10, "freeze": 1, "tail_hold": True},
            ],
            [
                {
                    "frame": 0,
                    "source_buffer": "pv_source_a.f32",
                    "frames": 1536,
                    "segment_frames": 128,
                    "impulse_offsets": [64, 63, 62, 65, 61, 66, 60, 67, 59, 68, 58, 69],
                    "pattern": "one at-most-half-scale positive binary-rational impulse at the declared near-center offset per FFT window",
                },
            ],
            source_assets=["pv_source_a.f32"],
        ),
        "pv_mag_smooth": pv_vector(
            "pv_mag_smooth.scd",
            "pv_mag_smooth.f32",
            "factor",
            [
                {"factor": 0.1, "state": "initialize"},
                {"factor": 0.1},
                {"factor": 0.25},
                {"factor": 0.75},
                {"factor": 1},
                {"factor": 0},
                {"factor": 0.5},
                {"factor": 0.2},
                {"factor": 0.2, "tail_hold": True},
                {"factor": 0.2, "tail_hold": True},
                {"factor": 0.2, "tail_hold": True},
            ],
            [
                {
                    "frame": 0,
                    "source_buffer": "pv_source_a.f32",
                    "frames": 1536,
                    "segment_frames": 128,
                    "impulse_offsets": [64, 63, 62, 65, 61, 66, 60, 67, 59, 68, 58, 69],
                    "pattern": "one at-most-half-scale positive binary-rational impulse at the declared near-center offset per FFT window",
                },
            ],
            source_assets=["pv_source_a.f32"],
        ),
        "pv_morph": pv_vector(
            "pv_morph.scd",
            "pv_morph.f32",
            "morph",
            [
                {"morph": 0},
                {"morph": 0.25},
                {"morph": 0.5},
                {"morph": 0.75},
                {"morph": 1},
                {"morph": 0.6},
                {"morph": 0.2},
                {"morph": 0.9},
                {"morph": 0.9, "tail_hold": True},
                {"morph": 0.9, "tail_hold": True},
                {"morph": 0.9, "tail_hold": True},
            ],
            [
                {
                    "frame": 0,
                    "source_a_buffer": "pv_source_a.f32",
                    "source_b_buffer": "pv_source_b.f32",
                    "frames": 1536,
                    "segment_frames": 128,
                    "source_a_impulse_offsets": [64, 63, 62, 65, 61, 66, 60, 67, 59, 68, 58, 69],
                    "source_b_impulse_offsets": [57, 70, 56, 71, 55, 72, 54, 73, 53, 74, 52, 75],
                    "pattern": "one distinct at-most-half-scale positive binary-rational impulse at each declared near-center offset per FFT window",
                },
            ],
            distinct_input_chains=True,
            source_assets=["pv_source_a.f32", "pv_source_b.f32"],
            source_phase_capture={
                "script": "pv_morph_source_phases.scd",
                "path": "pv_morph_source_phases.f32",
                "frames": 1536,
                "channels": 126,
                "decoded_frame_samples": [
                    128,
                    256,
                    384,
                    512,
                    640,
                    768,
                    896,
                    1024,
                    1152,
                    1280,
                    1408,
                ],
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
        ),
    }

    manifest = {
        "schema": 2,
        "format": {
            "vectors": "raw little-endian IEEE-754 f32",
            "flattening": "sample-frame-major, channel-interleaved",
        },
        "provenance": {
            "supercollider": {
                "repository": "https://github.com/supercollider/supercollider",
                "revision": command_output("git", "-C", str(sc_source), "rev-parse", "HEAD"),
                "sclang": {
                    "path": str(sclang),
                    "version": command_output(str(sclang), "-v"),
                    "sha256": sha256(sclang),
                },
                "scsynth": {
                    "path": str(scsynth),
                    "version": command_output(str(scsynth), "-v"),
                    "sha256": sha256(scsynth),
                },
                "approximate_polar_helper": {
                    "path": "include/plugin_interface/SC_Complex.h",
                    "sha256": sha256(
                        sc_source / "include/plugin_interface/SC_Complex.h"
                    ),
                },
                "core_plugin_build_commands": [
                    (
                        f"cmake -S {sc_source} -B {core_build_dir} -G Ninja "
                        "-DCMAKE_BUILD_TYPE=Release -DSC_QT=OFF -DSC_IDE=OFF "
                        "-DSUPERNOVA=OFF -DENABLE_TESTSUITE=OFF "
                        "-DSC_ABLETON_LINK=OFF -DSC_HIDAPI=OFF -DNO_X11=ON "
                        "-DNO_LIBSNDFILE=ON -DNATIVE=OFF -DUSE_CCACHE=OFF "
                        "-DSCLANG_SERVER=OFF"
                    ),
                    (
                        f"cmake --build {core_build_dir} --target "
                        "BinaryOpUGens DelayUGens DemandUGens FFT_UGens IOUGens "
                        "LFUGens MulAddUGens NoiseUGens OscUGens TriggerUGens "
                        "UnaryOpUGens UnpackFFTUGens --parallel 8"
                    ),
                ],
            },
            "sc3_plugins": {
                "repository": "https://github.com/supercollider/sc3-plugins",
                "revision": command_output("git", "-C", str(sc3_source), "rev-parse", "HEAD"),
                "build_commands": [
                    (
                        f"cmake -S {sc3_source} -B {build_dir} -G Ninja "
                        f"-DSC_PATH={sc_source} -DCMAKE_BUILD_TYPE=Release "
                        "-DSUPERNOVA=OFF -DNOVA_SIMD=OFF -DNOVA_DISK_IO=OFF "
                        "-DAY=OFF -DLADSPA=OFF -DHOA_UGENS=OFF "
                        "-DIN_PLACE_BUILD=ON -DUSE_CCACHE=OFF"
                    ),
                    (
                        f"cmake --build {build_dir} --target "
                        "AntiAliasingOscillators BhobFFT BhobFilt BlackrainUGens "
                        "DistortionUGens JoshPVUGens JoshUGens MCLDChaosUGens "
                        "MCLDFFTUGens NoiseRing SLUGens TJUGens --parallel 8"
                    ),
                ],
                "loaded_plugin_path": str(plugin_dir),
                "loaded_plugin_binaries": plugin_entries,
                "loaded_extension_classes": extension_class_entries,
            },
            "temporary_class_library_path": str(runtime_dir / "classes"),
            "capture_commands": capture_commands,
        },
        "settings": {
            "sample_rate_hz": 48000,
            "control_block_frames": 64,
            "fft_sizes": [128],
            "seeds": {"dnoise_ring": 1956},
            "dsp_tolerance": {
                "absolute": 0.0001,
                "relative": 0.001,
                "formula": "abs_error <= 1e-4 + 1e-3 * abs(reference)",
            },
            "spectral_comparison": {
                "ordinary_phase_distance": (
                    "abs(atan2(sin(actual - reference), "
                    "cos(actual - reference)))"
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
        },
        "vectors": vectors,
        "bmoog_public_functional_table": {
            "source": "https://www.musicdsp.org/en/latest/Filters/145-stilson-s-moog-filter-code.html",
            "site_revision": "43f15628",
            "values": 199,
            "json_path": "bmoog_gain_table.json",
            "json_sha256": sha256(ROOT / "bmoog_gain_table.json"),
            "f32_path": "bmoog_gain_table.f32",
            "f32_sha256": sha256(ROOT / "bmoog_gain_table.f32"),
        },
        "assets": [asset(path) for path in retained_paths],
    }

    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
