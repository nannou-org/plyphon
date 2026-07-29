#!/usr/bin/env python3
"""Generate provenance and hashes for the compact clean-room boundary pack."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent


def sha256(path: pathlib.Path) -> str:
    """Return one file's lowercase SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(*args: str) -> str:
    """Run a command and return its stripped standard output."""
    return subprocess.check_output(args, text=True).strip()


def asset(path: str) -> dict[str, object]:
    """Describe one retained boundary asset."""
    full_path = ROOT / path
    return {
        "path": path,
        "bytes": full_path.stat().st_size,
        "sha256": sha256(full_path),
    }


def vector(
    script: str,
    path: str,
    frames: int,
    channels: list[str],
    channel_count: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build common retained-vector metadata."""
    return {
        "script": script,
        "path": path,
        "frames": frames,
        "channels": channel_count if channel_count is not None else len(channels),
        "channels_order": channels,
        **extra,
    }


def capture_command(
    script: str,
    vector_path: str,
    script_args: list[str] | None = None,
) -> str:
    """Record one reproducible boundary-capture invocation."""
    rendered_args = [
        argument
        if argument.isdigit()
        else f"$FIXTURE_DIR/{argument}"
        for argument in (script_args or [])
    ]
    middle = "".join(f" {argument}" for argument in rendered_args)
    return (
        "$SCLANG -a --include-path $RUNTIME/classes "
        f"$FIXTURE_DIR/{script} $FIXTURE_DIR/{vector_path}{middle} "
        "$RUNTIME/plugins $SCSYNTH"
    )


def main() -> None:
    """Validate inputs and write the boundary-only manifest."""
    if len(sys.argv) != 9:
        raise SystemExit(
            "usage: generate_boundary_manifest.py PLUGINS SC3_BUILD CORE_BUILD "
            "RUNTIME SC3_SOURCE SC_SOURCE SCLANG SCSYNTH"
        )

    plugin_dir = pathlib.Path(sys.argv[1]).resolve()
    build_dir = pathlib.Path(sys.argv[2]).resolve()
    core_build_dir = pathlib.Path(sys.argv[3]).resolve()
    runtime_dir = pathlib.Path(sys.argv[4]).resolve()
    sc3_source = pathlib.Path(sys.argv[5]).resolve()
    sc_source = pathlib.Path(sys.argv[6]).resolve()
    sclang = pathlib.Path(sys.argv[7]).resolve()
    scsynth = pathlib.Path(sys.argv[8]).resolve()

    plugin_entries = []
    for loaded in sorted(plugin_dir.glob("*.scx")):
        sc3_built = build_dir / "source" / loaded.name
        core_built = core_build_dir / "server" / "plugins" / loaded.name
        if sc3_built.exists():
            built = sc3_built
            revision = "sc3_plugins"
        elif core_built.exists():
            built = core_built
            revision = "supercollider"
        else:
            raise SystemExit(f"loaded plugin has no pinned build output: {loaded.name}")
        plugin_entries.append(
            {
                "name": loaded.name,
                "source_revision": revision,
                "bytes": loaded.stat().st_size,
                "sha256": sha256(loaded),
                "built_sha256": sha256(built),
            }
        )

    vectors = {
        "decimator_boundaries": vector(
            "decimator_boundaries.scd",
            "decimator_boundaries.f32",
            512,
            [
                "source",
                "rate_negative_12000",
                "rate_zero",
                "rate_48000",
                "rate_96000",
            ],
            sample_rate_hz=48000,
            bits=24,
        ),
        "decimator_controls": vector(
            "decimator_controls.scd",
            "decimator_controls.f32",
            512,
            [
                "rate_nan",
                "rate_positive_infinity",
                "rate_negative_infinity",
                "bits_nan",
                "bits_positive_infinity",
                "bits_negative_infinity",
                "rate_zero_reference",
                "rate_sample_rate_reference",
                "bits_31_reference",
            ],
            sample_rate_hz=48000,
            score_invalid_frame=128,
            observed_invalid_block_frame=64,
            score_recovery_frame=256,
            observed_recovery_block_frame=192,
        ),
        **{
            f"bmoog_boundaries_{index}": vector(
                "bmoog_boundaries.scd",
                f"bmoog_boundaries_{index}.f32",
                512,
                [f"{label}_mode_{mode}" for mode in range(4)],
                script_args=[str(index)],
                cutoff=cutoff,
                q=q,
                isolated_nrt_process=True,
            )
            for index, (label, cutoff, q) in enumerate(
                [
                    ("cutoff_19_q_0_5", 19, 0.5),
                    ("cutoff_negative_100_q_0_5", -100, 0.5),
                    ("cutoff_24001_q_0_5", 24001, 0.5),
                    ("cutoff_440_q_negative_0_5", 440, -0.5),
                    ("cutoff_440_q_1_5", 440, 1.5),
                    ("cutoff_20_q_0_5", 20, 0.5),
                    ("cutoff_24000_q_0_5", 24000, 0.5),
                    ("cutoff_440_q_0", 440, 0),
                    ("cutoff_440_q_1", 440, 1),
                ]
            )
        },
        "bmoog_controls": vector(
            "bmoog_controls.scd",
            "bmoog_controls.f32",
            512,
            [
                "q_nan",
                "q_positive_infinity",
                "q_negative_infinity",
                "mode_nan",
                "mode_positive_infinity",
                "mode_negative_infinity",
                "mode_zero_reference",
            ],
            score_invalid_frame=128,
            observed_invalid_block_frame=64,
            score_recovery_frame=256,
            observed_recovery_block_frame=192,
        ),
        "perlin3_boundaries": vector(
            "perlin3_boundaries.scd",
            "perlin3_boundaries.f32",
            128,
            [
                "x_negative_256_minus_epsilon",
                "x_negative_256",
                "x_negative_256_plus_epsilon",
                "x_negative_epsilon",
                "x_zero",
                "x_positive_epsilon",
                "x_256_minus_epsilon",
                "x_256",
                "x_256_plus_epsilon",
                "base",
                "base_x_plus_256",
                "base_y_plus_256",
                "base_z_plus_256",
                "base_xyz_plus_256",
            ],
            epsilon=0.0009765625,
            fixed_y=0.375,
            fixed_z=-0.625,
            periodicity_base=[-0.25, 0.375, -0.625],
        ),
        "rossler_l_boundaries": vector(
            "rossler_l_boundaries.scd",
            "rossler_l_boundaries.f32",
            512,
            [
                f"freq_{frequency}_{coordinate}"
                for frequency in ("0", "negative_1", "0_0005", "48000_h_5")
                for coordinate in ("x", "y", "z")
            ],
            sample_rate_hz=48000,
            common_parameters={"a": 0.2, "b": 0.2, "c": 5.7, "xi": 0.1, "yi": 0, "zi": 0},
        ),
        "rossler_l_controls": vector(
            "rossler_l_controls.scd",
            "rossler_l_controls.f32",
            512,
            [
                f"{kind}_{control}_{coordinate}"
                for kind in ("nan", "positive_infinity", "negative_infinity")
                for control in ("freq", "a", "b", "c", "h", "xi", "yi", "zi")
                for coordinate in ("x", "y", "z")
            ]
            + [
                f"{reference}_{coordinate}"
                for reference in ("freq_48000_reference", "freq_0_001_reference")
                for coordinate in ("x", "y", "z")
            ],
            score_invalid_frame=128,
            observed_invalid_block_frame=64,
            score_recovery_frame=256,
            observed_recovery_block_frame=192,
        ),
        "pv_freeze_early": vector(
            "pv_freeze_early.scd",
            "pv_freeze_early.f32",
            768,
            [
                "ifft_audio",
                "k2a_interpolated_chain_token",
                "k2a_interpolated_ready_flag",
                "k2a_interpolated_freeze",
                "ready_positive_edge_pulse",
                "bin_0_magnitude",
                "bin_0_phase",
                "... alternating magnitude/phase through bin 64 ...",
                "bin_64_magnitude",
                "bin_64_phase",
            ],
            channel_count=135,
            fft_size=128,
            freeze=1,
            source_asset="pv_source_a.f32",
            decoded_channels=[5, 134],
        ),
        "pv_freeze_controls": vector(
            "pv_freeze_controls.scd",
            "pv_freeze_controls.f32",
            1536,
            [
                f"{kind}_{channel}"
                for kind in (
                    "nan",
                    "positive_infinity",
                    "negative_infinity",
                    "zero_reference",
                    "one_reference",
                )
                for channel in (
                    "ifft_audio",
                    "chain_token",
                    "ready_flag",
                    "freeze_control",
                    "ready_positive_edge_pulse",
                )
            ],
            fft_size=128,
            source_asset="pv_source_a.f32",
            score_invalid_frame=384,
            first_affected_ifft_frame=320,
            score_recovery_frame=768,
            first_recovered_ifft_frame=704,
        ),
        "signed_zero_boundaries": vector(
            "signed_zero_boundaries.scd",
            "signed_zero_boundaries.f32",
            512,
            [
                "positive_zero_reciprocal_witness",
                "negative_zero_reciprocal_witness",
            ]
            + [
                f"decimator_{control}_{sign}"
                for control in (
                    "source",
                    "source_reciprocal",
                    "rate",
                    "bits",
                )
                for sign in ("positive_zero", "negative_zero")
            ]
            + [
                f"bmoog_{control}_{sign}"
                for control in (
                    "source",
                    "source_reciprocal",
                    "cutoff",
                    "q",
                    "mode",
                )
                for sign in ("positive_zero", "negative_zero")
            ]
            + [
                f"perlin3_{control}_{sign}"
                for control in ("x", "y", "z")
                for sign in ("positive_zero", "negative_zero")
            ]
            + [
                f"rossler_l_{control}_{sign}_{coordinate}"
                for control in ("freq", "a", "b", "c", "h", "xi", "yi", "zi")
                for sign in ("positive_zero", "negative_zero")
                for coordinate in ("x", "y", "z")
            ]
            + [
                f"pv_freeze_{sign}_{channel}"
                for sign in ("positive_zero", "negative_zero")
                for channel in (
                    "ifft_audio",
                    "chain_token",
                    "ready_flag",
                    "ready_positive_edge_pulse",
                )
            ],
            sample_rate_hz=48000,
            fft_size=128,
            source_asset="pv_source_a.f32",
            reciprocal_witnesses=True,
        ),
    }

    asset_paths = [
        "BOUNDARIES.md",
        "capture_boundaries.sh",
        "generate_boundary_manifest.py",
        "verify_boundaries.py",
        "decimator_boundaries.scd",
        "decimator_boundaries.f32",
        "decimator_controls.scd",
        "decimator_controls.f32",
        "bmoog_boundaries.scd",
        *[f"bmoog_boundaries_{index}.f32" for index in range(9)],
        "bmoog_controls.scd",
        "bmoog_controls.f32",
        "perlin3_boundaries.scd",
        "perlin3_boundaries.f32",
        "rossler_l_boundaries.scd",
        "rossler_l_boundaries.f32",
        "rossler_l_controls.scd",
        "rossler_l_controls.f32",
        "pv_freeze_early.scd",
        "pv_freeze_early.f32",
        "pv_freeze_controls.scd",
        "pv_freeze_controls.f32",
        "signed_zero_boundaries.scd",
        "signed_zero_boundaries.f32",
        "pv_source_a.f32",
    ]
    capture_commands = [
        capture_command(
            entry["script"],
            entry["path"],
            [
                *entry.get("script_args", []),
                *(
                    [entry["source_asset"]]
                    if "source_asset" in entry
                    else []
                ),
            ],
        )
        for entry in vectors.values()
    ]
    manifest = {
        "schema": 1,
        "format": {
            "vectors": "raw little-endian IEEE-754 f32",
            "flattening": "sample-frame-major, channel-interleaved",
        },
        "provenance": {
            "supercollider": {
                "revision": command_output("git", "-C", str(sc_source), "rev-parse", "HEAD"),
                "sclang_version": command_output(str(sclang), "-v"),
                "sclang_sha256": sha256(sclang),
                "scsynth_version": command_output(str(scsynth), "-v"),
                "scsynth_sha256": sha256(scsynth),
            },
            "sc3_plugins": {
                "revision": command_output("git", "-C", str(sc3_source), "rev-parse", "HEAD"),
                "loaded_plugin_binaries": plugin_entries,
            },
            "capture_commands": capture_commands,
        },
        "vectors": vectors,
        "omitted_cases": [
            {
                "unit": "BMoog",
                "controls": ["freq=NaN", "freq=+infinity", "freq=-infinity"],
                "reason": (
                    "non-finite cutoff can reach a non-finite-to-index conversion or "
                    "out-of-range gain-table access; no black-box execution is safe enough "
                    "to make that undefined behavior normative"
                ),
            },
            {
                "unit": "PV_Freeze",
                "controls": ["live FFT size 128 -> 256 -> 128"],
                "reason": (
                    "an isolated NRT black-box probe terminated scsynth before a "
                    "render was produced; size-changing one live instance is not "
                    "safe enough to canonicalize or run in automatic capture"
                ),
            },
        ],
        "assets": [asset(path) for path in asset_paths],
    }
    (ROOT / "boundary_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
