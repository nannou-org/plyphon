#!/usr/bin/env python3
"""Generate deterministic f32 input buffers shared by all PV oracle captures."""

from __future__ import annotations

import pathlib
import struct


ROOT = pathlib.Path(__file__).resolve().parent
FRAMES = 1536
FFT_SIZE = 128

SOURCE_A_AMPLITUDES = [
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
]

SOURCE_A_OFFSETS = [64, 63, 62, 65, 61, 66, 60, 67, 59, 68, 58, 69]

SOURCE_B_AMPLITUDES = [
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
]

SOURCE_B_OFFSETS = [57, 70, 56, 71, 55, 72, 54, 73, 53, 74, 52, 75]


def source_a(index: int) -> float:
    window = index // FFT_SIZE
    if index % FFT_SIZE != SOURCE_A_OFFSETS[window]:
        return 0.0
    return SOURCE_A_AMPLITUDES[window]


def source_b(index: int) -> float:
    window = index // FFT_SIZE
    if index % FFT_SIZE != SOURCE_B_OFFSETS[window]:
        return 0.0
    return SOURCE_B_AMPLITUDES[window]


def write(path: pathlib.Path, values: list[float]) -> None:
    path.write_bytes(b"".join(struct.pack("<f", value) for value in values))


def main() -> None:
    write(ROOT / "pv_source_a.f32", [source_a(index) for index in range(FRAMES)])
    write(ROOT / "pv_source_b.f32", [source_b(index) for index in range(FRAMES)])


if __name__ == "__main__":
    main()
