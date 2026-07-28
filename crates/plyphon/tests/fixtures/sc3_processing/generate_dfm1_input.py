#!/usr/bin/env python3
"""Generate the deterministic, DC-biased input shared by DFM1 oracle captures."""

from __future__ import annotations

import pathlib
import struct


ROOT = pathlib.Path(__file__).resolve().parent
FRAMES = 640


def source(index: int) -> float:
    """Return a changing positive binary-rational sample with a strong DC bias."""
    alternating = 1 / 4 if index % 2 == 0 else -1 / 4
    ramp = ((index % 16) - 8) / 256
    return 5 / 8 + alternating + ramp


def main() -> None:
    values = [source(index) for index in range(FRAMES)]
    (ROOT / "dfm1_source.f32").write_bytes(
        b"".join(struct.pack("<f", value) for value in values)
    )


if __name__ == "__main__":
    main()
