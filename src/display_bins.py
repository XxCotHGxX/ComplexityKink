"""Explicit binning helpers for descriptive prompt-composite displays."""

from __future__ import annotations

import numpy as np


def half_open_integer_bin(values):
    """Assign score x to integer bin b representing [b - 0.5, b + 0.5).

    Exact half-point boundaries are assigned to the higher bin. Statistical
    breakpoint estimation uses the continuous score and must not call this
    display-only helper.
    """

    numeric = np.asarray(values, dtype=float)
    return np.floor(numeric + 0.5).astype(int)
