"""
melody_utils.py — Utility functions for melody analysis and encoding.

Provides helper functions used across the melody domain:

* **Primitive helpers** — array slicing and range generation called directly
  by melody primitives (``ranges``, etc.).

* **Pattern analysis** — functions to detect repetitions, periodic structure,
  and constant-shift relationships in melody arrays.  Used for exploratory
  data analysis and for constructing program candidates.

* **Encoding utilities** — run-length encoding (RLE), delta encoding, and
  their combination; used as simple compression baselines.
"""

from copy import copy
from math import exp, fabs
from pandas.core.algorithms import isin

from pandas.core.dtypes.missing import isneginf_scalar

import collections
import more_itertools
import numpy as np

from operator import sub, add, mul, truediv


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------

def get_sliced_array(array, start, end):
    """Return ``array[start:end]``, or ``False`` if the slice is invalid.

    Args:
        array: Sequence to slice.
        start: Inclusive start index (must be ≥ 0).
        end: Exclusive end index (must be ≥ 0, ≤ ``len(array)``).

    Returns:
        The requested slice, or ``False`` if any bound is violated.
    """
    if start < 0 or end < 0:
        return False
    if start > end:
        return False
    if end > len(array):
        return False
    return array[start:end]


def get_sliced_array_one_side(array, end):
    """Return ``array[:end]``, clamping ``end`` to the array length.

    Unlike :func:`get_sliced_array`, this never returns ``False`` — if ``end``
    exceeds the array length the full array is returned.

    Args:
        array: Sequence to slice.
        end: Exclusive end index.

    Returns:
        ``array[:end]`` if ``end ≤ len(array)``, otherwise ``array``.
    """
    if end > len(array):
        return array
    return array[:end]


def single_range(array: np.ndarray, times: int) -> np.ndarray:
    """Generate an ascending melodic scale from ``array`` with ``times`` steps.

    Starting from ``array``, each successive copy is shifted up by 1 (mod 6,
    with pitch values in {1,…,6}).  The result concatenates the original
    sequence with all shifted copies::

        output = [array, array+1, array+2, ..., array+times]   (all mod-6)

    Used by the ``ranges`` primitive to produce ascending scale fragments.

    Args:
        array: Starting note sequence with values in {1,…,6}.
        times: Number of additional +1-transposed copies to append.

    Returns:
        Concatenated array of length ``len(array) * (times + 1)`` with values
        wrapped into {1,…,6}.
    """
    last_array = array
    all_array = array
    for _ in range(times):
        all_array = np.concatenate([all_array, last_array + 1], -1)
        last_array = last_array + 1
    return (all_array - 1) % 6 + 1


# ---------------------------------------------------------------------------
# Pattern analysis helpers
# ---------------------------------------------------------------------------

def principal_period(s: str):
    """Return the principal (shortest) repeating period of string ``s``.

    Uses the classic ``(s + s).find(s, 1, -1)`` trick to detect the minimum-
    length substring whose repetition generates ``s``.

    Args:
        s: Input string to analyse.

    Returns:
        The shortest repeating unit if ``s`` has a non-trivial period,
        otherwise ``None``.
    """
    i = (s + s).find(s, 1, -1)
    return None if i == -1 else s[:i]


def repeatArr(s: np.ndarray):
    """Find the shortest contiguous sub-sequence that repeats in ``s``.

    Scans window sizes from largest to smallest, looking for the first window
    that appears more than once (non-overlapping occurrences are not required).

    Args:
        s: 1-D array to analyse.

    Returns:
        The shortest repeated window as a numpy array, or ``None`` if no
        repeated sub-sequence exists.
    """
    sequence = list(s)
    for size in reversed(range(2, len(s) // 2 + 1)):
        windows = [tuple(window) for window in more_itertools.windowed(sequence, size)]
        counter = collections.Counter(windows)
        for window, count in counter.items():
            if count > 1:
                return np.array(window)
    return None


def find_indexed_repeatsArr(s: np.ndarray, pattern: np.ndarray):
    """Find non-overlapping start indices of ``pattern`` within ``s``.

    Args:
        s: Sequence to search within.
        pattern: Subsequence to locate.

    Returns:
        List of start indices of non-overlapping occurrences if there are at
        least two, otherwise ``None``.
    """
    sequence = list(s)
    pattern = list(pattern)

    size = len(pattern)
    windows = [tuple(window) for window in more_itertools.windowed(sequence, size)]

    index = []
    for i in range(len(windows)):
        if list(windows[i]) == pattern:
            if not len(index):
                index.append(i)
            elif index[-1] + size > i:
                continue  # skip overlapping match
            else:
                index.append(i)

    return index if len(index) > 1 else None


def strict_diffArr(s: np.ndarray):
    """Search for pairs of windows related by a constant element-wise transformation.

    For all window sizes and all pairs of non-overlapping windows, checks
    whether the element-wise difference, ratio, or reverse-difference between
    the two windows is a constant (i.e., the windows are transpositions,
    scalings, or retrogrades of each other).

    Args:
        s: 1-D integer array to analyse.

    Returns:
        A dict with keys ``"diff"``, ``"div"``, ``"rev_diff"`` each containing
        a list of ``(i, j, value)`` triples for matching window pairs, or
        ``None`` if no such pairs exist.
    """
    sequence = list(s)
    for size in reversed(range(2, len(s) // 2 + 1)):
        windows = [tuple(window) for window in more_itertools.windowed(sequence, size)]

        potential_diff = {
            "diff": [],
            "div": [],
            "rev_diff": [],
        }
        for i in range(len(windows) - size):
            for j in range(i + size, len(windows)):
                diff = set(map(sub, windows[i], windows[j]))
                div = set(map(truediv, windows[i], windows[j]))
                rev_diff = set(map(sub, reversed(windows[i]), windows[j]))

                if len(diff) == 1:
                    potential_diff["diff"].append((i, j, diff))
                if len(div) == 1:
                    potential_diff["div"].append((i, j, div))
                if len(rev_diff) == 1:
                    potential_diff["rev_diff"].append((i, j, rev_diff))

    if (
        len(potential_diff["diff"])
        or len(potential_diff["div"])
        or len(potential_diff["rev_diff"])
    ):
        return potential_diff
    else:
        return None


# ---------------------------------------------------------------------------
# Encoding utilities (baselines)
# ---------------------------------------------------------------------------

def rle_encode(s):
    """Apply Run-Length Encoding to a sequence.

    Collapses consecutive equal elements into ``(value, count)`` pairs.

    Args:
        s: Sequence to encode.

    Returns:
        List of ``(value, count)`` tuples.  Empty list if ``s`` is empty.
    """
    sequence = list(s)

    if not sequence:
        return []

    out_list = [(sequence[0], 1)]

    for item in sequence[1:]:
        if item == out_list[-1][0]:
            out_list[-1] = (item, out_list[-1][1] + 1)
        else:
            out_list.append((item, 1))

    return out_list


def delta_encode(data):
    """Apply Delta Encoding to a sequence of values.

    Computes element-wise differences between successive elements.

    TODO: consider whether two digits should represent two bits.

    Args:
        data: Sequence of numeric values.

    Returns:
        List of element-wise differences ``data[i] - data[i-1]`` for
        ``i = 1, …, len(data) - 1``.
    """
    delta_encoded = []
    for i in range(1, len(data)):
        delta = np.subtract(data[i], data[i - 1])
        delta_encoded.append(delta)
    return delta_encoded


def combine_rle_delta(data):
    """Combine Run-Length Encoding and Delta Encoding for a sequence.

    Applies RLE first, then delta-encodes the RLE sequence values.  Each
    segment is tagged ``"RLE"`` if the run length is > 1, or ``"DELTA"``
    otherwise.

    Args:
        data: Sequence to encode.

    Returns:
        List of ``("RLE", value, count)`` or ``("DELTA", delta)`` tuples.
    """
    # Apply RLE
    rle_encoded = rle_encode(data)

    # Apply Delta Encoding on the sequence of RLE values
    delta_encoded = delta_encode([seq for seq, _ in rle_encoded])

    # Combine RLE and Delta tags
    combined = []
    for i, (seq, count) in enumerate(rle_encoded):
        if count > 1:
            combined.append(("RLE", seq, count))
        else:
            combined.append(("DELTA", delta_encoded[i - 1] if i > 0 else seq))
    return combined
