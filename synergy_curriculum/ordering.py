"""
Turn a synergy-selected library into a task *ordering*.

`curriculum.py` answers "which abstractions matter?". This answers "in what
order should melodies be presented so those abstractions are met sensibly?"

The link between the two is the occurrence matrix: melody i exercises library
program j exactly when X[i, j] == 1. Ordering melodies by what they exercise
turns a set of abstractions into a sequence of tasks.

Two orderings:

  coverage     Sort by how many library programs a melody exercises. Ascending
               is the usual easy-to-hard reading: melodies using one abstraction
               come before melodies that combine several.

  incremental  Build the sequence greedily so the library is introduced a few
               programs at a time -- at each step take the melody that adds the
               fewest *unseen* programs. This is the ordering that matches the
               synergy story: sources are met individually before they are met
               in combination.

Neither reorders the compressor's own melody indices in place; you get a
permutation to apply to your task list.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.append("..")


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared (the ranking Spearman needs)."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # Average the ranks within each group of tied values.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation, numpy-only."""
    ra, rb = _ranks(np.asarray(a)), _ranks(np.asarray(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


@dataclass
class OrderingResult:
    """A melody presentation order derived from a synergy-selected library."""

    order: List[int]
    coverage: np.ndarray
    introduces: List[List[str]]
    names: List[str]
    method: str
    outcome_corr: float = float("nan")
    outcome_name: str = ""
    n_uncovered: int = 0

    def to_frame(self) -> pd.DataFrame:
        """One row per melody, in presentation order."""
        return pd.DataFrame(
            {
                "position": np.arange(len(self.order)),
                "melody_index": self.order,
                "n_library_programs": self.coverage[self.order],
                "n_new": [len(items) for items in self.introduces],
                "introduces": ["; ".join(items) for items in self.introduces],
            }
        )

    def save(self, path: str) -> None:
        self.to_frame().to_csv(path, index=False)

    def __str__(self) -> str:
        frame = self.to_frame()
        lines = [
            f"Task ordering ({self.method})",
            f"  melodies      : {len(self.order)}",
            f"  library items : {len(self.names)}",
            f"  coverage range: {frame['n_library_programs'].min()}"
            f" -> {frame['n_library_programs'].max()} programs per melody",
        ]
        if self.n_uncovered:
            lines.append(
                f"  {self.n_uncovered} melody/melodies exercise none of the "
                "library (parked at the end)"
            )
        if not np.isnan(self.outcome_corr):
            lines.append(
                f"  rank corr with {self.outcome_name or 'outcome'}: "
                f"{self.outcome_corr:+.3f}"
            )
        lines.append("  first 10 melodies:")
        for _, row in frame.head(10).iterrows():
            introduced = row["introduces"][:60] or "-"
            lines.append(
                f"    {row['position']:3d}  melody {row['melody_index']:4d}  "
                f"uses {row['n_library_programs']}  new {row['n_new']}  {introduced}"
            )
        return "\n".join(lines)


def order_tasks(
    X: np.ndarray,
    names: Sequence[str],
    indices: Sequence[int],
    method: str = "incremental",
    descending: bool = False,
    outcome: Optional[Sequence[float]] = None,
    outcome_name: str = "",
    uncovered_last: bool = True,
    seed: int = 0,
) -> OrderingResult:
    """
    Order melodies by how they exercise a selected library.

    Parameters
    ----------
    X : (n_melodies, n_candidates) occurrence matrix from `build_dataset`.
    names : candidate names matching X's columns.
    indices : column indices of the selected library (`CurriculumResult.indices`).
    method : "incremental" or "coverage".
    descending : for "coverage", put the most complex melodies first.
    outcome : optional per-melody outcome (e.g. `data["y_continuous"]`) used
        only to report how the ordering lines up with measured difficulty.
    uncovered_last : park melodies that exercise none of the library at the
        end instead of letting them sort to the front.

    Returns
    -------
    OrderingResult with `order` -- a permutation of melody indices.
    """
    X = np.asarray(X)
    indices = list(indices)
    sub = X[:, indices]
    lib_names = [names[i] for i in indices]
    coverage = sub.sum(axis=1)
    rng = np.random.default_rng(seed)

    # Melodies exercising none of the library are outside its scope -- they
    # teach nothing about these abstractions. Keep them (the caller usually
    # still needs a full permutation) but park them at the end rather than
    # letting ascending-coverage sort them to the front as if they were the
    # gentlest introduction.
    covered = [i for i in range(sub.shape[0]) if coverage[i] > 0]
    uncovered = [i for i in range(sub.shape[0]) if coverage[i] == 0]
    if uncovered_last:
        pool, tail_uncovered = covered, uncovered
    else:
        pool, tail_uncovered = list(range(sub.shape[0])), []

    def _by_coverage(items: List[int]) -> List[int]:
        # Random tie-break so equal-coverage melodies are not ordered by their
        # arbitrary file index.
        jitter = {i: rng.random() for i in items}
        return sorted(items, key=lambda i: (coverage[i], jitter[i]))

    if method == "coverage":
        order = _by_coverage(pool)
        if descending:
            order = order[::-1]

    elif method == "incremental":
        remaining = set(pool)
        seen = np.zeros(sub.shape[1], dtype=bool)
        order = []

        while remaining:
            candidates = sorted(remaining)
            gains = {i: int((sub[i] & ~seen).sum()) for i in candidates}

            if max(gains.values()) == 0:
                # Library fully covered: everything left is a recombination of
                # what has already been introduced, so ramp by coverage.
                order.extend(_by_coverage(candidates))
                break

            # Among melodies that introduce something new, take the gentlest
            # step -- fewest new programs, then fewest total.
            best = min(
                (i for i in candidates if gains[i] > 0),
                key=lambda i: (gains[i], int(coverage[i]), i),
            )
            seen |= sub[best].astype(bool)
            order.append(int(best))
            remaining.discard(best)

    else:
        raise ValueError(f"Unknown ordering method: {method}")

    order = [int(i) for i in order] + [int(i) for i in tail_uncovered]

    # Which library programs each melody introduces, in presentation order.
    seen = np.zeros(sub.shape[1], dtype=bool)
    introduces: List[List[str]] = []
    for i in order:
        new = sub[i].astype(bool) & ~seen
        introduces.append([lib_names[j] for j in np.flatnonzero(new)])
        seen |= sub[i].astype(bool)

    corr = float("nan")
    if outcome is not None:
        # Does the ordering track measured difficulty? Positive means later
        # melodies scored higher on `outcome`.
        position = np.empty(len(order), dtype=float)
        position[np.asarray(order)] = np.arange(len(order))
        corr = spearman(position, np.asarray(outcome, dtype=float))

    return OrderingResult(
        order=order,
        coverage=coverage,
        introduces=introduces,
        names=lib_names,
        method=method,
        outcome_corr=corr,
        outcome_name=outcome_name,
        n_uncovered=len(tail_uncovered),
    )
