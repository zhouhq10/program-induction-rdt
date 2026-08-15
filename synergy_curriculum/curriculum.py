"""
Find the synergy-maximising program library -- the "synergy-based curriculum".

Given the occurrence matrix X (melodies x candidate subprograms) and a binned
outcome y, search for the size-k subset of subprograms whose joint occurrence
pattern carries the most *synergistic* information about outcome: programs that
predict performance in combination but not individually.

Two searches:

  random_search  -- the recovered original (`exp/archive/4_pid.ipynb` cell 13):
                    sample a library at random, keep it if synergy improved,
                    repeat. Faithful, but the odds of drawing a good size-30
                    subset out of hundreds of candidates are negligible.

  greedy_search  -- seeded with the best *pair* (synergy is undefined for a
                    single source, and pairs are where XOR-like structure
                    shows up), then greedily adding whichever candidate most
                    increases synergy. Far more sample-efficient, and with
                    random restarts it is what you want in practice.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.append("..")

from synergy_curriculum.pid import PID, check_sample_adequacy, synergy


@dataclass
class CurriculumResult:
    """The selected library plus the trace of how the search got there."""

    indices: List[int]
    names: List[str]
    synergy: float
    trace: List[float] = field(default_factory=list)
    decomposition: Optional[object] = None
    adequacy: Dict = field(default_factory=dict)
    method: str = ""

    def to_frame(self) -> pd.DataFrame:
        """One row per selected subprogram, ordered by the unique information."""
        df = pd.DataFrame({"index": self.indices, "term": self.names})
        df["n_args"] = df["term"].str.count(",")
        if self.decomposition is not None:
            df["unique_info"] = self.decomposition.unique
            df = df.sort_values("unique_info", ascending=False)
        return df.reset_index(drop=True)

    def save(self, path: str) -> None:
        self.to_frame().to_csv(path, index=False)

    def __str__(self) -> str:
        lines = [
            f"Synergy-based curriculum ({self.method})",
            f"  library size : {len(self.names)}",
            f"  synergy      : {self.synergy:.4f} bits",
        ]
        if self.decomposition is not None:
            lines.append(str(self.decomposition))
        lines.append("  programs:")
        lines += [f"    {name}" for name in self.names]
        return "\n".join(lines)


def _finalize(
    indices: Sequence[int],
    X: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    trace: List[float],
    method: str,
    debiased: bool,
    n_surrogates: int,
) -> CurriculumResult:
    indices = list(indices)
    selected_names = [names[i] for i in indices]
    X_sub = X[:, indices]

    adequacy = check_sample_adequacy(X_sub, y, verbose=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        pid = PID(X_sub, y)
        decomposition = pid.decomposition(
            debiased=debiased, n=n_surrogates, names=selected_names
        )

    return CurriculumResult(
        indices=indices,
        names=selected_names,
        synergy=decomposition.synergy,
        trace=trace,
        decomposition=decomposition,
        adequacy=adequacy,
        method=method,
    )


def random_search(
    X: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    lib_size: int = 20,
    n_iter: int = 500,
    seed: int = 0,
    debiased: bool = False,
    n_surrogates: int = 50,
    verbose: bool = True,
) -> CurriculumResult:
    """
    The recovered original search: sample libraries, keep the best.

    Kept for reproducing the 2024 analysis. `greedy_search` will beat it at
    equal budget for anything but a very small candidate pool.
    """
    rng = np.random.default_rng(seed)
    n_candidates = X.shape[1]
    lib_size = min(lib_size, n_candidates)

    best_synergy = -np.inf
    best_indices: List[int] = list(range(lib_size))
    trace: List[float] = []

    for iteration in range(n_iter):
        indices = rng.choice(n_candidates, size=lib_size, replace=False)
        value = synergy(X[:, indices], y)
        if value > best_synergy:
            best_synergy = value
            best_indices = sorted(int(i) for i in indices)
            trace.append(value)
            if verbose:
                print(f"  [{iteration:5d}] synergy -> {value:.4f}", flush=True)

    if not np.isfinite(best_synergy):
        raise RuntimeError(
            "No sampled library gave a finite synergy. The occurrence matrix is "
            "probably degenerate (constant columns, or a single outcome label)."
        )

    return _finalize(
        best_indices, X, y, names, trace, "random search", debiased, n_surrogates
    )


def _best_seed_pair(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    max_pairs: int,
    verbose: bool,
) -> List[int]:
    """
    Best-synergy pair, exhaustively if the pool is small enough.

    Greedy selection has to start from a pair: synergy is defined as
    I(X;y) - Imax over the (n-1)-subsets, which needs at least two sources.
    Starting from pairs is also what lets the search find XOR-like structure,
    where each source alone is uninformative.
    """
    n_candidates = X.shape[1]
    all_pairs = list(combinations(range(n_candidates), 2))

    if len(all_pairs) > max_pairs:
        chosen = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        pairs = [all_pairs[i] for i in chosen]
        if verbose:
            print(
                f"  seeding: sampling {max_pairs} of {len(all_pairs)} pairs",
                flush=True,
            )
    else:
        pairs = all_pairs
        if verbose:
            print(f"  seeding: scanning all {len(pairs)} pairs", flush=True)

    best_value, best_pair = -np.inf, pairs[0]
    for pair in pairs:
        value = synergy(X[:, list(pair)], y)
        if value > best_value:
            best_value, best_pair = value, pair

    if verbose:
        print(f"  seed pair synergy {best_value:.4f}", flush=True)
    return list(best_pair)


def greedy_search(
    X: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    lib_size: int = 20,
    seed: int = 0,
    max_pairs: int = 20000,
    restarts: int = 1,
    debiased: bool = False,
    n_surrogates: int = 50,
    verbose: bool = True,
) -> CurriculumResult:
    """
    Greedy forward selection on synergy.

    Start from the best pair, then repeatedly add the candidate that raises
    synergy most, until `lib_size` is reached. `restarts > 1` re-seeds from a
    random pair on subsequent runs and keeps the best result, which helps when
    the greedy path from the best pair is a poor one.
    """
    rng = np.random.default_rng(seed)
    n_candidates = X.shape[1]
    lib_size = min(lib_size, n_candidates)
    if lib_size < 2:
        raise ValueError("lib_size must be at least 2 for synergy to be defined")

    best_overall: Optional[tuple] = None

    for restart in range(restarts):
        if restart == 0:
            selected = _best_seed_pair(X, y, rng, max_pairs, verbose)
        else:
            selected = list(rng.choice(n_candidates, size=2, replace=False))
            if verbose:
                print(f"  restart {restart}: random seed pair", flush=True)

        current = synergy(X[:, selected], y)
        trace = [current]

        while len(selected) < lib_size:
            remaining = [i for i in range(n_candidates) if i not in selected]
            values = [(synergy(X[:, selected + [i]], y), i) for i in remaining]
            best_value, best_index = max(values, key=lambda t: t[0])

            # Greedy on synergy alone would stall at a local peak; keep going to
            # the requested size and let the trace show where it turned over.
            selected.append(int(best_index))
            current = best_value
            trace.append(current)
            if verbose:
                print(
                    f"  [{len(selected):3d}/{lib_size}] synergy {current:.4f} "
                    f"+= {names[best_index][:60]}",
                    flush=True,
                )

        # Report the prefix with the highest synergy, not necessarily the
        # full-size library -- adding sources eventually dilutes it.
        best_step = int(np.argmax(trace))
        selected = selected[: best_step + 2]

        if best_overall is None or trace[best_step] > best_overall[0]:
            best_overall = (trace[best_step], selected, trace)

    assert best_overall is not None
    _, selected, trace = best_overall

    if not np.isfinite(trace[0]):
        raise RuntimeError(
            "Greedy search found no finite synergy. Check that X has varying "
            "columns and y has at least two labels."
        )

    return _finalize(
        sorted(selected), X, y, names, trace, "greedy search", debiased, n_surrogates
    )


def find_curriculum(
    X: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    method: str = "greedy",
    **kwargs,
) -> CurriculumResult:
    """Dispatch to `greedy_search` or `random_search`."""
    if method == "greedy":
        return greedy_search(X, y, names, **kwargs)
    if method == "random":
        return random_search(X, y, names, **kwargs)
    raise ValueError(f"Unknown search method: {method}")
