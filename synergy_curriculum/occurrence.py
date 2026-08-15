"""
Build the binary occurrence matrix that the PID runs on.

The pipeline, per melody:

    prog_trajs DataFrame  ->  one concatenated program string
                          ->  all nested subprograms it contains
                          ->  X[melody, subprogram] in {0, 1}

plus an outcome y per melody (accuracy / distortion / rate / RD cost),
discretised into labels because the PID estimator indexes probability tables
by label.

Provenance
----------
`extract_nested_brackets` and the candidate filter are recovered from
`exp/archive/4_pid.ipynb` (cells 7-11), deleted in commit d7f76a8 and last
present at bd611f0. The rest is ported to the current data format: the old
notebooks used `prog.terms.tolist()` and `recon_errors` / `recon_length`
columns from the pre-refactor `Program_lib` API; today a melody is a
`prog_trajs` DataFrame with `term`, `distortion`, `recon_len` and `log_prob`.
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.append("..")


# ----- Loading melody compressions -----
def _load_obj(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_melody_programs(
    result_dir: str | Path,
    n_melodies: Optional[int] = None,
    task_start: int = 0,
) -> List[pd.DataFrame]:
    """
    Load one program-trajectory DataFrame per melody from a simulation run.

    Handles both save formats produced by `Compressor.save_result_per_task`:

    - PCFG greedy runs write `task_{i}_prog_trajs.obj`, a single DataFrame.
    - AG / HAG runs write `task_{i}_task_progs.obj`, a *list* of DataFrames
      (the lookback window over melodies); the last entry is the melody that
      task index actually corresponds to.

    Returns
    -------
    list of DataFrame, one per melody, each with at least the columns
    `term`, `distortion`, `recon_len`, `log_prob`.
    """
    result_dir = Path(result_dir)
    if not result_dir.is_dir():
        raise FileNotFoundError(f"No such result directory: {result_dir}")

    melodies: List[pd.DataFrame] = []
    index = task_start
    while n_melodies is None or len(melodies) < n_melodies:
        pcfg_path = result_dir / f"task_{index}_prog_trajs.obj"
        ag_path = result_dir / f"task_{index}_task_progs.obj"

        if pcfg_path.exists():
            obj = _load_obj(pcfg_path)
        elif ag_path.exists():
            obj = _load_obj(ag_path)
        else:
            break

        # AG/HAG store a list over the melody lookback window.
        if isinstance(obj, (list, tuple)):
            obj = next(
                (df for df in reversed(obj) if isinstance(df, pd.DataFrame) and len(df)),
                None,
            )
        if isinstance(obj, pd.DataFrame) and len(obj):
            melodies.append(obj.reset_index(drop=True))

        index += 1

    if not melodies:
        raise FileNotFoundError(
            f"No task_*_prog_trajs.obj or task_*_task_progs.obj found in {result_dir}"
        )
    return melodies


# ----- Subprogram extraction -----
def extract_nested_brackets(string: str) -> List[str]:
    """
    Every bracketed subexpression of a program term, innermost first.

    Recovered verbatim from `exp/archive/4_pid.ipynb`.

    >>> extract_nested_brackets('[CB,[B,repeat,[B,I,note_6]],[B,I,count_2]]')
    ['[B,I,note_6]', '[B,repeat,[B,I,note_6]]', '[B,I,count_2]', '[CB,...]']
    """
    stack: List[int] = []
    extracted: List[str] = []

    for i, char in enumerate(string):
        if char == "[":
            stack.append(i)
        elif char == "]" and stack:
            start = stack.pop()
            extracted.append(string[start : i + 1])

    return extracted


def melody_program_string(prog_trajs: pd.DataFrame, column: str = "term") -> str:
    """Concatenate a melody's program terms into the single string we match against."""
    return "".join(prog_trajs[column].astype(str).tolist())


def candidate_subprograms(
    program_strings: Iterable[str],
    min_args: int = 2,
    drop_trivial_concat: bool = True,
    drop_terms: Sequence[str] = ("concatenate",),
) -> List[str]:
    """
    The pool of subprograms the curriculum is selected from.

    Recovers the notebook's filter: drop fragments with fewer than `min_args`
    commas (too small to be a reusable abstraction) and drop bare
    concatenations of that size (structural glue, not content).

    Note: the original loop mutated the list it was iterating over
    (`for p in nested_all_progs: ... nested_all_progs.remove(p)`), which skips
    the element after every removal and so left short fragments in the pool.
    This version filters properly, so expect a smaller pool than the notebook's.
    """
    pool = set()
    for string in program_strings:
        pool.update(extract_nested_brackets(string))

    kept = []
    for prog in pool:
        n_args = prog.count(",")
        if n_args < min_args:
            continue
        if (
            drop_trivial_concat
            and n_args == min_args
            and any(term in prog for term in drop_terms)
        ):
            continue
        kept.append(prog)

    return sorted(kept)


def drop_constant_sources(
    X: np.ndarray, names: Sequence[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Remove subprograms that occur in every melody or in none.

    They carry no information, and a constant column makes the (n-1)-subset
    probability tables degenerate.
    """
    varies = (X.sum(axis=0) > 0) & (X.sum(axis=0) < X.shape[0])
    return X[:, varies], [n for n, keep in zip(names, varies) if keep]


def drop_duplicate_sources(
    X: np.ndarray,
    names: Sequence[str],
    prefer: str = "longest",
) -> Tuple[np.ndarray, List[str], Dict[str, List[str]]]:
    """
    Collapse subprograms whose occurrence patterns are identical.

    Nested subprograms are the reason this matters: `[B,I,note_3]` is a
    substring of `[B,repeat,[B,I,note_3]]`, so if the inner term only ever
    appears inside the outer one, the two columns are the same vector. PID
    cannot tell them apart, and the search would otherwise pick between them
    arbitrarily -- typically landing on the short, uninformative fragment.

    `prefer="longest"` keeps the most specific abstraction of each equivalence
    class, which is the one you want to name as a curriculum item.

    Returns the reduced matrix, the kept names, and a map from each kept name
    to the aliases it absorbed.
    """
    by_pattern: Dict[tuple, List[int]] = {}
    for j, name in enumerate(names):
        by_pattern.setdefault(tuple(X[:, j].tolist()), []).append(j)

    keep_indices: List[int] = []
    aliases: Dict[str, List[str]] = {}
    for group in by_pattern.values():
        if prefer == "longest":
            winner = max(group, key=lambda j: (len(names[j]), names[j]))
        elif prefer == "shortest":
            winner = min(group, key=lambda j: (len(names[j]), names[j]))
        else:
            raise ValueError(f"Unknown prefer: {prefer}")
        keep_indices.append(winner)
        aliases[names[winner]] = [names[j] for j in group if j != winner]

    keep_indices.sort()
    return (
        X[:, keep_indices],
        [names[j] for j in keep_indices],
        {names[j]: aliases[names[j]] for j in keep_indices},
    )


# ----- Occurrence matrix -----
def build_occurrence_matrix(
    melody_strings: Sequence[str],
    library: Sequence[str],
) -> np.ndarray:
    """
    X[i, j] = 1 if library program j appears in melody i's compression.

    This is a substring test, matching the original notebook
    (`prog in best_prog_string`), so a subprogram counts as present when it
    occurs nested inside a larger term -- which is the point: it means the
    melody's solution *used* that abstraction.
    """
    X = np.zeros((len(melody_strings), len(library)), dtype=int)
    for i, melody in enumerate(melody_strings):
        for j, prog in enumerate(library):
            if prog in melody:
                X[i, j] = 1
    return X


# ----- Outcomes -----
def melody_outcome(
    prog_trajs: pd.DataFrame,
    kind: str = "accuracy",
    beta: float = 1.0,
) -> float:
    """
    Scalar outcome for one melody.

    kind:
      accuracy   1 - total distortion / total reconstructed length.
                 The modern analogue of the notebook's
                 `1 - (recon_errors * recon_length).sum() / 50`.
      distortion total Levenshtein distortion.
      rate       description length, -sum(log_prob).
      rd_cost    distortion + beta * rate -- the objective the compressor
                 itself minimises (`Compressor._calculate_cost`).
    """
    distortion = float(prog_trajs["distortion"].sum())
    rate = float(-prog_trajs["log_prob"].sum())

    if kind == "accuracy":
        total_len = float(prog_trajs["recon_len"].sum())
        return 1.0 - distortion / total_len if total_len else np.nan
    if kind == "distortion":
        return distortion
    if kind == "rate":
        return rate
    if kind == "rd_cost":
        return distortion + beta * rate
    raise ValueError(f"Unknown outcome kind: {kind}")


def discretize(
    y: Sequence[float],
    n_bins: int = 2,
    strategy: str = "quantile",
) -> np.ndarray:
    """
    Bin a continuous outcome into integer labels.

    The PID estimator indexes probability tables by label, so a continuous y
    would give every melody its own label and saturate the mutual information.
    The old notebooks passed raw continuous accuracies straight in; binning is
    the fix.

    strategy: "quantile" for equal-count bins (robust, recommended) or
              "uniform" for equal-width bins.

    The quantile path bins by *average rank* rather than by `np.quantile`
    edges. Compressor outcomes are often heavily tied -- many melodies hit the
    same distortion -- and edge-based binning collapses to a single label when
    a quantile edge lands exactly on the modal value.
    """
    y = np.asarray(y, dtype=float)
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if y.size == 0:
        raise ValueError("Empty outcome vector")

    if strategy == "quantile":
        # Average-rank position of each distinct value, so tied melodies always
        # land in the same bin and bins stay as equal-count as ties allow.
        values, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
        mean_rank = np.cumsum(counts) - counts / 2.0
        value_labels = np.clip(
            (mean_rank / y.size * n_bins).astype(int), 0, n_bins - 1
        )
        labels = value_labels[inverse]
    elif strategy == "uniform":
        edges = np.linspace(y.min(), y.max(), n_bins + 1)[1:-1]
        labels = np.digitize(y, edges).astype(int)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    labels = np.asarray(labels, dtype=int)
    n_distinct = len(set(labels.tolist()))
    if n_distinct < 2:
        raise ValueError(
            f"Outcome collapsed to {n_distinct} label(s) -- every melody scored "
            f"the same ({len(np.unique(y))} distinct outcome value(s)), so there "
            "is no information to decompose."
        )
    return labels


# ----- End-to-end -----
def build_dataset(
    melodies: Sequence[pd.DataFrame],
    outcome: str = "accuracy",
    n_bins: int = 2,
    min_args: int = 2,
    beta: float = 1.0,
    strategy: str = "quantile",
    dedupe: bool = True,
) -> Dict:
    """
    Melody compressions -> (X, y, candidate names) ready for the PID.

    Returns a dict with keys `X`, `y`, `y_continuous`, `names`,
    `melody_strings`, `aliases`, `n_merged`.
    """
    melody_strings = [melody_program_string(df) for df in melodies]
    names = candidate_subprograms(melody_strings, min_args=min_args)
    if not names:
        raise ValueError(
            "No candidate subprograms survived filtering -- try lowering "
            "min_args, or check that the programs contain nested brackets."
        )

    X = build_occurrence_matrix(melody_strings, names)
    X, names = drop_constant_sources(X, names)

    if dedupe:
        n_before = len(names)
        X, names, aliases = drop_duplicate_sources(X, names)
        n_merged = n_before - len(names)
    else:
        aliases, n_merged = {}, 0

    if X.shape[1] < 2:
        raise ValueError(
            "Fewer than 2 varying subprograms across melodies; nothing to "
            "decompose. More melodies, or a more diverse run, are needed."
        )

    y_continuous = np.array(
        [melody_outcome(df, kind=outcome, beta=beta) for df in melodies]
    )
    y = discretize(y_continuous, n_bins=n_bins, strategy=strategy)

    return {
        "X": X,
        "y": y,
        "y_continuous": y_continuous,
        "names": names,
        "melody_strings": melody_strings,
        "aliases": aliases,
        "n_merged": n_merged,
    }
