"""
End-to-end demo on synthetic melody compressions, with a planted answer.

Generates melody program trajectories in the same on-disk format the real
compressors write (`task_{i}_prog_trajs.obj`), with a known synergistic pair
buried in a pool of distractors: outcome is high exactly when program A occurs
XOR program B occurs. Neither one predicts outcome on its own, so a search that
ranks programs individually cannot find them -- only synergy can.

Run it to check the pipeline works before pointing it at real results:

    python synergy_curriculum/demo_synthetic.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.append("..")
sys.path.append(str(Path(__file__).resolve().parents[1]))

from synergy_curriculum.curriculum import greedy_search
from synergy_curriculum.occurrence import build_dataset, load_melody_programs
from synergy_curriculum.pid import PID

# Subprograms built from the real melody primitives (memorize, up, down,
# reverse, repeat, concatenate, ranges) and note/count base terms.
PLANTED = [
    "[B,repeat,[B,I,note_3]]",
    "[CB,up,[B,I,note_5],[B,I,count_2]]",
]
DISTRACTORS = [
    "[B,reverse,[B,I,note_1]]",
    "[CB,down,[B,I,note_2],[B,I,count_1]]",
    "[B,repeat,[B,I,note_6]]",
    "[CB,ranges,[B,I,note_4],[B,I,count_3]]",
    "[B,reverse,[B,I,note_5]]",
    "[CB,up,[B,I,note_1],[B,I,count_4]]",
    "[B,repeat,[B,I,note_2]]",
    "[CB,down,[B,I,note_6],[B,I,count_2]]",
]

# In the `redundancy` and `mixed` regimes three of the distractors are promoted
# to roles. Library membership is unchanged across regimes -- only what drives
# the outcome differs -- so the regimes are comparable.
UNIQUE_TERM = "[B,repeat,[B,I,note_6]]"
REDUNDANT_PAIR = [
    "[CB,ranges,[B,I,note_4],[B,I,count_3]]",
    "[B,reverse,[B,I,note_5]]",
]

REGIMES = ("synergy", "redundancy", "mixed")


def _draw_synergy(n_melodies, library, rng, noise):
    """Pure synergy: outcome is A XOR B, so neither source predicts it alone."""
    present = np.zeros((n_melodies, len(library)), dtype=bool)
    good = np.zeros(n_melodies, dtype=bool)
    for i in range(n_melodies):
        row = rng.random(len(library)) < 0.5
        # Guarantee at least one term so the melody has a program at all.
        if not row.any():
            row[rng.integers(len(library))] = True
        a, b = bool(row[0]), bool(row[1])
        g = a ^ b
        if rng.random() < noise:
            g = not g
        present[i], good[i] = row, g
    return present, good


def _draw_redundancy(n_melodies, library, rng, noise, flip=0.08):
    """
    Pure redundancy: a latent cause drives the outcome and two library terms
    are independent noisy readouts of it, so each carries the *same* signal.

    The readouts are noisy rather than exact copies on purpose -- identical
    columns would be merged by `drop_duplicate_sources` before the PID sees
    them.
    """
    present = rng.random((n_melodies, len(library))) < 0.5
    latent = rng.random(n_melodies) < 0.5
    for term in REDUNDANT_PAIR:
        present[:, library.index(term)] = latent ^ (rng.random(n_melodies) < flip)
    good = latent ^ (rng.random(n_melodies) < noise)
    return present, good


def _draw_mixed(
    n_melodies,
    library,
    rng,
    noise,
    readout_flip=0.12,
    redundant_flip=0.1,
    w_shared=1.2,
    w_synergy=1.0,
    w_unique=1.2,
    w_redundant=1.2,
):
    """
    All three flavours at once -- what real data tends to look like.

    Three causes drive the outcome: a shared one read noisily by the XOR pair
    (giving them marginal information on top of their joint information), one
    carried by a single library term, and one read by two terms at once.
    """
    present = rng.random((n_melodies, len(library))) < 0.5
    shared = rng.random(n_melodies) < 0.5
    redundant = rng.random(n_melodies) < 0.5

    a = shared ^ (rng.random(n_melodies) < readout_flip)
    b = shared ^ (rng.random(n_melodies) < readout_flip)
    present[:, library.index(PLANTED[0])] = a
    present[:, library.index(PLANTED[1])] = b
    for term in REDUNDANT_PAIR:
        present[:, library.index(term)] = redundant ^ (
            rng.random(n_melodies) < redundant_flip
        )
    unique = present[:, library.index(UNIQUE_TERM)]

    score = (
        w_shared * shared
        + w_synergy * (a ^ b)
        + w_unique * unique
        + w_redundant * redundant
    )
    good = score > np.median(score)
    good = good ^ (rng.random(n_melodies) < noise)
    return present, good


_DRAW = {
    "synergy": _draw_synergy,
    "redundancy": _draw_redundancy,
    "mixed": _draw_mixed,
}


def generate_melodies(
    n_melodies: int = 200,
    save_dir: str | Path | None = None,
    seed: int = 0,
    noise: float = 0.05,
    regime: str = "synergy",
) -> Path:
    """
    Write synthetic prog_trajs with a planted information structure.

    `regime` selects what the outcome depends on: `synergy` (the XOR pair),
    `redundancy` (two noisy readouts of one cause), or `mixed` (both, plus a
    single-carrier cause). The library is the same in all three.
    """
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")

    rng = np.random.default_rng(seed)
    save_dir = Path(save_dir or tempfile.mkdtemp(prefix=f"synergy_demo_{regime}_"))
    save_dir.mkdir(parents=True, exist_ok=True)

    library = PLANTED + DISTRACTORS
    present_all, good_all = _DRAW[regime](n_melodies, library, rng, noise)

    for i in range(n_melodies):
        present = present_all[i]
        if not present.any():
            present[rng.integers(len(library))] = True
        good = bool(good_all[i])

        terms = [prog for prog, keep in zip(library, present) if keep]
        recon_len = np.full(len(terms), 8)
        # accuracy = 1 - distortion.sum() / recon_len.sum()
        target_accuracy = 0.9 if good else 0.4
        total_distortion = (1.0 - target_accuracy) * recon_len.sum()
        distortion = np.full(len(terms), total_distortion / len(terms))

        df = pd.DataFrame(
            {
                "term": terms,
                "distortion": distortion,
                "recon_len": recon_len,
                "log_prob": rng.normal(-12.0, 2.0, size=len(terms)),
            }
        )
        with open(save_dir / f"task_{i}_prog_trajs.obj", "wb") as fh:
            pickle.dump(df, fh)

    return save_dir


def planted_library(regime: str) -> list:
    """The terms the generator actually wired to the outcome, in `regime`."""
    if regime == "synergy":
        return list(PLANTED)
    if regime == "redundancy":
        return list(REDUNDANT_PAIR)
    return list(PLANTED) + [UNIQUE_TERM] + list(REDUNDANT_PAIR)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic end-to-end demo with a planted answer."
    )
    parser.add_argument(
        "--regime",
        choices=REGIMES,
        default="synergy",
        help="What the planted outcome depends on (default: synergy)",
    )
    parser.add_argument("--n_melodies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--debiased",
        action="store_true",
        help="Also report each measure minus its shuffled-surrogate mean",
    )
    parser.add_argument("--n_surrogates", type=int, default=50)
    args = parser.parse_args(argv)

    print(f"Generating synthetic melody compressions ({args.regime} regime) ...")
    result_dir = generate_melodies(
        n_melodies=args.n_melodies, seed=args.seed, regime=args.regime
    )
    print(f"  wrote to {result_dir}\n")

    melodies = load_melody_programs(result_dir)
    print(f"Loaded {len(melodies)} melodies")

    data = build_dataset(melodies, outcome="accuracy", n_bins=2, min_args=2)
    X, y, names = data["X"], data["y"], data["names"]
    print(f"Occurrence matrix: {X.shape[0]} melodies x {X.shape[1]} subprograms")
    print(f"  ({data['n_merged']} nested duplicates merged)")
    print(f"Outcome labels: {np.bincount(y)}\n")

    print("Searching for the synergy-maximising library ...")
    result = greedy_search(X, y, names, lib_size=4, seed=args.seed, verbose=True)
    print()
    print(result)

    # The search optimises synergy alone, so outside the `synergy` regime the
    # library it returns is not the planted one. Decompose the planted terms
    # directly to show the information structure that was actually wired in.
    planted = planted_library(args.regime)
    missing = [term for term in planted if term not in names]
    if missing:
        print(f"\nPlanted terms merged away or filtered out: {missing}")
    else:
        indices = [names.index(term) for term in planted]
        pid = PID(X[:, indices], y)
        print(f"\nDecomposition of the planted {args.regime} library:")
        print(pid.decomposition(
            debiased=args.debiased, n=args.n_surrogates, names=planted
        ))

    if args.regime == "synergy":
        found = [p for p in PLANTED if p in result.names]
        print(f"\nPlanted pair recovered: {len(found)}/2")
        for prog in PLANTED:
            print(f"  {'FOUND  ' if prog in result.names else 'missing'} {prog}")
        return 0 if len(found) == 2 else 1

    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
