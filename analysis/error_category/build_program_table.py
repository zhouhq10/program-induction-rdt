"""
Build the depth-2 program table used by the simpler-program error category.

Steps:
  1. enumerate every type-consistent note->note frame up to depth 2 with the
     grammar of inputs/task_pm_wo_mem.csv, dropping frames with memorize
     (as simulation_scripts/1_construct_frame.py does); each frame carries the
     log probability of its structure, including its combinators (frame_lp);
  2. among frames that differ only in their combinators (same primitives and
     holes in the same order), keep the most probable one;
  3. fill every note and count hole with every value (Grammar.unfold_frame on
     data/primitive/task_pm.csv), which scores the primitives, notes and counts
     (prog_lp; combinators are left to the frame);
  4. score each program by the full composition prior,
     log_prob = frame_lp + prog_lp;
  5. run every program with the current primitives to get its notes (recon).

The table depends only on the language, not on any model or participant.

Output: analysis/error_category/results/filled_prog_2.csv
        (columns term, comp_lp, prog_lp, frame_lp, log_prob, type_string, depth,
         frame, recon, recon_len)

Usage: python -m analysis.error_category.build_program_table [--out PATH]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.domain.melody.melody_primitive import MelodyProgram  # noqa: E402
from src.program.grammar import Grammar  # noqa: E402

from .common import HERE, REPO_ROOT  # noqa: E402

ROUTERS = {"B", "C", "S", "K"} | {a + b for a in "BCSK" for b in "BCSK"}
TYPE_SIGNATURE = [["note"], "note"]


def leaves(term: str) -> tuple:
    """Primitives and holes of a frame in order, combinators removed."""
    return tuple(e.strip() for e in re.split(r"[\[\],]", term)
                 if e.strip() and e.strip() not in ROUTERS)


def run_program(term: str) -> np.ndarray:
    with contextlib.redirect_stdout(io.StringIO()):
        out = MelodyProgram(term).run()
    if out is None:
        raise RuntimeError(f"program failed to run: {term}")
    out = np.asarray(out).ravel()
    if out.dtype == object:
        raise RuntimeError(f"program did not reduce to notes: {term} -> {out}")
    return out.astype(int)


def run(out: Path, depth: int = 2):
    """Generate the fixed depth-2 candidate table from the supplied priors."""
    if depth != 2:
        raise ValueError("This analysis uses a depth bound of 2")
    t0 = time.time()
    data = HERE / "inputs"

    # 1-2. frames
    frame_grammar = Grammar(production=pd.read_csv(data / "task_pm_wo_mem.csv", index_col=0))
    with contextlib.redirect_stdout(io.StringIO()):
        frames = frame_grammar.enumerate_typed_bfs(type_signature=TYPE_SIGNATURE, depth=depth)
    frames = frames[~frames["term"].str.contains("memorize")].reset_index(drop=True)
    n_enum = len(frames)
    frames = (frames.assign(leaves=frames["term"].apply(leaves))
              .sort_values("log_prob", ascending=False, kind="stable")
              .drop_duplicates("leaves")
              .sort_index()
              .reset_index(drop=True))
    print(f"frames: {n_enum} enumerated, {len(frames)} after keeping the most probable "
          f"combinator variant")

    # 3-4. programs, scored by frame (combinators) + primitives, notes and counts
    grammar = Grammar(production=pd.read_csv(REPO_ROOT / "data/primitive/task_pm.csv", index_col=0, na_filter=False))
    progs = []
    for term, type_string, frame_lp in zip(frames["term"], frames["type_string"],
                                           frames["log_prob"]):
        with contextlib.redirect_stdout(io.StringIO()):
            p = grammar.unfold_frame(term, type_string)
        p = p.rename(columns={"log_prob": "prog_lp"})
        p["frame_lp"] = frame_lp
        p["log_prob"] = p["prog_lp"] + frame_lp
        p["depth"] = depth
        p["frame"] = term
        progs.append(p)
    progs = pd.concat(progs, ignore_index=True)
    print(f"programs: {len(progs)}")

    # 5. reconstructions with the current primitives
    recon = [run_program(t) for t in progs["term"]]
    progs["recon"] = [np.array2string(r, max_line_width=10**6) for r in recon]
    progs["recon_len"] = [len(r) for r in recon]

    out.parent.mkdir(parents=True, exist_ok=True)
    progs.to_csv(out)
    print(f"distinct note sequences: {pd.Series([tuple(r) for r in recon]).nunique()}; "
          f"lengths {progs.recon_len.min()}-{progs.recon_len.max()}")
    print(f"saved {out} ({time.time() - t0:.0f} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "results/filled_prog_2.csv")
    args = ap.parse_args()
    run(args.out)


if __name__ == "__main__":
    main()
