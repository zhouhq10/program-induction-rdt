"""
3_compress_baseline.py — Greedy DP compression with baseline grammars.

Implements two baseline compression models that restrict the program search
space to test specific inductive biases:

* ``chunk`` — Chunking baseline: programs are restricted to type signatures
  with note arguments only (``note->note``, ``note_note->note``, etc.).
  This forces the model to chunk melodies into memorised subsequences via
  the ``memorize`` primitive, analogous to a simple chunking model.

* ``RLE`` — Run-Length Encoding baseline: programs are restricted to type
  signatures involving ``count`` tokens (``note_count->note``, etc.).
  This enables the model to encode repeated patterns efficiently, analogous
  to run-length encoding.

Usage (run from repo root):
    python scripts/3_compress_baseline.py --curriculum chunk --beta 1.0 \\
        --save_path results/ --experiname baseline_chunk
"""

import sys

sys.path.append("../")

import os
import argparse
import pandas as pd

from src.utils.general import prepare_task_data, create_save_path
from src.domain.melody.melody_primitive import *
from src.program.grammar import Grammar, AdaGrammar
from src.domain.melody.greedy_dp_compressor import (
    GreedyDP_PCFGCompressor,
    GreedyDP_AGCompressor,
)


def parse_arguments() -> argparse.Namespace:
    """Define and parse all command-line arguments for the baseline DP run."""
    parser = argparse.ArgumentParser(
        description="Greedy DP compression with baseline (chunk / RLE) grammars."
    )

    # ---- Curriculum --------------------------------------------------------
    parser.add_argument(
        "--curriculum",
        type=str,
        default="chunk",
        help="Baseline curriculum: 'chunk' (memorise-only) | 'RLE' (repeat-only).",
    )

    # ---- Computation / frame options ---------------------------------------
    parser.add_argument(
        "--search_budget",
        type=int,
        help="Number of program frames sampled per sub-melody.",
    )
    parser.add_argument(
        "--frame_depth",
        type=int,
        default=5,
        help="Maximum depth of the program tree (controls program complexity).",
    )
    parser.add_argument(
        "--frame_gen",
        type=str,
        default="fly",
        help="Frame generation strategy: 'fly' (on the fly) | 'sample' (pre-sampled files).",
    )

    # ---- Memory constraint -------------------------------------------------
    parser.add_argument(
        "--reuse_num_note_subtask",
        type=int,
        default=12,
        help=(
            "Maximum number of notes reused per sub-task for the memorise "
            "primitive.  Set to 12 because that is the minimum sub-melody "
            "length in the dataset."
        ),
    )
    parser.add_argument(
        "--melody_backtrack_budget",
        type=int,
        help=(
            "Outer backtracking window: how many previous melodies are "
            "reconsidered when updating the library (AG/HAG only)."
        ),
    )
    parser.add_argument(
        "--submelody_backtrack_budget",
        type=int,
        help="Inner backtracking window size in notes (0 = no backtracking).",
    )
    parser.add_argument(
        "--lib_size",
        type=int,
        default=10000,
        help="Maximum number of entries in the program library.",
    )
    # Pitman-Yor process parameters (global level)
    parser.add_argument(
        "--global_alpha",
        type=float,
        default=1.0,
        help="Concentration parameter for the global Pitman-Yor process.",
    )
    parser.add_argument(
        "--global_d",
        type=float,
        default=0.2,
        help="Discount parameter for the global Pitman-Yor process (0 ≤ d < 1).",
    )

    # ---- Task range --------------------------------------------------------
    parser.add_argument(
        "--task_start_ind",
        type=int,
        default=0,
        help="Index of the first task to process (useful for parallelisation).",
    )
    parser.add_argument(
        "--task_num",
        type=int,
        default=50,
        help="Number of tasks to process in this run.",
    )

    # ---- Rate-distortion hyper-parameters ----------------------------------
    parser.add_argument(
        "--beta",
        type=float,
        help=(
            "Rate-distortion trade-off parameter β.  "
            "Higher β → more weight on description length."
        ),
    )

    # ---- Output paths ------------------------------------------------------
    parser.add_argument(
        "--save_path",
        type=str,
        help="Root directory for saving results.",
    )
    parser.add_argument(
        "--experiname",
        type=str,
        help="Experiment name; used as a sub-directory under save_path.",
    )
    parser.add_argument(
        "--folder_name",
        type=str,
        default="['beta', 'random_seed', 'global_alpha']",
        help=(
            "Python list literal of argument names whose values are used to "
            "construct the leaf folder name."
        ),
    )

    # ---- Random seeds ------------------------------------------------------
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Seed for sub-sampling tasks from the dataset.",
    )
    parser.add_argument(
        "--random_seq",
        type=int,
        default=0,
        help="Seed for shuffling the presentation order of tasks.",
    )
    parser.add_argument(
        "--random_round",
        type=int,
        default=0,
        help="Seed for randomness within a single compression round.",
    )

    # ---- Compatibility arguments -------------------------------------------
    # Accepted but not used by baselines; present so that baseline runs can
    # share the same launch scripts as full model runs.
    parser.add_argument(
        "--lossless",
        type=int,
        default=0,
        help="(Compatibility) Not used by baselines; reserved for full model.",
    )
    parser.add_argument(
        "--mem",
        type=int,
        default=0,
        help="(Compatibility) Not used by baselines; reserved for full model.",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point: load data, initialise the baseline compressor, and run."""
    args = parse_arguments()
    # lossless_error and frames are runtime flags not exposed as CLI args
    args.lossless_error = False
    args.frames = None
    print(args, flush=True)

    # Load melody train tasks
    base_path = "../data/primitive"
    task_path = "../data/task/train_notes_wo_break_104.obj"
    task_data = prepare_task_data(task_path, args.random_seed, args.random_seq)

    # Create output directory
    full_folder_path = create_save_path(args)

    # Load curriculum-specific production table; each baseline uses a
    # restricted set of primitives encoded in a separate CSV.
    init_pm = pd.read_csv(
        os.path.join(base_path, f"task_pm_{args.curriculum}.csv"),
        index_col=0,
        na_filter=False,
    )
    pl = Grammar(production=init_pm)

    # Restrict type signatures to match the baseline's program space
    if args.curriculum == "chunk":
        # Chunking: only note-typed arguments — forces memorise-primitive use
        pl.type_strings = [
            "note->note",
            "note_note->note",
            "note_note_note->note",
        ]
    elif args.curriculum == "RLE":
        # RLE: count-typed arguments — enables repeat-primitive encoding
        pl.type_strings = [
            "note_count->note",
            "note_note_count->note",
            "note_count_note->note",
        ]

    program_lib = AdaGrammar(
        production=init_pm,
        lib_size=args.lib_size,
        global_alpha=args.global_alpha,
        global_d=args.global_d,
    )
    compressor = GreedyDP_AGCompressor(program_lib=program_lib, args=args)
    compressor.run(tasks=task_data, save_path=full_folder_path)


if __name__ == "__main__":
    main()
