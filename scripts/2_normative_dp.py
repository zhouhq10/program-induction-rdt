"""
2_normative_dp.py — Normative (optimal) DP compression over melody sequences.

Runs the full backtracking dynamic-programming compressor over a set of train
tasks.  For each task the compressor searches for the program decomposition that
minimises the rate-distortion cost. 

Usage (run from repo root):
    python scripts/2_normative_dp.py --curriculum pcfg --beta 1.0 \\
        --save_path results/ --experiname exp1 --task_num 50
"""

import sys

sys.path.append("../")

import argparse
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd

from src.program.grammar import *
from src.domain.melody.melody_primitive import *


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def initialize_program_library(
    curriculum: str,
    init_pm: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[object, object]:
    """Instantiate the grammar and its corresponding DP compressor.

    Args:
        curriculum: One of ``"pcfg"``, ``"count_ag"``, or ``"hier_ag"``.
        init_pm: Initial production table (DataFrame) loaded from
            ``data/primitive/task_pm.csv``.
        args: Parsed command-line arguments forwarded to the compressor
            constructor (beta, search_budget, library hyper-parameters, etc.).

    Returns:
        A ``(program_library, compressor)`` tuple.  The program library is a
        :class:`Grammar` (or subclass) instance; the compressor is the
        matching :class:`DPCompressor` subclass instance.

    Raises:
        ValueError: If ``curriculum`` is not one of the three supported values.
    """
    if curriculum == "pcfg":
        pl = Grammar(production=init_pm)
        from src.domain.melody.dp_compressor import DP_PCFGCompressor

        return pl, DP_PCFGCompressor(program_lib=pl, args=args)

    elif curriculum == "count_ag":
        pl = AdaGrammar(
            production=init_pm,
            lib_size=args.lib_size,
            global_alpha=args.global_alpha,
            global_d=args.global_d,
        )
        from src.domain.melody.dp_compressor import DP_AGCompressor

        return pl, DP_AGCompressor(program_lib=pl, args=args)

    elif curriculum == "hier_ag":
        pl = HierAdaGrammar(
            production=init_pm,
            lib_size=args.lib_size,
            global_alpha=args.global_alpha,
            global_d=args.global_d,
            local_alpha=args.local_alpha,
            local_d=args.local_d,
            local_pattern=args.local_pattern,
        )
        from src.domain.melody.dp_compressor import DP_HAGCompressor

        return pl, DP_HAGCompressor(program_lib=pl, args=args)

    else:
        raise ValueError(f"Invalid curriculum type {curriculum}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Define and parse all command-line arguments for the normative DP run."""
    parser = argparse.ArgumentParser(
        description="Normative (optimal) DP compression of melody sequences."
    )

    # ---- Curriculum --------------------------------------------------------
    parser.add_argument(
        "--curriculum",
        type=str,
        help="Grammar curriculum: 'pcfg' | 'count_ag' | 'hier_ag'.",
    )
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
        help="Frame generation strategy: 'fly' (on the fly) | 'sample' (pre-sampled files) | 'extract'.",
    )

    # ---- Memory constraint -------------------------------------------------
    parser.add_argument(
        "--reuse_num_note_subtask",
        type=int,
        default=12,
        help=(
            "Maximum number of notes that can be reused from a sub-task for "
            "the memorise primitive.  Set to 12 because that is the minimum "
            "sub-melody length in the dataset."
        ),
    )

    # ---- Adaptor Grammar (AG / HAG) options --------------------------------
    parser.add_argument(
        "--lib_size",
        type=int,
        default=100,
        help="Maximum number of entries in the learned program library (AG/HAG only).",
    )
    parser.add_argument(
        "--local_lib_update",
        type=float,
        default=1,
        help="Scaling factor for local library updates (HAG only).",
    )
    parser.add_argument(
        "--global_lib_update",
        type=float,
        default=1,
        help="Scaling factor for global library updates (AG/HAG only).",
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
    # Pitman-Yor process parameters (local / per-task level, HAG only)
    parser.add_argument(
        "--local_alpha",
        type=float,
        default=1.0,
        help="Concentration parameter for the local (per-task) Pitman-Yor process (HAG only).",
    )
    parser.add_argument(
        "--local_d",
        type=float,
        default=0.2,
        help="Discount parameter d for the local Pitman-Yor process (HAG only).",
    )
    parser.add_argument(
        "--local_pattern",
        type=bool,
        default=0,
        help="Whether to use local pattern sharing in the HAG (HAG only).",
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
        help="Number of tasks to process in this run.",
    )

    # ---- Inference options -------------------------------------------------
    parser.add_argument(
        "--melody_backtrack_budget",
        type=int,
        help=(
            "Backtracking window size: how many previous melodies are "
            "reconsidered when updating the library"
        ),
    )
    parser.add_argument(
        "--submelody_backtrack_budget",
        type=int,
        default=12,
        help=(
            "Maximum length (in notes) of a sub-melody chunk considered by "
            "the inner DP.  Caps the inner backtracking window."
        ),
    )

    # ---- Rate-distortion hyper-parameters ----------------------------------
    parser.add_argument(
        "--beta",
        type=float,
        help=(
            "Rate-distortion trade-off parameter β.  "
            "Higher β → more weight on description length (shorter programs); "
            "β = 0 → distortion only."
        ),
    )
    parser.add_argument(
        "--lossless",
        type=int,
        help="If 1, only programs with zero distortion are accepted.",
    )
    parser.add_argument(
        "--mem",
        type=int,
        help="If 1, include the 'memorise' primitive (required for RLE baseline).",
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
        help=(
            "Python list literal of argument names whose values are used to "
            "construct the leaf folder name, e.g. \"['beta', 'random_seed']\"."
        ),
    )

    # ---- Continuation ------------------------------------------------------
    parser.add_argument(
        "--continue_infer",
        type=int,
        help=(
            "If 1, resume a previous run by loading the last saved checkpoint "
            "and continuing from the next task index."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point: load data, initialise the compressor, and run."""
    args = parse_arguments()
    print(args, flush=True)

    # Load melody train tasks
    task_path = "../data/task/train_notes_wo_break_104.obj"
    task_data = prepare_task_data(task_path, args.random_seed, args.random_seq)

    # Create output directory
    full_folder_path = create_save_path(args)

    # Load primitive production table and initialise grammar + compressor
    init_pm = pd.read_csv("../data/melody/task_pm.csv", index_col=0, na_filter=False)
    _, compressor = initialize_program_library(args.curriculum, init_pm, args)

    # Pre-load sampled program frames if frame_gen == "sample";
    frames = (
        [
            pd.read_csv(f"../data/melody/task_frames_{i}.csv", index_col=0)
            for i in range(1, args.frame_depth + 1)
        ]
        if args.frame_gen == "sample"
        else None
    )
    args.frames = frames

    # Run the compression pipeline over all tasks
    compressor.run(tasks=task_data, save_path=full_folder_path)


if __name__ == "__main__":
    main()
