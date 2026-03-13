"""
3_compress_dp.py — Greedy DP compression over melody sequences.

Runs the greedy dynamic-programming compressor over a set of training tasks.
Unlike the normative compressor (``2_normative_dp.py``), the program at each
sub-melody step is selected *stochastically* (sampled proportional to
exp(value)) rather than exhaustively, making inference faster at the cost of
optimality.

Usage (run from repo root):
    python scripts/3_compress_dp.py --curriculum pcfg --beta 1.0 \\
        --save_path results/ --experiname exp1 --task_num 50
"""

import sys

sys.path.append("../")

import os
import argparse
import pandas as pd

from src.utils.general import prepare_task_data, create_save_path
from src.domain.melody.melody_primitive import *
from src.program.grammar import *
from src.domain.melody.greedy_dp_compressor import (
    GreedyDP_PCFGCompressor,
    GreedyDP_AGCompressor,
    GreedyDP_HAGCompressor,
)


def initialize_program_library(
    curriculum: str, init_pm: pd.DataFrame, args: argparse.Namespace
) -> object:
    """Instantiate the grammar and its corresponding greedy DP compressor.

    Args:
        curriculum: One of ``"pcfg"``, ``"count_ag"``, or ``"hier_ag"``.
        init_pm: Initial production table (DataFrame) loaded from
            ``data/melody/task_pm.csv``.
        args: Parsed command-line arguments forwarded to the compressor
            constructor (beta, frame budget, library hyper-parameters, etc.).

    Returns:
        The matching :class:`GreedyDPCompressor` subclass instance.
    """
    if curriculum == "pcfg":
        program_lib = Grammar(production=init_pm)
        compressor = GreedyDP_PCFGCompressor(program_lib=program_lib, args=args)
    elif curriculum == "count_ag":
        program_lib = AdaGrammar(
            production=init_pm,
            lib_size=args.lib_size,
            global_alpha=args.global_alpha,
            global_d=args.global_d,
        )
        compressor = GreedyDP_AGCompressor(program_lib=program_lib, args=args)
    elif curriculum == "hier_ag":
        program_lib = HierAdaGrammar(
            production=init_pm,
            lib_size=args.lib_size,
            global_alpha=args.global_alpha,
            global_d=args.global_d,
            local_alpha=args.local_alpha,
            local_d=args.local_d,
            local_pattern=args.local_pattern,
            local_lib_size=args.local_lib_size,
        )
        compressor = GreedyDP_HAGCompressor(program_lib=program_lib, args=args)
    else:
        raise ValueError(f"Invalid curriculum type {curriculum}")
    return compressor


def parse_arguments() -> argparse.Namespace:
    """Define and parse all command-line arguments for the greedy DP run."""
    parser = argparse.ArgumentParser(
        description="Greedy DP compression of melody sequences."
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
        help="Frame generation strategy: 'fly' (on the fly) | 'sample' (pre-sampled files).",
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
        default=10000,
        help="Maximum number of entries in the global program library (AG/HAG only).",
    )
    parser.add_argument(
        "--local_lib_size",
        type=int,
        default=10000,
        help="Maximum number of entries in the per-task local library (HAG only).",
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
        help="Discount parameter for the local Pitman-Yor process (HAG only).",
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
            "Outer backtracking window: how many previous melodies are "
            "reconsidered when updating the library (AG/HAG only)."
        ),
    )
    parser.add_argument(
        "--submelody_backtrack_budget",
        type=int,
        help=(
            "Inner backtracking window: maximum number of notes within a "
            "melody considered by the greedy DP segmentation step."
        ),
    )

    # ---- Rate-distortion hyper-parameters ----------------------------------
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help=(
            "Rate-distortion trade-off parameter β.  "
            "Higher β → more weight on description length (shorter programs)."
        ),
    )
    parser.add_argument(
        "--lossless",
        type=int,
        default=0,
        help="If 1, only programs with zero distortion are accepted.",
    )
    parser.add_argument(
        "--mem",
        type=int,
        default=1,
        help="If 1, include the 'memorise' primitive (required for chunking).",
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

    return parser.parse_args()


def main() -> None:
    """Main entry point: load data, initialise the compressor, and run."""
    args = parse_arguments()
    # lossless_error is a runtime flag (not a CLI arg) that enables per-note
    # correctness masking for human-experiment analysis; disabled by default.
    args.lossless_error = False
    print(args, flush=True)

    # Load melody train tasks
    base_path = "../data/primitive"
    task_path = "../data/task/train_notes_wo_break_104.obj"
    task_data = prepare_task_data(task_path, args.random_seed, args.random_seq)

    # Create output directory
    full_folder_path = create_save_path(args)

    # Load primitive production table and initialise grammar + compressor
    init_pm = pd.read_csv(
        os.path.join(base_path, "task_pm.csv"), index_col=0, na_filter=False
    )
    compressor = initialize_program_library(args.curriculum, init_pm, args)

    # Pre-load sampled program frames if frame_gen == "sample";
    # frames[i] contains frames of depth i+1 (1-indexed)
    frames = (
        [
            pd.read_csv(os.path.join(base_path, f"task_frames_{i}.csv"), index_col=0)
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
