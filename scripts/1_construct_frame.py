import sys

sys.path.append("..")

import numpy as np
import pandas as pd
import argparse, re, os

from src.program.grammar import Grammar
from src.program.primitive import Placeholder


def check_remove_memorize(frames: pd.DataFrame) -> pd.DataFrame:
    """Removes rows where 'term' contains 'memorize'."""
    frames = frames[~frames["term"].str.contains("memorize")]
    return frames.reset_index(drop=True)


def power_law_dist(N: int, alpha: float = 1) -> np.ndarray:
    """
    Returns a power-law probability distribution over integers 1, 2, ..., N.

    Parameters
    ----------
    N : int
        Maximum integer value.
    alpha : float
        Exponent; higher alpha concentrates mass on smaller values.
    """
    integers = np.arange(1, N + 1).astype(float)
    probabilities = np.power(integers, -alpha)
    probabilities /= probabilities.sum()
    return probabilities


def main():
    parser = argparse.ArgumentParser(
        description="Generate program frames for the melody task."
    )
    parser.add_argument("--task", type=str, default="melody", help="Task name.")
    parser.add_argument(
        "--depth",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Depths of programs to generate (enumerate mode).",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum program depth (sample mode).",
    )
    parser.add_argument(
        "--frame_gen",
        type=str,
        default="sample",
        help="Frame generation method: 'enumerate' or 'sample'.",
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=20,
        help="Number of frames to sample per sub-task.",
    )
    parser.add_argument(
        "--num_task",
        type=int,
        default=50,
        help="Number of tasks (sample mode only).",
    )
    parser.add_argument(
        "--task_len",
        type=int,
        default=120,
        help="Number of sub-tasks per task (sample mode only).",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/pcfg_frame",
        help="Base path for saving frames.",
    )
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    # Load primitive model and build grammar
    pm_init = pd.read_csv(f"../data/{args.task}/task_pm.csv", index_col=0)
    pl = Grammar(production=pm_init)

    # Input-output type signatures
    t0 = [["note"], "note"]
    t1 = [["note", "count"], "note"]
    t2 = [["note", "note"], "note"]
    t3 = [["note", "note", "note"], "note"]
    t4 = [["note", "note", "count"], "note"]
    t5 = [["note", "count", "note"], "note"]

    types = [t0, t1, t2, t3, t4, t5]
    type_strings = [Placeholder.complete_typelist_to_string(t[0], t[1]) for t in types]

    if args.frame_gen == "sample":
        np.random.seed(args.random_seed)
        depth_prob = power_law_dist(args.max_depth, alpha=1)
        src_pth = (
            f"{args.save_path}/rd_curve/frame_num_{args.frame_num}_{args.random_seed}"
        )
        os.makedirs(src_pth, exist_ok=True)

        for i in range(args.num_task):
            save_path_cur_task = f"{src_pth}/task_{i}"
            os.makedirs(save_path_cur_task, exist_ok=True)
            for j in range(args.task_len):
                sampled_depths = [
                    np.random.choice(np.arange(1, args.max_depth + 1), p=depth_prob)
                    for _ in range(args.frame_num)
                ]
                sampled_types = [
                    np.random.choice(type_strings) for _ in range(args.frame_num)
                ]

                prog_list = []
                for depth, typestring in zip(sampled_depths, sampled_types):
                    type_list = Placeholder.string_to_typelist(typestring)
                    while True:
                        prog = pl.enumerate_one_typed_bfs(
                            type_signature=type_list, depth=depth
                        )
                        if not prog.empty:
                            if (
                                prog["term"][0].count("note")
                                + prog["term"][0].count("count")
                                <= 7
                            ):
                                prog["depth"] = depth
                                prog["frame"] = prog["term"]
                                prog_list.append(prog)
                                break
                prog_list = pd.concat(prog_list).reset_index(drop=True)
                prog_list.to_csv(f"{save_path_cur_task}/index_{j}.csv")

    else:
        for depth in args.depth:
            rfs = []
            for t in types:
                progs = pl.enumerate_typed_bfs(type_signature=t, depth=depth)
                print(t)
                rfs.append(progs)
            combined = pd.concat(rfs).reset_index(drop=True)
            combined = check_remove_memorize(combined)
            combined.to_csv(
                f"{args.save_path}/all_frames_given_depth_and_typestring/task_frames_{depth}.csv"
            )


if __name__ == "__main__":
    main()
