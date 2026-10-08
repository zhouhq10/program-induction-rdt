import sys

sys.path.append("..")

import numpy as np
import pandas as pd
import argparse, re, os, random
from multiprocessing import Pool
from pathlib import Path

from src.program.grammar import Grammar
from src.program.primitive import Placeholder

REPO_ROOT = Path(__file__).resolve().parents[1]


def check_remove_memorize(frames: pd.DataFrame) -> pd.DataFrame:
    """Removes rows where 'term' contains 'memorize'."""
    frames = frames[~frames["term"].str.contains("memorize")]
    return frames.reset_index(drop=True)


_worker_grammar = None


def _init_worker(pm_path):
    global _worker_grammar
    _worker_grammar = Grammar(production=pd.read_csv(pm_path, index_col=0))


def _sample_batch(job):
    """Draw ``n`` frames of ``depth`` in a worker process (duplicates included)."""
    seed, types, depth, n = job
    np.random.seed(seed)
    random.seed(seed)
    rows = []
    for _ in range(n):
        t = types[np.random.randint(len(types))]
        progs = _worker_grammar.enumerate_one_typed_bfs(type_signature=t, depth=depth)
        if not progs.empty:
            rows.append(check_remove_memorize(progs))
    return pd.concat(rows) if rows else pd.DataFrame()


def sample_frames(pm_path, types, depth, num_frames, num_workers, seed, batch=200):
    """Sample up to num_frames unique frames (memorize-free) of the given depth.

    Workers draw frames in parallel; stops early once a whole round of batches
    yields no new frame, i.e. when the space of frames is close to exhausted.
    """
    seen = {}
    round_ = 0
    with Pool(num_workers, initializer=_init_worker, initargs=(pm_path,)) as pool:
        while len(seen) < num_frames:
            jobs = [(seed * 100003 + round_ * num_workers + w, types, depth, batch)
                    for w in range(num_workers)]
            round_ += 1
            before = len(seen)
            for frames in pool.imap_unordered(_sample_batch, jobs):
                for _, row in frames.iterrows():
                    seen.setdefault((row["term"], row["type_string"]), row)
            print(f"depth {depth}: {len(seen)} frames", flush=True)
            if len(seen) == before:
                break
    frames = pd.DataFrame(list(seen.values())).reset_index(drop=True)
    return frames.head(num_frames)


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
        help="The depths of the programs to generate (enumerate mode), e.g. --depth 1 2.",
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
        default="outputs/pcfg_frame",
        help="Base path for saving frames.",
    )
    parser.add_argument(
        "--max_enum_depth",
        type=int,
        default=1,
        help="Depths above this are sampled instead of enumerated (enumerate mode).",
    )
    parser.add_argument(
        "--num_sampled_frames",
        type=int,
        default=30000,
        help="Number of unique frames to sample per depth above --max_enum_depth.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Worker processes for sampling frames.",
    )
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()
    save_path = Path(args.save_path)
    if not save_path.is_absolute():
        save_path = REPO_ROOT / save_path

    # Load primitive model and build grammar
    # Seed programs are needed so that frames contain PM(...) slots
    pm_path = REPO_ROOT / "data" / args.task / "task_pm_with_seeds.csv"
    pm_init = pd.read_csv(pm_path, index_col=0)
    pl = Grammar(production=pm_init)

    # Input-output type signatures
    t0 = [["note"], "note"]
    t1 = [["note", "count"], "note"]
    t2 = [["note", "note"], "note"]
    t3 = [["note", "note", "note"], "note"]
    t4 = [["note", "note", "count"], "note"]
    t5 = [["note", "count", "note"], "note"]
    t6 = [["note", "count", "count"], "note"]

    types = [t0, t1, t2, t3, t4, t5, t6]
    type_strings = [Placeholder.complete_typelist_to_string(t[0], t[1]) for t in types]

    if args.frame_gen == "sample":
        np.random.seed(args.random_seed)
        depth_prob = power_law_dist(args.max_depth, alpha=1)
        src_pth = (
            save_path / "rd_curve" / f"frame_num_{args.frame_num}_{args.random_seed}"
        )
        os.makedirs(src_pth, exist_ok=True)

        for i in range(args.num_task):
            save_path_cur_task = src_pth / f"task_{i}"
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
                prog_list.to_csv(save_path_cur_task / f"index_{j}.csv")

    else:
        frame_dir = save_path / "all_frames_given_depth_and_typestring"
        frame_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)  # router choice in enumerate_one_typed_bfs
        for depth in args.depth:
            if depth > args.max_enum_depth:
                combined = sample_frames(
                    pm_path, types, depth, args.num_sampled_frames,
                    args.num_workers, args.random_seed,
                )
            else:
                rfs = []
                for t in types:
                    progs = pl.enumerate_typed_bfs(type_signature=t, depth=depth)
                    print(t)
                    rfs.append(progs)
                combined = pd.concat(rfs).reset_index(drop=True)
                combined = check_remove_memorize(combined)
            print(f"depth {depth}: {len(combined)} frames")
            combined.to_csv(
                frame_dir / f"task_frames_{depth}.csv"
            )


if __name__ == "__main__":
    main()
