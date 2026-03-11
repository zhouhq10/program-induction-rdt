import sys

sys.path.append("../")


import re
import os
import pickle
import random
import numpy as np
import pandas as pd
from random import sample
from multiprocessing import Pool


from src.domain.melody.melody_primitive import *
from src.program.grammar import *
from src.domain.melody.greedy_dp_compressor import (
    GreedyDP_PCFGCompressor,
    GreedyDP_AGCompressor,
    GreedyDP_HAGCompressor,
)

import argparse

default_params = {
    "lib_size": 10000,
    "local_lib_size": 10000,
    "global_alpha": 1.0,
    "global_d": 0.2,
    "local_alpha": 1.0,
    "local_d": 0.2,
    "lossless": 0,
    "mem": 1,
    "frame_gen": "fly",
    "frame_num": 20,
    "frame_depth": 5,
    "reuse_num_note_subtask": 12,
    "max_melody_num": 0,
    "max_submelody_length": 1,
    "lossless_error": False,
}


def constrain_iter_sample(
    pl: Grammar,
    test_task: list,
    compressor: GreedyDP_PCFGCompressor,
    init_pm: pd.DataFrame,
    save_path: str,
    args: argparse.Namespace,
) -> None:

    best_dict = None
    best_rd = 1e6
    for _ in range(args.iter_num_per_task):
        compressor.init_pm = init_pm
        compressor.lib.production = init_pm
        prog_trajs = compressor.run(test_task, save_path=False)
        distortions = prog_trajs["distortion"].sum()
        rates = -prog_trajs["log_prob"].sum()
        value = distortions + args.beta * rates
        if value < best_rd:
            best_rd = value
            best_dict = prog_trajs
    save_dict = {"prog_trajs": best_dict}
    compressor.save_result_per_task(args.task_start_ind, save_dict, save_path)


def constrain_iter_sample_hag(
    test_task: list,
    init_pm: pd.DataFrame,
    init_global_table: dict,
    init_global_lib: pd.DataFrame,
    save_path: str,
    args: argparse.Namespace,
) -> None:

    for _ in range(
        args.task_start_ind, args.task_start_ind + args.task_num_out_compressor
    ):
        best_dict = None
        best_rd = 1e6

        for _ in range(args.iter_num_per_task):
            # Initialize the grammar
            pl = HierAdaGrammar(
                production=init_pm,
                lib_size=args.lib_size,
                local_lib_size=args.local_lib_size,
                global_alpha=args.global_alpha,
                global_d=args.global_d,
                local_alpha=args.local_alpha,
                local_d=args.local_d,
            )
            pl.history_table = init_global_table
            pl.global_production = init_global_lib
            pl.init_pm = init_pm
            compressor = GreedyDP_HAGCompressor(program_lib=pl, args=args)

            # Run the compressor
            prog_trajs, loc_lib, global_lib, global_table = compressor.run(
                test_task, save_path=False
            )
            distortions = prog_trajs["distortion"].sum()
            rates = -prog_trajs["log_prob"].sum()
            value = distortions + args.beta * rates
            if value < best_rd:
                best_rd = value
                best_dict = prog_trajs

        save_dict = {"prog_trajs": best_dict}
        compressor.save_result_per_task(args.task_start_ind, save_dict, save_path)
        args.task_start_ind += 1


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--explog_base_path", type=str, help="The base path to the results."
    )
    parser.add_argument(
        "--explog_name",
        type=str,
        default="beta_rdt_greedy_fly",
        help="The name of the experiment.",
    )
    parser.add_argument(
        "--explog_params",
        type=str,
        default="beta_0.1_random_seed_0_random_round_0",
        help="The parameters of the experiment.",
    )
    parser.add_argument(
        "--explog_task_progress",
        type=int,
        default=49,
        help="The task progress to evaluate.",
    )

    parser.add_argument(
        "--curriculum",
        type=str,
        default="pcfg",
        help="curriculum type, choose from ['pcfg','count_ag','hier_ag']",
    )

    # Random param search
    parser.add_argument(
        "--iter_num_per_task",
        type=int,
        default=10,
        help="The number of iterations per task.",
    )
    parser.add_argument(
        "--task_start_ind", type=int, default=0, help="The start index of the task."
    )
    parser.add_argument("--task_num", type=int, default=1, help="The number of tasks.")
    parser.add_argument(
        "--task_num_out_compressor",
        type=int,
        default=50,
        help="The number of tasks to output from the compressor.",
    )
    parser.add_argument(
        "--random_seed", type=int, default=0, help="The random seed for sampling."
    )
    parser.add_argument(
        "--random_seq", type=int, default=0, help="The random sequence for sampling."
    )

    args = parser.parse_args()
    print(args, flush=True)

    # Prep data
    base_path = "../data/melody"

    # Initialize task data
    task_data = load_pickle("../data/task/evaluation_notes_wo_break_52.obj")
    task_data = [np.array(x) for x in task_data]
    # Set a random seed for reproducibility
    random.seed(args.random_seed)
    task_data = random.sample(task_data, 50)[:50]
    # Shuffle the list
    random.seed(args.random_seq)
    random.shuffle(task_data)

    # ----- Save options
    # Extract model and experiment name
    folder_name = args.explog_params
    pattern = r"([a-zA-Z]+(?:_[a-zA-Z]+)*)_([\d.]+)"
    matches = re.findall(pattern, folder_name)
    params = {
        param: float(value) if "." in value else int(value) for param, value in matches
    }

    # Add params to args
    for key, value in params.items():
        setattr(args, key, value)
    for key, value in default_params.items():
        if key not in args:
            setattr(args, key, value)

    # Evaluate paths
    train_path = f"{args.explog_base_path}/{args.explog_name}/{folder_name}"
    save_path = f"{train_path}/eval_iter_{args.iter_num_per_task}_task_progress_{args.explog_task_progress}"
    os.makedirs(save_path, exist_ok=True)

    # Load the initial production model
    if "pcfg" in args.explog_base_path:
        # Prep frame and primitive data
        init_pm = pd.read_csv(
            os.path.join(base_path, "task_pm.csv"), index_col=0, na_filter=False
        )
        pl = Grammar(production=init_pm)
        compressor = GreedyDP_PCFGCompressor(pl, args)
        constrain_iter_sample(
            pl=pl,
            compressor=compressor,
            init_pm=init_pm,
            test_task=task_data,
            save_path=save_path,
            args=args,
        )
    elif "hag" in args.explog_base_path:
        global_lib_path = (
            f"{train_path}/task_{args.explog_task_progress}_global_lib_list.obj"
        )
        global_lib = load_pickle(global_lib_path)[-1]
        global_table_path = (
            f"{train_path}/task_{args.explog_task_progress}_global_table_list.obj"
        )
        global_table = load_pickle(global_table_path)[-1]
        init_pm = pd.read_csv(
            os.path.join(base_path, "task_pm.csv"), index_col=0, na_filter=False
        )
        constrain_iter_sample_hag(
            init_pm=init_pm,
            init_global_table=global_table,
            init_global_lib=global_lib,
            test_task=task_data,
            save_path=save_path,
            args=args,
        )
    elif "ag" in args.explog_base_path:
        global_lib_path = (
            f"{train_path}/task_{args.explog_task_progress}_global_lib_list.obj"
        )
        global_lib = load_pickle(global_lib_path)[-1]
        pl = AdaGrammar(
            production=global_lib,
            lib_size=args.lib_size,
            global_alpha=args.global_alpha,
            global_d=args.global_d,
        )
        compressor = GreedyDP_AGCompressor(program_lib=pl, args=args)
        constrain_iter_sample(
            pl=pl,
            compressor=compressor,
            init_pm=global_lib,
            test_task=task_data,
            save_path=save_path,
            args=args,
        )
    else:
        raise ValueError(f"Invalid curriculum type")


if __name__ == "__main__":
    main()
