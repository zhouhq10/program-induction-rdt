"""
CLI: melody compressions -> occurrence matrix -> PID -> synergy-based curriculum.

    python synergy_curriculum/run_synergy_curriculum.py \
        --result_dir /path/to/simulation_result/hag/greedy/.../ \
        --lib_size 12 --outcome accuracy --n_bins 2 \
        --out curriculum.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append("..")
sys.path.append(str(Path(__file__).resolve().parents[1]))

from synergy_curriculum.curriculum import find_curriculum
from synergy_curriculum.occurrence import build_dataset, load_melody_programs
from synergy_curriculum.ordering import order_tasks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find the synergy-maximising program library for a run."
    )
    # Input
    p.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Directory holding task_*_prog_trajs.obj or task_*_task_progs.obj",
    )
    p.add_argument("--n_melodies", type=int, default=None, help="Cap melodies loaded")
    p.add_argument("--task_start", type=int, default=0)

    # Occurrence matrix
    p.add_argument(
        "--min_args",
        type=int,
        default=2,
        help="Drop subprograms with fewer than this many commas",
    )
    p.add_argument(
        "--outcome",
        type=str,
        default="accuracy",
        choices=["accuracy", "distortion", "rate", "rd_cost"],
    )
    p.add_argument("--beta", type=float, default=1.0, help="Only for --outcome rd_cost")
    p.add_argument("--n_bins", type=int, default=2, help="Bins for the outcome")
    p.add_argument(
        "--bin_strategy", type=str, default="quantile", choices=["quantile", "uniform"]
    )

    # Search
    p.add_argument("--method", type=str, default="greedy", choices=["greedy", "random"])
    p.add_argument("--lib_size", type=int, default=12, help="Curriculum size")
    p.add_argument("--n_iter", type=int, default=500, help="random search only")
    p.add_argument("--restarts", type=int, default=1, help="greedy only")
    p.add_argument("--max_pairs", type=int, default=20000, help="greedy seeding budget")
    p.add_argument("--seed", type=int, default=0)

    # Debiasing
    p.add_argument(
        "--debiased",
        action="store_true",
        help="Report measures with the shuffled-surrogate mean subtracted",
    )
    p.add_argument("--n_surrogates", type=int, default=50)

    # Task ordering
    p.add_argument(
        "--order",
        type=str,
        default=None,
        choices=["incremental", "coverage"],
        help="Also derive a melody presentation order from the selected library",
    )
    p.add_argument(
        "--order_descending",
        action="store_true",
        help="For --order coverage: most complex melodies first",
    )

    # Output
    p.add_argument("--out", type=str, default=None, help="CSV path for the curriculum")
    p.add_argument(
        "--order_out", type=str, default=None, help="CSV path for the task ordering"
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    verbose = not args.quiet

    # ----- 1. Load melody compressions
    melodies = load_melody_programs(
        args.result_dir, n_melodies=args.n_melodies, task_start=args.task_start
    )
    print(f"Loaded {len(melodies)} melodies from {args.result_dir}")

    # ----- 2. Occurrence matrix + binned outcome
    data = build_dataset(
        melodies,
        outcome=args.outcome,
        n_bins=args.n_bins,
        min_args=args.min_args,
        beta=args.beta,
        strategy=args.bin_strategy,
    )
    X, y, names = data["X"], data["y"], data["names"]
    print(f"Occurrence matrix: {X.shape[0]} melodies x {X.shape[1]} subprograms")
    print(f"  ({data['n_merged']} nested duplicates merged into their outer term)")
    print(
        f"Outcome '{args.outcome}': "
        f"mean {data['y_continuous'].mean():.4f}, "
        f"label counts {np.bincount(y).tolist()}"
    )

    # ----- 3. Search for the synergy-maximising library
    search_kwargs = dict(
        lib_size=args.lib_size,
        seed=args.seed,
        debiased=args.debiased,
        n_surrogates=args.n_surrogates,
        verbose=verbose,
    )
    if args.method == "greedy":
        search_kwargs.update(restarts=args.restarts, max_pairs=args.max_pairs)
    else:
        search_kwargs.update(n_iter=args.n_iter)

    print(f"\nSearching ({args.method}) for a size-{args.lib_size} library ...")
    result = find_curriculum(X, y, names, method=args.method, **search_kwargs)

    # ----- 4. Report
    print()
    print(result)
    print(f"\nSample adequacy: {result.adequacy}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(out))
        print(f"\nWrote curriculum to {out}")

    # ----- 5. Optional: turn the library into a task ordering
    if args.order:
        ordering = order_tasks(
            X,
            names,
            result.indices,
            method=args.order,
            descending=args.order_descending,
            outcome=data["y_continuous"],
            outcome_name=args.outcome,
            seed=args.seed,
        )
        print()
        print(ordering)

        if args.order_out:
            order_out = Path(args.order_out)
            order_out.parent.mkdir(parents=True, exist_ok=True)
            ordering.save(str(order_out))
            print(f"\nWrote task ordering to {order_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
