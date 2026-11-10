"""Generate the program table, classify recall errors, and draw the bar plots."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import HERE, REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "build", "classify", "summarize", "figures"], default="all")
    parser.add_argument("--human-json", type=Path, default=REPO_ROOT / "data/human/chirp.json")
    parser.add_argument("--stimuli-dir", type=Path, default=REPO_ROOT / "docs/chirp-demo/training")
    parser.add_argument("--program-table", type=Path, help="Use an existing table; otherwise generate one during the full run")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=12, help="Maximum window length; 0 scans the full aligned segment")
    parser.add_argument("--font", default="DejaVu Sans", help="Use Verdana to match the paper font if installed")
    args = parser.parse_args()
    if args.min_len < 3 or args.max_len < 0 or (args.max_len and args.max_len < args.min_len):
        parser.error("Require min-len >= 3 and max-len = 0 or >= min-len")
    args.output.mkdir(parents=True, exist_ok=True)
    table = args.program_table if args.program_table is not None else args.output / "filled_prog_2.csv"

    if args.stage == "build" or (args.stage == "all" and args.program_table is None):
        from .build_program_table import run
        run(table)
    if args.stage == "build":
        return

    gts = pressed = None
    if args.stage in ("all", "classify", "summarize"):
        from .common import load_dataset
        gts, pressed = load_dataset(args.human_json, args.stimuli_dir)
        print(f"Loaded {len(gts)} completed participants.")
    if args.stage in ("all", "classify"):
        if not table.is_file():
            parser.error(f"Missing program table: {table}. Run --stage build first.")
        from .common import load_programs
        from .classify import run
        run(gts, pressed, load_programs(table), args.output, args.min_len, args.max_len)
    if args.stage in ("all", "summarize"):
        from .summarize import run
        run(gts, pressed, args.output, args.min_len)
    if args.stage in ("all", "figures"):
        from .figures import run
        run(args.output, args.min_len, args.font)


if __name__ == "__main__":
    main()
