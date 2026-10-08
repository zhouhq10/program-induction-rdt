import sys

sys.path.append("..")

import pandas as pd
import pickle, argparse
from pathlib import Path

from src.program.primitive import *
from src.program.grammar import Grammar
from src.domain.melody.melody_primitive import melody_primitive_list, global_melody_pms

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="Process input and output file paths for PM tasks."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="primitive",
        help="Task name.",
    )
    parser.add_argument(
        "--output_pickle_path",
        type=str,
        default="data/{}/task_pm.obj",
        help="Path for the output pickle file.",
    )
    args = parser.parse_args()

    # ----- Initial primitives set up -----
    pm_terms = []

    # Add base primitives
    for i in range(1, 7):
        pm_terms.append(global_melody_pms[f"note_{i}"])

    for i in range(1, 7):
        pm_terms.append(global_melody_pms[f"count_{i}"])

    # Add function-level primitive
    pm_terms = pm_terms + melody_primitive_list

    # Add program-level primitive
    note_list, program_list = [], []
    for i in range(1, 7):
        note_list.append(f"note_{i}")

    for note in note_list:
        program_list.append(
            {
                "term": f"[B,I,{note}]",
                "arg_type": "note",
                "ret_type": "note",
                "type_string": "note->note",
                "ctype": "program",
            }
        )
        program_list.append(
            {
                "term": f"[BK,I,{note}]",
                "arg_type": "note_count",
                "ret_type": "note",
                "type_string": "note_count->note",
                "ctype": "program",
            }
        )
        program_list.append(
            {
                "term": f"[BK,I,{note}]",
                "arg_type": "note_note",
                "ret_type": "note",
                "type_string": "note_note->note",
                "ctype": "program",
            }
        )
        program_list.append(
            {
                "term": f"[BKK,I,{note}]",
                "arg_type": "note_count_count",
                "ret_type": "note",
                "type_string": "note_count_count->note",
                "ctype": "program",
            }
        )

    pm_terms = pm_terms + program_list

    # ----- Initial primitives and argument types set up -----
    pm_setup = []

    for pt in pm_terms:
        if isinstance(pt, dict):
            term = pt["term"]
            ctype = pt["ctype"]
            arg_type = pt["arg_type"]
            ret_type = pt["ret_type"]
            type_string = pt["type_string"]
        elif pt.ctype == "primitive":
            term = pt.name
            ctype = pt.ctype
            arg_type = pt.arg_type
            ret_type = pt.ret_type
            type_string = pt.type_string
        else:
            # base term
            term = pt.name
            ctype = "base_term"
            arg_type = ""
            ret_type = pt.ctype
            type_string = pt.type_string

        pm_setup.append(
            {
                "term": term,
                "arg_type": arg_type,
                "ret_type": ret_type,
                "type_string": type_string,
                "ctype": ctype,
                "count": 1,
            }
        )

    # ----- Construct and save primitive dataframe -----
    pm_task = pd.DataFrame.from_records(pm_setup).reset_index(drop=1)
    pm_task["is_init"] = int(1)
    pm_task["frame"] = ""
    pm_task["depth"] = 0

    # ----- Compute priors over primitives -----
    # Compute priors over primitives, now everything is uniform given paired input-output types
    # prior_uniform_per_type only returns base terms and primitives, so seed programs are added back
    grammar = Grammar(production=pm_task)
    seed_progs = pm_task[pm_task["ctype"] == "program"].assign(comp_lp=0.0)
    grammar.production = pd.concat(
        [grammar.prior_uniform_per_type(), seed_progs], ignore_index=True
    )

    # Compute adaptor priors over programs
    # Since we are initializing the priors, we do not consider AG here (count-based)
    grammar.production["adaptor_lp"] = 0
    grammar.production = grammar.update_overall_lp()
    pm_task = grammar.production

    # ----- Save the updated pm_task -----
    output_path = Path(args.output_pickle_path.format(args.task))
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # The initial library for compression has no programs; seed programs are
    # only used by 1_construct_frame.py so that frames contain PM(...) slots
    pm_lib = pm_task[pm_task["ctype"] != "program"].reset_index(drop=True)
    with open(output_path, "wb") as f:
        pickle.dump(pm_lib, f)
    pm_lib.to_csv(output_path.with_suffix(".csv"))
    pm_task.to_csv(output_path.with_name(output_path.stem + "_with_seeds.csv"))


if __name__ == "__main__":
    main()
