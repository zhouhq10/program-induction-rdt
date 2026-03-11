import sys

sys.path.append("..")

import pandas as pd
import pickle, argparse

from src.program.primitive import *
from src.program.grammar import Grammar
from src.domain.melody.melody_primitive import melody_primitive_list


def main():
    parser = argparse.ArgumentParser(
        description="Process input and output file paths for PM tasks."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="melody",
        help="Task name.",
    )
    parser.add_argument(
        "--output_pickle_path",
        type=str,
        default="../data/{}/task_pm.obj",
        help="Path for the output pickle file.",
    )
    args = parser.parse_args()

    # ----- Initial primitives set up -----
    pm_terms = []

    # Add base primitives
    for i in range(1, 7):
        note_var_name = f"note_{i}"
        if note_var_name in globals():
            pm_terms.append(globals()[note_var_name])

    for i in range(1, 7):
        count_var_name = f"count_{i}"
        if count_var_name in globals():
            pm_terms.append(globals()[count_var_name])

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

    # ----- Compute priors over primitives -----
    # Compute priors over primitives, now everything is uniform given paired input-output types
    grammar = Grammar(production=pm_task)
    grammar.production = grammar.prior_uniform_per_type()

    # Compute adaptor priors over programs
    # Since we are initializing the priors, we do not consider AG here (count-based)
    grammar.production["adaptor_lp"] = 0
    grammar.production = grammar.update_overall_lp()

    # ----- Save the updated pm_task -----
    output_path = args.output_pickle_path.format(args.task)
    with open(output_path, "wb") as f:
        pickle.dump(pm_task, f)

    # For visualization
    pm_task[
        ["term", "arg_type", "ret_type", "type_string", "ctype", "is_init", "count"]
    ].to_csv(output_path.replace(".obj", ".csv"))


if __name__ == "__main__":
    main()
