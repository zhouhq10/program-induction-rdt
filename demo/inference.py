"""
inference.py — Run the greedy DP compressor on user-provided melodies.

Thin wrapper around ``src.domain.melody.greedy_dp_compressor`` used by the
demo.  The input is one melody split into segments, processed in order as in
``run_submelody_human``: HAG updates its local library after each segment, and
AG/HAG update the global library once the whole melody is done.
"""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.program.grammar import Grammar, AdaGrammar, HierAdaGrammar
from src.domain.melody.melody_utils import set_num_notes
from src.domain.melody.greedy_dp_compressor import (
    GreedyDP_PCFGCompressor,
    GreedyDP_AGCompressor,
    GreedyDP_HAGCompressor,
)

# In the built site the table sits next to this file; in the repo it is under data/primitive
_HERE = Path(__file__).resolve().parent
PM_PATH = next(
    p
    for p in [_HERE / "data" / "task_pm.csv",
              _HERE.parent / "data" / "primitive" / "task_pm.csv"]
    if p.exists()
)
CURRICULA = {"PCFG": "pcfg", "AG": "count_ag", "HAG": "hier_ag"}


def parse_segments(text: str, max_len: int = 40, num_notes: int = 6) -> list:
    """Parse one segment per line, notes in {1,…,num_notes} separated by spaces or commas."""
    melodies = []
    for line in text.strip().splitlines():
        notes = line.replace(",", " ").split()
        if not notes:
            continue
        melody = np.array([int(n) for n in notes])
        if not ((melody >= 1) & (melody <= num_notes)).all():
            raise ValueError(f"Notes must be in 1–{num_notes}: {line}")
        if len(melody) > max_len:
            raise ValueError(f"Segments can have at most {max_len} notes: {line}")
        melodies.append(melody)
    if not melodies:
        raise ValueError("Enter at least one segment.")
    return melodies


def initial_pm(num_notes: int = 6) -> pd.DataFrame:
    """The primitive table, with note_1 … note_{num_notes} as base notes."""
    pm = pd.read_csv(PM_PATH, index_col=0, na_filter=False)
    is_note = pm["term"].str.fullmatch(r"note_\d+")
    if is_note.sum() == num_notes:
        return pm
    template = pm[is_note].iloc[0]
    notes = pd.DataFrame([dict(template, term=f"note_{i}")
                          for i in range(1, num_notes + 1)])
    notes["comp_lp"] = notes["log_prob"] = np.log(1 / num_notes)
    return pd.concat([notes, pm[~is_note]], ignore_index=True)


def build_compressor(model: str, beta: float, search_budget: int,
                     frame_depth: int, submelody_backtrack_budget: int,
                     global_alpha: float = 1.0, global_d: float = 0.2,
                     local_alpha: float = 1.0, local_d: float = 0.2,
                     lossless: bool = False, selection: str = "best",
                     num_notes: int = 6):
    curriculum = CURRICULA[model]
    args = SimpleNamespace(
        curriculum=curriculum, beta=beta, search_budget=search_budget,
        frame_depth=frame_depth, frame_gen="fly",
        submelody_backtrack_budget=submelody_backtrack_budget,
        melody_backtrack_budget=0, reuse_num_note_subtask=12,
        lossless=int(lossless), lossless_error=False, mem=1,
        task_start_ind=0, task_num=None,
        lib_size=10000, local_lib_size=10000,
        global_alpha=global_alpha, global_d=global_d,
        local_alpha=local_alpha, local_d=local_d,
        local_pattern=0, selection=selection,
    )
    init_pm = initial_pm(num_notes)
    if curriculum == "pcfg":
        return GreedyDP_PCFGCompressor(Grammar(production=init_pm), args)
    if curriculum == "count_ag":
        lib = AdaGrammar(init_pm, lib_size=args.lib_size,
                         global_alpha=args.global_alpha, global_d=args.global_d)
        return GreedyDP_AGCompressor(lib, args)
    lib = HierAdaGrammar(
        init_pm, args.lib_size, local_lib_size=args.local_lib_size,
        global_alpha=args.global_alpha, global_d=args.global_d,
        local_alpha=args.local_alpha, local_d=args.local_d,
        local_pattern=args.local_pattern,
    )
    return GreedyDP_HAGCompressor(lib, args)


def library_table(production: pd.DataFrame) -> pd.DataFrame:
    """Learned programs in the library, most used first."""
    lib = production[production["ctype"] == "program"]
    if lib.empty:
        return pd.DataFrame(columns=["program", "type", "count", "log_prob"])
    lib = lib.sort_values(["count", "log_prob"], ascending=False)
    return pd.DataFrame({
        "program": lib["term"].values,
        "type": lib["type_string"].values,
        "count": lib["count"].astype(int).values,
        "log_prob": lib["log_prob"].astype(float).round(2).values,
    })


def program_table(progs: pd.DataFrame) -> pd.DataFrame:
    """Selected sub-programs of one melody, in order."""
    return pd.DataFrame({
        "program": progs["term"].values,
        "output": [" ".join(str(int(n)) for n in r) for r in progs["recon"]],
        "distortion": progs["distortion"].astype(int).values,
        "log_prob": progs["log_prob"].astype(float).round(2).values,
    })


def iter_infer(segments, model="HAG", beta=1.0, search_budget=5,
               frame_depth=2, submelody_backtrack_budget=2, seed=0,
               global_alpha=1.0, global_d=0.2, local_alpha=1.0, local_d=0.2,
               lossless=False, selection="best", num_notes=6):
    """Compress the segments of one melody in order, yielding after each segment.

    ``library`` is the library used for the next segment (HAG: local, AG: global).
    The last result also has ``global_library``, the global library after the
    melody (AG/HAG).
    """
    # The worker reuses one interpreter for all runs, so set this every time
    set_num_notes(num_notes)
    np.random.seed(seed)
    random.seed(seed)
    comp = build_compressor(model, beta, search_budget, frame_depth,
                            submelody_backtrack_budget, global_alpha, global_d,
                            local_alpha, local_d, lossless, selection, num_notes)
    if model == "HAG":
        comp.lib.production = comp.init_pm
        comp.lib.global_production = comp.init_pm
        comp.lib.history_table = {t: 0 for t in comp.lib.type_strings}

    all_progs = []
    for n, segment in enumerate(segments):
        if model == "PCFG":
            rate, distortion, progs = comp.run_per_task(segment)
        else:
            rate, distortion, progs = GreedyDP_AGCompressor.run_inner_dp_per_task(
                comp, segment
            )
        if model == "HAG":
            for i in range(len(progs)):
                comp.lib.update_local_lib(progs.iloc[[i]])
        all_progs.append(progs)

        result = {
            "index": n,
            "melody": segment,
            "programs": program_table(progs),
            "rate": float(rate),
            "distortion": float(distortion),
            "library": None if model == "PCFG" else library_table(comp.lib.production),
            "global_library": None,
        }

        # The global library is updated once the whole melody is done
        if n == len(segments) - 1 and model != "PCFG":
            for progs in all_progs:
                for i in range(len(progs)):
                    if model == "HAG":
                        comp.lib.update_post_lib(progs.iloc[[i]], comp.lib.production)
                    else:
                        comp.lib.update_post_lib(progs.iloc[[i]])
            glob = comp.lib.global_production if model == "HAG" else comp.lib.production
            result["global_library"] = library_table(glob)

        yield result


def result_to_json(result: dict) -> str:
    """Serialise one result of :func:`iter_infer` for the browser."""
    return json.dumps({
        "index": result["index"],
        "melody": [int(n) for n in result["melody"]],
        "programs": result["programs"].to_dict(orient="records"),
        "rate": result["rate"],
        "distortion": result["distortion"],
        "library": _records(result["library"]),
        "global_library": _records(result["global_library"]),
    }, default=lambda x: x.item())


def _records(df):
    return None if df is None else df.to_dict(orient="records")
