"""Public JSON input, fixed program-table loading, and shared helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
NUM_KEYS, NUM_MELODY, N_SEG = 6, 5, 5
KEYS = ["C", "D", "E", "F", "A", "B"]
KEY_MAPPING = dict(zip("SDFJKL", range(1, 7)))
CATEGORIES = ["vertical", "temporal", "reuse_notes", "reuse_segment", "reuse_melody", "simpler"]
LABELS = {
    "vertical": "Vertical shift", "temporal": "Temporal shift", "reuse": "Reuse",
    "reuse_notes": "Reuse: previous note", "reuse_segment": "Reuse: earlier segment",
    "reuse_melody": "Reuse: earlier melody", "simpler": "Simpler program",
}


def load_dataset(human_json, stimuli_dir):
    """Match the original notebook: completed > 0, file order, cyclic shift.

    Read only stimulus indices, shifts and Phase-2 keys. Exported analysis tables
    use zero-based cohort positions; no account IDs or survey fields are copied.
    Ground truth includes six segments, responses only the five recall segments.
    """
    entries = json.loads(Path(human_json).read_text())
    if not isinstance(entries, list):
        raise ValueError("Expected chirp.json to contain a list of session records")
    cohort = [e for e in entries if int(e["completed"]) > 0]
    if not cohort:
        raise ValueError("No completed sessions")
    targets, responses = [], []
    for p, entry in enumerate(cohort):
        exp = json.loads(entry["experimentData"])
        if len(exp["sampled_indices"]) != NUM_MELODY or len(exp["shift"]) != NUM_MELODY:
            raise ValueError(f"Participant {p}: expected five melodies and shifts")
        gt, recall = [], []
        for i, index in enumerate(exp["sampled_indices"]):
            segments, recalled = [], []
            for j in range(N_SEG + 1):
                path = Path(stimuli_dir) / f"{index}-{j}.json"
                values = np.asarray(json.loads(path.read_text()))
                if (values.ndim != 1 or not len(values) or
                        not np.isin(values, range(1, NUM_KEYS + 1)).all()):
                    raise ValueError(f"Invalid stimulus: {path}")
                target = (values.astype(int) + int(exp["shift"][i]) - 1) % NUM_KEYS + 1
                segments.append(target)
                if j < N_SEG:
                    keys = exp["roundKeyPhase"][f"{i}-{j}"]["phase2"]
                    try:
                        response = np.asarray([KEY_MAPPING[k] for k in keys], dtype=int)
                    except (KeyError, TypeError) as error:
                        raise ValueError(f"Invalid Phase-2 keys at {p}/{i}/{j}") from error
                    recalled.append(response)
            gt.append(segments)
            recall.append(recalled)
        targets.append(gt)
        responses.append(recall)
    return targets, responses


def load_programs(path):
    """Best exact program by per-note log probability; retain first on ties.

    Use a generated table including frame/combinator and primitive/literal costs.
    """
    df = pd.read_csv(path, index_col=0)
    required = {"recon", "log_prob", "term", "prog_lp", "frame_lp"}
    if not required.issubset(df.columns):
        raise ValueError(f"Program table requires columns {sorted(required)}")
    if not np.isfinite(df[["log_prob", "prog_lp", "frame_lp"]].to_numpy()).all():
        raise ValueError("Program score components must be finite")
    if not np.allclose(df["log_prob"], df["prog_lp"] + df["frame_lp"], rtol=0, atol=1e-12):
        raise ValueError("Program log_prob must equal prog_lp + frame_lp")
    best = {}
    for recon, lp, term in zip(df["recon"], df["log_prob"], df["term"]):
        key = tuple(int(x) for x in recon.strip("[]").split())
        if not key or any(x not in range(1, NUM_KEYS + 1) for x in key) or not np.isfinite(lp):
            raise ValueError("Invalid reconstruction or log probability in program table")
        v = lp / len(key)
        if key not in best or v > best[key][0]:
            best[key] = (v, term)
    return best


def ngrams(segs, n: int) -> set:
    out = set()
    for s in segs:
        a = np.asarray(s, int)
        for i in range(len(a) - n + 1):
            out.add(tuple(a[i:i + n].tolist()))
    return out
