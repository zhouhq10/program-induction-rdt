"""Exclusive wrong-note proportions for four pooled and six split categories."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import CATEGORIES, LABELS, N_SEG, NUM_MELODY


def run(gts, pressed, output, min_len=3):
    """Pool reuse BEFORE testing uniqueness; average participant proportions."""
    df = pd.read_csv(output / "error_notes.csv")
    df = df[df.len >= min_len]
    # total wrong notes per participant (positions up to the shorter length)
    total = {}
    for p in range(len(gts)):
        c = 0
        for i in range(NUM_MELODY):
            for j in range(N_SEG):
                t, r = np.asarray(gts[p][i][j], int), np.asarray(pressed[p][i][j], int)
                L = min(len(t), len(r))
                c += int((t[:L] != r[:L]).sum())
        total[p] = c

    key = ["participant", "melody", "segment", "note_id"]
    out = []
    for pooled in (False, True):
        d = df.copy()
        if pooled:
            d["category"] = d.category.replace({"reuse_notes": "reuse", "reuse_segment": "reuse",
                                                "reuse_melody": "reuse"})
        cats = ["vertical", "temporal", "reuse", "simpler"] if pooled else CATEGORIES
        sets = d.groupby(key).category.agg(lambda s: frozenset(s))
        rows = []
        for p in range(len(gts)):
            s = sets[sets.index.get_level_values(0) == p] if p in sets.index.get_level_values(0) else []
            counts = {c: 0 for c in cats}
            for cs in s:
                if len(cs) == 1:
                    counts[next(iter(cs))] += 1
            rows.append({"participant": p, **{c: (counts[c] / total[p] if total[p] else np.nan) for c in cats},
                         "n_wrong": total[p]})
        per = pd.DataFrame(rows)
        for c in cats:
            out.append({"split": "pooled" if pooled else "six", "category": c,
                        "mean": per[c].mean(), "sem": per[c].sem(), "n": int(per[c].notna().sum())})
        per.to_csv(output / f"proportions_per_participant_{'pooled' if pooled else 'six'}_min{min_len}.csv",
                   index=False)
    res = pd.DataFrame(out)
    res.to_csv(output / f"proportions_min{min_len}.csv", index=False)

    L = [f"unique-error proportions (windows >= {min_len} notes), mean +/- sem over "
         f"{len(gts)} participants", ""]
    for split, g in res.groupby("split", sort=False):
        L.append(split)
        for _, r in g.iterrows():
            L.append(f"  {LABELS.get(r.category, r.category):<26} {r['mean']:.3f} +/- {r['sem']:.3f}")
        L.append("")
    # how many notes carry more than one label
    sets = df.groupby(key).category.agg(lambda s: frozenset(s))
    multi = sum(len(s) > 1 for s in sets)
    L.append(f"error notes with any label: {len(sets)}; with >1 label (not counted): {multi}; "
             f"total wrong notes: {sum(total.values())}")
    txt = "\n".join(L)
    print(txt)
    (output / f"step2_summary_min{min_len}.txt").write_text(txt + "\n")
