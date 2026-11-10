"""Window classification; predicates preserved from the figure-generating analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import CATEGORIES, N_SEG, NUM_KEYS, NUM_MELODY, ngrams


def windows_with_error(resp, target, min_len=3, max_len=12):
    L = min(len(resp), len(target))
    wrong = resp[:L] != target[:L]
    for n in range(min_len, min(max_len or L, L) + 1):
        for s0 in range(L - n + 1):
            if wrong[s0:s0 + n].any():
                yield s0, n, np.flatnonzero(wrong[s0:s0 + n]) + s0


def first_occurrence(win, segs):
    """(segment index, offset) of the first verbatim occurrence of win."""
    n = len(win)
    for k, s in enumerate(segs):
        a = np.asarray(s, int)
        for i in range(len(a) - n + 1):
            if np.array_equal(a[i:i + n], win):
                return k, i
    return None


def classify_trial(p, i, j, gts, pressed, progs, min_len=3, max_len=12):
    target = np.asarray(gts[p][i][j], int)
    resp = np.asarray(pressed[p][i][j], int)
    L = min(len(resp), len(target))
    if L == 0 or np.array_equal(resp[:L], target[:L]):
        return []
    cur_ng = {n: ngrams([target], n) for n in range(min_len, min(max_len or L, L) + 1)}
    within = [np.asarray(s, int) for s in gts[p][i][:j]]
    across = [(ii, k, np.asarray(gts[p][ii][k], int))
              for ii in range(i) for k in range(N_SEG)]
    rows = []

    def add(cat, s0, n, errs, **extra):
        for e in errs:
            rows.append(dict(participant=p, melody=i, segment=j, note_id=int(e),
                             category=cat, start=s0, len=n, **extra))

    for s0, n, errs in windows_with_error(resp, target, min_len, max_len):
        w, t = resp[s0:s0 + n], target[s0:s0 + n]
        tup = tuple(w.tolist())

        # vertical shift: constant non-zero cyclic offset
        d = (w - t) % NUM_KEYS
        if d[0] != 0 and np.all(d == d[0]):
            add("vertical", s0, n, errs, offset=int(d[0]))

        # temporal shift: same notes elsewhere in the target
        for k in range(len(target) - n + 1):
            if k != s0 and np.array_equal(target[k:k + n], w):
                add("temporal", s0, n, errs, lag=int(s0 - k))
                break

        # run repetition of the preceding recalled note
        if s0 > 0 and np.all(w == resp[s0 - 1]):
            add("reuse_notes", s0, n, errs, note=int(w[0]))

        # reuse from earlier material, temporal displacement excluded
        if tup not in cur_ng[n]:
            hit = first_occurrence(w, within)
            if hit is not None:
                add("reuse_segment", s0, n, errs, src_melody=i, src_segment=hit[0],
                    src_offset=hit[1])
            else:
                for ii, k, s in across:
                    a = s
                    found = None
                    for off in range(len(a) - n + 1):
                        if np.array_equal(a[off:off + n], w):
                            found = off
                            break
                    if found is not None:
                        add("reuse_melody", s0, n, errs, src_melody=ii, src_segment=k,
                            src_offset=found)
                        break

        # simpler program: exact depth-2 program with higher per-note log prob
        vr, vt = progs.get(tup), progs.get(tuple(t.tolist()))
        if len(set(tup)) > 1 and vr is not None and vt is not None and vr[0] > vt[0]:
            add("simpler", s0, n, errs, gain=float(vr[0] - vt[0]),
                prog_recall=vr[1], prog_target=vt[1])
    return rows


def run(gts, pressed, progs, output, min_len=3, max_len=12):
    """Write one row per wrong note, matching window and category."""
    rows = []
    for p in range(len(gts)):
        for i in range(NUM_MELODY):
            for j in range(N_SEG):
                rows.extend(classify_trial(p, i, j, gts, pressed, progs, min_len, max_len))
        print(f"Classified participant {p + 1}/{len(gts)}", flush=True)
    columns = ["participant", "melody", "segment", "note_id", "category", "start", "len",
               "offset", "lag", "note", "src_melody", "src_segment", "src_offset",
               "gain", "prog_recall", "prog_target"]
    pd.DataFrame(rows, columns=columns).to_csv(output / "error_notes.csv", index=False)
