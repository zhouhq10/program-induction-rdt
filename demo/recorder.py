"""
recorder.py — Precompute the "Bad recorder" tab.

Compresses each preset tune once per player (β) with seven notes and writes
demo/recorder.json, so the presets play back instantly.  Tunes typed in the
browser are compressed live by the worker with the same parameters.

    python demo/recorder.py
"""

import json
import re
import sys
import warnings
from pathlib import Path

DEMO = Path(__file__).resolve().parent
sys.path[:0] = [str(DEMO.parent), str(DEMO)]
warnings.filterwarnings("ignore")

from inference import iter_infer, parse_segments, result_to_json  # noqa: E402

NOTE_NAMES = "CDEFGAB"

# β 0.3–0.7 keeps the tune and slips a few notes, which is what a bad recorder
# sounds like.  At β ≥ 1 cheap scales beat exact recall (the value is averaged
# per note), so the last player shows the model pushed past that point.
PLAYERS = [
    {"name": "The music student", "desc": "The odd slip", "beta": 0.3},
    {"name": "The school recorder", "desc": "Some wrong notes", "beta": 0.5},
    {"name": "From memory, twenty years later", "desc": "Keeps the tune, loses notes", "beta": 0.7},
    {"name": "Pushed too far", "desc": "Short programs win over the tune: mostly scales, "
                                      "though repeated phrases still repeat", "beta": 3.0},
]

# Fixed settings for every player; only β changes
PARAMS = {
    "model": "HAG", "search_budget": 5, "frame_depth": 2,
    "submelody_backtrack_budget": 2, "seed": 0, "global_alpha": 1.0,
    "global_d": 0.2, "local_alpha": 1.0, "local_d": 0.2, "lossless": False,
    "selection": "best", "num_notes": 7,
}

# Public-domain tunes, one phrase per line.  ' and , move a note an octave up
# or down; they only change playback, the model sees the seven note names.
TUNES = [
    {"name": "Twinkle Twinkle", "text": "C C G G A A G\nF F E E D D C\nG G F F E E D\n"
                                        "G G F F E E D\nC C G G A A G\nF F E E D D C"},
    {"name": "Frère Jacques", "text": "C D E C C D E C\nE F G E F G\n"
                                      "G A G F E C G A G F E C\nC G, C C G, C"},
    {"name": "Ode to Joy", "text": "E E F G G F E D\nC C D E E D D\n"
                                   "E E F G G F E D\nC C D E D C C"},
    {"name": "Mary Had a Little Lamb", "text": "E D C D E E E\nD D D E G G\n"
                                               "E D C D E E E\nE D D E D C"},
]


def pitch_classes(text: str) -> str:
    """Tune text → one line of note numbers 1–7 per phrase (as app.js parses it)."""
    lines = []
    for line in text.strip().splitlines():
        notes = [t.rstrip("',") for t in line.split()]
        lines.append(" ".join(n if n.isdigit() else str(NOTE_NAMES.index(n.upper()) + 1)
                              for n in notes))
    return "\n".join(lines)


def cache_key(pcs: str, beta: float) -> str:
    """Must match recCacheKey in app.js."""
    return re.sub(r"\n", "|", pcs) + "@" + format(beta, "g")


def main() -> None:
    cache = {}
    for tune in TUNES:
        pcs = pitch_classes(tune["text"])
        segments = parse_segments(pcs, num_notes=PARAMS["num_notes"])
        for player in PLAYERS:
            params = dict(PARAMS, beta=player["beta"])
            results = [json.loads(result_to_json(r)) for r in iter_infer(segments, **params)]
            cache[cache_key(pcs, player["beta"])] = results
            rate = sum(r["rate"] for r in results)
            dist = sum(r["distortion"] for r in results)
            print(f"{tune['name']:<24} β={player['beta']:<4} rate {rate:5.1f}  wrong {dist:3.0f}")

    out = {"players": PLAYERS, "params": PARAMS, "tunes": TUNES, "cache": cache}
    (DEMO / "recorder.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"Wrote {DEMO / 'recorder.json'}")


if __name__ == "__main__":
    main()
