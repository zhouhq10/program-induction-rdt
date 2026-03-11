# Data

This directory contains all data files used by the simulations and analyses.

---

## Directory structure

```
data/
├── task/        # Melody task stimuli (pickle)
├── primitive/   # Primitive model tables and program frames (CSV)
└── human/       # Human behavioral data (JSON)
```

---

## `data/task/` — Melody task stimuli

Each `.obj` file is a Python pickle containing a list of melody tasks. Each task is a melody sequence represented as a NumPy array of integers in `{1, 2, 3, 4, 5, 6}`, corresponding to the 6 note types.

| File | Split | N tasks | Description |
|------|-------|---------|-------------|
| `train_notes_wo_break_104.obj` | Train | 104 | Primary training set used by compression scripts |
| `train_notes_wo_break_197.obj` | Train | 197 | Extended training set |
| `validation_notes_wo_break_63.obj` | Validation | 63 | Held-out validation set |
| `evaluation_notes_wo_break_52.obj` | Evaluation | 52 | Primary held-out test set used by `4_evaluate.py` |
| `evaluation_notes_wo_break_75.obj` | Evaluation | 75 | Extended held-out test set |

**Loading example:**

```python
import pickle

with open("data/task/train_notes_wo_break_104.obj", "rb") as f:
    train_tasks = pickle.load(f)
# train_tasks: list of numpy arrays, each array is a melody sequence
```

**"wo_break"** means melodies without rests between sub-melodies. The number in the filename is the task count.

---

## `data/primitive/` — Primitive model and program frames

These CSV files are pre-built outputs of `scripts/0_construct_pm.py` and `scripts/1_construct_frame.py` and are provided for convenience so you do not need to regenerate them.

| File | Description |
|------|-------------|
| `task_pm.csv` | Production table for the PCFG/AG/HAG primitive model (uniform prior). Each row is a production rule with its associated weight. |
| `task_pm_chunk.csv` | Production table for the **chunk** baseline model. |
| `task_pm_RLE.csv` | Production table for the **RLE** (run-length encoding) baseline model. |
| `task_frames_1.csv` | Pre-sampled program frames at depth 1 (20 frames, seed 0). |
| `task_frames_2.csv` | Pre-sampled program frames at depth 2 (20 frames, seed 0). |

**Production table columns:** `lhs` (left-hand side non-terminal), `rhs` (right-hand side), `weight` (unnormalized probability), and optionally `type` (input/output types).

**Frame CSV columns:** Each row encodes one program tree frame as a sequence of primitive names and combinator slots.

---

## `data/human/` — Human behavioral data

| File | Description |
|------|-------------|
| `chirp.json` | Behavioral data from the melody recall experiment. |

**`chirp.json` structure:**

The file is a JSON object keyed by participant-session identifiers. Each entry contains:

- `pressed_note` — dict keyed by `"{melody}-{submelody}"` (e.g. `"0-3"`), with phases `phase1`, `phase2`, `phase3`, each containing the sequence of keys pressed by the participant (mapped to integers 1–6 via `S→1, D→2, F→3, J→4, K→5, L→6`)
- `score` — accuracy scores per melody × submelody × phase
- `reaction_time` — response times (seconds) per melody × submelody × phase
- `sampled_indices` — indices into the ground-truth melody pool used for this session
- `shift` — pitch-shift applied to each melody for this participant

Melodies have 5 melodies × 6 sub-melodies per session. The ground-truth melodies are encoded as integer arrays over `{1, …, 6}`.