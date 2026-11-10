# Recall error-category bar plots

Reproduce Fig. 3c and the SI bar plot.

## Run

From the repository root, using the project environment with Python 3.10+:

```bash
python -m analysis.error_category
```

The analysis uses NumPy, pandas, Matplotlib, SciPy and more-itertools, available
through the repository's package dependencies. The default command rebuilds the
program table each time. All generated tables and plots are saved under
`analysis/error_category/results/`, which is ignored by git.

The plots are:

- `results/figures/fig_categories_main_panel.pdf` — four categories for Fig. 3c.
- `results/figures/fig_categories_si.pdf` — six categories with reuse split by source.

Both plots are also saved as PNGs. The default font is DejaVu Sans; use
`--font Verdana` if installed to match the paper.

Intermediate outputs include `filled_prog_2.csv`, `error_notes.csv`,
`proportions_min3.csv`, participant-level proportions, and a text summary.

## Data paths

Defaults use existing repository data:

- `data/human/chirp.json`: keep completed sessions (`int(completed) > 0`) in
  input file order. The raw file contains JSON-encoded `experimentData` records.
- `docs/chirp-demo/training/`: stimulus files named `<index>-<segment>.json`.
  Resolve each session's `sampled_indices`, apply its cyclic pitch shifts, and
  map Phase-2 keys `S D F J K L` to notes 1–6.
- `inputs/task_pm_wo_mem.csv` and `data/primitive/task_pm.csv`: fixed production tables
  used to generate and score candidates. These are grammar inputs, not results.

Override the participant and stimulus paths as needed:

```bash
python -m analysis.error_category \
    --human-json /path/to/chirp.json \
    --stimuli-dir /path/to/training \
    --output analysis/error_category/results_custom
```

## Program enumeration and scoring

`build_program_table.py` enumerates frames with a depth bound of 2 using `task_pm_wo_mem.csv`. It excludes `memorize`. 
Programs are evaluated using the current primitives in `src/`.

The combined log probability is `prog_lp + frame_lp`: primitive, note and count
costs plus frame/combinator costs. For each reconstruction, retain the exact
program with the highest combined log probability per note.

## Classification and counting

Comparison uses the common prefix of target and response; unmatched suffix
positions are not included in the wrong-note denominator.

| Category | Rule |
|---|---|
| Vertical shift | Constant nonzero pitch offset modulo six keys. |
| Temporal shift | Recalled window matches material at a different position in the current target. |
| Reuse: previous note | Window repeats the immediately preceding recalled note. |
| Reuse: earlier segment | Window matches an earlier presented segment of this melody and occurs nowhere in the current target. |
| Reuse: earlier melody | Window matches an earlier melody's presented recall segment, not the current target or an earlier segment of the current melody. |
| Simpler program | Both windows have exact candidate programs and recall has a strictly higher best log probability per note; constant-note recall windows are excluded. |


## Individual stages

```bash
python -m analysis.error_category --stage build
python -m analysis.error_category --stage classify
python -m analysis.error_category --stage summarize
python -m analysis.error_category --stage figures
```

To reuse a previously generated table during a full run:

```bash
python -m analysis.error_category \
    --program-table analysis/error_category/results/filled_prog_2.csv
```