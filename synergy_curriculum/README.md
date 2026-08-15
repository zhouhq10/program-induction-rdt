# Synergy-based curriculum

Recovers the synergy / partial information decomposition (PID) analysis that was
deleted from this repository and completes it into a runnable pipeline:

```
melody compressions  ->  occurrence matrix  ->  PID  ->  synergy-maximising library
```

A program library is treated as a curriculum when its programs predict task
outcome in combination but not individually. Synergy in a partial information
decomposition quantifies that property. The pipeline therefore searches for the
size-k subset of subprograms whose joint occurrence pattern is maximally
synergistic with respect to how well melodies were compressed.

---

## Provenance

| Piece | Where it came from |
|---|---|
| `PIDCalculator.synergy()` | Originally `compression/program/pid_calculator.py`, written in commit `adddda5` ("Finish the synergy computation.", 2024-01-03). Vendored verbatim into `pid.py`. |
| `extract_nested_brackets`, candidate filter, random search | `exp/archive/4_pid.ipynb` cells 7-13, deleted in `d7f76a8` ("clean archive"), last present at `bd611f0`. |
| `redundancy` / `unique` / `mutual` / `decomposition` | Not present in the original code. The archived notebooks call them but no implementation exists, which is why those calls are commented out. Implemented here following Williams & Beer (2010) using the `_Imin` / `_Imax` helpers in `pid_helpers.py`. |
| Everything else | New. |


---

## Files

| File | Purpose |
|---|---|
| `occurrence.py` | Load melody compressions, extract candidate subprograms, build the binary occurrence matrix and the binned outcome. |
| `pid.py` | Partial information decomposition. Holds the recovered `PIDCalculator` (probability tables and `synergy()`) and `PID`, which subclasses it to add redundancy, unique, mutual, `decomposition()`, shuffled-surrogate debiasing and sample-adequacy checks. |
| `pid_helpers.py` | Probability-table and information-theoretic helpers (`_Imin`, `_Imax`, joint and conditional tables), vendored from the old `compression/program/` package. |
| `curriculum.py` | The search for the synergy-maximising library (greedy and random). |
| `ordering.py` | Turn a selected library into a melody presentation order. |
| `run_synergy_curriculum.py` | CLI tying it together. |
| `demo_synthetic.py` | End-to-end demo on synthetic data with a planted answer, in three information regimes (`--regime synergy\|redundancy\|mixed`). |

No new dependencies; the code builds on `pid_helpers.py`, a trimmed
[pidpy](https://github.com/pietromarchesi/pidpy). `pymorton` is required only by
the non-binary path, which the curriculum does not use.

---

## Quick start

To check the pipeline on synthetic melodies with a known planted answer:

```sh
python synergy_curriculum/demo_synthetic.py
```

The demo buries an XOR pair (outcome is high when program A occurs XOR program B
occurs) among distractors. Neither program predicts outcome alone, so only a
synergy-based search recovers them. The expected report is
`Planted pair recovered: 2/2`.

`--regime redundancy` and `--regime mixed` plant different information
structures in the same library; see [Reading the output](#reading-the-output).

On a real run:

```sh
python synergy_curriculum/run_synergy_curriculum.py \
    --result_dir /path/to/simulation_result/hag/greedy/<experiname>/<params> \
    --lib_size 12 \
    --outcome accuracy \
    --n_bins 2 \
    --debiased \
    --out curriculum.csv
```

Or from Python:

```python
from synergy_curriculum import load_melody_programs, build_dataset, greedy_search

melodies = load_melody_programs("/path/to/run")
data = build_dataset(melodies, outcome="accuracy", n_bins=2)
result = greedy_search(data["X"], data["y"], data["names"], lib_size=12)

print(result)
result.save("curriculum.csv")
```

---

## The three stages

### 1. Occurrence

`--result_dir` should hold the `.obj` files written by
`Compressor.save_result_per_task`. Both save formats are handled:
`task_{i}_prog_trajs.obj` (PCFG greedy, one DataFrame) and
`task_{i}_task_progs.obj` (AG/HAG, a list over the melody lookback window).

Each melody's `term` column is concatenated into one string, every bracketed
subexpression is extracted with `extract_nested_brackets`, and
`X[melody, subprogram] = 1` when that subprogram appears as a substring.
Substring matching is the intended semantics: it indicates that the melody's
solution used the abstraction, possibly nested inside a larger term.

Two cleanups are applied automatically:

- Constant columns are dropped. A subprogram present in every melody or in none
  carries no information and makes the (n-1)-subset tables degenerate.
- Duplicate columns are merged. `[B,I,note_3]` is a substring of
  `[B,repeat,[B,I,note_3]]`, so if the inner term appears only inside the outer
  one, their columns are identical. The most specific (longest) term is kept and
  absorbed aliases are returned in `data["aliases"]`. Without this step the
  search reports uninformative fragments rather than abstractions.

### 2. PID

`--outcome` selects the quantity the decomposition is about:

| value | meaning |
|---|---|
| `accuracy` | `1 - total distortion / total reconstructed length` (the analogue of the notebook's `1 - (recon_errors*recon_length).sum()/50`) |
| `distortion` | total Levenshtein distortion |
| `rate` | description length, `-sum(log_prob)` |
| `rd_cost` | `distortion + beta * rate`, the objective the compressor minimises |

The outcome is binned into `--n_bins` labels because the estimator indexes
probability tables by label. The original notebooks passed raw continuous
accuracies, which gives nearly every melody its own label and saturates the
mutual information; binning corrects this. Bins are formed by average rank, so
heavily tied outcomes, which are common because many melodies reach identical
distortion, still split cleanly.

### 3. Search

- `--method greedy` (default) is seeded with the highest-synergy pair, since
  synergy requires at least two sources and pair structure is where XOR-like
  dependence appears, then adds whichever candidate raises synergy most. It
  reports the highest-synergy prefix, so the returned library is often smaller
  than `--lib_size`; adding sources eventually dilutes synergy and the trace
  records where this occurs. `--restarts` re-seeds from random pairs.
- `--method random` reproduces the 2024 notebook: sample a library, keep it if
  synergy improved, repeat `--n_iter` times. It is retained for reproducibility
  but scales poorly beyond a small candidate pool. On the demo (15 candidates,
  `lib_size=4`, 500 iterations) it reaches 0.051 bits against 0.709 bits for
  greedy. The exact figure depends on `--seed` and `--n_iter`; the order of
  magnitude does not.

---

## Reading the output

The blocks below come from `demo_synthetic.py`, not from real data. The demo
plants a different information structure for each `--regime`, so each measure
can be inspected against a known ground truth. The library is the same ten terms
in all three regimes; only the outcome-generating process changes. Each block is
the decomposition of the planted terms, which the demo prints after the search.

### Pure synergy: `--regime synergy` (default)

```sh
python synergy_curriculum/demo_synthetic.py --debiased
```

```
  mutual      I(X;y)  =  0.7210 bits
  redundancy          =  0.0002 bits
  synergy             =  0.7087 bits
  unique (sum)        =  0.0121 bits
  mutual     debiased =  0.7107 bits (shuffled  0.0103 +/- 0.0065, z = 110.16)
  redundancy debiased = -0.0010 bits (shuffled  0.0012 +/- 0.0017, z =  -0.56)
  synergy    debiased =  0.7039 bits (shuffled  0.0047 +/- 0.0038, z = 186.57)
```

Outcome is `A XOR B`, so A and B are each marginally independent of it.
Redundancy is `Imin` over the single sources and unique information is
`I(X_i;y) - redundancy`; for XOR both are approximately zero by construction and
all information resides in the pair. This is a limiting case, useful as a
correctness check but not representative of real runs.

### Pure redundancy: `--regime redundancy`

```sh
python synergy_curriculum/demo_synthetic.py --regime redundancy --debiased
```

```
  mutual      I(X;y)  =  0.6015 bits
  redundancy          =  0.4321 bits
  synergy             =  0.1044 bits
  unique (sum)        =  0.0650 bits
  mutual     debiased =  0.5914 bits (shuffled  0.0101 +/- 0.0066, z =  88.98)
  redundancy debiased =  0.4305 bits (shuffled  0.0016 +/- 0.0024, z = 178.78)
  synergy    debiased =  0.0986 bits (shuffled  0.0058 +/- 0.0060, z =  16.57)
```

The opposite case: one latent cause drives the outcome and two library terms are
independent noisy readouts of it. Most of the information is available from
either term alone, which is what large redundancy indicates. The readouts are
noisy rather than exact copies because identical columns are merged by
`drop_duplicate_sources` before the PID is computed.

### Mixed structure: `--regime mixed`

```sh
python synergy_curriculum/demo_synthetic.py --regime mixed --n_melodies 1200 --debiased
```

```
  mutual      I(X;y)  =  0.5840 bits
  redundancy          =  0.0796 bits
  synergy             =  0.0551 bits
  unique (sum)        =  0.1426 bits
  mutual     debiased =  0.5646 bits (shuffled  0.0194 +/- 0.0044, z = 128.77)
  redundancy debiased =  0.0795 bits (shuffled  0.0001 +/- 0.0001, z = 728.58)
  synergy    debiased =  0.0470 bits (shuffled  0.0081 +/- 0.0025, z =  18.81)
  top unique sources:
     0.0663  [B,repeat,[B,I,note_6]]
     0.0362  [CB,ranges,[B,I,note_4],[B,I,count_3]]
     0.0358  [B,reverse,[B,I,note_5]]
     0.0043  [CB,up,[B,I,note_5],[B,I,count_2]]
    -0.0000  [B,repeat,[B,I,note_3]]
```

Three causes act simultaneously: one read noisily by the XOR pair, so those
terms carry marginal as well as joint information; one carried by a single term;
and one read by two terms. All three measures are small but clearly nonzero,
with unique information the largest. This is the pattern to expect on real runs
rather than the XOR case. Redundancy is `Imin`, a minimum over sources, so a
single uninformative program in the library holds it near zero regardless of how
much the remaining programs share; this is why the synergy block reports 0.0002.

The command specifies 1200 melodies deliberately. At the default 200 the same
regime gives `synergy = 0.0595` at z = 1.9, indistinguishable from the shuffled
surrogates and therefore attributable to finite-sample bias, although the raw
value appears comparable. Redundancy and unique information remain detectable at
that sample size; synergy does not.

### Interpretation

The relation `mutual >= redundancy + sum(unique) + synergy` is not an identity
here. Williams & Beer's terms are computed independently by this estimator and
sum exactly only in the two-source case. The four quantities should be read as
separate diagnostics rather than as a partition.

With `--debiased`, each measure is also reported minus its shuffled-surrogate
mean, which is the value the estimator returns from finite-sample bias alone,
together with a z score against the surrogate spread. In the synergy block,
synergy lies 187 surrogate standard deviations above chance while redundancy
lies at z = -0.6. The debiased numbers should always be checked before a synergy
value is interpreted: with many sources and few melodies the raw value can be
almost entirely bias, as the mixed regime demonstrates.

---

## Task ordering (optional)

The library search identifies which abstractions matter. `ordering.py` converts
that result into a melody presentation order using the same occurrence matrix:
melody `i` exercises library program `j` when `X[i, j] == 1`.

```sh
python synergy_curriculum/run_synergy_curriculum.py \
    --result_dir /path/to/run --lib_size 8 \
    --order incremental --order_out order.csv
```

```python
from synergy_curriculum import order_tasks

ordering = order_tasks(data["X"], data["names"], result.indices,
                       method="incremental", outcome=data["y_continuous"])
print(ordering)
ordering.save("order.csv")
```

| method | behaviour |
|---|---|
| `incremental` (default) | Greedy: at each step take the melody introducing the fewest unseen library programs. Abstractions are encountered individually before they are encountered in combination, matching the synergy account. |
| `coverage` | Sort by how many library programs each melody exercises. `--order_descending` reverses to hardest-first. |

`order` is a full permutation of melody indices. Melodies exercising none of the
library are placed at the end rather than at the front, since they fall outside
the library's scope; their count is reported. The result also reports the
Spearman correlation between presentation position and measured outcome, a
sanity check on whether the ordering tracks difficulty. It will not always do
so, which is itself informative.

On the demo, over a 6-program library, `incremental` covers every program by
position 5 while `coverage` requires until position 8. The demo's own selected
library contains only 2 programs, since greedy reports the best-synergy prefix;
the 6-program library used here is the first six candidate columns, chosen to
make the contrast visible.

---

## Caveats

- Undersampling is the principal threat. With n binary sources there are 2^n
  possible occurrence patterns. When nearly every melody has a distinct pattern,
  `I(X;y)` saturates and synergy reflects sample size rather than structure.
  This affected the original analysis (approximately 30 sources, 100 melodies).
  `check_sample_adequacy` warns when the saturation ratio exceeds 0.9 or when
  there are fewer than 10 melodies per source; both warnings fire during the
  search. Small libraries (5-15) are preferable unless many melodies are
  available.
- Sources are capped at 62, because `pid_helpers._map_binary_par` packs each row
  into an int64.
- The library search and the task ordering are distinct claims. The PID selects
  a set of abstractions, which is what the original analysis did. The ordering
  in `ordering.py` is a defensible derivation from that set, but nothing in the
  PID validates a presentation order; it is a design choice built on the
  occurrence matrix, not a result.
- Greedy search is a heuristic. Synergy is not submodular, so forward selection
  carries no optimality guarantee. Use `--restarts` and compare.
- `synergy_dit()` in `pid.py` is an untested cross-check against the `dit`
  package, which is not installed here. Exact PID in `dit` is exponential in the
  number of sources and is practical only for roughly five or fewer.
