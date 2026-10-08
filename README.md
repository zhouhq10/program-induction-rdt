# Program Induction and Rate-Distortion Theory for Human Sequence Learning

This repository contains the simulation code for the paper. We model how humans learn and recall melody sequences using **program induction** under a **rate-distortion** (RD) framework. The model finds symbolic program representations of melodies that trade off description length (rate) against reconstruction error (distortion), using Probabilistic Context-Free Grammars (PCFG), Adaptor Grammars (AG), and Hierarchical Adaptor Grammars (HAG).

> 🎵 **Try it in your browser** — [**Hugging Face Space →**](https://huggingface.co/spaces/zhouhq10/program-induction-rdt)
> - [**Model demo**](https://zhouhq10-program-induction-rdt.static.hf.space/index.html): compress your own melodies into programs, sweep β and watch the library grow.
> - [**Experiment**](https://zhouhq10-program-induction-rdt.static.hf.space/experiment/index.html): the online melody-learning task participants completed (preview mode — no data is saved).

---

## Overview

Melodies are sequences of 6 note types (`note_1`–`note_6`). The model learns a grammar-based compression of melody sequences via dynamic programming. Three grammar types are compared:

| Model | Description |
|-------|-------------|
| `pcfg` | Probabilistic Context-Free Grammar — fixed prior |
| `count_ag` | Adaptor Grammar — builds a shared global library across tasks |
| `hier_ag` | Hierarchical Adaptor Grammar — global + local (per-task) library |

Compression runs over train tasks; learned libraries are then evaluated on held-out tasks. Model predictions are compared against human behavioral data from a melody recall experiment.

---

## Repository structure

```
program-induction-rdt/
├── src/
│   ├── program/          # Core grammar engine (PCFG/AG/HAG, primitives, types)
│   └── domain/melody/    # Melody-specific compressors and primitives
├── scripts/              # Runnable pipeline scripts
├── src/utils/            # Data loading, preprocessing, evaluation helpers
├── data/                 # Task stimuli and human data (see data/README.md)
├── demo/                 # In-browser model demo (Hugging Face Space)
├── docs/chirp-demo/      # Static copy of the online experiment
├── environment.yml       # Conda environment specification
└── requirements.txt      # Pip-installable dependencies
```

---

## Installation
Python 3.8 is required.

We recommend creating a clean conda environment first:

```bash
conda create -n rdt python=3.8 -y
conda activate rdt
```

Then install the package from the project root (where the `pyproject.toml` lives):
```bash
pip install .
```
---

## Quickstart

### 1. Build the primitive model

```bash
python scripts/0_construct_pm.py --task primitive
```

### 2. Generate program frames

```bash
python scripts/1_construct_frame.py --task primitive --frame_num 20 --max_depth 5
```

### 3. Run compression (choose one)

```bash
# Normative (optimal) DP — PCFG
python scripts/2_normative_dp.py --curriculum pcfg \
    --save_path results/ --experiname pcfg_run --task_num 50 \
    --search_budget 20 --melody_backtrack_budget 0 --beta 1.0 \
    --folder_name "['beta', 'random_seed']"

# Greedy DP — Adaptor Grammar
python scripts/3_compress_dp.py --curriculum count_ag \
    --save_path results/ --experiname ag_run --task_num 50 \
    --lib_size 10000 --search_budget 5 --melody_backtrack_budget 1 --submelody_backtrack_budget 2 --beta 1.0 \
    --folder_name "['beta', 'global_alpha', 'random_seed']"

# Greedy DP — Hierarchical Adaptor Grammar
python scripts/3_compress_dp.py --curriculum hier_ag \
    --save_path results/ --experiname hag_run --task_num 50 \
    --lib_size 10000 --search_budget 5 --melody_backtrack_budget 1 --submelody_backtrack_budget 2 --beta 1.0 \
    --folder_name "['beta', 'global_alpha', 'local_alpha', 'random_seed']"

# Chunk baselines
python scripts/3_compress_baseline.py --curriculum chunk \
    --save_path results/ --experiname chunk_run --task_num 50 \
    --lib_size 10000 --search_budget 5 --melody_backtrack_budget 0 --submelody_backtrack_budget 2 --beta 1.0 \
    --folder_name "['beta', 'random_seed']"

# RLE baselines
python scripts/3_compress_baseline.py --curriculum RLE --mem 0 \
    --save_path results/ --experiname rle_run --task_num 50 \
    --lib_size 10000 --search_budget 1 --melody_backtrack_budget 0 --submelody_backtrack_budget 1 --beta 1.0 \
    --folder_name "['beta', 'random_seed']"
```

### 4. Evaluate on held-out tasks

```bash
python scripts/4_evaluate.py \
    --explog_base_path results/pcfg \
    --curriculum pcfg \
    --task_num 10 \
    --iter_num_per_task 5
```

---

## Interactive demo

[`demo/`](demo/) is a static website that runs the greedy compressor from `src/` in the
browser with [Pyodide](https://pyodide.org), so no server is needed. It has three tabs:

- **Explore** — compress melodies with PCFG, AG or HAG, change β and the library
  parameters, sweep β to trace a rate–distortion curve, and inspect the programs and library.
- **Tutorial** — the primitives, combinators, rate–distortion objective and libraries.
- **Bad recorder** — playing a tune from memory as lossy compression: players with
  increasing β play public-domain tunes (or your own) with more and more wrong notes.
  This tab uses seven notes (C–B) instead of the experiment's six.

The site also includes the online experiment from [`docs/chirp-demo/`](docs/chirp-demo/)
under `experiment/`.

```bash
python demo/recorder.py           # precompute the Bad recorder presets (only after changing them)
python demo/build.py              # assemble the site into demo/site/
python -m http.server -d demo/site 8000

# Publish to the Hugging Face Space (once: pip install huggingface_hub && hf auth login)
python demo/publish_hf.py         # → <your username>/program-induction-rdt
```

---

## Data

See [`data/README.md`](data/README.md) for a full description of all data files.

- `data/task/` — melody task stimuli (train / validation / evaluation splits)
- `data/primitive/` — pre-built primitive model tables and program frames
- `data/human/` — human behavioral data (`chirp.json`)

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{zhou2024harmonizing,
  title={Harmonizing Program Induction with Rate-Distortion Theory},
  author={Zhou, Hanqi and Nagy, David G and Wu, Charley M},
  booktitle={Proceedings of the Annual Meeting of the Cognitive Science Society},
  volume={46},
  year={2024}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
