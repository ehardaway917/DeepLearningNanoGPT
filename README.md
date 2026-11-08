# nanoGPT hyperparameter study (CSCE 5218, nanoGPT assignment)

Character-level Tiny Shakespeare with [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (commit `3adf61e`, MIT license, vendored in `nanoGPT/`). This repo trains all 2^7 = 128 combinations of the assignment's grid, plus 8 seed repeats, analyzes them, and contains the report.

| Factor | Levels |
|---|---|
| block_size | 64, 128 |
| n_layer | 4, 6 |
| n_head | 4, 8 |
| n_embd | 128, 256 |
| batch_size | 8, 16 |
| max_iters | 1000, 2000 (lr_decay_iters = max_iters) |
| dropout | 0.1, 0.2 |

## Layout

| Path | Contents |
|---|---|
| `nanoGPT/` | nanoGPT with `train.py` patched (seed argument, final checkpoint + run summary); the diff is in `train_py.patch` |
| `upstream/` | the unmodified `train.py` and the upstream commit hash |
| `grid.py` | the grid and seed repeats; runs `train.py` once per config, sequentially, skipping finished runs |
| `runs/<name>/` | `train.log` (full stdout), `summary.json` (config, wall-clock, peak GPU memory), `eval.json` (exact losses, per-position loss, generated sample) |
| `runs_slow_archive/` | first-pass logs of the [128, 6] slice, slowed down by other GPU load and therefore rerun |
| `nanogpt_experiments.py` | analysis source; `build_notebook.py` turns it into `nanogpt_experiments.ipynb`, `run_notebook.py` executes it |
| `results.csv`, `results.json` | one row per grid run; all numbers quoted in the report |
| `figures/` | report figures |
| `report.md` → `nanoGPT_report.docx` | the report (`python build_report.py`) |

Run names encode the config: `b128_l6_h8_e256_bs16_it2000_d0.1` means block 128, 6 layers, 8 heads, n_embd 256, batch 16, 2000 iterations and dropout 0.1. A `_s1338` suffix marks a seed repeat.

## Reproduce

```
pip install torch numpy pandas matplotlib requests tiktoken nbformat nbclient python-docx
cd nanoGPT && python data/shakespeare_char/prepare.py && cd ..
python grid.py            # ~30 min of training on an RTX 5090; finished runs are skipped
python run_notebook.py    # exact evaluation, tables, figures -> results.json, figures/
python build_report.py    # report.md -> nanoGPT_report.docx
```

Checkpoints (`runs/*/ckpt.pt`) are not committed. `grid.py` recreates them.
