# PAD reproduction kit

| File | What it is |
|---|---|
| [`PAD_Kaggle_Reproduction.ipynb`](PAD_Kaggle_Reproduction.ipynb) | **Self-contained Kaggle notebook.** Re-implements the official PAD code (no GitHub download), auto-discovers and converts the datasets, trains the 5-domain lifelong sequence (AKA order-1), and evaluates seen and unseen domains. Supports checkpoint/resume across 12 h sessions. |
| [`REPORT.md`](REPORT.md) | Plain-language methodology report: the idea, every component with equations and code pointers, the training/eval protocol, reported numbers, code-vs-paper observations, and how the port was verified. |
| `tools/verify_against_official.py` | Loads the official `pad` package next to the notebook and checks that weights, forward passes, evaluator, and optimiser groups match exactly (CPU, synthetic weights). |
| `tools/make_fake_data.py`, `tools/run_notebook.py` | Build a synthetic `/kaggle/input` (all dataset formats) and execute the notebook headlessly for testing. |
| `tools/notebook_src/`, `tools/build_notebook.py` | Notebook source as plain `.py` cells, and the script that rebuilds the `.ipynb`. |

Quick start on Kaggle: upload the notebook, choose *GPU T4 x2* with *Internet on*, attach the four Kaggle datasets
(`pengcw1/market-1501`, `whurobin/dukemtmcreid`, `achrafabid7/msmt17`, `priyanagda/cuhk03`; the default `PROTOCOL = "kaggle-4"`
runs Market → Duke → MSMT17 → CUHK03, i.e. the paper's order without CUHK-SYSU, see REPORT.md §8.1). Keep Internet on: the
notebook also downloads the publicly available unseen test sets (VIPeR, GRID, i-LIDS, SenseReID) by itself, run once with `RUN_MODE = "quick"`, then set `RUN_MODE = "full"` and use *Save & Run All*.
