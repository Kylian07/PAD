# %% [markdown]
# # PAD reproduction on Kaggle: Prompt-Anchored Vision–Text Distillation for Lifelong Person Re-ID (CVPR 2026)
#
# This notebook **re-implements the official PAD code (github.com/zu-zi/PAD) inside one file**, so nothing has to be
# cloned or downloaded from GitHub. The training logic, hyper-parameters, losses, prompt pool, and evaluation
# protocol follow the official `configs/pad.yml`, `train_lifelong.py`, `test_lifelong.py`, and `pad/*` modules.
# Only the infrastructure is new: Kaggle data discovery and conversion, checkpoint/resume across sessions, a
# memory-safe evaluator that gives identical numbers, and result tables and plots.
#
# **What it reproduces**: pick one with `PROTOCOL` in the config cell.
# * **`"kaggle-4"` (default):** uses only datasets that are on Kaggle, i.e. **Market-1501 → CUHK-SYSU →
#   DukeMTMC-reID → CUHK03**. This is AKA order-1 with MSMT17 left out (no Kaggle copy of the MSMT17 images exists).
#   Every hyper-parameter, loss, and per-domain setting stays exactly as in the paper's code, so the only difference
#   is one fewer domain in the sequence.
# * **`"paper-5"`:** the full AKA order-1 sequence, Market-1501 → CUHK-SYSU → DukeMTMC-reID → MSMT17 → CUHK03. It
#   needs MSMT17, which you have to request from its authors and upload yourself.
# * After every step the notebook evaluates all domains seen so far (mAP / Rank-1). No stored exemplars are used.
# * **Unseen domains (optional):** CUHK01, CUHK02, GRID, i-LIDS, PRID, SenseReID, and VIPeR. None of these are on
#   Kaggle, so this part is skipped automatically unless you upload them.
#
# **How to run on Kaggle**
# 1. *Settings → Accelerator → GPU T4 x2* (the notebook pins **one** GPU, as the authors did for every reported
#    number) and *Internet → On* (needed for the CLIP ViT-B/16 weights).
# 2. *Add Input* → attach the datasets listed in the next cell. The notebook finds them anywhere under `/kaggle/input`.
# 3. Do a quick smoke test with `RUN_MODE = "quick"`, then set `RUN_MODE = "full"` and use **Save Version → Save & Run All
#    (commit)** so it runs in the background (up to 12 h).
# 4. If a session ends early, attach this notebook's own output as an input and run again. Domains that already
#    finished are found and skipped (see `RESUME`).
#
# **Estimated full-run time on one T4:** `kaggle-4` takes about 6–8 h; `paper-5` takes about 9–11 h (MSMT17 alone
# is about 2.5 h). The first pass over each dataset is slower because Kaggle reads `/kaggle/input` from network
# storage. If the 12 h limit is hit, use the resume step above.

# %% [markdown]
# ## Datasets to attach (Kaggle "Add Input")
#
# **For the default `kaggle-4` protocol, attach these four Kaggle datasets** (set the notebook to `RUN_MODE = "full"` for real results):
#
# | Step | Domain | Kaggle dataset (search the slug in *Add Input*) | Notes |
# |---|---|---|---|
# | 1 | Market-1501 | [`pengcw1/market-1501`](https://www.kaggle.com/datasets/pengcw1/market-1501) (alt: `rayiooo/reid_market-1501`) | standard `bounding_box_train / query / bounding_box_test` (confirmed on Kaggle) |
# | 2 | CUHK-SYSU | [`manaschaiaonon/cuhk-sysu`](https://www.kaggle.com/datasets/manaschaiaonon/cuhk-sysu) | raw person-search release is converted to the AKA `cuhksysu4reid` splits automatically. If its layout is not recognised, the data cell prints its folder tree |
# | 3 | DukeMTMC-reID | [`whurobin/dukemtmcreid`](https://www.kaggle.com/datasets/whurobin/dukemtmcreid) (alt: `igorkrashenyi/dukemtmc-reid`) | standard layout (confirmed on Kaggle) |
# | 4 | CUHK03 | [`priyanagda/cuhk03`](https://www.kaggle.com/datasets/priyanagda/cuhk03) | raw `cuhk-03.mat` (confirmed on Kaggle), converted here with the CUHK03-NP (new-protocol, *detected*) split. The small split file `cuhk03_new_protocol_config_detected.mat` is used if attached, otherwise it is downloaded from its original source (Zhong et al., person-re-ranking), so keep Internet on |
#
# **Do not attach `achrafabid7/msmt17`:** despite its name it contains a pretrained CLIP-ReID *model*, not images.
#
# **Only needed for `paper-5` / extra unseen sets (not on Kaggle; upload as private datasets):**
#
# | Domain | Source | Notes |
# |---|---|---|
# | MSMT17 | request form at http://www.pkuvmc.com/publications/msmt17.html | V1 (`train/ test/ list_*.txt`) or V2 (`mask_train_v2/ mask_test_v2/`); both handled |
# | Unseen sets | DualNorm (BMVC'19) test-set bundle: Google Drive `1-5JqXxqQ14MCngGLMPqiA-zbI2CUBFxV` | VIPeR `cam_a/cam_b`, GRID `probe/gallery + features_and_partitions.mat`, PRID `single_shot`, i-LIDS `i-LIDS_Pedestrian/Persons`, CUHK01 `campus`, CUHK02 `Dataset/P5`, SenseReID `test_probe/test_gallery` |
#
# If auto-discovery picks the wrong folder, set `DATA_OVERRIDES` in the config cell. These slugs were found by web
# search, so check that the folder layout matches before you run.

# %%
# ---- MUST be the first code cell: pin a single GPU (the authors report all numbers on one GPU; with >1 visible GPU
# ---- the official code wraps nn.DataParallel, which the README warns weakens distillation).
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("PAD_GPU", "0")

import copy, csv, glob, json, logging, math, random, re, shutil, statistics, sys, time, urllib.request, hashlib
import os.path as osp
from bisect import bisect_right
from collections import OrderedDict, defaultdict
from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

try:
    import timm
    from timm.data.random_erasing import RandomErasing
    from timm.scheduler.cosine_lr import CosineLRScheduler
except ImportError:  # timm is pre-installed on Kaggle; install it only if missing (PyPI, not GitHub)
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)
    import timm
    from timm.data.random_erasing import RandomErasing
    from timm.scheduler.cosine_lr import CosineLRScheduler

import warnings
# The official code steps the LR scheduler at the start of each epoch (kept for fidelity) -> silence PyTorch's notice.
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()` before")
warnings.filterwarnings("ignore", category=FutureWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
print(f"torch {torch.__version__} | timm {timm.__version__} | device {DEVICE} | "
      f"GPUs visible: {torch.cuda.device_count()} "
      f"{torch.cuda.get_device_name(0) if DEVICE == 'cuda' else ''}")

# %% [markdown]
# ## Run configuration (edit this cell)

# %%
RUN_MODE = os.environ.get("PAD_RUN_MODE", "quick")   # "quick" = smoke test (minutes) | "full" = paper protocol

# Which seen domains to train, in order.
#   "kaggle-4": Market-1501 -> CUHK-SYSU -> DukeMTMC-reID -> CUHK03 (all on Kaggle; AKA order-1 without MSMT17,
#               whose images are not on Kaggle)
#   "paper-5" : Market-1501 -> CUHK-SYSU -> DukeMTMC-reID -> MSMT17 -> CUHK03 (AKA order-1, as in the paper;
#               MSMT17 must be requested from its authors and uploaded as a private dataset)
PROTOCOL = os.environ.get("PAD_PROTOCOL", "kaggle-4")
TRAIN_ORDER = {
    "kaggle-4": ["market1501", "cuhksysu", "dukemtmcreid", "cuhk03"],
    "paper-5": ["market1501", "cuhksysu", "dukemtmcreid", "msmt17", "cuhk03"],
}[PROTOCOL]
SKIP_MISSING_DOMAINS = True  # True: leave out any domain whose data isn't attached (with a warning) instead of stopping
RUN_UNSEEN_EVAL = True      # evaluate the final model on the unseen datasets that are available
# Unseen test sets that are publicly downloadable (no request form) are fetched automatically:
# VIPeR, GRID, i-LIDS (original university links, with an Internet Archive fallback) and SenseReID (Google Drive).
# CUHK01/CUHK02 need a request form and PRID2011 has no stable direct file link, so attach those yourself if you have them.
DOWNLOAD_PUBLIC_UNSEEN = True
PRID_ZIP_URL = ""           # optional: paste the prid_2011.zip link from the TU Graz PRID page to fetch PRID too
RESUME = True               # skip domains whose checkpoint already exists (in OUT_DIR or in an attached old output)

# Faithfulness switch for intermediate evaluations (see the report, Sec. "Implementation observations"):
# the official test script rebuilds VA-Prompt "active slots" from non-zero prompt weights, which counts ALL 36
# (randomly initialised) slots even after step 1. True = do exactly what the official code does.
OFFICIAL_EVAL_SLOT_RECONSTRUCTION = True

# Paths (Kaggle defaults; can be overridden with env vars for local runs)
INPUT_ROOT = os.environ.get("PAD_INPUT_ROOT", "/kaggle/input")
WORK_DIR = os.environ.get("PAD_WORK_DIR", "/kaggle/working")
_tmp_base = "/kaggle/temp" if osp.isdir("/kaggle/temp") else "/tmp"
DATA_ROOT = os.environ.get("PAD_DATA_ROOT", osp.join(_tmp_base, "pad_data"))   # symlinks + converted data
OUT_DIR = osp.join(WORK_DIR, "PAD_outputs")                                     # checkpoints, logs, csv
CLIP_CACHE = os.environ.get("PAD_CLIP_CACHE", osp.join(_tmp_base, "clip_cache"))
os.makedirs(DATA_ROOT, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(CLIP_CACHE, exist_ok=True)

# Manually point a dataset to a folder if auto-discovery fails, e.g.
# DATA_OVERRIDES = {"market1501": "/kaggle/input/market-1501/Market-1501-v15.09.15"}
DATA_OVERRIDES = {}

NUM_WORKERS = min(4, os.cpu_count() or 2)   # Kaggle has 4 vCPUs (official config uses 8/12; only affects speed)

# Quick mode: tiny schedule so the whole pipeline runs end-to-end in a few minutes.
QUICK = dict(stage1_epochs=1, stage2_epochs=1, max_iters_per_epoch=10, eval_max_query=300, eval_max_gallery=3000)
if RUN_MODE != "full":
    print("!" * 100 + "\n!! RUN_MODE = \"quick\": SMOKE TEST ONLY (1 short epoch, subsampled eval). Numbers are NOT results. "
          "Set RUN_MODE = \"full\" for the real experiment.\n" + "!" * 100)
print("PROTOCOL =", PROTOCOL, "| order:", " -> ".join(TRAIN_ORDER), "| RUN_MODE =", RUN_MODE, "| DATA_ROOT =", DATA_ROOT, "| OUT_DIR =", OUT_DIR)
