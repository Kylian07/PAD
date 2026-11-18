# %% [markdown]
# # LiteLReID: a lightweight, exemplar-free lifelong person Re-ID student distilled from a PAD teacher (Kaggle)
#
# **Goal.** A ReID model that is **cheap at test time**, learns new camera domains **one after another without
# storing any old images**, and can learn from **few labels** in each new domain.
#
# **Design (two models).**
# * **Teacher (used only during training):** the PAD model (CLIP ViT-B/16 with TA/VA prompts, CVPR 2026), trained on
#   each domain in turn. The code is the verified PAD reproduction in this repository, unchanged.
# * **Student (the model you deploy):** **MobileCLIP-S1**'s image encoder (Apple, CVPR 2024), or another small
#   backbone (`STUDENT_ARCH`). After the teacher learns domain *t*, the student learns domain *t* from:
#   1. **labels** of domain *t* (ID loss + triplet loss), on the labelled identities only;
#   2. **teacher feature and relation distillation** on every image of domain *t*, labelled or not;
#   3. **text anchors**: the teacher's per-identity text embeddings (PAD's TA-Prompt), which the student, being
#      CLIP-aligned, can match directly;
#   4. **exemplar-free old knowledge**: a frozen copy of the previous student plus **identity prototypes** (one mean
#      feature vector per old identity, no images). The new student must keep the old student's similarity to
#      those prototypes.
#
# **Protocol.** By default Market-1501 → CUHK-SYSU → DukeMTMC-reID → CUHK03 (`PROTOCOL = "kaggle-4"`, AKA order-1
# without MSMT17), plus the unseen sets that can be downloaded (GRID, i-LIDS). After each step, the teacher and the
# student are evaluated on every domain seen so far. At the end, the efficiency (parameters, GFLOPs, latency,
# throughput) of both is measured.
#
# **This is a research prototype.** The loss weights below are reasonable starting points, not tuned values. Run the
# smoke test first.
#
# **How to run on Kaggle**
# 1. *Accelerator → GPU T4 x2* (one GPU is used) and *Internet → On* (CLIP and MobileCLIP weights, unseen sets).
# 2. *Add Input:* `pengcw1/market-1501`, `manaschaiaonon/cuhk-sysu`, `whurobin/dukemtmcreid`, `priyanagda/cuhk03`.
# 3. Run the smoke test (`RUN_MODE = "quick"`, a few minutes), then set `RUN_MODE = "full"` and **Save & Run All**.
# 4. Expect about 6 h for the teacher and about 3–4 h for the student on one T4. If the session limit is reached,
#    the notebook stops cleanly. Attach this notebook's output as an input and run again: finished teacher domains,
#    finished student domains, and the student's last saved epoch are picked up automatically.

# %%
import time
T_NOTEBOOK_START = time.time()   # used to stop the student cleanly before Kaggle's 12 h limit
