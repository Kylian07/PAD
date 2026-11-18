# %% [markdown]
# ## LiteLReID configuration (edit this cell)
# The cells above configure the **teacher** (the PAD reproduction: `RUN_MODE`, `PROTOCOL`, data paths). This cell
# configures the **student** and the few-label setting.

# %%
STUDENT_ARCH = os.environ.get("LITE_ARCH", "MobileCLIP-S1")
#   "MobileCLIP-S1" (21.5 M) | "MobileCLIP2-S0" (11.4 M) | "MobileCLIP-S2" | "MobileCLIP2-S2" | "MobileCLIP-B"
#                       (open_clip, Apple weights from the HF hub; CLIP-aligned, so the teacher's text anchors apply)
#   "timm:resnet18" | "timm:mobilenetv3_large_100" | "timm:resnet50" | any timm model name with ImageNet weights
#                                                          (not CLIP-aligned: a learned 512-d projection is used)
STUDENT_PRETRAINED = os.environ.get("LITE_PRETRAINED", "1") == "1"   # keep True; False = random init (offline tests only)
# Which students to train. "ours" = the full method. "naive_ft" = the same student fine-tuned on each domain in turn
# with ID + triplet loss only (no teacher, no anti-forgetting): the lower bound reviewers expect.
STUDENT_METHODS = os.environ.get("LITE_METHODS", "ours").split(",")

# Few-label setting: the fraction of identities in each new domain that have labels (1.0 = fully supervised).
# The teacher is trained on the labelled identities only. The student uses labels for those identities and
# distillation (no labels) for all other images.
LABEL_FRACTION = float(os.environ.get("LITE_LABEL_FRACTION", "1.0"))
LABEL_SEED = 0
assert 0 < LABEL_FRACTION <= 1.0

STUDENT_EPOCHS = {"first": 60, "later": 30}     # the first domain trains the student from ImageNet/CLIP weights
STUDENT_LR = {"first": 3e-4, "later": 1e-4}      # AdamW; a lower lr later limits drift on old domains
STUDENT_WD = 5e-4
STUDENT_WARMUP_EPOCHS = 5
STUDENT_UNLABELLED_BATCH = 32                    # images per step from the unlabelled pool (if LABEL_FRACTION < 1)

# Loss weights (untuned starting points; see the method cell for what each term does)
LAMBDA = dict(id=1.0, tri=1.0, i2t=0.5,          # supervised: labelled identities of the current domain
              fkd=1.0, rkd=1.0,                  # teacher distillation: every current-domain image
              proto=1.0, old=0.5)                # anti-forgetting: previous student + old identity prototypes
TAU = dict(i2t=0.05, rkd=0.1, proto=0.1)         # temperatures

STUDENT_CKPT_EVERY = 5                           # save a resumable student state every N epochs
MAX_SESSION_HOURS = float(os.environ.get("LITE_MAX_HOURS", "11.3"))   # stop cleanly before Kaggle's 12 h limit
if LABEL_FRACTION < 1.0:   # a few-label teacher must never be confused with a fully supervised one
    OUT_DIR = osp.join(WORK_DIR, f"PAD_outputs_lf{LABEL_FRACTION:g}")
    os.makedirs(OUT_DIR, exist_ok=True)
LITE_OUT = osp.join(WORK_DIR, "LiteLReID_outputs")
os.makedirs(LITE_OUT, exist_ok=True)

for m in STUDENT_METHODS:
    assert m in ("ours", "naive_ft"), m
print(f"STUDENT_ARCH={STUDENT_ARCH} | methods={STUDENT_METHODS} | LABEL_FRACTION={LABEL_FRACTION} | "
      f"epochs={STUDENT_EPOCHS} | lr={STUDENT_LR} | LAMBDA={LAMBDA}")
