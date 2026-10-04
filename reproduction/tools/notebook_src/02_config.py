# %% [markdown]
# ## Hyper-parameters: a 1:1 port of the official `configs/pad.yml`
# Shared settings come first. `DOMAINS` holds the per-domain overrides, merged exactly like `pad/config.py`.

# %%
class CN(dict):
    """Tiny attribute-dict so the ported code can keep `cfg.SOLVER.STAGE1.MAX_EPOCHS` syntax."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v

    @staticmethod
    def wrap(d):
        return CN({k: CN.wrap(v) if isinstance(v, dict) else v for k, v in d.items()})


BASE_CFG = {
    "MODEL": {"NAME": "ViT-B-16", "STRIDE_SIZE": [16, 16], "ID_LOSS_WEIGHT": 0.25, "TRIPLET_LOSS_WEIGHT": 1.0,
              "I2T_LOSS_WEIGHT": 1.0, "LABEL_SMOOTH": True},
    "INPUT": {"SIZE_TRAIN": [256, 128], "SIZE_TEST": [256, 128], "PROB": 0.5, "RE_PROB": 0.5, "PADDING": 10,
              "PIXEL_MEAN": [0.5, 0.5, 0.5], "PIXEL_STD": [0.5, 0.5, 0.5]},
    "DATALOADER": {"SAMPLER": "softmax_triplet", "NUM_INSTANCE": 4, "NUM_WORKERS": 8},
    "DATASETS": {"NAMES": ("market1501",), "ROOT_DIR": DATA_ROOT},
    "SOLVER": {
        "SEED": 42, "MARGIN": 0.3,
        "STAGE1": {"IMS_PER_BATCH": 64, "OPTIMIZER": "Adam", "BASE_LR": 0.00035, "WEIGHT_DECAY": 1e-4,
                   "MAX_EPOCHS": 12, "LOG_PERIOD": 50, "WARMUP_EPOCHS": 3, "WARMUP_LR_INIT": 1e-5, "LR_MIN": 1e-6},
        "STAGE2": {"IMS_PER_BATCH": 64, "OPTIMIZER": "Adam", "BASE_LR": 5e-6, "WEIGHT_DECAY": 1e-4,
                   "MAX_EPOCHS": 25, "LOG_PERIOD": 50, "STEPS": [14, 22], "GAMMA": 0.1, "WARMUP_ITERS": 10,
                   "WARMUP_FACTOR": 0.1, "WARMUP_METHOD": "linear"},
    },
    "TEST": {"IMS_PER_BATCH": 128, "FEAT_NORM": "yes", "NECK_FEAT": "before"},
    "TA_PROMPT": {"N_CTX": 4, "N_CLS_CTX": 4},
    "VA_PROMPT": {"ENABLE": True, "G_LEN": 6, "E_LEN": 6, "POOL_SIZE": 36, "TOP_K": 4, "EMBEDDING_KEY": "mean",
                  "KEY_LEARNABLE": True, "STRIP_AFTER_BLOCK": True, "NEW_SLOTS_PER_DOMAIN": 8},
    "UNFREEZE": {"LAST_BLOCKS": 4},
    "OPT": {"PROMPT_LR": 0.005, "PROMPT_WD": 0.0},
    "DISTILL": {"ENABLE": True, "LAMBDA_FEAT": 0.5, "LAMBDA_LOGIT": 0.5, "TEMP": 4.0, "EMA_MOMENTUM": 0.997},
    "TEXTKD": {"ENABLE": True, "LAMBDA_LOGIT": 0.5, "TEMP": 0.07, "LOGITS_SCALE": 7.0, "NEG_PER_BATCH": 256},
    "EVAL": {"SPLIT_ID": 0, "SEED": 123, "DIRECTION": "auto"},
    "OUTPUT_DIR": "",
}

DOMAIN_OVERRIDES = {
    # First domain: no teacher -> full visual-branch tuning, longer schedule, KD off.
    "market1501": {"SOLVER": {"STAGE1": {"MAX_EPOCHS": 20}, "STAGE2": {"MAX_EPOCHS": 30, "STEPS": [18, 26]}},
                   "DISTILL": {"ENABLE": False}, "TEXTKD": {"ENABLE": False}, "OPT": {"PROMPT_LR": 0.002}},
    "cuhksysu": {},
    "dukemtmcreid": {},
    # Larger dataset -> slightly longer schedule.
    "msmt17": {"DATALOADER": {"NUM_WORKERS": 12},
               "SOLVER": {"STAGE1": {"MAX_EPOCHS": 15}, "STAGE2": {"MAX_EPOCHS": 32, "STEPS": [18, 28]}}},
    # Small dataset -> smaller batch, stronger erasing, fewer new prompt slots.
    "cuhk03": {"INPUT": {"RE_PROB": 0.6},
               "SOLVER": {"STAGE1": {"IMS_PER_BATCH": 48, "MAX_EPOCHS": 10},
                          "STAGE2": {"IMS_PER_BATCH": 48, "MAX_EPOCHS": 28, "STEPS": [16, 24]}},
               "VA_PROMPT": {"NEW_SLOTS_PER_DOMAIN": 4}},
}


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def build_cfg(domain_name, is_first_domain):
    """Shared block + that domain's overrides (== pad.config.load_cfg_from_yaml)."""
    raw = _deep_merge(BASE_CFG, DOMAIN_OVERRIDES.get(domain_name, {}))
    raw["DATASETS"]["NAMES"] = (domain_name,)
    if not is_first_domain and domain_name == "market1501":
        # Market as a *later* domain (other orders) should use the shared schedule + KD like the others.
        raw = _deep_merge(BASE_CFG, {}); raw["DATASETS"]["NAMES"] = (domain_name,)
    cfg = CN.wrap(raw)
    cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
    if RUN_MODE == "quick":
        cfg.SOLVER.STAGE1.MAX_EPOCHS = QUICK["stage1_epochs"]
        cfg.SOLVER.STAGE2.MAX_EPOCHS = QUICK["stage2_epochs"]
        cfg.SOLVER.STAGE1.LOG_PERIOD = cfg.SOLVER.STAGE2.LOG_PERIOD = 5
    return cfg


print(json.dumps(build_cfg("msmt17", False)["SOLVER"], indent=1))
