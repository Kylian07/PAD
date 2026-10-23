# %% [markdown]
# ## Lifelong driver (port of `train_lifelong.py` + `test_lifelong.py` + `scripts/run_lifelong.sh`)
# For each domain *t* in order:
# 1. Build the model from CLIP and load the **visual branch** of the previous step's checkpoint (backbone, VA-Prompt,
#    BN necks). The classifier and TA-Prompt are re-created because the identity space changes.
# 2. `allocate_new_domain_slots`: freeze old E-Prompt slots (gradient mask) and open new ones.
# 3. Keep `model_old` = a frozen copy, which serves as the TEXKD anchor and the initial EMA teacher.
# 4. Stage 1 (TA-Prompt), then Stage 2 (visual), then save the checkpoint.
# 5. Evaluate on every domain seen so far, and append the result to `seq_eval.csv`.

# %%
def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _set_all_requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad_(flag)


def _set_by_name(model, substrings, flag):
    keys = tuple(s.lower() for s in ((substrings,) if isinstance(substrings, str) else substrings))
    for n, p in model.named_parameters():
        if any(k in n.lower() for k in keys):
            p.requires_grad_(flag)


def _unfreeze_last_blocks(model, n_last):
    blocks = list(model.image_encoder.transformer.resblocks)
    for i, blk in enumerate(blocks):
        if i >= max(0, len(blocks) - int(n_last)):
            for p in blk.parameters():
                p.requires_grad_(True)


def _log_params(model, logger, tag):
    total = sum(p.numel() for p in model.parameters()); tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[{tag}] trainable params: {tr:,}/{total:,} ({tr / total:.2%})")


def _filter_visual_state_dict(sd):
    """Keep only the visual-branch weights (backbone + VA-Prompt + BN necks) == official _filter_resume_state_dict."""
    out = {}
    for k, v in sd.items():
        while k.startswith("module."):
            k = k[len("module."):]
        lk = k.lower()
        if lk.startswith("image_encoder") or "bottleneck" in lk or \
                (("ln_post" in lk) and ("text" not in lk)) or (lk.endswith(".proj") and "text" not in lk):
            out[k] = v
    return out


def _register_va_grad_mask(model):
    """Zero-out gradients on VA-Prompt slots owned by previous domains."""
    va = getattr(model.image_encoder, "va_prompt", None)
    if va is None:
        return

    def _make_hook(fmask):
        def hook(grad):
            expand = fmask.view(-1, 1, 1) if fmask.ndim == 1 else fmask
            expand = expand.to(grad.device)
            grad = grad.clone()
            grad[expand.expand_as(grad)] = 0
            return grad
        return hook

    for li in range(va.num_layers):
        fmask = va.frozen_masks[li].detach().cpu()
        va.e_prompt[li].register_hook(_make_hook(fmask))                 # [pool, e_len, C]
        if va.key_learnable:
            va.e_key[li].register_hook(_make_hook(fmask.view(-1, 1)))    # [pool, C]


def ckpt_path(domain):
    return osp.join(OUT_DIR, domain, "ViT-B-16_visual.pth")


def find_existing_ckpt(domain):
    """Checkpoint in this session's OUT_DIR, or in an attached previous run's output (Kaggle input)."""
    p = ckpt_path(domain)
    if osp.isfile(p):
        return p
    for d in ALL_INPUT_DIRS:
        q = osp.join(d, "ViT-B-16_visual.pth")
        if osp.basename(d) == domain and "pad_outputs" in d.lower() and osp.isfile(q):
            os.makedirs(osp.dirname(p), exist_ok=True)
            shutil.copy(q, p)
            print(f"[resume] copied {q} -> {p}")
            for f in ("train_log.txt",):
                if osp.isfile(osp.join(d, f)):
                    shutil.copy(osp.join(d, f), osp.join(osp.dirname(p), f))
            return p
    return None


def train_one_domain(domain, prev_ckpt, is_first):
    cfg = build_cfg(domain, is_first)
    set_seed(cfg.SOLVER.SEED)
    out_dir = osp.join(OUT_DIR, domain)
    logger = setup_logger(out_dir, "train_log.txt")
    logger.info(f"[Lifelong] dataset={domain} output_dir={out_dir} RUN_MODE={RUN_MODE}")

    train_loader_stage2, train_loader_stage1, _, _, num_classes = make_dataloader(cfg)
    model = PADModel(num_classes, cfg).to(DEVICE)

    if prev_ckpt:
        filt = _filter_visual_state_dict(torch.load(prev_ckpt, map_location="cpu"))
        ret = model.load_state_dict(filt, strict=False)
        logger.info(f"[Resume] from {prev_ckpt}: loaded={len(filt)} missing={len(ret.missing_keys)} unexpected={len(ret.unexpected_keys)}")
    else:
        logger.info("[Resume] first domain -- starting from CLIP pretrain; KD disabled (no teacher)")
        cfg.DISTILL.ENABLE = False
        cfg.TEXTKD.ENABLE = False
    logger.info(f"[Switches] TEXTKD={cfg.TEXTKD.ENABLE} VISKD={cfg.DISTILL.ENABLE}")

    if hasattr(model.image_encoder, "va_prompt"):
        slots = int(cfg.VA_PROMPT.NEW_SLOTS_PER_DOMAIN)
        model.image_encoder.va_prompt.allocate_new_domain_slots(slots)
        _register_va_grad_mask(model)
        logger.info(f"[VA-Prompt] +{slots} slots/layer; active={int(model.image_encoder.va_prompt.active_sizes[0, 0])}"
                    f"/{cfg.VA_PROMPT.POOL_SIZE}")

    model_old = copy.deepcopy(model).cpu()
    teacher_for_textkd = model_old if (prev_ckpt and cfg.TEXTKD.ENABLE) else None
    loss_func = make_loss(cfg, num_classes=num_classes)

    # ---- Stage 1: TA-Prompt only ----
    _set_all_requires_grad(model, False)
    _set_by_name(model, "prompt_learner", True)
    _log_params(model, logger, "STAGE1")
    optimizer_1 = make_optimizer(cfg, model, stage="STAGE1")
    sched_1 = make_cosine_scheduler(optimizer_1, num_epochs=cfg.SOLVER.STAGE1.MAX_EPOCHS, lr_min=cfg.SOLVER.STAGE1.LR_MIN,
                                    warmup_lr_init=cfg.SOLVER.STAGE1.WARMUP_LR_INIT, warmup_t=cfg.SOLVER.STAGE1.WARMUP_EPOCHS)
    do_train_stage1(cfg, model, train_loader_stage1, optimizer_1, sched_1, teacher_model=teacher_for_textkd)

    # ---- Stage 2: visual branch ----
    if not prev_ckpt:
        _set_all_requires_grad(model, True)
        _set_by_name(model, "prompt_learner", False)
        _log_params(model, logger, "STAGE2 (first domain)")
    else:
        _set_all_requires_grad(model, False)
        _set_by_name(model, "prompt_learner", False)
        _set_by_name(model, ("classifier", "bottleneck"), True)
        if cfg.VA_PROMPT.ENABLE:
            _set_by_name(model, ("va_prompt", "g_prompt", "e_prompt", "e_key"), True)
        _unfreeze_last_blocks(model, int(cfg.UNFREEZE.LAST_BLOCKS))
        _log_params(model, logger, "STAGE2")
    optimizer_2 = make_optimizer(cfg, model, stage="STAGE2")
    sched_2 = WarmupMultiStepLR(optimizer_2, cfg.SOLVER.STAGE2.STEPS, cfg.SOLVER.STAGE2.GAMMA,
                                cfg.SOLVER.STAGE2.WARMUP_FACTOR, cfg.SOLVER.STAGE2.WARMUP_ITERS, cfg.SOLVER.STAGE2.WARMUP_METHOD)
    do_train_stage2(cfg, model, train_loader_stage2, optimizer_2, sched_2, loss_func,
                    teacher_model=(model_old if prev_ckpt else None))

    p = ckpt_path(domain)
    torch.save(_filter_visual_state_dict(model.state_dict()), p)
    logger.info(f"[Save] {p}")
    del model_old, optimizer_1, optimizer_2
    return model, cfg


# %% [markdown]
# ### Evaluation helper (== `test_lifelong._load_visual_state_dict` + `do_inference`)
# The official test script loads the visual weights into a fresh model and then **re-derives the number of active
# prompt slots from the non-zero E-Prompt weights**. Because unused slots are randomly initialised (not zero), this
# always gives 36, so at test time the top-k is chosen from the whole pool. `OFFICIAL_EVAL_SLOT_RECONSTRUCTION`
# reproduces that behaviour (default). Set it to False to evaluate with the slot count used during training.

# %%
def prepare_eval_model(model):
    m = copy.deepcopy(model).to(DEVICE).eval()
    if OFFICIAL_EVAL_SLOT_RECONSTRUCTION and hasattr(m.image_encoder, "va_prompt"):
        va = m.image_encoder.va_prompt
        with torch.no_grad():
            for li in range(va.num_layers):
                used = int((va.e_prompt[li].detach().abs().sum(dim=(1, 2)) > 1e-6).sum().item())
                va.active_sizes[li, 0] = min(used, va.pool_size)
    return m


def model_from_ckpt(ckpt, domain):
    cfg = build_cfg(domain, False)
    m = PADModel(1, cfg).to(DEVICE)
    m.load_state_dict(_filter_visual_state_dict(torch.load(ckpt, map_location="cpu")), strict=False)
    return m


def evaluate_domains(model, eval_domains, trained_domain, csv_out=None, extra_cfg=None):
    logger = logging.getLogger("pad")
    m = prepare_eval_model(model)
    rows = []
    for d in eval_domains:
        cfg = build_cfg(d, False)
        if extra_cfg:
            for k, v in extra_cfg.items():
                cfg.EVAL[k] = v
        cfg.DATASETS.NAMES = (d,)
        ds = build_dataset(d, cfg, verbose=False)
        _, _, val_loader, num_query, _ = make_dataloader(cfg, dataset=ds)
        t0 = time.time()
        r1, r5, mAP = do_inference(cfg, m, val_loader, num_query)
        logger.info(f"[Eval] trained={trained_domain} eval={d} Rank-1={r1:.4f} Rank-5={r5:.4f} mAP={mAP:.4f} "
                    f"({time.time() - t0:.0f}s)")
        row = {"trained_domain": trained_domain, "eval_domain": d, "rank1": f"{r1:.4f}", "rank5": f"{r5:.4f}",
               "map": f"{mAP:.4f}", "split_id": (extra_cfg or {}).get("SPLIT_ID", "")}
        rows.append(row)
        if csv_out:
            new = not osp.exists(csv_out)
            with open(csv_out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new:
                    w.writeheader()
                w.writerow(row)
    del m
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return rows


# %% [markdown]
# ## ▶ Run the lifelong sequence
# This is the long cell. Each finished domain leaves `PAD_outputs/<domain>/ViT-B-16_visual.pth`, and with `RESUME=True`
# a rerun skips finished domains and their evaluations.

# %%
SEQ_CSV = osp.join(OUT_DIR, "seq_eval.csv")


def _done_evals():
    if not osp.exists(SEQ_CSV):
        return set()
    with open(SEQ_CSV) as f:
        return {(r["trained_domain"], r["eval_domain"]) for r in csv.DictReader(f)}


prev_ckpt, t_all = None, time.time()
for idx, domain in enumerate(TRAIN_ORDER):
    if domain not in PREPARED:
        print(f"!! stopping: dataset '{domain}' is not available (see the data cell).")
        break
    existing = find_existing_ckpt(domain) if RESUME else None
    model = None
    if existing:
        print(f"[skip] {domain}: found checkpoint {existing}")
    else:
        t0 = time.time()
        model, _ = train_one_domain(domain, prev_ckpt, is_first=(idx == 0))
        print(f"[time] {domain} trained in {timedelta(seconds=int(time.time() - t0))}")
        existing = ckpt_path(domain)
    seen = TRAIN_ORDER[: idx + 1]
    todo = [d for d in seen if (domain, d) not in _done_evals() and d in PREPARED]
    if todo:
        if model is None:
            model = model_from_ckpt(existing, domain)
        setup_logger(osp.join(OUT_DIR, domain), "test_log.txt")
        evaluate_domains(model, todo, domain, csv_out=SEQ_CSV)
    prev_ckpt = existing
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
print(f"[DONE] total wall time {timedelta(seconds=int(time.time() - t_all))}; results -> {SEQ_CSV}")

# Extra final evaluation when the prompt pool is not full (e.g. kaggle-4 fills 8+8+8+4 = 28 of 36 slots):
# the official test script counts all 36 slots, so 8 never-trained random slots also take part in top-k selection.
# Here the final model is ALSO evaluated with only the slots it actually trained, and both results are reported.
SEQ_CSV_TRAINSLOTS = osp.join(OUT_DIR, "final_eval_trained_slots.csv")
_done_domains = [d for d in TRAIN_ORDER if find_existing_ckpt(d)]
if _done_domains and _done_domains[-1] == TRAIN_ORDER[-1] and not osp.exists(SEQ_CSV_TRAINSLOTS):
    _fm = model_from_ckpt(ckpt_path(TRAIN_ORDER[-1]), TRAIN_ORDER[-1])
    _va = getattr(_fm.image_encoder, "va_prompt", None)
    if _va is not None and int(_va.active_sizes[0, 0]) < _va.pool_size:
        print(f"[extra eval] pool used {int(_va.active_sizes[0, 0])}/{_va.pool_size} slots -> evaluating with trained slots only")
        _flag = OFFICIAL_EVAL_SLOT_RECONSTRUCTION
        OFFICIAL_EVAL_SLOT_RECONSTRUCTION = False
        try:
            setup_logger(osp.join(OUT_DIR, TRAIN_ORDER[-1]), "test_log.txt")
            evaluate_domains(_fm, [d for d in TRAIN_ORDER if d in PREPARED], TRAIN_ORDER[-1], csv_out=SEQ_CSV_TRAINSLOTS)
        finally:
            OFFICIAL_EVAL_SLOT_RECONSTRUCTION = _flag
    del _fm


# %% [markdown]
# ## Unseen-domain generalisation (port of `scripts/eval_unseen.sh`)
# Protocol per dataset, matching Table 1's Unseen-Avg: CUHK02 and SenseReID use 1 deterministic split, GRID / i-LIDS /
# PRID average 10 splits, and VIPeR / CUHK01 average 20 runs (10 identity splits × 2 directions). Seed 123.
# Unseen-Avg is the mean over the datasets that are available.

# %%
UNSEEN_PROTOCOL = {"cuhk02": 1, "sensereid": 1, "grid": 10, "ilids": 10, "prid": 10, "viper": 20, "cuhk01": 20}
UNSEEN_CSV = osp.join(OUT_DIR, "unseen_raw.csv")
unseen_summary = {}
last_domain = [d for d in TRAIN_ORDER if find_existing_ckpt(d)]
avail = [d for d in UNSEEN_PROTOCOL if d in PREPARED]
if RUN_UNSEEN_EVAL and last_domain and avail:
    last_domain = last_domain[-1]
    um = model_from_ckpt(ckpt_path(last_domain), last_domain)
    setup_logger(OUT_DIR, "unseen_log.txt")
    if osp.exists(UNSEEN_CSV):
        os.remove(UNSEEN_CSV)
    for d in avail:
        n = UNSEEN_PROTOCOL[d] if RUN_MODE == "full" else 1
        rows = []
        for sid in range(n):
            rows += evaluate_domains(um, [d], last_domain, csv_out=UNSEEN_CSV,
                                     extra_cfg={"SPLIT_ID": sid, "DIRECTION": "auto", "SEED": 123})
        unseen_summary[d] = (100 * statistics.mean(float(r["rank1"]) for r in rows),
                             100 * statistics.mean(float(r["map"]) for r in rows))
    del um
    for d, (r1, mp) in unseen_summary.items():
        print(f"{d:10s} Rank-1 {r1:6.2f}  mAP {mp:6.2f}")
    print(f"Unseen-Avg ({len(unseen_summary)} sets): Rank-1 {statistics.mean(v[0] for v in unseen_summary.values()):.2f}  "
          f"mAP {statistics.mean(v[1] for v in unseen_summary.values()):.2f}")
else:
    print("unseen evaluation skipped (no unseen datasets attached, or no trained checkpoint yet)")
