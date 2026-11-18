# %% [markdown]
# ## Student training for one domain
# For domain *t*, each step takes one PK batch of labelled images (P identities × 4), plus, when `LABEL_FRACTION < 1`,
# a batch of unlabelled images. The losses (weights in `LAMBDA`) are:
#
# | Term | Data | What it does |
# |---|---|---|
# | `id` | labelled | label-smoothed cross-entropy on the BNNeck classifier (current identities) |
# | `tri` | labelled | batch-hard triplet loss, margin 0.3, on f |
# | `i2t` | labelled | cross-entropy between the student's 512-d embedding and the teacher's **text anchors** (cosine / τ) |
# | `fkd` | all | 1 − cos(W·f, teacher's projected CLS feature) (W is a training-only linear layer) |
# | `rkd` | all | KL between the student's and the teacher's in-batch similarity distributions (relation distillation) |
# | `old` | all, t > 1 | 1 − cos(f, f_old), where f_old comes from the frozen previous student |
# | `proto` | all, t > 1 | KL between the similarity of f and of f_old to the **stored identity prototypes** of all earlier domains |
#
# **Prototype memory.** After each domain, the student stores one L2-normalised mean feature per labelled identity
# (`protos.pt`, a few hundred kB). No images are kept. The `naive_ft` baseline uses only `id` + `tri`.
#
# The schedule is AdamW with cosine decay and warm-up, AMP, and the teacher's augmentation. A resumable state is saved
# every `STUDENT_CKPT_EVERY` epochs. Training stops cleanly before `MAX_SESSION_HOURS`.

# %%
RUN_TAG = f"{STUDENT_ARCH.replace(':', '-')}_lf{LABEL_FRACTION:g}"


def lite_dir(method, domain):
    return osp.join(LITE_OUT, f"{method}_{RUN_TAG}", domain)


def find_lite_file(method, domain, fname):
    """A student file from this session, or from an attached earlier output (copied in, with its folder)."""
    p = osp.join(lite_dir(method, domain), fname)
    if osp.isfile(p):
        return p
    pattern = osp.join(INPUT_ROOT, "**", "LiteLReID_outputs", f"{method}_{RUN_TAG}", domain, fname)
    for q in glob.glob(pattern, recursive=True):
        os.makedirs(osp.dirname(p), exist_ok=True)
        for f in glob.glob(osp.join(osp.dirname(q), "*")):
            if osp.isfile(f) and not osp.exists(osp.join(osp.dirname(p), osp.basename(f))):
                shutil.copy(f, osp.dirname(p))
        print(f"[resume] copied {osp.dirname(q)} -> {osp.dirname(p)}")
        return p
    return None


def _hours_used():
    return (time.time() - T_NOTEBOOK_START) / 3600


def _kl_rows(student_logits, teacher_logits):
    return F.kl_div(F.log_softmax(student_logits, dim=1), F.softmax(teacher_logits, dim=1), reduction="batchmean")


def _sim_logits(f, tau):
    f = F.normalize(f, dim=1)
    s = f @ f.t() / tau
    return s.masked_fill(torch.eye(len(f), dtype=torch.bool, device=f.device), -1e4)


@torch.no_grad()
def compute_prototypes(student, ds, cfg):
    _, loader, _, _, num_classes = _pad_make_dataloader(cfg, dataset=ds)
    student.eval()
    sums = torch.zeros(num_classes, student.feat_dim)
    cnt = torch.zeros(num_classes)
    for it, (img, pid, _, _) in enumerate(loader):
        if RUN_MODE == "quick" and it >= QUICK["max_iters_per_epoch"]:
            break
        with _autocast():
            f = student(img.to(DEVICE)).float().cpu()
        sums.index_add_(0, pid, F.normalize(f, dim=1)); cnt.index_add_(0, pid, torch.ones(len(pid)))
    keep = cnt > 0
    return F.normalize(sums[keep] / cnt[keep, None], dim=1)


def _cycle(loader):
    while True:
        for b in loader:
            yield b


def train_student_domain(method, domain, step_idx, prev_domain):
    """Train the student on one domain. Returns the trained student, or None if stopped by the session limit."""
    cfg = build_cfg(domain, step_idx == 0)
    set_seed(cfg.SOLVER.SEED)
    torch.backends.cudnn.benchmark = True       # the student is not part of the PAD fidelity check
    ddir = lite_dir(method, domain); os.makedirs(ddir, exist_ok=True)
    lg = setup_logger(ddir, "student_log.txt")
    ds = labelled_view(build_dataset(domain, cfg, verbose=False), domain)
    train_loader, _, _, _, num_classes = _pad_make_dataloader(cfg, dataset=ds)
    if len(train_loader) == 0:
        raise ValueError(f"{domain}: {num_classes} labelled identities cannot fill one PK batch of "
                         f"{train_loader.batch_size} ({train_loader.batch_size // cfg.DATALOADER.NUM_INSTANCE} ids x "
                         f"{cfg.DATALOADER.NUM_INSTANCE}); raise LABEL_FRACTION")
    unl_iter = None
    if ds.unlabelled and method == "ours":
        unl_loader = DataLoader(ImageDataset(ds.unlabelled, train_loader.dataset.transform),
                                batch_size=STUDENT_UNLABELLED_BATCH, shuffle=True, drop_last=True,
                                num_workers=NUM_WORKERS, collate_fn=_train_collate, pin_memory=True)
        unl_iter = _cycle(unl_loader)

    phase = "first" if step_idx == 0 else "later"
    epochs = QUICK.get("student_epochs", 1) if RUN_MODE == "quick" else STUDENT_EPOCHS[phase]
    lr = STUDENT_LR[phase]
    student = StudentReID(STUDENT_ARCH, num_classes).to(DEVICE)
    if prev_domain:
        sd = torch.load(find_lite_file(method, prev_domain, "student.pth"), map_location="cpu")
        miss = student.load_state_dict(sd, strict=False)
        assert all(k.startswith("classifier.") for k in miss.missing_keys) and not miss.unexpected_keys, miss
    lg.info(f"[{method}] domain={domain} step={step_idx + 1} labelled ids={num_classes} imgs={len(ds.train)} "
            f"unlabelled imgs={len(ds.unlabelled)} epochs={epochs} lr={lr} batch={train_loader.batch_size}")

    teacher = bank = old_student = protos = None
    kd_proj = nn.Linear(student.feat_dim, 512, bias=False).to(DEVICE)
    if method == "ours":
        base = model_from_ckpt(find_existing_ckpt(domain), domain)
        teacher = prepare_eval_model(base); del base
        for p in teacher.parameters():
            p.requires_grad_(False)
        tb = find_text_bank(domain)
        if tb:
            bank = F.normalize(torch.load(tb, map_location="cpu").float(), dim=1).to(DEVICE)
            if bank.shape[0] != num_classes:
                lg.info(f"!! text bank has {bank.shape[0]} ids, expected {num_classes}: i2t disabled"); bank = None
        else:
            lg.info("!! no text bank for this domain (teacher trained elsewhere?): i2t disabled")
        if prev_domain:
            old_student = copy.deepcopy(student).eval()
            for p in old_student.parameters():
                p.requires_grad_(False)
            plist = [find_lite_file(method, d, "protos.pt") for d in TRAIN_ORDER[:step_idx]]
            protos = F.normalize(torch.cat([torch.load(p) for p in plist]).float(), dim=1).to(DEVICE)
            lg.info(f"[memory] {len(protos)} old identity prototypes ({protos.numel() * 4 / 1024:.0f} kB, no images)")

    params = [p for p in student.parameters() if p.requires_grad] + list(kd_proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=STUDENT_WD)
    scheduler = CosineLRScheduler(optimizer, t_initial=epochs, lr_min=lr * 0.01, warmup_t=min(STUDENT_WARMUP_EPOCHS, epochs - 1),
                                  warmup_lr_init=lr * 0.01, warmup_prefix=True)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    xent, tri = CrossEntropyLabelSmooth(num_classes), TripletLoss(0.3)

    start = 1
    sp = find_lite_file(method, domain, "state.pth")
    if sp:
        st = torch.load(sp, map_location="cpu", weights_only=False)
        student.load_state_dict(st["student"]); kd_proj.load_state_dict(st["kd_proj"])
        optimizer.load_state_dict(st["optimizer"]); scaler.load_state_dict(st["scaler"])
        random.setstate(st["rng_py"]); np.random.set_state(st["rng_np"]); torch.set_rng_state(st["rng_torch"])
        start = st["epoch"] + 1
        lg.info(f"[resume] from epoch {st['epoch']}")

    def save_state(epoch):
        torch.save({"student": student.state_dict(), "kd_proj": kd_proj.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "rng_py": random.getstate(),
                    "rng_np": np.random.get_state(), "rng_torch": torch.get_rng_state()}, osp.join(ddir, "state.pth.tmp"))
        os.replace(osp.join(ddir, "state.pth.tmp"), osp.join(ddir, "state.pth"))

    times = []
    for epoch in range(start, epochs + 1):
        if times and _hours_used() + 1.3 * np.mean(times) / 3600 + 0.3 > MAX_SESSION_HOURS:
            save_state(epoch - 1)
            lg.info(f"[time] {_hours_used():.1f} h used: stopping at epoch {epoch - 1}. Attach this output and re-run.")
            return None
        t0 = time.time()
        scheduler.step(epoch - 1)
        student.train()
        meters = defaultdict(float); n_seen = 0; it = 0
        for it, (img, pid, _, _) in enumerate(train_loader):
            if RUN_MODE == "quick" and it >= QUICK["max_iters_per_epoch"]:
                break
            img, pid = img.to(DEVICE, non_blocking=True), pid.to(DEVICE, non_blocking=True)
            x = img
            if unl_iter is not None:
                x = torch.cat([img, next(unl_iter)[0].to(DEVICE, non_blocking=True)])
            nl = img.size(0)
            with _autocast():
                score, f = student(x)
                parts = {"id": xent(score[:nl].float(), pid), "tri": tri(f[:nl], pid)[0]}
                if teacher is not None:
                    with torch.no_grad():
                        t = teacher(x).float()                     # [CLS 768 ; projected CLS 512]
                    if bank is not None:
                        logits = F.normalize(student.to_text(f[:nl]), dim=1) @ bank.t() / TAU["i2t"]
                        parts["i2t"] = F.cross_entropy(logits.float(), pid)
                    parts["fkd"] = (1 - F.cosine_similarity(kd_proj(f), t[:, 768:], dim=1)).mean()
                    parts["rkd"] = _kl_rows(_sim_logits(f, TAU["rkd"]), _sim_logits(t, TAU["rkd"]))
                if old_student is not None:
                    with torch.no_grad():
                        f_old = old_student(x).float()
                    parts["old"] = (1 - F.cosine_similarity(f, f_old, dim=1)).mean()
                    parts["proto"] = _kl_rows(F.normalize(f, dim=1) @ protos.t() / TAU["proto"],
                                              F.normalize(f_old, dim=1) @ protos.t() / TAU["proto"])
                loss = sum(LAMBDA[k] * v for k, v in parts.items())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            for k, v in parts.items():
                meters[k] += float(v)
            meters["loss"] += float(loss); meters["acc"] += float((score[:nl].argmax(1) == pid).float().mean())
            n_seen += x.size(0)
        n_it = max(it + 1, 1)
        times.append(time.time() - t0)
        lg.info(f"epoch {epoch}/{epochs} lr {optimizer.param_groups[0]['lr']:.2e} | "
                + " ".join(f"{k} {v / n_it:.3f}" for k, v in meters.items())
                + f" | {times[-1]:.0f}s ({n_seen / max(times[-1], 1e-9):.0f} img/s)")
        if epoch == start:
            lg.info(f"[estimate] ~{times[-1] * (epochs - epoch) / 3600:.1f} h left for this domain")
        if epoch % STUDENT_CKPT_EVERY == 0 and epoch != epochs:
            save_state(epoch)

    torch.save(student_state_for_saving(student), osp.join(ddir, "student.pth"))
    if method == "ours":
        pr = compute_prototypes(student, ds, cfg)
        torch.save(pr, osp.join(ddir, "protos.pt"))
        lg.info(f"[memory] stored {len(pr)} prototypes for {domain} ({pr.numel() * 4 / 1024:.0f} kB)")
    if osp.isfile(osp.join(ddir, "state.pth")):
        os.remove(osp.join(ddir, "state.pth"))
    del teacher, old_student
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return student


def load_student(method, domain):
    m = StudentReID(STUDENT_ARCH, 1).to(DEVICE)
    m.load_state_dict(torch.load(find_lite_file(method, domain, "student.pth"), map_location="cpu"), strict=False)
    return m.eval()


def evaluate_student(student, eval_domains, method, trained_domain, csv_out, extra_cfg=None):
    lg = logging.getLogger("pad")
    rows = []
    for d in eval_domains:
        cfg = build_cfg(d, False)
        for k, v in (extra_cfg or {}).items():
            cfg.EVAL[k] = v
        ds = build_dataset(d, cfg, verbose=False)
        _, _, val_loader, num_query, _ = _pad_make_dataloader(cfg, dataset=ds)
        r1, r5, mAP = do_inference(cfg, student, val_loader, num_query)
        lg.info(f"[Eval student {method}] trained={trained_domain} eval={d} Rank-1={r1:.4f} mAP={mAP:.4f}")
        row = {"method": method, "arch": STUDENT_ARCH, "label_fraction": LABEL_FRACTION, "trained_domain": trained_domain,
               "eval_domain": d, "rank1": f"{r1:.4f}", "rank5": f"{r5:.4f}", "map": f"{mAP:.4f}",
               "split_id": (extra_cfg or {}).get("SPLIT_ID", "")}
        rows.append(row)
        new = not osp.exists(csv_out)
        with open(csv_out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
    return rows
