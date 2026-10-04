# %% [markdown]
# ## Utilities: EMA teacher and evaluator (port of `pad/utils.py`)
# `EMATeacher` implements θ_tea ← m·θ_tea + (1−m)·θ_stu with m = 0.997, applied to parameters **and** float buffers
# (BN statistics) after every optimiser step.
#
# `R1_mAP_eval` computes the same CMC/mAP as the official evaluator: features are L2-normalised, ranked by Euclidean
# distance, and gallery images with the same identity **and** the same camera as the query are ignored. The official
# version builds the full query×gallery matrix in RAM, which for MSMT17 is 11,659 × 82,161 and needs about 15 GB,
# risky on Kaggle. This version processes queries in chunks on the GPU and gives identical numbers.

# %%
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / max(self.count, 1)


class EMATeacher(nn.Module):
    r"""Momentum teacher: theta_tea = m * theta_tea + (1 - m) * theta_stu  (PAD Eq. 6)."""

    def __init__(self, student, momentum=0.997, device=None):
        super().__init__()
        self.momentum = float(momentum)
        self.teacher = copy.deepcopy(student)
        self.teacher.to(device or next(student.parameters()).device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    @torch.no_grad()
    def update(self, student, momentum=None):
        m = self.momentum if momentum is None else float(momentum)
        cur_params, cur_buffers = dict(student.named_parameters()), dict(student.named_buffers())
        for name, p in self.teacher.named_parameters():
            if name in cur_params and p.shape == cur_params[name].shape and torch.is_floating_point(p):
                p.mul_(m).add_(cur_params[name].detach().to(device=p.device, dtype=p.dtype), alpha=1.0 - m)
        for name, b in self.teacher.named_buffers():
            if name in cur_buffers and b.shape == cur_buffers[name].shape and torch.is_floating_point(b):
                b.mul_(m).add_(cur_buffers[name].detach().to(device=b.device, dtype=b.dtype), alpha=1.0 - m)

    def forward(self, *args, **kwargs):
        return self.teacher(*args, **kwargs)


@torch.no_grad()
def eval_func_chunked(qf, gf, q_pids, g_pids, q_camids, g_camids, max_rank=50, chunk=256):
    """Same maths as the official `_eval_func` (CMC + mAP with same-id/same-cam removal), computed per query chunk."""
    dev = DEVICE
    gf = gf.to(dev).float()
    g_p, g_c = torch.as_tensor(np.asarray(g_pids), device=dev), torch.as_tensor(np.asarray(g_camids), device=dev)
    q_p, q_c = torch.as_tensor(np.asarray(q_pids), device=dev), torch.as_tensor(np.asarray(q_camids), device=dev)
    num_g = gf.shape[0]
    max_rank = min(max_rank, num_g)
    gg = gf.pow(2).sum(1)
    cmc_sum, aps = torch.zeros(max_rank, device=dev), []
    ranks = torch.arange(max_rank, device=dev)
    for s in range(0, qf.shape[0], chunk):
        q = qf[s:s + chunk].to(dev).float()
        dist = q.pow(2).sum(1, keepdim=True) + gg[None, :] - 2.0 * q @ gf.t()
        order = torch.argsort(dist, dim=1)
        gp, gc = g_p[order], g_c[order]
        qp, qc = q_p[s:s + chunk, None], q_c[s:s + chunk, None]
        match = gp == qp
        keep = ~(match & (gc == qc))
        mk = match & keep
        num_rel = mk.sum(1)
        valid = num_rel > 0
        kept_rank = torch.cumsum(keep.int(), dim=1)                         # 1-based rank among kept gallery items
        first = torch.where(mk, kept_rank, torch.full_like(kept_rank, num_g + 1)).min(1).values
        cmc_sum += ((first[valid, None] - 1) <= ranks[None, :]).float().sum(0)
        prec = torch.where(mk, torch.cumsum(mk.int(), dim=1).float() / kept_rank.clamp(min=1).float(),
                           torch.zeros((), device=dev))
        aps.append((prec.sum(1) / num_rel.clamp(min=1).float())[valid].cpu())
    aps = torch.cat(aps)
    assert aps.numel() > 0, "All query identities are missing from the gallery."
    return (cmc_sum / aps.numel()).cpu().numpy(), float(aps.mean())


class R1_mAP_eval:
    def __init__(self, num_query, max_rank=50, feat_norm="yes"):
        self.num_query, self.max_rank, self.feat_norm = num_query, max_rank, feat_norm
        self.reset()

    def reset(self):
        self.feats, self.pids, self.camids = [], [], []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat.cpu()); self.pids.extend(np.asarray(pid)); self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == "yes":
            feats = F.normalize(feats, dim=1, p=2)
        nq = self.num_query
        return eval_func_chunked(feats[:nq], feats[nq:], self.pids[:nq], self.pids[nq:],
                                 self.camids[:nq], self.camids[nq:], self.max_rank)


def setup_logger(save_dir, fname):
    logger = logging.getLogger("pad")
    logger.setLevel(logging.INFO)
    logger.handlers.clear(); logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fh = logging.FileHandler(osp.join(save_dir, fname), mode="a"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


def _max_iters():
    return QUICK["max_iters_per_epoch"] if RUN_MODE == "quick" else None


def _autocast():
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP)


# %% [markdown]
# ## Training engine (port of `pad/engine.py`)
# **Stage 1, TA-Prompt warm-up (paper Sec. 3.3).** The image encoder is frozen, so image features (512-d projected
# CLS) are cached once. Only the per-identity text tokens are trained, with symmetric SupCon (image→text +
# text→image). From the 2nd domain on, **TEXKD** is added. A frozen copy of the model made *before* Stage 1 gives
# fixed text anchors T_e for all classes. For each batch the positive classes plus 256 random negative classes are
# taken, and the student's image→text logit distribution is pulled toward the anchor's:
# KL(softmax(7·S·T_sᵀ / 0.07) ‖ softmax(7·S·T_eᵀ / 0.07))·0.07², weighted 0.5.
#
# **Stage 2, visual branch (paper Sec. 3.4).** The trained TA-Prompt gives a fixed per-class text bank. The visual
# branch trains with ID + triplet + image→text CE. From the 2nd domain on, only the VA-Prompt slots of the current
# domain, the last 4 ViT blocks, and the BN/classifier heads are trainable. **VISKD** distils from an **EMA teacher**
# (initialised from the previous-domain model): 0.5·MSE over [penultimate CLS, last CLS, projected CLS] plus
# 0.5·T²·KL over the image→text logits with T = 4.

# %%
def do_train_stage1(cfg, model, train_loader_stage1, optimizer, scheduler, teacher_model=None):
    logger = logging.getLogger("pad")
    logger.info("[Stage1] start")
    epochs, log_period = cfg.SOLVER.STAGE1.MAX_EPOCHS, cfg.SOLVER.STAGE1.LOG_PERIOD
    textkd_on = bool(cfg.TEXTKD.ENABLE) and (teacher_model is not None)
    if textkd_on:
        kd_text = KDLosses(lambda_feat=0.0, lambda_logit=float(cfg.TEXTKD.LAMBDA_LOGIT), temperature=float(cfg.TEXTKD.TEMP))
        logit_scale = float(cfg.TEXTKD.LOGITS_SCALE)
    else:
        kd_text, logit_scale = None, 1.0
    model.to(DEVICE)
    student_ref = model
    total = sum(p.numel() for p in model.parameters()); trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[Stage1] trainable params: {trainable}/{total} ({trainable / max(total, 1):.2%})")
    loss_meter, scaler, supcon = AverageMeter(), torch.amp.GradScaler("cuda", enabled=USE_AMP), SupConLoss(DEVICE)
    t_start = time.monotonic()

    # Cache image features once (encoder is frozen during Stage 1).
    logger.info("[Stage1] caching image features ...")
    model.eval()
    feats_all, labels_all = [], []
    with torch.no_grad():
        for it, (img, vid, _, _) in enumerate(train_loader_stage1):
            if _max_iters() and it >= _max_iters() * 4:
                break
            img, vid = img.to(DEVICE, non_blocking=True), vid.to(DEVICE, non_blocking=True)
            with _autocast():
                feats_all.append(model(img, vid, get_image=True).detach().float().cpu())
            labels_all.append(vid.detach().cpu())
    feats_all = torch.cat(feats_all).to(DEVICE).float()
    labels_all = torch.cat(labels_all).to(DEVICE)
    N = labels_all.size(0)
    model.train()
    batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
    iters_per_epoch = (N + batch - 1) // batch
    logger.info(f"[Stage1] cached {N} samples, {iters_per_epoch} iters/epoch")

    # Frozen text-anchor bank for TEXKD.
    Te_all = None
    if textkd_on:
        teacher = teacher_model.to(DEVICE).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        C = int(student_ref.prompt_learner.num_class)
        with torch.no_grad(), _autocast():
            Te_all = teacher(label=torch.arange(C, device=DEVICE, dtype=torch.long), get_text=True).detach().float()
        Te_all = Te_all / Te_all.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
    neg_K = int(cfg.TEXTKD.NEG_PER_BATCH)

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()
        kd_running = None
        idx_all = torch.randperm(N, device=DEVICE)
        for it in range(iters_per_epoch):
            s, e = it * batch, min((it + 1) * batch, N)
            if s >= e:
                continue
            idx = idx_all[s:e]
            cur_feats, cur_labels = feats_all[idx], labels_all[idx]
            optimizer.zero_grad(set_to_none=True)
            with _autocast():
                text_feats = model(label=cur_labels, get_text=True).float()
                loss = supcon(cur_feats, text_feats, cur_labels, cur_labels) + supcon(text_feats, cur_feats, cur_labels, cur_labels)
            if textkd_on:
                with torch.no_grad():
                    uniq_pos, _ = torch.unique(cur_labels, sorted=False, return_inverse=True)
                    C = Te_all.size(0)
                    mask = torch.ones(C, dtype=torch.bool, device=Te_all.device); mask[uniq_pos] = False
                    neg_pool = torch.nonzero(mask, as_tuple=False).squeeze(1)
                    if neg_K > 0 and neg_pool.numel() > 0:
                        neg_idx = neg_pool if neg_pool.numel() <= neg_K else neg_pool[torch.randperm(neg_pool.numel(), device=neg_pool.device)[:neg_K]]
                        cls_idx = torch.cat([uniq_pos, neg_idx], dim=0)
                    else:
                        cls_idx = uniq_pos
                with _autocast():
                    Ts_sub = model(label=cls_idx, get_text=True)
                Ts_sub = Ts_sub.float(); Ts_sub = Ts_sub / Ts_sub.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
                Te_sub = Te_all.index_select(0, cls_idx)
                S = cur_feats / cur_feats.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
                kl = kd_text.kl((S @ Ts_sub.t()) * logit_scale, (S @ Te_sub.t()) * logit_scale)
                loss = loss + kd_text.lambda_logit * kl
                v = float(kl.item()); kd_running = v if kd_running is None else 0.9 * kd_running + 0.1 * v
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_meter.update(loss.item(), cur_feats.size(0))
            if (it + 1) % log_period == 0:
                msg = (f"Stage1 Epoch[{epoch}] Iter[{it + 1}/{iters_per_epoch}] Loss {loss_meter.avg:.3f} "
                       f"LR {optimizer.param_groups[0]['lr']:.2e}")
                if textkd_on and kd_running is not None:
                    msg += f" TEXKD {kd_running:.3f}"
                logger.info(msg)
        logger.info(f"[Stage1] epoch {epoch}/{epochs} loss {loss_meter.avg:.4f}")
    logger.info(f"[Stage1] total time {timedelta(seconds=int(time.monotonic() - t_start))}")


def do_train_stage2(cfg, model, train_loader_stage2, optimizer, scheduler, loss_fn, teacher_model=None):
    logger = logging.getLogger("pad")
    logger.info("[Stage2] start")
    epochs, log_period = cfg.SOLVER.STAGE2.MAX_EPOCHS, cfg.SOLVER.STAGE2.LOG_PERIOD
    model.to(DEVICE)
    student_ref, num_classes = model, model.num_classes
    distill_on = bool(cfg.DISTILL.ENABLE) and (teacher_model is not None)
    ema_teacher = kd_criterion = None
    if distill_on:
        ema_teacher = EMATeacher(teacher_model.to(DEVICE), momentum=float(cfg.DISTILL.EMA_MOMENTUM), device=torch.device(DEVICE))
        kd_criterion = KDLosses(lambda_feat=float(cfg.DISTILL.LAMBDA_FEAT), lambda_logit=float(cfg.DISTILL.LAMBDA_LOGIT),
                                temperature=float(cfg.DISTILL.TEMP))
    loss_meter, acc_meter, scaler = AverageMeter(), AverageMeter(), torch.amp.GradScaler("cuda", enabled=USE_AMP)
    t_start = time.monotonic()

    # Per-class text bank from the trained TA-Prompt (computed once).
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    text_bank = []
    with torch.no_grad():
        for s in range(0, num_classes, batch):
            with _autocast():
                text_bank.append(model(label=torch.arange(s, min(s + batch, num_classes), device=DEVICE), get_text=True).cpu())
    text_bank = torch.cat(text_bank).to(DEVICE).float()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        loss_meter.reset(); acc_meter.reset()
        scheduler.step()
        model.train()
        n_iter = 0
        for n_iter, (img, vid, _, _) in enumerate(train_loader_stage2):
            if _max_iters() and n_iter >= _max_iters():
                break
            img, target = img.to(DEVICE, non_blocking=True), vid.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast():
                score, feat, image_features = model(x=img, label=target)
                logits = image_features @ text_bank.t()
                loss = loss_fn(score, feat, target, logits)
                if distill_on:
                    with torch.no_grad():
                        _, t_feat, t_image_features = ema_teacher(x=img, label=target, return_train_outputs=True)
                        t_logits = t_image_features @ text_bank.t()
                    kd_total, kd_feat_val, kd_kl_val = kd_criterion(student_feats=feat, teacher_feats=t_feat,
                                                                    student_logits=logits, teacher_logits=t_logits)
                    loss = loss + kd_total
                else:
                    kd_feat_val = kd_kl_val = None
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            if distill_on:
                ema_teacher.update(student_ref)
            acc = (logits.max(1)[1] == target).float().mean().item()
            loss_meter.update(loss.item(), img.size(0)); acc_meter.update(acc, 1)
            if (n_iter + 1) % log_period == 0:
                msg = (f"Stage2 Epoch[{epoch}] Iter[{n_iter + 1}/{len(train_loader_stage2)}] Loss {loss_meter.avg:.3f} "
                       f"Acc {acc_meter.avg:.3f} LR {optimizer.param_groups[0]['lr']:.2e}")
                if distill_on and kd_feat_val is not None:
                    msg += f" KD_feat {kd_feat_val.item():.3f} KD_kl {kd_kl_val.item():.3f}"
                logger.info(msg)
        dt = (time.time() - t_epoch) / max(n_iter + 1, 1)
        logger.info(f"[Stage2] epoch {epoch}/{epochs} loss {loss_meter.avg:.4f} acc {acc_meter.avg:.3f} "
                    f"({dt:.3f}s/batch, {train_loader_stage2.batch_size / max(dt, 1e-6):.0f} img/s)")
    logger.info(f"[Stage2] total time {timedelta(seconds=int(time.monotonic() - t_start))}")


@torch.no_grad()
def do_inference(cfg, model, val_loader, num_query):
    """fp32 feature extraction, exactly like the official do_inference (no autocast)."""
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    model.to(DEVICE).eval()
    for img, pid, camid, _, _, _ in val_loader:
        evaluator.update((model(img.to(DEVICE)), pid, camid))
    cmc, mAP = evaluator.compute()
    return float(cmc[0]), float(cmc[4]), float(mAP)
