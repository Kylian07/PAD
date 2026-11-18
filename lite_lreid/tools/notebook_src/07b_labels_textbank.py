# %% [markdown]
# ## Few-label split and teacher text anchors
# * **Labelled view.** When `LABEL_FRACTION < 1`, a fixed random subset of each domain's training identities keeps its
#   labels (seed `LABEL_SEED`); their images are relabelled 0..K−1. The other images become an *unlabelled pool*.
#   The teacher's training loader is wrapped so that **the teacher only ever sees the labelled identities**.
#   Evaluation (query/gallery) is unchanged.
# * **Text bank.** Right after the teacher finishes a domain, its trained TA-Prompt gives one 512-d text embedding
#   per (labelled) identity: "A photo of a [X1..X4] person." These are saved as `text_bank.pt` next to the teacher
#   checkpoint, because the checkpoint itself keeps only the visual branch.

# %%
class _DatasetView:
    def __init__(self, train, query, gallery, unlabelled):
        self.train, self.query, self.gallery, self.unlabelled = train, query, gallery, unlabelled
        self.num_train_pids = max(1, len({p for _, p, _, _ in train}))


def labelled_view(dataset, name):
    """Keep labels for LABEL_FRACTION of the training identities (deterministic per domain)."""
    if LABEL_FRACTION >= 1.0:
        return _DatasetView(dataset.train, dataset.query, dataset.gallery, [])
    pids = sorted({p for _, p, _, _ in dataset.train})
    rng = random.Random(f"{LABEL_SEED}-{name}")
    keep = set(rng.sample(pids, max(2, int(round(LABEL_FRACTION * len(pids))))))
    relabel = {p: i for i, p in enumerate(sorted(keep))}
    train = [(path, relabel[p], c, v) for path, p, c, v in dataset.train if p in keep]
    unl = [(path, -1, c, v) for path, p, c, v in dataset.train if p not in keep]
    return _DatasetView(train, dataset.query, dataset.gallery, unl)


if "_pad_make_dataloader" not in globals():      # wrap once, even if this cell is re-run
    _pad_make_dataloader = make_dataloader


def make_dataloader(cfg, dataset=None):
    """PAD's make_dataloader; the teacher's training data is restricted to the labelled identities."""
    if dataset is None:
        name = cfg.DATASETS.NAMES[0]
        dataset = labelled_view(build_dataset(name, cfg), name)
    return _pad_make_dataloader(cfg, dataset=dataset)


def text_bank_path(domain):
    return osp.join(OUT_DIR, domain, "text_bank.pt")


@torch.no_grad()
def save_text_bank(model, domain):
    model.eval()
    feats = []
    for s in range(0, model.num_classes, 64):
        lab = torch.arange(s, min(s + 64, model.num_classes), device=DEVICE)
        with _autocast():
            feats.append(model(label=lab, get_text=True).float().cpu())
    bank = torch.cat(feats)
    torch.save(bank, text_bank_path(domain))
    print(f"[text bank] {domain}: {tuple(bank.shape)} -> {text_bank_path(domain)}")


def find_text_bank(domain):
    p = text_bank_path(domain)
    if osp.isfile(p):
        return p
    for d in ALL_INPUT_DIRS:
        q = osp.join(d, "text_bank.pt")
        if osp.basename(d) == domain and osp.basename(osp.dirname(d)) == osp.basename(OUT_DIR) and osp.isfile(q):
            os.makedirs(osp.dirname(p), exist_ok=True); shutil.copy(q, p)
            return p
    return None


print(f"LABEL_FRACTION={LABEL_FRACTION}: "
      + ("fully supervised" if LABEL_FRACTION >= 1 else f"{LABEL_FRACTION:.0%} of identities labelled per domain"))
