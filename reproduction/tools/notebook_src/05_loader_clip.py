# %% [markdown]
# ## DataLoaders (port of `pad/data/loader.py`)
# * **Stage-2 loader:** strong augmentation (flip, pad+crop, random erasing) plus the **PK identity sampler**
#   (P = batch/4 identities × K = 4 images each). The batch-hard triplet loss needs this.
# * **Stage-1 loader:** test-time transform, shuffled. Used only once, to cache frozen image features.
# * **Val loader:** query followed by gallery.

# %%
def _train_collate(batch):
    imgs, pids, camids, viewids, _ = zip(*batch)
    return (torch.stack(imgs, dim=0), torch.tensor(pids, dtype=torch.int64),
            torch.tensor(camids, dtype=torch.int64), torch.tensor(viewids, dtype=torch.int64))


def _val_collate(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    return (torch.stack(imgs, dim=0), pids, camids, torch.tensor(camids, dtype=torch.int64),
            torch.tensor(viewids, dtype=torch.int64), img_paths)


def _quick_subsample_eval(query, gallery, seed=0):
    """Quick mode only: shrink query/gallery so evaluation takes seconds (metrics are NOT comparable)."""
    rng = random.Random(seed)
    q = rng.sample(query, min(len(query), QUICK["eval_max_query"]))
    qp = {x[1] for x in q}
    g_pos = [x for x in gallery if x[1] in qp]
    g_neg = [x for x in gallery if x[1] not in qp]
    g = g_pos + rng.sample(g_neg, max(0, min(len(g_neg), QUICK["eval_max_gallery"] - len(g_pos))))
    return q, g


def make_dataloader(cfg, dataset=None):
    train_tf = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=InterpolationMode.BICUBIC),   # == interpolation=3
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"),
    ])
    val_tf = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])
    name = cfg.DATASETS.NAMES[0]
    if dataset is None:
        dataset = build_dataset(name, cfg)
    query, gallery = dataset.query, dataset.gallery
    if RUN_MODE == "quick":
        query, gallery = _quick_subsample_eval(query, gallery)

    nw = cfg.DATALOADER.NUM_WORKERS
    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_tf)
    train_set_normal = ImageDataset(dataset.train, val_tf)
    val_set = ImageDataset(query + gallery, val_tf)

    train_loader_stage2 = train_loader_stage1 = None
    if len(dataset.train) > 0:
        train_loader_stage2 = DataLoader(
            train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=nw, collate_fn=_train_collate, pin_memory=True, persistent_workers=nw > 0)
        train_loader_stage1 = DataLoader(
            train_set_normal, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True,
            num_workers=nw, collate_fn=_train_collate, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
                            num_workers=nw, collate_fn=_val_collate, pin_memory=True)
    return train_loader_stage2, train_loader_stage1, val_loader, len(query), num_classes


# %% [markdown]
# ## CLIP ViT-B/16 backbone (port of `pad/clip/model.py`, ViT part only)
# This is the standard OpenAI CLIP ViT with one PAD change in `VisionTransformer.forward`. At **every**
# transformer block, the VA-Prompt tokens `[G-Prompt ; top-k E-Prompts]` are inserted right after `[CLS]`, the block
# runs, and the prompt tokens are **stripped** again. Patch positions therefore never shift, and each layer gets
# fresh prompts. The positional embedding is bilinearly resized from 14×14 to 16×8 patches for 256×128 inputs.

# %%
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x):
        orig_type = x.dtype
        return super().forward(x.type(torch.float32)).type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)), ("gelu", QuickGELU()), ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.width, self.layers = width, layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    """CLIP ViT visual encoder with optional per-layer VA-Prompt injection (PAD Sec. 3.4.2)."""

    def __init__(self, h_resolution, w_resolution, patch_size, stride_size, width, layers, heads, output_dim):
        super().__init__()
        self.h_resolution, self.w_resolution, self.output_dim = h_resolution, w_resolution, output_dim
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=stride_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(h_resolution * w_resolution + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.va_strip_after_block = True

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls_tok = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls_tok, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x).permute(1, 0, 2)                     # NLD -> LND

        use_va = hasattr(self, "va_prompt") and (self.va_prompt is not None)
        idx_pen = min(10, len(self.transformer.resblocks) - 1)  # penultimate block output, used by VISKD
        x_pen_lnd = None
        for li, block in enumerate(self.transformer.resblocks):
            if use_va:
                x_nld = x.permute(1, 0, 2)
                out = self.va_prompt.forward_for_layer(li, x_nld, cls_feat=None)
                prompts, strip_len = out["batched"], out["strip_len"]
                x_cat = torch.cat([x_nld[:, :1, :], prompts, x_nld[:, 1:, :]], dim=1)   # [CLS | prompts | patches]
                x_after = block(x_cat.permute(1, 0, 2)).permute(1, 0, 2)
                if self.va_strip_after_block:
                    x_after = torch.cat([x_after[:, :1, :], x_after[:, 1 + strip_len:, :]], dim=1)
                x = x_after.permute(1, 0, 2)
            else:
                x = block(x)
            if li == idx_pen:
                x_pen_lnd = x.clone()
        x_last_lnd = x
        x_pen = (x_pen_lnd if x_pen_lnd is not None else x_last_lnd).permute(1, 0, 2)
        x_last = self.ln_post(x_last_lnd.permute(1, 0, 2))
        x_proj = x_last @ self.proj if self.proj is not None else x_last
        return x_pen, x_last, x_proj


class CLIP(nn.Module):
    def __init__(self, embed_dim, vision_layers, vision_width, vision_patch_size, vision_stride_size,
                 context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
                 h_resolution, w_resolution):
        super().__init__()
        self.context_length = context_length
        self.visual = VisionTransformer(h_resolution, w_resolution, vision_patch_size, vision_stride_size,
                                        vision_width, vision_layers, vision_width // 64, embed_dim)
        self.transformer = Transformer(transformer_width, transformer_layers, transformer_heads,
                                       attn_mask=self.build_attention_mask())
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype


def resize_pos_embed(posemb, posemb_new, hight, width):
    ntok_new = posemb_new.shape[0] - 1
    posemb_token, posemb_grid = posemb[:1], posemb[1:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    print(f"Resized position embedding: {tuple(posemb.shape)} -> {tuple(posemb_new.shape)} (grid {gs_old}x{gs_old} -> {hight}x{width})")
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    return torch.cat([posemb_token, posemb_grid.squeeze()], dim=0)


def convert_weights(model):
    """(Official name kept) -- actually casts Linear/Conv/MHA/projection weights to fp32."""
    def _f(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                t = getattr(l, attr)
                if t is not None:
                    t.data = t.data.float()
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                a = getattr(l, name)
                if a is not None:
                    a.data = a.data.float()
    model.apply(_f)


def build_model(state_dict, h_resolution, w_resolution, vision_stride_size):
    state_dict = {k: v for k, v in state_dict.items() if k not in ("input_resolution", "context_length", "vocab_size")}
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    model = CLIP(embed_dim, vision_layers, vision_width, vision_patch_size, vision_stride_size, context_length,
                 vocab_size, transformer_width, transformer_width // 64, transformer_layers, h_resolution, w_resolution)
    pe = state_dict["visual.positional_embedding"]
    try:   # official code interpolates in the checkpoint dtype (fp16 for the OpenAI JIT file)
        state_dict["visual.positional_embedding"] = resize_pos_embed(pe, model.visual.positional_embedding, h_resolution, w_resolution)
    except RuntimeError:   # older CPU kernels lack fp16 bilinear
        state_dict["visual.positional_embedding"] = resize_pos_embed(pe.float(), model.visual.positional_embedding, h_resolution, w_resolution)
    convert_weights(model)
    expected = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in expected}   # drop extras some mirrors carry
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


# ---- CLIP weights: tried in order; the first that works is used -----------------------------------------------------
CLIP_SHA = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
CLIP_URLS = [f"https://openaipublic.azureedge.net/clip/models/{CLIP_SHA}/ViT-B-16.pt",          # official (as in repo)
             f"https://openaipublic.blob.core.windows.net/clip/models/{CLIP_SHA}/ViT-B-16.pt"]  # same file, origin host


def _clip_state_dict_from_file(path):
    try:
        return torch.jit.load(path, map_location="cpu").eval().state_dict()
    except RuntimeError:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        return sd.get("state_dict", sd) if isinstance(sd, dict) else sd.state_dict()


_CLIP_SD_CACHE = {}


def load_clip_state_dict():
    if "sd" in _CLIP_SD_CACHE:
        return {k: v.clone() for k, v in _CLIP_SD_CACHE["sd"].items()}
    sd = None
    # 1) a copy attached as a Kaggle input (any file named ViT-B-16.pt), or cached from an earlier cell run
    local = [osp.join(CLIP_CACHE, "ViT-B-16.pt")] + [osp.join(d, "ViT-B-16.pt") for d in ALL_INPUT_DIRS]
    for p in local:
        if osp.isfile(p):
            sd = _clip_state_dict_from_file(p); print(f"[CLIP] loaded {p}"); break
    # 2) the official OpenAI URL used by the repo (sha256-verified)
    if sd is None:
        target = osp.join(CLIP_CACHE, "ViT-B-16.pt")
        for url in CLIP_URLS:
            try:
                print(f"[CLIP] downloading {url}")
                urllib.request.urlretrieve(url, target + ".part")
                if hashlib.sha256(open(target + ".part", "rb").read()).hexdigest() != CLIP_SHA:
                    raise RuntimeError("sha256 mismatch")
                os.replace(target + ".part", target)
                sd = _clip_state_dict_from_file(target); break
            except Exception as e:  # noqa: BLE001
                print(f"[CLIP]   failed: {e}")
    # 3) the same OpenAI weights mirrored on the Hugging Face hub (open_clip format, identical key names)
    if sd is None:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("timm/vit_base_patch16_clip_224.openai", "open_clip_pytorch_model.bin", cache_dir=CLIP_CACHE)
        sd = torch.load(p, map_location="cpu", weights_only=False); print(f"[CLIP] loaded HF mirror {p}")
    _CLIP_SD_CACHE["sd"] = {k: v.clone() for k, v in sd.items()}
    return sd


# %% [markdown]
# ## PAD model (port of `pad/model.py`)
# * **`VAPromptPool`** (visual side): for each of the 12 layers there is one shared **G-Prompt** (6 tokens) and a pool
#   of 36 **E-Prompt** slots (6 tokens each) with keys. A query (L2-normalised mean token of the layer input) picks
#   the **top-4** slots by cosine similarity. Each new domain opens fresh slots (8, or 4 for CUHK03), and slots of
#   earlier domains are **frozen** with a gradient mask.
# * **`TAPromptLearner`** (text side): `"A photo of a [X1][X2][X3][X4] person."`, where the 4 `[X]` tokens are
#   learnable **per identity**. The CLIP text encoder stays frozen.
# * **`PADModel`**: two BNNeck heads (768-d CLS and 512-d projected CLS), ID classifiers, and the text branch.
#
# The prompt template is tokenised once **offline**, so the notebook needs neither the CLIP BPE vocabulary file nor
# `ftfy`/`regex`. The IDs below are exactly what `clip.tokenize("A photo of a X X X X person.")` returns.

# %%
# <|startoftext|> a photo of a x x x x person . <|endoftext|>  (padded with zeros to 77)
_PROMPT_TOKEN_IDS = [49406, 320, 1125, 539, 320, 343, 343, 343, 343, 2533, 269, 49407]


def tokenize_prompt_template(context_length=77):
    t = torch.zeros(1, context_length, dtype=torch.long)
    t[0, :len(_PROMPT_TOKEN_IDS)] = torch.tensor(_PROMPT_TOKEN_IDS)
    return t


def _l2(x, dim=-1, eps=1e-12):
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp(min=eps)


class VAPromptPool(nn.Module):
    """Per-layer {G-Prompt, E-Prompt pool} with lifelong slot allocation."""

    def __init__(self, num_layers, embed_dim=768, g_len=6, e_len=6, pool_size=36, top_k=4,
                 embedding_key="mean", key_learnable=True):
        super().__init__()
        assert num_layers > 0 and 0 < top_k <= pool_size and embedding_key in ("mean", "cls")
        self.num_layers, self.embed_dim, self.g_len, self.e_len = num_layers, embed_dim, g_len, e_len
        self.pool_size, self.top_k, self.embedding_key, self.key_learnable = pool_size, top_k, embedding_key, key_learnable
        self.g_prompt = nn.ParameterList(nn.Parameter(self._init((g_len, embed_dim))) for _ in range(num_layers))
        self.e_prompt = nn.ParameterList(nn.Parameter(self._init((pool_size, e_len, embed_dim))) for _ in range(num_layers))
        if key_learnable:
            self.e_key = nn.ParameterList(nn.Parameter(self._init((pool_size, embed_dim))) for _ in range(num_layers))
        else:
            self.e_key = nn.ParameterList(nn.Parameter(torch.empty(0), requires_grad=False) for _ in range(num_layers))
        self.register_buffer("active_sizes", torch.zeros(num_layers, 1, dtype=torch.long))
        self.register_buffer("frozen_masks", torch.zeros(num_layers, pool_size, dtype=torch.bool))
        self.register_buffer("trainable_masks", torch.zeros(num_layers, pool_size, dtype=torch.bool))

    @staticmethod
    def _init(shape):
        t = torch.zeros(shape)
        nn.init.uniform_(t, a=-0.02, b=0.02)
        return t

    def _query(self, x_noprompt, cls_feat):
        q = x_noprompt.mean(dim=1) if self.embedding_key == "mean" else (cls_feat if cls_feat is not None else x_noprompt[:, 0, :])
        return _l2(q, dim=-1)

    def _keys(self, layer_id):
        k = self.e_key[layer_id] if self.key_learnable else self.e_prompt[layer_id].mean(dim=1)
        return _l2(k, dim=-1)

    @torch.no_grad()
    def _topk(self, layer_id, q):
        keys = self._keys(layer_id)
        used = int(self.active_sizes[layer_id, 0].item())
        if used <= 0:
            used = min(self.pool_size, self.top_k)
        sim = q @ keys[:used].t()
        _, idx = torch.topk(sim, k=min(self.top_k, used), dim=1)
        return idx

    def forward_for_layer(self, layer_id, x_noprompt, cls_feat=None):
        B, _, C = x_noprompt.shape
        q = self._query(x_noprompt, cls_feat)
        idx = self._topk(layer_id, q)
        ep = self.e_prompt[layer_id][idx].reshape(B, -1, C).to(device=x_noprompt.device, dtype=x_noprompt.dtype)
        gp = self.g_prompt[layer_id].unsqueeze(0).expand(B, -1, -1).to(device=x_noprompt.device, dtype=x_noprompt.dtype)
        batched = torch.cat([gp, ep], dim=1)
        return {"batched": batched, "strip_len": batched.size(1), "topk_idx": idx}

    @torch.no_grad()
    def allocate_new_domain_slots(self, slots_per_layer):
        """Freeze previously-used slots and open `slots_per_layer` fresh ones."""
        for li in range(self.num_layers):
            used_before = int(self.active_sizes[li, 0].item())
            used_after = min(used_before + slots_per_layer, self.pool_size)
            frozen = self.frozen_masks[li].clone()
            if used_before > 0:
                frozen[:used_before] = True
            self.frozen_masks[li] = frozen
            trainable = torch.zeros_like(self.trainable_masks[li])
            if used_after > used_before:
                trainable[used_before:used_after] = True
            self.trainable_masks[li] = trainable
            self.active_sizes[li, 0] = used_after


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        eot = tokenized_prompts.argmax(dim=-1)
        return x[torch.arange(x.shape[0]), eot] @ self.text_projection


class TAPromptLearner(nn.Module):
    """Class-conditional "A photo of a X X X X person." tokens (PAD Sec. 3.3)."""

    def __init__(self, num_class, dtype, token_embedding, n_ctx=4, n_cls_ctx=4):
        super().__init__()
        assert n_ctx == 4 and n_cls_ctx == 4, "pre-tokenised template assumes the official N_CTX = N_CLS_CTX = 4"
        ctx_dim = 512
        tokenized = tokenize_prompt_template().to(token_embedding.weight.device)
        with torch.no_grad():
            embedding = token_embedding(tokenized).type(dtype)
        self.tokenized_prompts = tokenized
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)
        self.register_buffer("token_prefix", embedding[:, : n_ctx + 1, :])           # SOT a photo of a
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])  # person . EOT ...
        self.num_class, self.n_cls_ctx = num_class, n_cls_ctx

    def forward(self, label):
        b = label.shape[0]
        return torch.cat([self.token_prefix.expand(b, -1, -1), self.cls_ctx[label], self.token_suffix.expand(b, -1, -1)], dim=1)


def _init_kaiming(m):
    cls = m.__class__.__name__
    if cls.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out"); nn.init.constant_(m.bias, 0.0)
    elif cls.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif cls.find("BatchNorm") != -1 and m.affine:
        nn.init.constant_(m.weight, 1.0); nn.init.constant_(m.bias, 0.0)


def _init_classifier(m):
    if m.__class__.__name__.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class PADModel(nn.Module):
    def __init__(self, num_classes, cfg):
        super().__init__()
        self.model_name, self.num_classes, self.neck_feat = cfg.MODEL.NAME, num_classes, cfg.TEST.NECK_FEAT
        self.in_planes, self.in_planes_proj = 768, 512
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False); self.classifier.apply(_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, num_classes, bias=False); self.classifier_proj.apply(_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes); self.bottleneck.bias.requires_grad_(False); self.bottleneck.apply(_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj); self.bottleneck_proj.bias.requires_grad_(False); self.bottleneck_proj.apply(_init_kaiming)

        h_res = (cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1
        w_res = (cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1
        clip_model = build_model(load_clip_state_dict(), h_res, w_res, cfg.MODEL.STRIDE_SIZE[0]).to(DEVICE)
        self.image_encoder = clip_model.visual
        if cfg.VA_PROMPT.ENABLE:
            self.image_encoder.va_prompt = VAPromptPool(
                num_layers=int(self.image_encoder.transformer.layers), embed_dim=self.in_planes,
                g_len=int(cfg.VA_PROMPT.G_LEN), e_len=int(cfg.VA_PROMPT.E_LEN), pool_size=int(cfg.VA_PROMPT.POOL_SIZE),
                top_k=int(cfg.VA_PROMPT.TOP_K), embedding_key=str(cfg.VA_PROMPT.EMBEDDING_KEY),
                key_learnable=bool(cfg.VA_PROMPT.KEY_LEARNABLE)).to(DEVICE)
            self.image_encoder.va_strip_after_block = bool(cfg.VA_PROMPT.STRIP_AFTER_BLOCK)
        self.prompt_learner = TAPromptLearner(num_classes, clip_model.dtype, clip_model.token_embedding,
                                              n_ctx=int(cfg.TA_PROMPT.N_CTX), n_cls_ctx=int(cfg.TA_PROMPT.N_CLS_CTX))
        self.text_encoder = TextEncoder(clip_model)

    def forward(self, x=None, label=None, get_image=False, get_text=False, return_train_outputs=False):
        if get_text:
            return self.text_encoder(self.prompt_learner(label), self.prompt_learner.tokenized_prompts)
        if get_image:
            _, _, proj = self.image_encoder(x)
            return proj[:, 0]
        feat_pen, feat_last, feat_proj = self.image_encoder(x)
        f_pen, f, f_proj = feat_pen[:, 0], feat_last[:, 0], feat_proj[:, 0]
        feat, feat_p = self.bottleneck(f), self.bottleneck_proj(f_proj)
        if self.training or return_train_outputs:
            return [self.classifier(feat), self.classifier_proj(feat_p)], [f_pen, f, f_proj], f_proj
        if self.neck_feat == "after":
            return torch.cat([feat, feat_p], dim=1)
        return torch.cat([f, f_proj], dim=1)          # test feature: [CLS(768) ; proj CLS(512)] before BN
