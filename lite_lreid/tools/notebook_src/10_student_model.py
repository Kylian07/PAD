# %% [markdown]
# ## Student model
# `backbone → global feature f → BNNeck → ID classifier`, as in standard ReID baselines.
# * **MobileCLIP-S1/S2/B and MobileCLIP2-S0/S2/S4**: the image tower of Apple's MobileCLIP (open_clip; weights from the
#   Hugging Face hub). Its 512-d output lives in CLIP's image–text space, so it is compared with the teacher's text anchors
#   directly (`to_text` = identity).
# * **timm backbones**: ImageNet-pretrained CNNs; a learned linear layer maps f to the 512-d text space.
#
# The data pipeline is the teacher's (normalised with mean = std = 0.5). The student converts each batch to its own
# normalisation internally, so teacher and student always see exactly the same augmented images. Test feature = f
# (before BNNeck), as for the teacher.

# %%
try:
    import open_clip
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=True)
    import open_clip

MOBILECLIP_TAGS = {"MobileCLIP-S1": "datacompdr", "MobileCLIP-S2": "datacompdr", "MobileCLIP-B": "datacompdr",
                   "MobileCLIP2-S0": "dfndr2b", "MobileCLIP2-S2": "dfndr2b", "MobileCLIP2-S4": "dfndr2b"}


def _build_backbone(arch):
    if arch in MOBILECLIP_TAGS:
        clip_m, _, _ = open_clip.create_model_and_transforms(arch, pretrained=MOBILECLIP_TAGS[arch] if STUDENT_PRETRAINED else None,
                                                             cache_dir=osp.join(CLIP_CACHE, "open_clip"))
        pp = open_clip.get_model_preprocess_cfg(clip_m)
        visual = clip_m.visual
        with torch.no_grad():
            dim = visual(torch.zeros(1, 3, 256, 128)).shape[-1]
        del clip_m
        return visual, dim, pp["mean"], pp["std"], True
    if arch.startswith("timm:"):
        m = timm.create_model(arch[5:], pretrained=STUDENT_PRETRAINED, num_classes=0)
        cfg_ = m.pretrained_cfg
        return m, m.num_features, cfg_.get("mean", (0.485, 0.456, 0.406)), cfg_.get("std", (0.229, 0.224, 0.225)), False
    raise ValueError(f"unknown STUDENT_ARCH {arch!r}")


class StudentReID(nn.Module):
    def __init__(self, arch, num_classes, text_dim=512):
        super().__init__()
        self.backbone, self.feat_dim, mean, std, self.clip_aligned = _build_backbone(arch)
        self.register_buffer("in_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("in_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.to_text = (nn.Identity() if (self.clip_aligned and self.feat_dim == text_dim)
                        else nn.Linear(self.feat_dim, text_dim, bias=False))
        self.bottleneck = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(_init_kaiming)
        self.reset_classifier(num_classes)

    def reset_classifier(self, num_classes):
        dev = self.bottleneck.weight.device
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.feat_dim, num_classes, bias=False).to(dev)
        self.classifier.apply(_init_classifier)

    def _renorm(self, x):
        # teacher normalisation (x - 0.5) / 0.5  ->  [0, 1]  ->  student normalisation
        return ((x * 0.5 + 0.5) - self.in_mean) / self.in_std

    def forward(self, x):
        f = self.backbone(self._renorm(x))
        if isinstance(f, (tuple, list)):
            f = f[0]
        f = f.float()
        if self.training:
            return self.classifier(self.bottleneck(f)), f
        return f


def student_state_for_saving(model):
    """Everything except the per-domain classifier (the identity space changes at each step)."""
    return {k: v for k, v in model.state_dict().items() if not k.startswith("classifier.")}


def count_params(module):
    return sum(p.numel() for p in module.parameters())


_probe = StudentReID(STUDENT_ARCH, 10)
if not STUDENT_PRETRAINED:
    print("!! STUDENT_PRETRAINED=False: random student weights (offline test only)")
print(f"[student] {STUDENT_ARCH}: feature dim {_probe.feat_dim}, CLIP-aligned={_probe.clip_aligned}, "
      f"params (inference) {count_params(_probe.backbone) / 1e6:.1f} M")
del _probe
