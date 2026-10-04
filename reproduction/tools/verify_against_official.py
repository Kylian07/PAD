"""Numerically compare the notebook port against the official `pad` package (CPU, synthetic weights).

Usage (from repo root):
  pip install torch torchvision timm yacs h5py scipy nbformat nbclient ipykernel matplotlib
  export PAD_INPUT_ROOT=/tmp/fake_input FAKE_ROOT=/tmp/fake_input
  python reproduction/tools/make_fake_data.py
  PAD_WORK_DIR=/tmp/w PAD_DATA_ROOT=/tmp/d PAD_CLIP_CACHE=/tmp/c PAD_RUN_MODE=quick python reproduction/tools/verify_against_official.py
"""
import os, sys, json, numpy as np, torch, torch.nn as nn
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FAKE = os.environ["PAD_INPUT_ROOT"]
# ---- 1) load notebook definitions (all cells up to, not including, the lifelong run cell) ----
nb = json.load(open(os.path.join(REPO, "reproduction", "PAD_Kaggle_Reproduction.ipynb")))
ns = {}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if "prev_ckpt, t_all = None" in src:
        break
    exec(compile(src, "nbcell", "exec"), ns)
print("notebook cells loaded")

# ---- 2) official package with cuda redirected to cpu ----
_to = nn.Module.to
def _mto(self, *a, **k):
    a = tuple("cpu" if (isinstance(x, str) and x.startswith("cuda")) else x for x in a)
    return _to(self, *a, **k)
nn.Module.to = _mto
torch.Tensor.cuda = lambda self, *a, **k: self
nn.Module.cuda = lambda self, *a, **k: self
sys.path.insert(0, REPO)
from pad.clip import clip as oclip
oclip._download = lambda url, root=None: f"{FAKE}/clip-weights/ViT-B-16.pt"
import pad.model as omodel, pad.utils as outils
from pad.config import cfg as ocfg, load_cfg_from_yaml
ocfg.defrost(); load_cfg_from_yaml(ocfg, os.path.join(REPO, "configs", "pad.yml"), domain_idx=1); ocfg.freeze()

torch.manual_seed(0)
off = omodel.PADModel(50, ocfg)
mine = ns["PADModel"](50, ns["build_cfg"]("cuhksysu", False))
sd_off = off.state_dict(); sd_mine = mine.state_dict()
assert set(sd_off) == set(sd_mine), (set(sd_off) ^ set(sd_mine))
print(f"state_dict keys identical: {len(sd_off)}")
# the CLIP-derived weights must be identical before copying anything
clip_keys = [k for k in sd_off if k.startswith("image_encoder.") and "va_prompt" not in k or k.startswith("text_encoder.")]
print("max |diff| CLIP-derived weights:", max((sd_off[k].float() - sd_mine[k].float()).abs().max().item() for k in clip_keys))
print("token_prefix/suffix diff:", (sd_off["prompt_learner.token_prefix"] - sd_mine["prompt_learner.token_prefix"]).abs().max().item(),
      (sd_off["prompt_learner.token_suffix"] - sd_mine["prompt_learner.token_suffix"]).abs().max().item(),
      "tokenized equal:", torch.equal(off.prompt_learner.tokenized_prompts.cpu(), mine.prompt_learner.tokenized_prompts.cpu()))
mine.load_state_dict(sd_off)
for m in (off, mine):
    m.image_encoder.va_prompt.allocate_new_domain_slots(8)
x = torch.randn(6, 3, 256, 128); lab = torch.randint(0, 50, (6,))
off.train(); mine.train()
torch.manual_seed(1); so, fo, po = off(x, lab)
torch.manual_seed(1); sm, fm, pm = mine(x, lab)
print("train fwd |diff| scores/feats/proj:", max((a - b).abs().max().item() for a, b in zip(so, sm)),
      max((a - b).abs().max().item() for a, b in zip(fo, fm)), (po - pm).abs().max().item())
off.eval(); mine.eval()
print("eval feature |diff|:", (off(x) - mine(x)).abs().max().item(), "| text |diff|:",
      (off(label=lab, get_text=True) - mine(label=lab, get_text=True)).abs().max().item())

# ---- 3) evaluator: official _eval_func (numpy) vs notebook chunked GPU/CPU version ----
rng = np.random.RandomState(0)
nq, ng = 300, 2000
qf = torch.randn(nq, 64); gf = torch.randn(ng, 64)
qp = rng.randint(0, 120, nq); gp = rng.randint(0, 120, ng); qc = rng.randint(0, 6, nq); gc = rng.randint(0, 6, ng)
qn, gn = torch.nn.functional.normalize(qf, dim=1), torch.nn.functional.normalize(gf, dim=1)
dist = (qn.pow(2).sum(1, keepdim=True) + gn.pow(2).sum(1)[None] - 2 * qn @ gn.t()).numpy()
cmc_o, map_o = outils._eval_func(dist, qp, gp, qc, gc, 50)
cmc_m, map_m = ns["eval_func_chunked"](qn, gn, qp, gp, qc, gc, 50, chunk=37)
print(f"official mAP {map_o:.6f} vs notebook {map_m:.6f} | max CMC diff {np.abs(cmc_o - cmc_m).max():.2e}")

# ---- 4) trainable-parameter sets and optimiser groups (stage 1 and stage 2, later domain) ----
import train_lifelong as otl, pad.optim as ooptim
def official_stage(m, stage):
    otl._set_all_requires_grad(m, False)
    if stage == 1:
        otl._set_by_name(m, "prompt_learner", True)
    else:
        otl._set_by_name(m, ("classifier", "bottleneck"), True)
        otl._set_by_name(m, ("va_prompt", "g_prompt", "e_prompt", "e_key"), True)
        otl._unfreeze_last_blocks(m, 4)
def mine_stage(m, stage):
    ns["_set_all_requires_grad"](m, False)
    if stage == 1:
        ns["_set_by_name"](m, "prompt_learner", True)
    else:
        ns["_set_by_name"](m, "prompt_learner", False)
        ns["_set_by_name"](m, ("classifier", "bottleneck"), True)
        ns["_set_by_name"](m, ("va_prompt", "g_prompt", "e_prompt", "e_key"), True)
        ns["_unfreeze_last_blocks"](m, 4)
for st in (1, 2):
    official_stage(off, st); mine_stage(mine, st)
    to = {n for n, p in off.named_parameters() if p.requires_grad}; tm = {n for n, p in mine.named_parameters() if p.requires_grad}
    go = ooptim.make_optimizer(ocfg, off, f"STAGE{st}"); gm = ns["make_optimizer"](ns["build_cfg"]("cuhksysu", False), mine, f"STAGE{st}")
    print(f"stage{st}: trainable sets equal={to == tm} ({len(to)} tensors) | groups official="
          f"{[(g['name'], g['lr'], sum(p.numel() for p in g['params'])) for g in go.param_groups]} notebook="
          f"{[(g['name'], g['lr'], sum(p.numel() for p in g['params'])) for g in gm.param_groups]}")
