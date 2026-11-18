# %% [markdown]
# ## Efficiency: teacher vs student (what you would deploy)
# For each model at the ReID input size 256×128:
# * **Params:** inference parameters only (teacher: image encoder + BN necks; student: backbone + BNNeck).
# * **GFLOPs:** counted with `torch.utils.flop_counter` (one multiply-add = 2 FLOPs; many papers report GMACs =
#   GFLOPs / 2, so state which one you use).
# * **GPU latency** (batch 1, fp16), **GPU throughput** (batch 64, fp16), and **CPU latency** (batch 1, fp32). These
#   depend on the hardware, so report them as "on a Kaggle T4" or "on this CPU".
# * **Memory kept between domains:** the student keeps only identity prototypes, no images.

# %%
import contextlib
from torch.utils.flop_counter import FlopCounterMode


@torch.no_grad()
def _gflops(model, dev):
    x = torch.randn(1, 3, 256, 128, device=dev)
    fc = FlopCounterMode(display=False)
    with fc:
        model(x)
    return fc.get_total_flops() / 1e9


@torch.no_grad()
def _latency_ms(model, dev, bs, n, fp16):
    x = torch.randn(bs, 3, 256, 128, device=dev)
    ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.float16)) if fp16 else contextlib.nullcontext
    for _ in range(3):
        with ctx():
            model(x)
    ts = []
    for _ in range(n):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with ctx():
            model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts))


def measure_model(name, model, inference_params):
    model = model.eval()
    res = {"model": name, "params_M": inference_params / 1e6}
    cpu_model = copy.deepcopy(model).float().cpu()
    res["GFLOPs"] = _gflops(cpu_model, "cpu")
    res["cpu_ms_bs1"] = _latency_ms(cpu_model, "cpu", 1, 3 if RUN_MODE == "quick" else 20, False)
    del cpu_model
    if DEVICE == "cuda":
        model = model.to(DEVICE)
        res["gpu_ms_bs1"] = _latency_ms(model, "cuda", 1, 50, True)
        res["gpu_img_s_bs64"] = 64 * 1000 / _latency_ms(model, "cuda", 64, 20, True)
    return res


EFFICIENCY = []
_last = TRAIN_ORDER[-1]
if find_existing_ckpt(_last):
    _t = prepare_eval_model(model_from_ckpt(find_existing_ckpt(_last), _last))
    _tp = count_params(_t.image_encoder) + count_params(_t.bottleneck) + count_params(_t.bottleneck_proj)
    EFFICIENCY.append(measure_model("Teacher: PAD (CLIP ViT-B/16)", _t, _tp)); del _t
_seen = set()
for method in STUDENT_METHODS:
    if find_lite_file(method, _last, "student.pth") and STUDENT_ARCH not in _seen:
        _s = load_student(method, _last)
        r = measure_model(f"Student: {STUDENT_ARCH}", _s, count_params(_s.backbone) + count_params(_s.bottleneck))
        pr = [find_lite_file("ours", d, "protos.pt") for d in TRAIN_ORDER] if method == "ours" else []
        r["memory_kB"] = sum(osp.getsize(p) for p in pr if p) / 1024 if pr else 0.0
        EFFICIENCY.append(r); _seen.add(STUDENT_ARCH); del _s
for r in EFFICIENCY:
    print(f"{r['model']:34s} params {r['params_M']:6.1f} M | {r['GFLOPs']:6.2f} GFLOPs | "
          f"CPU {r['cpu_ms_bs1']:7.1f} ms | " +
          (f"GPU {r['gpu_ms_bs1']:6.2f} ms, {r['gpu_img_s_bs64']:6.0f} img/s" if "gpu_ms_bs1" in r else "GPU n/a"))
if len(EFFICIENCY) == 2:
    t, s = EFFICIENCY
    print(f"student vs teacher: {t['params_M'] / s['params_M']:.1f}x fewer params, {t['GFLOPs'] / s['GFLOPs']:.1f}x fewer FLOPs, "
          f"{t['cpu_ms_bs1'] / s['cpu_ms_bs1']:.1f}x faster on CPU")
json.dump(EFFICIENCY, open(osp.join(LITE_OUT, f"efficiency_{RUN_TAG}.json"), "w"), indent=1)
