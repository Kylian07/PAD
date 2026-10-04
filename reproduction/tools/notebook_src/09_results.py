# %% [markdown]
# ## Results: forgetting matrix, final table, and comparison with the paper
# * **Matrix**: row = after training step *t*, column = evaluation domain. Values on the diagonal are just-learned
#   performance. Reading down a column shows how much of that domain is **forgotten** later.
# * **Final row** (after CUHK03): the numbers the paper reports in Table 1, i.e. per-domain mAP/R1 and Seen-Avg.
#
# `PAPER_REPORTED` holds the PAD numbers for AKA order-1 as quoted from the paper (arXiv 2605.05027). **Check them
# against Table 1 of the PDF before you report anything.** The paper also gives a 7-seed mean of 70.30 ± 0.49 mAP /
# 80.98 ± 0.31 R1 for Seen-Avg, so a single-seed reproduction within about ±1 point is in the expected range.

# %%
import matplotlib.pyplot as plt

PAPER_REPORTED = {  # (mAP, Rank-1) in %, PAD, AKA order-1, final step
    "market1501": (77.5, 89.8), "cuhksysu": (93.9, 94.9), "dukemtmcreid": (71.1, 82.6),
    "msmt17": (37.7, 64.0), "cuhk03": (66.4, 68.5),
}
PAPER_SEEN_AVG_7SEEDS = (70.30, 80.98)
SHORT = {"market1501": "Market", "cuhksysu": "CUHK-SYSU", "dukemtmcreid": "Duke", "msmt17": "MSMT17", "cuhk03": "CUHK03"}

res = defaultdict(dict)
if osp.exists(SEQ_CSV):
    with open(SEQ_CSV) as f:
        for r in csv.DictReader(f):
            res[r["trained_domain"]][r["eval_domain"]] = (100 * float(r["map"]), 100 * float(r["rank1"]))
steps = [d for d in TRAIN_ORDER if d in res]

if steps:
    hdr = "after step".ljust(12) + "".join(f"{SHORT[d]:>18s}" for d in TRAIN_ORDER) + f"{'Seen-Avg':>18s}"
    print("mAP / Rank-1 (%)\n" + hdr)
    for s in steps:
        vals = [res[s].get(d) for d in TRAIN_ORDER]
        got = [v for v in vals if v]
        line = SHORT[s].ljust(12) + "".join(f"{v[0]:8.1f} / {v[1]:5.1f} " if v else f"{'-':>18s}" for v in vals)
        line += f"{np.mean([v[0] for v in got]):8.1f} / {np.mean([v[1] for v in got]):5.1f}"
        print(line)

    final = steps[-1]
    print(f"\nFinal step ({SHORT[final]}) vs paper:")
    print(f"{'domain':12s}{'ours mAP':>10s}{'paper':>8s}{'Δ':>7s}{'ours R1':>10s}{'paper':>8s}{'Δ':>7s}")
    ours_m, ours_r = [], []
    for d in TRAIN_ORDER:
        if d in res[final]:
            m, r = res[final][d]; pm, pr = PAPER_REPORTED[d]; ours_m.append(m); ours_r.append(r)
            print(f"{SHORT[d]:12s}{m:10.1f}{pm:8.1f}{m - pm:+7.1f}{r:10.1f}{pr:8.1f}{r - pr:+7.1f}")
    if len(ours_m) == 5:
        pm = np.mean([v[0] for v in PAPER_REPORTED.values()]); pr = np.mean([v[1] for v in PAPER_REPORTED.values()])
        print(f"{'Seen-Avg':12s}{np.mean(ours_m):10.1f}{pm:8.1f}{np.mean(ours_m) - pm:+7.1f}"
              f"{np.mean(ours_r):10.1f}{pr:8.1f}{np.mean(ours_r) - pr:+7.1f}   (paper 7-seed: {PAPER_SEEN_AVG_7SEEDS})")

    # Forgetting: best-ever mAP on a domain minus its final mAP (only domains seen before the last step).
    print("\nForgetting (best mAP during the sequence − final mAP):")
    for d in TRAIN_ORDER[:-1]:
        hist = [res[s][d][0] for s in steps if d in res[s]]
        if len(hist) > 1:
            print(f"  {SHORT[d]:10s} {max(hist) - hist[-1]:5.1f} pts  (history: {', '.join(f'{h:.1f}' for h in hist)})")

    # ---- Figure 1: mAP matrix heatmap (sequential single hue) ----
    M = np.full((len(steps), len(TRAIN_ORDER)), np.nan)
    for i, s in enumerate(steps):
        for j, d in enumerate(TRAIN_ORDER):
            if d in res[s]:
                M[i, j] = res[s][d][0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=9,
                        color="white" if M[i, j] > 60 else "#0b0b0b")
    ax.set_xticks(range(len(TRAIN_ORDER)), [SHORT[d] for d in TRAIN_ORDER], rotation=20)
    ax.set_yticks(range(len(steps)), [f"after {SHORT[s]}" for s in steps])
    ax.set_title("mAP (%) after each lifelong step", loc="left", fontsize=11)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    # ---- Figure 2: final per-domain mAP, this run vs paper (grouped bars) ----
    ax = axes[1]
    doms = [d for d in TRAIN_ORDER if d in res[final]]
    x = np.arange(len(doms)); w = 0.38
    ours = [res[final][d][0] for d in doms]; paper = [PAPER_REPORTED[d][0] for d in doms]
    b1 = ax.bar(x - w / 2 - 0.01, ours, w, color="#2a78d6", label="This run")
    b2 = ax.bar(x + w / 2 + 0.01, paper, w, color="#eb6834", label="Paper (reported)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}",
                    ha="center", va="bottom", fontsize=8, color="#52514e")
    ax.set_xticks(x, [SHORT[d] for d in doms])
    ax.set_ylabel("mAP (%)"); ax.set_ylim(0, 105)
    ax.set_title(f"Final mAP after {SHORT[final]}: reproduction vs paper", loc="left", fontsize=11)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    fig.savefig(osp.join(OUT_DIR, "results.png"), dpi=150)
    plt.show()
else:
    print("no results yet")

if RUN_MODE == "quick":
    print("\n(quick mode: tiny schedule + subsampled evaluation, so these numbers are NOT comparable to the paper)")

# %% [markdown]
# ## Outputs
# Everything lives in `/kaggle/working/PAD_outputs/` (kept after a committed run):
# `seq_eval.csv` (all seen-domain evals), `unseen_raw.csv`, `results.png`, `<domain>/train_log.txt`, and
# `<domain>/ViT-B-16_visual.pth` (about 350 MB each; needed to resume or to re-run the unseen evaluation).
