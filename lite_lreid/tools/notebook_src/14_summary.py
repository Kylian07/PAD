# %% [markdown]
# ## Results: teacher vs student(s)
# * **Final step** (after the last domain): per-domain mAP / Rank-1, Seen-Avg, and the unseen sets.
# * **Forgetting matrix** of each student: row = after step *t*, column = evaluation domain.
# * **Accuracy vs cost plot:** Seen-Avg mAP against GFLOPs, i.e. the trade-off the paper would argue for.

# %%
def _read_rows(path, key):
    out = defaultdict(dict)
    if osp.exists(path):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                out[key(r)][r["eval_domain"]] = (100 * float(r["map"]), 100 * float(r["rank1"]))
    return out


teacher_res = _read_rows(SEQ_CSV, lambda r: ("teacher", r["trained_domain"]))
student_res = _read_rows(STUDENT_CSV, lambda r: (r["method"], r["trained_domain"]))
final = TRAIN_ORDER[-1]
models = [("teacher", "Teacher PAD (ViT-B/16)")] + [(m, f"Student {m} ({STUDENT_ARCH})") for m in STUDENT_METHODS]
summary = {}
print("Final step, mAP / Rank-1 (%)")
print("model".ljust(40) + "".join(f"{SHORT[d]:>16s}" for d in TRAIN_ORDER) + f"{'Seen-Avg':>16s}")
for key, label in models:
    res = (teacher_res if key == "teacher" else student_res).get((key, final), {})
    vals = [res.get(d) for d in TRAIN_ORDER]
    if not any(vals):
        print(label.ljust(40) + "  (not finished yet)"); continue
    got = [v for v in vals if v]
    avg = (np.mean([v[0] for v in got]), np.mean([v[1] for v in got]))
    summary[key] = {"label": label, "per_domain": dict(zip(TRAIN_ORDER, vals)), "seen_avg": avg}
    print(label.ljust(40) + "".join(f"{v[0]:7.1f} /{v[1]:5.1f}  " if v else f"{'-':>16s}" for v in vals)
          + f"{avg[0]:7.1f} /{avg[1]:5.1f}")

unseen_t = {}
if osp.exists(osp.join(OUT_DIR, "unseen_summary.csv")):
    with open(osp.join(OUT_DIR, "unseen_summary.csv")) as fh:
        unseen_t = {r["dataset"]: (float(r["map"]), float(r["rank1"])) for r in csv.DictReader(fh)}
if unseen_t or STUDENT_UNSEEN:
    print("\nUnseen sets, mAP / Rank-1 (%)")
    for d in sorted(set(unseen_t) | {d for _, d in STUDENT_UNSEEN}):
        line = f"  {d:10s} teacher " + (f"{unseen_t[d][0]:5.1f}/{unseen_t[d][1]:5.1f}" if d in unseen_t else "  -  ")
        for m in STUDENT_METHODS:
            if (m, d) in STUDENT_UNSEEN:
                r1, mp = STUDENT_UNSEEN[(m, d)]
                line += f" | {m} {mp:5.1f}/{r1:5.1f}"
        print(line)

for m in STUDENT_METHODS:
    steps = [d for d in TRAIN_ORDER if (m, d) in student_res]
    if steps:
        print(f"\nForgetting matrix, student '{m}' (mAP %)")
        print("after step".ljust(12) + "".join(f"{SHORT[d]:>11s}" for d in TRAIN_ORDER))
        for s in steps:
            print(SHORT[s].ljust(12) + "".join(f"{student_res[(m, s)][d][0]:11.1f}" if d in student_res[(m, s)] else f"{'-':>11s}"
                                               for d in TRAIN_ORDER))

if summary and EFFICIENCY:
    eff = {("teacher" if r["model"].startswith("Teacher") else "student"): r for r in EFFICIENCY}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = {"teacher": "#52514e", "ours": "#2a78d6", "naive_ft": "#eb6834"}
    for key, s in summary.items():
        e = eff.get("teacher" if key == "teacher" else "student")
        if e:
            ax.scatter(e["GFLOPs"], s["seen_avg"][0], s=90, color=colors.get(key, "#2a78d6"), zorder=3)
            ax.annotate(s["label"], (e["GFLOPs"], s["seen_avg"][0]), textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("GFLOPs per image (256×128, log scale)"); ax.set_ylabel("Seen-Avg mAP (%)")
    ax.set_title("Accuracy vs inference cost", loc="left", fontsize=11); ax.grid(color="#e6e5e0"); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout(); fig.savefig(osp.join(LITE_OUT, f"accuracy_vs_cost_{RUN_TAG}.png"), dpi=150); plt.show()

json.dump({"run_tag": RUN_TAG, "protocol": PROTOCOL, "run_mode": RUN_MODE, "label_fraction": LABEL_FRACTION,
           "lambda": LAMBDA, "tau": TAU, "epochs": STUDENT_EPOCHS, "lr": STUDENT_LR,
           "final": {k: {"per_domain": v["per_domain"], "seen_avg": v["seen_avg"]} for k, v in summary.items()},
           "unseen_teacher": unseen_t, "unseen_student": {f"{m}/{d}": v for (m, d), v in STUDENT_UNSEEN.items()},
           "efficiency": EFFICIENCY}, open(osp.join(LITE_OUT, f"summary_{RUN_TAG}.json"), "w"), indent=1)
if RUN_MODE == "quick":
    print("\n(quick mode: 1 short epoch per domain and a subsampled evaluation, so these numbers are NOT results)")

# %% [markdown]
# ## Outputs (`/kaggle/working/`)
# * `PAD_outputs/`: teacher checkpoints, `text_bank.pt` per domain, `seq_eval.csv`, `unseen_summary.csv`.
# * `LiteLReID_outputs/<method>_<arch>_lf<fraction>/<domain>/`: `student.pth` (deployable weights), `protos.pt`
#   (the identity prototypes kept as memory), and the logs.
# * `LiteLReID_outputs/student_seq_eval_<tag>.csv`, `efficiency_<tag>.json`, `summary_<tag>.json`, and
#   `accuracy_vs_cost_<tag>.png`.
