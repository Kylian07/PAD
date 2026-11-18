# %% [markdown]
# ## ▶ Train the student(s) through the same domain sequence
# This needs the teacher checkpoints from the cells above (`ours`; `naive_ft` needs no teacher). After each domain,
# the student is evaluated on every domain seen so far (`student_seq_eval_<tag>.csv`). Finished domains are skipped
# when you re-run with the previous output attached.

# %%
STUDENT_CSV = osp.join(LITE_OUT, f"student_seq_eval_{RUN_TAG}.csv")
STUDENT_STOPPED = False


def _student_done_evals():
    if not osp.exists(STUDENT_CSV):
        return set()
    with open(STUDENT_CSV) as fh:
        return {(r["method"], r["trained_domain"], r["eval_domain"]) for r in csv.DictReader(fh)}


t_all = time.time()
for method in STUDENT_METHODS:
    prev = None
    for idx, domain in enumerate(TRAIN_ORDER):
        if domain not in PREPARED:
            print(f"!! {method}: dataset '{domain}' is not available; stopping this sequence.")
            break
        if method == "ours" and not find_existing_ckpt(domain):
            print(f"!! {method}: no teacher checkpoint for '{domain}' yet (teacher stopped early?). Re-run to continue.")
            break
        student = None
        if find_lite_file(method, domain, "student.pth"):
            print(f"[skip] {method}/{domain}: student already trained")
        else:
            t0 = time.time()
            student = train_student_domain(method, domain, idx, prev)
            if student is None:
                STUDENT_STOPPED = True
                break
            print(f"[time] {method}/{domain} trained in {timedelta(seconds=int(time.time() - t0))}")
        todo = [d for d in TRAIN_ORDER[: idx + 1] if (method, domain, d) not in _student_done_evals() and d in PREPARED]
        if todo:
            student = student or load_student(method, domain)
            setup_logger(lite_dir(method, domain), "student_test_log.txt")
            evaluate_student(student.eval(), todo, method, domain, STUDENT_CSV)
        prev = domain
        del student
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    if STUDENT_STOPPED:
        print("[stop] session time budget reached: attach this notebook's output as an input and run again.")
        break
print(f"[student] wall time {timedelta(seconds=int(time.time() - t_all))}; results -> {STUDENT_CSV}")

# ---- unseen domains (same protocol as the teacher cell above) ----
STUDENT_UNSEEN = {}
if RUN_UNSEEN_EVAL and not STUDENT_STOPPED and avail:
    unseen_csv = osp.join(LITE_OUT, f"student_unseen_raw_{RUN_TAG}.csv")
    if osp.exists(unseen_csv):
        os.remove(unseen_csv)
    for method in STUDENT_METHODS:
        if not find_lite_file(method, TRAIN_ORDER[-1], "student.pth"):
            continue
        sm = load_student(method, TRAIN_ORDER[-1])
        setup_logger(LITE_OUT, "student_unseen_log.txt")
        for d in avail:
            n = UNSEEN_PROTOCOL[d] if RUN_MODE == "full" else 1
            rows = []
            for sid in range(n):
                rows += evaluate_student(sm, [d], method, TRAIN_ORDER[-1], unseen_csv,
                                         extra_cfg={"SPLIT_ID": sid, "DIRECTION": "auto", "SEED": 123})
            STUDENT_UNSEEN[(method, d)] = (100 * statistics.mean(float(r["rank1"]) for r in rows),
                                           100 * statistics.mean(float(r["map"]) for r in rows))
        del sm
    for (m, d), (r1, mp) in STUDENT_UNSEEN.items():
        print(f"{m:9s} {d:10s} Rank-1 {r1:6.2f}  mAP {mp:6.2f}")
