# PAD: Prompt-Anchored Vision–Text Distillation for Lifelong Person Re-ID
### A plain-language methodology report and reproduction guide

> **Paper:** Wen Wen, Hao Chen, Shiliang Zhang. *Prompt-Anchored Vision–Text Distillation for Lifelong Person
> Re-identification.* CVPR 2026, pp. 18503–18512. arXiv: [2605.05027](https://arxiv.org/abs/2605.05027).
> Code: [github.com/zu-zi/PAD](https://github.com/zu-zi/PAD) (this fork is identical to upstream commit `ca2b557`).
>
> **Companion notebook:** [`PAD_Kaggle_Reproduction.ipynb`](PAD_Kaggle_Reproduction.ipynb) is a single,
> self-contained Kaggle notebook that re-implements the official code and runs the full experiment.
>
> **How this report was made:** from a line-by-line reading of the official code, plus the paper's abstract and
> numbers as quoted in public search-engine extracts. The PDF itself could not be opened from the environment
> used to prepare this report. Where the report states *what the code does*, it has been checked against the code.
> Where it quotes *paper numbers*, check them against Table 1 of the PDF.

---

## 1. The problem in one paragraph

**Person re-identification (ReID)** means finding the same pedestrian across different, non-overlapping cameras:
given a *query* image, rank a *gallery* so that images of the same identity come first. **Lifelong ReID (LReID)**
asks a model to learn from datasets ("domains") that arrive **one after another**, e.g. Market-1501, then
CUHK-SYSU, then DukeMTMC, then MSMT17, then CUHK03, and to stay good on **all** of them. The difficulty is
**catastrophic forgetting**: fine-tuning on a new domain overwrites what was learned on old ones. PAD studies the
**exemplar-free** setting, where *no images from old domains may be stored* (privacy and storage constraints).
Only the model is carried forward.

## 2. The key idea

PAD is built on **CLIP** (an image encoder and a text encoder pretrained to agree on image–text pairs). The
authors observe an *asymmetry*:

* The **text encoder** can stay **frozen** forever. Text like *"A photo of a … person."* lives in a fixed semantic
  space that does not drift when new camera domains arrive, so it can act as a **stable anchor** across domains.
* The **image encoder** must adapt to each new domain (lighting, resolution, viewpoints), but it should adapt in
  a controlled way.

PAD therefore:
1. learns **text-side prompts (TA-Prompt)** that describe each identity through the frozen text encoder, and
   regularises them with **fixed textual distillation (TEXKD)**;
2. adapts the visual side through a growing **pool of visual prompts (VA-Prompt)**, where new slots are opened
   per domain and old slots are frozen, together with a few unfrozen top layers;
3. limits visual drift with **EMA-teacher visual distillation (VISKD)** at the feature and logit level.

```mermaid
flowchart LR
  subgraph Text side (frozen encoder)
    L[identity label y] --> TA["TA-Prompt<br/>'A photo of a [X1][X2][X3][X4] person.'<br/>(4 learnable tokens per identity)"]
    TA --> TE[CLIP text encoder ❄]
    TE --> T[text feature t_y]
  end
  subgraph Visual side (adapts)
    I[image] --> VE["CLIP ViT-B/16<br/>+ VA-Prompt pool at every layer<br/>(last 4 blocks trainable)"]
    VE --> V[image feature v]
  end
  T -- "SupCon (Stage 1)" --- V
  T -- "image→text logits = v·tᵀ (Stage 2 CE)" --- V
  FT["frozen copy: Fixed-T"] -. TEXKD .-> TA
  EMA["EMA teacher (from previous domain)"] -. VISKD .-> VE
```

## 3. Building block: the CLIP-ReID two-stage recipe

PAD's training loop comes from **CLIP-ReID** (Li et al., AAAI 2023). There are no text captions in ReID datasets,
only identity labels, so CLIP-ReID **learns a text description per identity** first and then uses those
descriptions to supervise the image encoder:

| | Stage 1: learn text prompts | Stage 2: learn the image encoder |
|---|---|---|
| Trainable | only the per-identity prompt tokens | image side (backbone or prompts, BN, classifiers) |
| Frozen | image encoder + text encoder | text encoder, TA-Prompt |
| Loss | image↔text contrastive (SupCon) | ID + triplet + image→text cross-entropy |

PAD repeats both stages **for every new domain** and adds the lifelong components (VA-Prompt pool, TEXKD, VISKD).

## 4. Components, with equations and code pointers

Notation: $B$ = batch size, $v_i$ = 512-d projected image feature (CLS token after `ln_post @ proj`), $t_y$ = text
feature of identity $y$, $\mathcal{P}(i)$ = the batch items with the same identity as $i$.

### 4.1 TA-Prompt (Text-Anchored prompt), paper §3.3 · `pad/model.py: TAPromptLearner`
For identity $y$, the text input is the token sequence

$$\texttt{[SOS] a photo of a } [X]^y_1 [X]^y_2 [X]^y_3 [X]^y_4 \texttt{ person . [EOS]}$$

The four $[X]^y_j \in \mathbb{R}^{512}$ are **learnable word embeddings, unique to each identity** (initialised
$\mathcal{N}(0, 0.02^2)$). The rest are fixed CLIP token embeddings. The frozen text encoder maps this to $t_y$.

**Stage-1 loss: symmetric supervised contrastive (SupCon)** · `losses.py: SupConLoss`, `engine.py: do_train_stage1`

$$\mathcal{L}_{i2t} = -\frac{1}{B}\sum_i \frac{1}{|\mathcal{P}(i)|}\sum_{p\in\mathcal{P}(i)} \log\frac{\exp(v_i^\top t_p)}{\sum_{j}\exp(v_i^\top t_j)},\qquad \mathcal{L}_{supcon}=\mathcal{L}_{i2t}+\mathcal{L}_{t2i}$$

Note that the code uses temperature 1 and **unnormalised** features, as in CLIP-ReID. The image encoder is frozen in
this stage, so image features are computed **once** and cached, which makes Stage 1 cheap.

### 4.2 TEXKD (fixed textual distillation), paper §3.3.2 · `engine.py: do_train_stage1` (`textkd_on` branch)
From the 2nd domain on, a frozen copy of the model provides **anchor text features** $T^e$ (Fixed-T in the
figure). For each batch, the classes present plus **256 random negative classes** are taken, both the student's
and the anchor's text features are L2-normalised, and image→text similarity distributions are compared:

$$\mathcal{L}_{TEXKD}= \lambda_{t}\,\tau^2\,\mathrm{KL}\!\left(\mathrm{softmax}\!\left(\tfrac{s\,\hat v\,\hat T^{e\top}}{\tau}\right)\Big\|\ \mathrm{softmax}\!\left(\tfrac{s\,\hat v\,\hat T^{s\top}}{\tau}\right)\right),\quad s=7,\ \tau=0.07,\ \lambda_t=0.5$$

(implemented as `F.kl_div(log_softmax(student/τ), softmax(teacher/τ))·τ²`). **Intuition:** the new identity prompts
may move only in ways that keep the image–text similarity *structure* close to the fixed anchor, so the text space
stays a consistent reference across domains.

### 4.3 VA-Prompt (Visual Adaptive prompt pool), paper §3.4.2 · `model.py: VAPromptPool`, `clip/model.py: VisionTransformer.forward`
At **each of the 12 ViT layers** $\ell$:
* one **G-Prompt** $g^\ell \in \mathbb{R}^{6\times768}$ (general, shared by all domains);
* a pool of **36 E-Prompt slots** $e^\ell_k \in\mathbb{R}^{6\times768}$ (expert), each with a key $k^\ell_k\in\mathbb{R}^{768}$.

Layer forward:
1. Query $q^\ell = \mathrm{L2}(\text{mean of the layer's input tokens})$.
2. Choose the **top-4** active slots by cosine similarity $q^{\ell\top}\hat k^\ell_k$.
3. Insert $[g^\ell;\ e^\ell_{k_1};\dots;e^\ell_{k_4}]$ (30 tokens) **right after [CLS]**, run the block, then
   **remove** those 30 tokens again. The prompts influence CLS and patches through attention, but the patch
   layout and positional embeddings are untouched, and every layer gets fresh prompts ("deep" prompting).

**Lifelong slot allocation** (`allocate_new_domain_slots` + gradient hook `_register_va_grad_mask`):

| Domain | Market | CUHK-SYSU | Duke | MSMT17 | CUHK03 |
|---|---|---|---|---|---|
| new slots per layer | 8 | 8 | 8 | 8 | 4 |
| active slots after step | 8 | 16 | 24 | 32 | **36 (full)** |
| slots trainable in this step | 0–7 | 8–15 | 16–23 | 24–31 | 32–35 |

Old slots get **zero gradient** from then on, so the knowledge parametrised in them is preserved exactly.
Top-k selection still searches *all* active slots, so old experts can be reused for similar new data.

### 4.4 Selective unfreezing, paper §3.4.3 · `train_lifelong.py`
* **Domain 1 (Market):** everything on the visual side is trainable (full fine-tuning of ViT-B/16 + prompts).
* **Domains 2–5:** trainable = current VA-Prompt slots + G-Prompts + **last 4 transformer blocks** + BNNeck and
  classifier heads (≈ 4 × 7.1 M ≈ 28 M parameters for the blocks, about a third of the ViT). The patch embedding, blocks 1–8,
  `ln_post`, and `proj` stay frozen.

### 4.5 EMA teacher and VISKD, paper §3.4.1 · `utils.py: EMATeacher`, `losses.py: KDLosses`
The teacher starts as the previous-domain model and follows the student slowly:

$$\theta_{tea} \leftarrow m\,\theta_{tea} + (1-m)\,\theta_{stu},\qquad m=0.997\ \text{(after every iteration; BN statistics too)}$$

VISKD has two parts:

$$\mathcal{L}_{VISKD}=\underbrace{0.5\cdot\tfrac13\sum_{f\in\{f_{pen},f_{cls},f_{proj}\}}\mathrm{MSE}(f^{s},f^{t})}_{\text{feature level (Eq. 7)}}+\underbrace{0.5\cdot T^2\,\mathrm{KL}\!\left(\mathrm{softmax}(z^t/T)\,\|\,\mathrm{softmax}(z^s/T)\right)}_{\text{logit level (Eq. 8)},\ T=4}$$

where $f_{pen}$ = CLS after block 11, $f_{cls}$ = final CLS, $f_{proj}$ = projected CLS, and $z = v\,T_{bank}^\top$
are image→text logits against the current domain's text bank. **Why EMA rather than a frozen old model?** A frozen
teacher pins the student to the past and blocks plasticity. An EMA teacher drifts slowly, so it damps sudden
semantic drift without preventing adaptation.

### 4.6 Stage-2 objective · `losses.py: make_loss` + `engine.py: do_train_stage2`

$$\mathcal{L}_{S2}= 0.25\,[\mathrm{CE}_{ls}(W f_{cls}^{bn})+\mathrm{CE}_{ls}(W' f_{proj}^{bn})] + [\mathrm{Tri}(f_{pen})+\mathrm{Tri}(f_{cls})+\mathrm{Tri}(f_{proj})] + \mathrm{CE}_{ls}(v\,T_{bank}^\top) + \mathcal{L}_{VISKD}$$

* $\mathrm{CE}_{ls}$: cross-entropy with label smoothing $\varepsilon=0.1$. $f^{bn}$: after BNNeck.
* Tri: batch-hard triplet with margin 0.3. Batches are **PK-sampled**: 16 identities × 4 images (12 × 4 for CUHK03).
* The text bank $T_{bank}$ is computed once from the trained TA-Prompt and acts as a **text classifier**.

**Test feature:** $[f_{cls}\ (768) \,;\, f_{proj}\ (512)]$ *before* BN, L2-normalised, Euclidean ranking.

## 5. Training protocol (from `configs/pad.yml`)

| | Stage 1 (TA-Prompt) | Stage 2 (visual) |
|---|---|---|
| Optimiser | Adam; **prompts at 5e-3** (2e-3 on Market) | Adam; backbone/heads 5e-6, prompts 5e-3 (2e-3 on Market) |
| Schedule | cosine, 3 warm-up epochs | linear warm-up 10 epochs (×0.1 start), step ×0.1 at milestones |
| Epochs: Market / SYSU / Duke / MSMT / C03 | 20 / 12 / 12 / 15 / 10 | 30 / 25 / 25 / 32 / 28 (milestones 18-26 / 14-22 / 14-22 / 18-28 / 16-24) |
| Batch | 64 (48 on CUHK03) | 64 (48 on CUHK03) |
| Input | 256×128, flip, pad 10 + crop, random erasing p = 0.5 (0.6 on CUHK03), mixed precision (fp16) | |
| Hardware | **1 GPU** (README: multi-GPU DataParallel weakens distillation), seed 42 | |

**Evaluation.** After each step the model is evaluated on every domain seen so far (mAP and Rank-1). The headline is
the final step's per-domain numbers and their mean (**Seen-Avg**). Generalisation is measured on 7 **unseen**
datasets: GRID, PRID, and i-LIDS averaged over 10 splits; VIPeR and CUHK01 over 20 runs (10 splits × 2
directions); CUHK02 and SenseReID on one fixed split. Their mean is **Unseen-Avg**.

## 6. Reported results (AKA order-1, final step)

| | Market | CUHK-SYSU | Duke | MSMT17 | CUHK03 | Seen-Avg |
|---|---|---|---|---|---|---|
| PAD mAP | 77.5 | 93.9 | 71.1 | 37.7 | 66.4 | 69.3* |
| PAD Rank-1 | 89.8 | 94.9 | 82.6 | 64.0 | 68.5 | 80.0* |

\*Mean of the five columns. The paper also reports a **7-seed** result of **70.30 ± 0.49 mAP / 80.98 ± 0.31 R1**
for Seen-Avg. These values come from search-engine extracts of the arXiv HTML. Check them, and the
Unseen-Avg and order-2 numbers, in Tables 1–2 of the PDF, then edit `PAPER_REPORTED` in the notebook if needed.

**What counts as a successful replication?** The 7-seed std is about 0.5 mAP, so a single run within about **±1 point
of Seen-Avg** is consistent. Per-domain gaps of 1–2 points are normal for one seed. GPU type (T4 vs the authors'
GPU), cuDNN non-determinism, and dataset version differences also move numbers slightly.

## 7. Implementation observations (code vs. paper description)

These came from reading the official code carefully. They are not necessarily bugs, but they are worth raising
with your guide and in any write-up. The notebook reproduces the official behaviour in every case, so its numbers
stay comparable.

1. **The "Fixed-T" text anchor is the *initial* prompt of the current domain, not the previous domain's text.**
   On resume, only the visual branch is loaded (`_filter_resume_state_dict`), so the TA-Prompt of each new domain
   is freshly random. `model_old` is copied **before** Stage 1, so TEXKD pulls the student's similarity
   distribution toward that of the *untrained* prompts, i.e. toward the frozen encoder's response to roughly
   *"A photo of a person"*. That fits the paper's thesis (the frozen text encoder is the anchor), but it is a
   regulariser toward a generic text prior rather than a transfer of old identities' text features.
2. **E-Prompt keys are never trained.** Top-k selection runs under `@torch.no_grad()` and there is no
   key–query matching loss (unlike L2P/DualPrompt). The keys stay at their random initialisation, so slot
   selection works like a fixed random projection of the layer's mean token. Only the prompt values learn.
3. **Test-time slot count.** `test_lifelong.py` rebuilds `active_sizes` by counting slots with non-zero
   weights. Unused slots are randomly initialised (not zero), so this always yields **36**. After step *t* < 5,
   evaluation therefore lets top-k pick from slots that were never trained. The *final* result is unaffected
   (all 36 are allocated by then), but the intermediate forgetting curve is affected. The notebook switch
   `OFFICIAL_EVAL_SLOT_RECONSTRUCTION=False` evaluates with the training-time slot count, which makes a nice small
   ablation.
4. **CUHK-SYSU camera IDs.** The README says to rename all images to `{pid}_c1_{idx}.jpg`. If query and gallery
   both use camera 1, the standard evaluator (which ignores same-ID **and** same-camera gallery hits) discards
   every true match and fails. The notebook gives the query split camera 2, as the AKA/PTKP loaders do.
5. **Stage 1 trains prompts at `PROMPT_LR`** (5e-3 / 2e-3), not at `STAGE1.BASE_LR` (3.5e-4), because the optimiser
   routes every parameter whose name contains `prompt_learner` into the prompt group.
6. **VISKD includes the penultimate-layer CLS**, and the triplet loss is applied to all three features.
7. **Unseen data path.** The README shows `data/Unseen/<set>`, but the loaders read `data/<set>`.
8. **Single GPU matters.** With more than one GPU visible, the code silently switches to `nn.DataParallel` without
   SyncBN, which the authors warn increases forgetting. The notebook pins one GPU (Kaggle's T4 x2 exposes two).

## 8. The reproduction notebook

**What is identical to the official code:** model, prompt pool, losses, optimiser groups, schedulers, sampler,
augmentations, training order, per-domain overrides, seeds, deterministic cuDNN, fp32 evaluation, and the unseen
split protocol.

**Verified numerically.** A test loads the *official* `pad` package (CUDA redirected to CPU) next to the notebook
code and compares them. All of the following matched **exactly (difference 0.0)**:
* all CLIP-derived weights after positional-embedding resizing, and the TA-Prompt prefix/suffix embeddings (the
  notebook ships pre-computed token IDs, `[49406, 320, 1125, 539, 320, 343, 343, 343, 343, 2533, 269, 49407]`,
  so it needs no tokenizer vocabulary file);
* training-mode forward pass (ID scores, the three features, projection), eval feature, and text features;
* CMC and mAP from the notebook's chunked evaluator vs the official `_eval_func`;
* the set of trainable tensors and the optimiser groups/LRs in Stage 1 and Stage 2 of a later domain.

The whole notebook was also executed end to end on synthetic data in every supported dataset format: 5 domains,
all evaluations, 6 unseen sets, results table and figure, and resume from an attached previous output. A resumed
run reproduced the same metrics exactly.

**What is different (infrastructure only):**

| Change | Why | Effect on results |
|---|---|---|
| Kaggle dataset auto-discovery + symlinks | Kaggle datasets have arbitrary layouts | none |
| CUHK03 converted with h5py, mapped by the CUHK03-NP file list | the official converter assumes a scipy-readable `.mat`; the real file is v7.3/HDF5 | same split, same naming |
| CUHK-SYSU converter (raw or cuhksysu4reid input) | no converter shipped in the repo | see §7.4 |
| Chunked evaluator | the official one needs ~15 GB RAM for MSMT17 | identical numbers (verified) |
| Saves visual-branch weights only | the official resume loads only these anyway | none |
| CLIP weights: OpenAI URL → HF mirror fallback | Kaggle network reliability | same weights |
| 4 data-loader workers | Kaggle has 4 vCPUs | speed only |

**Running it:** see the notebook's first cells. In short: T4 x2 + Internet on, attach the four Kaggle datasets (§8.1), run once with
`RUN_MODE="quick"` (about 5 min), then `RUN_MODE="full"` with *Save & Run All*. Expect about 6–8 h on one T4
(MSMT17 is the longest step). If the 12 h limit is hit, attach the run's output as an input and rerun, and finished
domains are skipped.

**Datasets** (Kaggle slugs found by web search; check the layout before running):
Market-1501 `pengcw1/market-1501` · DukeMTMC-reID `whurobin/dukemtmcreid` · MSMT17 `achrafabid7/msmt17` ·
CUHK03 `priyanagda/cuhk03` · CUHK-SYSU: official site (not found on Kaggle; upload privately) · unseen test sets:
the DualNorm release (Google Drive id `1-5JqXxqQ14MCngGLMPqiA-zbI2CUBFxV`). Note that DukeMTMC-reID was withdrawn
by its creators in 2019. Check your institution's policy on using it.

### 8.1 Kaggle-only protocol (`PROTOCOL = "kaggle-4"`, the notebook default)

CUHK-SYSU and the seven unseen test sets are not on Kaggle, so by default the notebook runs the paper's
**AKA order-1 sequence with CUHK-SYSU left out**:

**Market-1501 → DukeMTMC-reID → MSMT17 → CUHK03** (Kaggle: `pengcw1/market-1501`, `whurobin/dukemtmcreid`,
`achrafabid7/msmt17`, `priyanagda/cuhk03`)

| What stays the same as the paper | What changes |
|---|---|
| CLIP ViT-B/16 start, both stages, every loss and weight, all LRs, epochs, batch sizes, and per-domain overrides (Market as the first domain without KD; MSMT17 longer schedule; CUHK03 batch 48, erasing 0.6, 4 new slots), seed, single GPU, and evaluation metric | 4 steps instead of 5, so there is one fewer domain to forget through |
| Prompt-pool mechanics (8 new slots per domain, 4 for CUHK03, old slots frozen) | the pool ends at **28 / 36** slots instead of full |
| | no Unseen-Avg (no unseen sets on Kaggle); this part is skipped automatically |

**How to read the results.** The paper has no published 4-domain numbers, so the notebook compares against the
paper's 5-domain numbers **for the same four domains** (paper average over Market/Duke/MSMT17/CUHK03: 63.2 mAP /
76.2 R1). Expected direction of the gap:
* **Market and Duke** should come out *equal or a little higher* than the paper, because the model passes through
  one fewer later domain and has less chance to forget them.
* **MSMT17 and CUHK03** are trained in the same position relative to the end of the sequence, so they should land
  *close to* the paper's numbers.
* A result clearly *below* the paper on the last two domains, or far below on Market/Duke, would be a meaningful
  sign of a reproduction problem. That is the useful check to report to your guide.

**Prompt slots when the pool is not full.** Because the pool ends at 28/36, observation §7.3 now affects the
*final* numbers: the official test script lets top-k also pick the 8 never-trained random slots. The notebook
therefore evaluates the final model **twice** and prints both: (a) the official way (all 36 slots, for fidelity),
and (b) with only the 28 trained slots (`final_eval_trained_slots.csv`). Report (a) as "official code behaviour" and
(b) as a sanity variant. A large gap between them is itself a finding worth mentioning.

To run the full paper protocol later, upload CUHK-SYSU as a private Kaggle dataset and set `PROTOCOL = "paper-5"`.

## 9. Suggested follow-up experiments (cheap, informative)

* **Seed variance:** 2–3 seeds (`BASE_CFG["SOLVER"]["SEED"]`) to compare against the paper's ±0.49.
* **Ablations by switching off one component:** `TEXTKD.ENABLE=False`, `DISTILL.ENABLE=False`,
  `VA_PROMPT.ENABLE=False`, `UNFREEZE.LAST_BLOCKS=0`. Each changes one line in `BASE_CFG`.
* **Observation §7.3:** rerun the evaluation with `OFFICIAL_EVAL_SLOT_RECONSTRUCTION=False` (no retraining needed).
* **Observation §7.2:** add a key–query pull loss and see whether learned routing helps.

## 10. Sources
* PAD paper, arXiv 2605.05027 ([abs](https://arxiv.org/abs/2605.05027), [html](https://arxiv.org/html/2605.05027)) and code ([zu-zi/PAD](https://github.com/zu-zi/PAD)).
* CLIP-ReID (Li, Sun, Li; AAAI 2023): the two-stage text-prompt recipe PAD builds on.
* AKA benchmark (Pu et al., CVPR 2021; [TPCD/LifelongReID](https://github.com/TPCD/LifelongReID)): the LReID protocol and `cuhksysu4reid` preprocessing.
* DualPrompt (Wang et al., ECCV 2022): the G-/E-prompt design.
* CUHK03-NP split (Zhong et al., CVPR 2017; [person-re-ranking](https://github.com/zhunzhong07/person-re-ranking)).
* Kaggle datasets: [Market-1501](https://www.kaggle.com/datasets/pengcw1/market-1501), [DukeMTMC-reID](https://www.kaggle.com/datasets/whurobin/dukemtmcreid), [MSMT17](https://www.kaggle.com/datasets/achrafabid7/msmt17), [CUHK03](https://www.kaggle.com/datasets/priyanagda/cuhk03).
