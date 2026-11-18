# LiteLReID: lightweight, exemplar-free lifelong person Re-ID (Kaggle prototype)

A research prototype for **low-resource lifelong person Re-ID**:
* **cheap inference:** the deployed model is a small student (MobileCLIP-S1 by default);
* **no stored images:** only one prototype vector per old identity is kept between domains;
* **few labels:** `LABEL_FRACTION` controls how many identities of each new domain are labelled.

| File | What it is |
|---|---|
| [`LiteLReID_Kaggle.ipynb`](LiteLReID_Kaggle.ipynb) | **The Kaggle notebook.** Part 1 trains the PAD teacher (the verified reproduction in `../reproduction/`, unchanged except that it also saves the teacher's per-identity text embeddings). Part 2 trains the student through the same domain sequence, evaluates both after every step and on the unseen sets, and measures parameters, GFLOPs, latency and throughput. |
| `tools/notebook_src/*.py` | The student cells (method, training, evaluation, efficiency, summary). |
| `tools/build_notebook.py` | Builds the notebook from the PAD cells plus the student cells. |

## Method in one paragraph
After the PAD teacher learns domain *t*, the student learns domain *t* from:
* the labels (ID and triplet losses);
* the teacher's features and in-batch similarity structure (feature and relation distillation, on all images);
* the teacher's per-identity **text anchors** (the student is CLIP-aligned, so it matches them directly);
* for *t* > 1, a frozen copy of the previous student together with **identity prototypes** of all earlier domains,
  which keep the old feature geometry without replaying any image.

The `naive_ft` baseline fine-tunes the same student domain by domain with ID and triplet loss only.

## Running on Kaggle
1. GPU T4 x2, Internet on. Attach `pengcw1/market-1501`, `manaschaiaonon/cuhk-sysu`, `whurobin/dukemtmcreid` and
   `priyanagda/cuhk03`.
2. `RUN_MODE = "quick"` for a smoke test, then `"full"` with *Save & Run All*.
3. Budget: about 6 h for the teacher plus about 3–4 h for the student on one T4. The notebook stops cleanly
   before 12 h; attach its output and run again to continue.

Main switches (in the "LiteLReID configuration" cell):

| Switch | Values |
|---|---|
| `STUDENT_ARCH` | `"MobileCLIP-S1"` (default, 21.5 M), `"MobileCLIP2-S0"` (11.4 M), `"MobileCLIP-S2"`, `"MobileCLIP-B"`, `"timm:resnet18"`, any `timm:` model |
| `STUDENT_METHODS` | `["ours"]`; add `"naive_ft"` for the lower-bound baseline (about 1.5 h more) |
| `LABEL_FRACTION` | `1.0` (fully supervised), or e.g. `0.25`. The teacher then also sees only the labelled 25 % |
| `LAMBDA`, `TAU` | loss weights and temperatures (untuned starting points) |

## Status
The pipeline was checked end to end on CPU with synthetic data and randomly initialised weights: teacher, student,
the `naive_ft` baseline, the few-label path, resume, evaluation, efficiency and summary. It has **not yet been run on a
GPU with real weights**, so the accuracy numbers are still to come. MSMT17 and CUHK01/02 are not included: they need
licence requests.
