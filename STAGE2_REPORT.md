# Stage 2 Report — Full FV Generalizability Sweep

**Project:** BlackboxNLP 2026 Reproducibility Challenge — stress-testing "Function Vectors in Large Language Models" (Todd et al., ICLR 2024) on the *generalizability* theme.
**Question:** Do FV findings — (1) a small set of attention heads carries a compact task representation, (2) injecting the summed head outputs at ~L/3 induces zero-shot task behavior — transfer from GPT-J-6B to modern GQA models (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B-v0.3)?
**Status:** ✅ Stage 2 complete. All 4 models × 6 tasks + k-ablation. Committed + pushed (`main` @ `850c1b4`).
**Date:** 2026-07-18/19. Hardware: single RTX A6000 (48 GB).

> **Metric note (important):** all transfer numbers below are reported as **FV effect = (zero-shot accuracy with FV) − (zero-shot accuracy, no intervention)**, at each task's best-effect layer. Raw "+FV" accuracy alone is misleading because it includes the zero-shot baseline (which is nontrivial for some tasks/models). Effect isolates what the FV actually contributes.

---

## Methodology (held fixed from the original repo)

- Per (model, task): mean head activations over 100 ten-shot prompts → indirect effect (AIE) per head over 25 shuffled-label trials → build FV from top-10 heads → sweep injection over all layers, eval zero-shot top-1 accuracy vs a no-intervention baseline. Seed 42, prompt format `Q:/A:`.
- Test set filtered to examples the model gets right at 10-shot (standard for this method).
- Modern models loaded bf16; GPT-J fp32 (repo default). GQA head-splitting valid for all four (head_dim × n_heads = hidden_size).
- k-ablation: k ∈ {5,10,20,50} at each task's best layer (not a full re-sweep).

---

## T1 — Reproduction anchor (GPT-J-6B)

All six tasks reproduce a strong FV effect, validating the pipeline:

| Task | best L | no-interv | +FV | **effect** | 10-shot ICL |
|---|---|---|---|---|---|
| antonym | 11 | 0.9% | 60.5% | **+0.596** | 63% |
| capitalize | 10 | 0.0% | 77.5% | **+0.775** | 99% |
| country-capital | 8 | 4.9% | 90.2% | **+0.854** | 98% |
| english-french | 5 | 1.5% | 72.5% | **+0.710** | 80% |
| present-past | 3 | 1.7% | 55.2% | **+0.534** | 95% |
| singular-plural | 13 | 21.4% | 83.3% | **+0.619** | 98% |

Cross-check vs paper: GPT-J antonym FV at the paper's operating layer (~L9) = **49.2%** in our run vs the paper's reported **48.2 ± 2.0%** — within noise. (Best-over-sweep is 60.5% at L11.)

---

## T2 — Generalization (FV effect = intervention − clean, at best layer)

| Task | GPT-J | Llama-3.1 | Qwen2.5 | Mistral |
|---|---|---|---|---|
| antonym | +0.60 | +0.47 | +0.64 | +0.07 |
| capitalize | +0.78 | +0.88 | +0.94 | +0.01 |
| country-capital | +0.85 | +0.43 | +0.33 | +0.00 |
| english-french | +0.71 | +0.05 | +0.03 | +0.00 |
| present-past | +0.53 | +0.02 | +0.47 | +0.00 |
| singular-plural | +0.62 | +0.78 | +0.88 | +0.00 |
| **tasks transferring (>0.3)** | **6/6** | **4/6** | **5/6** | **0/6** |

Best-effect layers per (model,task): GPT-J L3–13, Llama-3.1 L8–14, Qwen2.5 L16–19, Mistral (no meaningful peak).

---

## Key findings

1. **The core FV claim generalizes to Llama-3.1 and Qwen2.5** — 4–6 of 6 tasks, with effects often matching or exceeding GPT-J (e.g. capitalize +0.88/+0.94, singular-plural +0.78/+0.88). Modern GQA architecture does not, in general, break function vectors.

2. **english-french fails on *all* three modern models** (+0.03–0.05) while working strongly on GPT-J (+0.71). A **task-general boundary condition** — the translation FV does not transfer off the 2023-era model. (French answers are multi-token; the first-token metric is a candidate contributor, but this held across two different tokenizer families, so it is not purely tokenization.)

3. **present-past is model-specific** — works on GPT-J (+0.53) and Qwen2.5 (+0.47) but fails on Llama-3.1 (+0.02). A Llama-3.1 idiosyncrasy, not a universal failure. (Contrast: singular-plural, also morphological, transfers on Llama at +0.78 — so it is not "morphology fails.")

4. **Mistral-7B-v0.3 is the null case at top-10 heads** — effect ≤ +0.07 on every task. The FV *does* perturb the model (it degrades the zero-shot baseline by up to −0.19/−0.42 at some layers on present-past/singular-plural), so injection works; the top-10-head vector simply fails to carry the task. **But see the k-ablation below** — Mistral partially recovers with more heads, suggesting a more *distributed* task representation rather than a dead pipeline. Bug-vs-genuine is still being verified (open item).

5. **The "L/3" heuristic does not hold uniformly.** Optimal injection layer drifts *later* with newer architectures: GPT-J ~L3–13 (of 28), Llama-3.1 ~L8–14 (of 32), **Qwen2.5 ~L16–19 (of 28) ≈ 0.6× depth**. The paper's "inject around the first third" is GPT-J-specific.

6. **Optimal head count is task- and model-dependent** (top-10 is not universally best):

| Model | antonym (k5/k10/k20/k50) | country-capital (k5/k10/k20/k50) |
|---|---|---|
| GPT-J | 0.55 / 0.61 / 0.70 / 0.64 | 0.73 / 0.90 / 0.93 / 0.93 |
| Llama-3.1 | 0.58 / 0.48 / 0.69 / 0.73 | 0.12 / 0.48 / 0.29 / 0.93 |
| Qwen2.5 | 0.43 / 0.64 / 0.62 / 0.50 | 0.31 / 0.38 / 0.81 / 0.93 |
| Mistral | 0.09 / 0.12 / 0.16 / **0.34** | 0.05 / 0.05 / 0.00 / 0.05 |

   Notably, **Mistral antonym rises to 0.34 at k=50** (from 0.12 at k=10) — its task signal is spread over *more* heads than the paper's top-10 captures. This challenges the "small set of causal heads" claim for Mistral (at least for some tasks; country-capital stays null even at k=50).

---

## Runtimes (single A6000)

| Model | 6 main tasks | notes |
|---|---|---|
| Llama-3.1-8B (bf16) | 3.8 h | ~3× faster than GPT-J |
| Qwen2.5-7B (bf16) | 2.6 h | fastest |
| Mistral-7B-v0.3 (bf16) | 3.9 h | 32 layers |
| GPT-J-6B (fp32) | 6.8 h | slow (fp32); english-french alone 3.5 h |
| **Total main + k-ablation** | **~17.5 h** | ≈ $25–35 at RunPod rates; well under the 24–36 h budget |

The original 43 h projection was conservative (it assumed modern models ran at GPT-J's fp32 speed; they do not).

---

## Infrastructure / integrity

- **Load-once driver** (`run_sweep.py`) was equivalence-verified against per-task CLI: recomputed GPT-J/antonym mean activations through the driver matched the cached smoke-test file **exactly** (`allclose`, maxdiff 0.0).
- **BOS preflight** on Llama-3.1 and Qwen2.5: single leading BOS token each, no double-BOS.
- **3 bugs found & fixed mid-run** (committed `e71f052`): Qwen/Mistral were missing from two model-architecture dispatch sites (`intervention_utils` `new_output`, `extract_utils.compute_function_vector` `out_proj`) — their o_proj is `nn.Linear(bias=False)`, same as Llama; and Mistral's tokenizer needed `protobuf`. Recovered from cache with ~30 min lost. All other models' results are unaffected (they were correct throughout).
- **Provenance:** every number above traces to a committed `zs_results_layer_sweep.json` / `model_baseline.json` / `*_indirect_effect.pt` under `results_repro/<model>/<task>/`. Mean-activation tensors are gitignored (large, regenerable).
- Git history: `c8636df` scripts → `29234da` Llama → `e71f052` fixes → `5d8466a` Qwen → `1fad6bb` Mistral → `850c1b4` GPT-J. All pushed.

---

## Open questions / next steps

1. **Mistral bug-vs-genuine (highest priority).** The k=50 partial recovery (antonym 0.34) argues *against* a dead pipeline and *toward* a genuinely more-distributed representation, but capitalize's total collapse (0.6% vs 92–99% elsewhere) still warrants a GPU check: compare ‖FV‖ across models, decode the FV to vocab (`fv_to_vocab`) to see if Mistral's FV points at task-relevant tokens, and test injection scaling. Cache makes a re-run cheap.
2. **Stage 3 — analysis:** `make_tables.py` for T1–T4 + F1 layer-sweep figure (acc vs layer per model) + T4 AIE concentration (share of total AIE in top-10 heads per model — directly tests the "small set of heads" claim, and Mistral's k-ablation predicts it will be *lower* for Mistral).
3. **Stage 4 — paper (4–6 pp, EMNLP template):** the negative results (Mistral null / distributed heads, english-french boundary, layer drift, non-universal head count) are the substance and frame cleanly as boundary conditions. Deadline **July 24 AoE**.

---

## One-paragraph summary for discussion

Function vectors reproduce cleanly on GPT-J-6B (all 6 tasks, matching the paper) and **generalize well to Llama-3.1-8B (4/6) and Qwen2.5-7B (5/6)** — the core mechanism survives the move to modern GQA models. But the transfer is **not uniform**: english-french fails on every modern model (task-general), present-past fails only on Llama-3.1 (model-specific), the optimal injection layer drifts from ~L/3 (GPT-J) toward ~0.6× depth (Qwen), and the top-10-head heuristic is not universally optimal. **Mistral-7B-v0.3 is the striking negative case** — near-zero transfer at top-10 heads, though it partially recovers with 50 heads, hinting its task representation is more distributed than the FV method assumes. Net: FV findings are real but architecture-contingent, which is exactly the boundary-condition story the reproducibility challenge targets.
