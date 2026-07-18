# Stage 1 Report — GPT-J-6B + Antonym Smoke Test

**Project:** BlackboxNLP 2026 Reproducibility Challenge — Function Vectors generalizability study
**Target paper:** Todd et al., "Function Vectors in Large Language Models" (ICLR 2024)
**Date:** 2026-07-18
**Hardware:** RTX A6000 (48 GB) — *note: not the A100-80GB the plan assumed*
**Status:** ✅ Smoke test complete. Stopped, awaiting confirmation before full sweep.

---

## Environment (all verified)

| Item | Value |
|---|---|
| `HF_HOME` | `/workspace/hf_cache` (persisted in `~/.bashrc`) |
| torch / CUDA | 2.4.1+cu124, CUDA available, RTX A6000 |
| transformers | 4.49.0 (pinned, per authors — kept for GPT-J anchor) |
| HF auth | logged in as `Zachbot168` |
| GPU during run | 100% util, ~23.9 GB used (GPT-J fp32 resident) |

**Git hygiene done** (before any results existed): `*_mean_head_activations.pt` added to `.gitignore` (large/regenerable, 44 MB each); `*indirect_effect*.pt` (46 KB) and eval JSONs confirmed **trackable**; `hf_cache` lives outside the repo so it can never be committed.

---

## 1. FV accuracy vs zero-shot baseline (our run)

Task = antonym. Test set 504 examples; **319 pass the 10-shot filter** (model ranks the gold answer #1); FV accuracy is measured on that filtered set (standard for this method).

| Metric (top-1) | Value |
|---|---|
| Zero-shot, **no intervention** | **0.9%** |
| Zero-shot, **+FV at best layer (L11)** | **60.5%** |
| Zero-shot, +FV at **L9** (paper's operating layer ≈ L/3) | 49.2% |
| Pure model 0-shot baseline (full test set) | 0.6% |
| 10-shot ICL baseline (no FV) | 63.3% |
| 10-shot **shuffled-label**: no-FV → +FV | 46.4% → **86.8%** |

**Zero-shot layer sweep (top-1 accuracy, +FV at each layer):**

```
L 0: 0.423   L 7: 0.498   L14: 0.348   L21: 0.016
L 1: 0.436   L 8: 0.542   L15: 0.276   L22: 0.016
L 2: 0.439   L 9: 0.492   L16: 0.044   L23: 0.013
L 3: 0.545   L10: 0.455   L17: 0.034   L24: 0.013
L 4: 0.455   L11: 0.605*  L18: 0.016   L25: 0.013
L 5: 0.524   L12: 0.602   L19: 0.016   L26: 0.013
L 6: 0.558   L13: 0.376   L20: 0.016   L27: 0.009
```
(clean / no-intervention ≈ 0.009 at every layer; * = best)

**Shape:** rises from L0, peaks at **L11–12**, collapses to baseline after ~L15. For a 28-layer model L/3 ≈ 9.3 — squarely in the peak band. This is the textbook FV curve and confirms the paper's "inject around the first third of the network" finding.

---

## 2. Comparison to Todd et al. (ICLR 2024) — actual reported values

Values looked up directly from the arXiv HTML (2310.15213v2), **not estimated**:

| Quantity | Todd et al. (reported) | Our run | Verdict |
|---|---|---|---|
| GPT-J antonym, zero-shot **+FV** (their operating layer) | **48.2 ± 2.0%** | **49.2%** at L9 | **Within noise — match** |
| GPT-J antonym, zero-shot baseline | ~floor (6-task avg 5.5%) | 0.9% | Consistent (floor) |
| 6-task-avg zero-shot: baseline → +FV | 5.5% → 57.5% | antonym 0.9% → 60.5% (best) | Consistent regime |
| Shuffled-label 10-shot + FV (6-task avg) | ~90.8% | antonym 86.8% | Consistent |

**Apples-to-apples:** injecting at the layer the paper uses (~L9) gives **49.2% vs their 48.2%** — indistinguishable. Our best-over-sweep (60.5% at L11) is higher only because we report the sweep maximum rather than a single fixed layer.

**Source table:** the paper's per-task GPT-J zero-shot number (48.2%) is in the appendix results table; the 5.5% → 57.5% figures are the 6-task macro-average from the main results table.

**Conclusion: reproduction is solid.** No BOS / tokenization / transformers-version red flags. Safe to proceed to the modern-model generalization runs.

---

## 3. Runtime + full-sweep projection

**Measured breakdown for this one (model, task) pair — total 100.2 min (6010 s):**

| Stage | Time |
|---|---|
| Model load (GPT-J **fp32**, ~24 GB) + 10-shot filter (216 valid + 504 test) | ~5 min |
| Mean head activations (100 trials) | 0.25 min |
| Indirect effect (25 shuffled trials × 28 layers × 16 heads) | ~26 min |
| **Layer sweep (zero-shot + shuffled, all 28 layers)** | **~59 min ← dominant** |
| Model baseline (0→10 shot, no FV) | ~10 min |

### Projection for the full sweep

| Scope | Estimate | Notes |
|---|---|---|
| **Main runs (T1 + T2 + F1):** 4 models × 6 tasks × ~100 min | **~40 h** | Modern models are bf16 (faster per-op) but 28–32 layers, so per-run cost ≈ comparable to GPT-J |
| **k-ablation as `run_experiments.sh` is currently written** (`--edit_layer -1` → full 59-min sweep per run, 4×2×3 = 24 runs) | **+~20 h (wasted)** | Ablation only needs a *single* layer, not a full sweep |
| **Total, as-written** | **~60 h** | |
| **Total, with fixes below** | **~43 h** | k-ablation at fixed best-layer + skip redundant baselines ≈ 3–4 h instead of 20 h |

**vs the plan's 24–36 h estimate:** that assumed an **A100-80GB**; this box is an **RTX A6000 (48 GB)** and the layer sweep is heavier than budgeted. At RunPod rates (~$1.20–1.80/h), ~43 h ≈ **$50–75** — within the stated budget. ~1.8 days is comfortable before the **July 24 AoE** deadline.

**Caveat:** the projection for the 3 modern models is extrapolated from GPT-J only; their real per-run time won't be known until the first modern-model run. Llama-3.1-8B and Mistral-7B-v0.3 have 32 layers, Qwen2.5-7B has 28.

---

## Fixes I will apply to `run_experiments.sh` before launching Stage 2

1. **Correct the path-doubling.** The scripts append `/{dataset_name}` internally, but `run_experiments.sh` also passes the task in `--save_path_root`, so results would land in a doubled `results_repro/<model>/<task>/<task>/`. Fix to the canonical `results_repro/<model>/<task>/` — which also matches this smoke run, so GPT-J/antonym is **reused, not recomputed**.
2. **k-ablation at the fixed best layer** (`--edit_layer=<L>` from the main run) instead of a full sweep. ~20 h saved.
3. **Load each model once** and loop its tasks in-process, avoiding ~11 redundant fp32 model reloads per model.
4. **Fix the broken cache-check** (`if [ ! -f "$OUT/${TASK}_indirect_effect.pt" ]` looks at the wrong path) so completed indirect-effects are skipped on resume.

**Optional trim:** dropping the *shuffled-label* half of the layer sweep (keeping zero-shot, the headline metric) cuts the main runs from ~40 h to **~28 h** (total ~31 h with fixes). Loses the shuffled-label generalization data.

---

## Open questions for the user (before Stage 2)

1. **Full plan (~43 h) or zero-shot-only trim (~31 h)?**
2. **Keep all 4 models × 6 tasks**, or use the plan's fallback (2 modern models, or 4 tasks)?
3. **Pre-flight token-ID check** on one constructed prompt to confirm no double-BOS on Llama-3 (CLAUDE.md gotcha) before the long modern-model runs — do it? (~1 min)

**Nothing is committed yet.** Per the git rule, GPT-J results will be committed after all its tasks finish (not just this smoke test). Smoke-test outputs are on disk at `results_repro/EleutherAI_gpt-j-6b/antonym/` and gitignore-verified.

---

## Provenance (every number above traces to a file)

All in `results_repro/EleutherAI_gpt-j-6b/antonym/`:

| File | Contains |
|---|---|
| `zs_results_layer_sweep.json` | Zero-shot per-layer clean vs +FV accuracy (→ best-layer 60.5%, L9 49.2%) |
| `fs_shuffled_results_layer_sweep.json` | 10-shot shuffled per-layer clean vs +FV (→ 46.4% → 86.8%) |
| `model_baseline.json` | 0→10-shot no-FV baseline (→ 0-shot 0.6%, 10-shot 63.3%) |
| `fs_results_layer_sweep.json` | 10-shot filter (→ 319/504 pass) |
| `antonym_indirect_effect.pt` | Per-head AIE, 25 trials (→ FV head selection, T4) |
| `antonym_mean_head_activations.pt` | Mean head activations (gitignored, 44 MB) |
| `fv_eval_args.txt` | Exact args/config for this run |
| `../smoke_gptj_antonym.log` | Full stdout + `ELAPSED_SEC=6010` timing marker |
