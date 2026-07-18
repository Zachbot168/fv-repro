# BlackboxNLP 2026 Reproducibility Challenge — Function Vectors Generalizability Study

**Target paper:** Todd et al., "Function Vectors in Large Language Models" (ICLR 2024)
**Theme:** Generalizability (+ light Ablation)
**Deadline:** July 24, 2026, 11:59 PM AoE — submit via OpenReview BlackboxNLP Special Track
**Venue page:** https://blackboxnlp.github.io/2026/reproducibility

## Research question

Do the core FV findings — (1) a small set of attention heads carries a compact task
representation, (2) injecting the summed head outputs at layer ~L/3 induces zero-shot
task behavior — transfer from 2023-era models (GPT-J-6B, Llama-2) to modern
GQA-based models (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B-v0.3)? Do the paper's
hyperparameter heuristics (top-10/20 heads, edit layer ≈ L/3) remain optimal?

## Models

| Model | HF name | Role | Notes |
|---|---|---|---|
| GPT-J-6B | EleutherAI/gpt-j-6b | Reproduction anchor | Match paper's Table 1/2 numbers |
| Llama-3.1-8B | meta-llama/Llama-3.1-8B | Generalization | bf16 (patched); gated repo — accept license on HF first |
| Qwen2.5-7B | Qwen/Qwen2.5-7B | Generalization | new branch in model_utils patch |
| Mistral-7B-v0.3 | mistralai/Mistral-7B-v0.3 | Generalization | same branch; also gated |

All satisfy head_dim × n_heads = hidden_size (required by the codebase's per-head
splitting; assert added in patch). Gemma-2/3 do NOT — excluded, and this
architectural constraint is itself a reportable observation.

## Tasks (6 headline tasks from the paper, all present in dataset_files/)

antonym, capitalize, country-capital, english-french, present-past, singular-plural

Stretch: + person-instrument, product-company, sentiment (relation-style tasks).

## Pipeline per (model, task)

1. `compute_average_activations.py` — mean head activations over 100 10-shot prompts
2. `compute_indirect_effect.py` — AIE per head, 25 shuffled-label trials
3. `evaluate_function_vector.py --edit_layer -1` — build FV from top-k heads, sweep
   injection over all layers, eval zero-shot + shuffled-shot accuracy vs baselines

## Experiments → paper tables

- **T1 (reproduction):** GPT-J zero-shot FV accuracy on 6 tasks vs paper's reported
  numbers. Success = within noise.
- **T2 (generalization headline):** same 6 tasks × 3 modern models. Zero-shot
  accuracy: no-intervention baseline vs +FV, at best layer.
- **F1 (layer sweep):** accuracy vs injection layer per model — does the L/3 rule
  hold for modern architectures?
- **T3/F2 (head-count ablation):** k ∈ {5, 10, 20, 50} top heads — is top-10/20
  still the right operating point?
- **T4 (AIE concentration):** what fraction of total indirect effect is in the top
  10 heads per model — is the "small set of causal heads" claim architecture-general?

## Compute budget

Single A100-80GB (RunPod/Lambda, ~$1.20–1.80/hr). Per (model, task):
~100 fwd passes (mean acts) + 25×n_heads-lite AIE pass + layer-sweep eval.
Estimate 0.5–1.5 hr per (model, task) → 4 models × 6 tasks ≈ 24–36 GPU-hours
≈ $40–70. Cache mean_activations and indirect_effects to disk (scripts do this)
so re-runs are cheap. If tight: cut to 2 modern models or 4 tasks — still a paper.

## Paper skeleton (4–6 pp, EMNLP template)

1. Intro — why FVs matter, why generalizability of interp findings is the challenge theme
2. Background — FV method summary (½ page)
3. Setup — models, tasks, what we changed (patch), what we held fixed
4. Results — T1 reproduction, T2 generalization, layer/head ablations
5. Discussion — which heuristics transfer, which don't; the head_dim architectural
   constraint; negative results framed as boundary conditions
6. Limitations

## Risks / mitigations

- Llama/Mistral gated repos → accept licenses + `huggingface-cli login` day 1
- Tokenizer quirks (Llama-3 BOS handling) → verify prompt token counts before long runs
- GPT-J numbers don't match paper → check transformers version pinning
  (repo pins 4.49.0); report honestly either way — it's a repro challenge
- Time → T1+T2 alone is a submittable paper; ablations are upside
