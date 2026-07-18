# CLAUDE.md — FV Reproducibility Challenge (deadline July 24 AoE)

## What this project is
BlackboxNLP 2026 Reproducibility Challenge submission (OpenReview Special Track,
https://blackboxnlp.github.io/2026/reproducibility). We stress-test "Function
Vectors in Large Language Models" (Todd et al., ICLR 2024) along the
**generalizability** theme: do FV findings and hyperparameter heuristics transfer
to modern GQA models (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B-v0.3)?
Read EXPERIMENT_PLAN.md first — it defines the tables the paper needs.

## State of the repo
- Upstream: github.com/ericwtodd/function_vectors, already cloned, with
  `modern_models.patch` applied to src/utils/model_utils.py:
  - new branch for qwen/mistral (bf16, o_proj hooks, head_dim assert)
  - Llama-3.x routed to bf16 instead of fp32
- Verified on CPU: all imports work, datasets load (antonym: 1678/216/504
  train/valid/test), prompt construction produces correct Q:/A: format.
- NOT yet done: any GPU run, any HF model download.

## Environment setup (do first)
1. `pip install torch transformers==4.49.0 accelerate datasets bitsandbytes sentencepiece scikit-learn matplotlib seaborn`
   then `pip install git+https://github.com/davidbau/baukit@main`
   (transformers 4.49.0 is what the authors pin — keep it for the GPT-J anchor run)
2. `huggingface-cli login` — user must have accepted Llama-3.1 and Mistral licenses
3. Sanity: `python -c "import torch; print(torch.cuda.get_device_name(0))"`

## Execution order
1. **Smoke run (30 min):** GPT-J + antonym only. Verify
   results_repro/EleutherAI_gpt-j-6b/antonym/ contains indirect_effect + eval jsons,
   and zero-shot FV accuracy is in the ballpark of the paper (~0.5+ on antonym at
   best layer vs ~0 baseline). If numbers are wildly off, STOP and debug before
   burning GPU hours: check transformers version, BOS handling, prompt format.
2. **Full sweep:** `bash run_experiments.sh` (models × 6 tasks, then k-ablation).
   ~24-36 A100-hours. Run under tmux/nohup. Stages cache to disk; re-runs skip
   completed indirect-effect computations.
3. **Analysis:** write analysis/make_tables.py to produce:
   T1 repro table, T2 generalization table, F1 layer-sweep figure (acc vs layer
   per model), T3 head-count ablation, T4 AIE concentration (share of total AIE
   in top-10 heads). Matplotlib, one figure per file, PDF output.
4. **Paper:** 4-6 pages, EMNLP 2026 template (github.com/acl-org/acl-style-files).
   Structure is in EXPERIMENT_PLAN.md. Negative results are welcome and should be
   reported prominently, framed as boundary conditions.

## Known technical gotchas
- Per-head splitting assumes head_dim = hidden_size/n_heads (assert added).
  Gemma-family models violate this — they are intentionally excluded; mention in paper.
- Llama-3 tokenizer BOS: llama branch sets prepend_bos=True; verify no double-BOS
  in constructed prompts (print token ids for one prompt before long runs).
- Qwen2.5 tokenizer has no pad token by default; scripts use last-token logits so
  it shouldn't matter, but set tokenizer.pad_token = tokenizer.eos_token if needed.
- fv_eval_sweep.py in eval_scripts/ shows the authors' own sweep invocations —
  consult it if evaluate_function_vector.py args are ambiguous.
- If VRAM-limited (<40GB): all four models fit in bf16 on 24GB EXCEPT GPT-J which
  the repo loads in fp32 by default — pass smaller batch or load fp16 via code edit.

## Style/report rules
- Every number in the paper must trace to a json/pt file in results_repro/ —
  keep a provenance note in analysis/.
- This is archival ACL Anthology publication: original paper must be cited, our
  repo must be open-sourced (add LICENSE + README before submission).
- Author: Zach Lee, UIUC. Single-author unless he says otherwise.
