# Provenance: every paper number → committed source file

All source files live under `results_repro/` in this repo. Seed 42 throughout;
transformers 4.49.0; single RTX A6000. `<M>` ∈ {EleutherAI_gpt-j-6b,
meta-llama_Llama-3.1-8B, Qwen_Qwen2.5-7B, mistralai_Mistral-7B-v0.3}, `<T>` ∈ the six tasks.

| Artifact | Derived from | By |
|---|---|---|
| T1 (GPT-J vs paper) | `results_repro/EleutherAI_gpt-j-6b/<T>/zs_results_layer_sweep.json`; paper column = Todd et al. Table 10 (arXiv 2310.15213v2) | `analysis/make_tables.py` |
| T2 (FV effect matrix) | `results_repro/<M>/<T>/zs_results_layer_sweep.json` (effect = top-1 `intervention_topk` − `clean_topk`, max over layers) | `analysis/make_tables.py` |
| T3 (k-ablation) | k=10 from the main sweep; k∈{5,20,50} from `results_repro/<M>/k<K>/<T>/zs_results_editlayer_<L>.json`; k=100 (Mistral antonym) from `results_repro/mistralai_Mistral-7B-v0.3/k100/antonym/zs_results_editlayer_14.json` | `analysis/make_tables.py` |
| T4 / F2 (AIE concentration) | `results_repro/<M>/<T>/<T>_indirect_effect.pt` (mean over 25 trials; share = top-k sum / positive sum) | `analysis/make_tables.py` |
| F1 / F1b (layer sweeps) | `results_repro/<M>/<T>/zs_results_layer_sweep.json` | `analysis/make_tables.py` |
| ICL baselines | `results_repro/<M>/<T>/model_baseline.json` (`'0'`/`'10'` → `clean_topk` top-1) | direct read |
| FV norms / vocab decode | cached mean-activations + IE tensors → `analysis/fv_probe_results.json` | `analysis/fv_probe.py` |
| Mistral verification | all of the above + `results_repro/sweep_progress.log` (epochs) + git history | `analysis/mistral_verification.md` |
| Runtimes | `results_repro/sweep_progress.log` (`elapsed=` fields); smoke-test total in `results_repro/smoke_gptj_antonym.log` (`ELAPSED_SEC=6010`) | grep |

Notes:
- Ours-vs-paper caveat (T1): paper reports FV accuracy at *their chosen layer*, mean±std over
  seeds; we report best-over-layer-sweep at seed 42. Our GPT-J antonym at the paper's layer
  (~L9) was 49.2% vs their 48.2±2.0 (layer key "9" in
  `results_repro/EleutherAI_gpt-j-6b/antonym/zs_results_layer_sweep.json`).
- `*_mean_head_activations.pt` files are **gitignored** (44 MB each, regenerable via
  `compute_indirect_effect.py` / the driver from the committed configs). They existed on the
  run pod only. Everything needed to *recompute* them (dataset, seed, args json) is committed.
- Filter sets: `results_repro/<M>/<T>/fs_results_layer_sweep.json` (10-shot clean ranks).
- Exact run args per (model, task): `results_repro/<M>/<T>/fv_eval_args.txt`.
