# Do Function Vectors Transfer? A Reproducibility Study on Modern GQA Language Models

Reproducibility study of **"Function Vectors in Large Language Models"**
(Todd et al., ICLR 2024, [arXiv:2310.15213](https://arxiv.org/abs/2310.15213)),
submitted to the [BlackboxNLP 2026 Reproducibility Challenge](https://blackboxnlp.github.io/2026/reproducibility)
(generalizability theme).

We reproduce the core FV results on GPT-J-6B and stress-test their transfer to
three modern grouped-query-attention models — Llama-3.1-8B, Qwen2.5-7B, and
Mistral-7B-v0.3 — across the paper's six headline tasks. In brief: the
reproduction succeeds and the core mechanism transfers to Llama-3.1 (4/6
tasks) and Qwen2.5 (5/6), but five findings prove GPT-J-specific: the optimal
injection layer drifts from ~0.32× depth to ~0.61× depth, top-10 heads is not
a universal operating point, english-french fails on all modern models,
Mistral-7B-v0.3 is a systematic null at top-10 heads (recovering monotonically
with head count, consistent with a more distributed task representation), and
FV vocabulary decodability does not generalize. See `paper/` for the full
write-up and `analysis/mistral_verification.md` for the Mistral bug-vs-genuine
analysis.

## Repository layout

| Path | Contents |
|---|---|
| `function_vectors/` | Upstream codebase (vendored, MIT © Eric Todd) with our modernization changes: GQA model dispatch, bf16 loading, head-dim assertion, load-once sweep driver (`src/run_sweep.py`) |
| `modern_models.patch` | The core model-support patch as a standalone diff |
| `run_experiments.sh` | Full sweep entry point (4 models × 6 tasks + k-ablation) |
| `results_repro/` | **All raw results** (layer-sweep evals, baselines, indirect-effect tensors, exact run args per cell) |
| `analysis/` | `make_tables.py` (tables/figures/summary from `results_repro/`), `fv_probe.py` (FV norm + vocab decode), `provenance.md` (number → file map), `mistral_verification.md` |
| `paper/` | LaTeX source; `prepare_tables.sh` wires the generated tables into the paper |
| `EXPERIMENT_PLAN.md` | The pre-registered experiment plan |

## Reproducing

**Environment.** Python ≥3.10, CUDA GPU (~48 GB VRAM; GPT-J loads fp32).
`pip install -r requirements.txt` — `transformers` is pinned to 4.49.0 (the
original authors' pin). Accept the Llama-3.1 and Mistral licenses on Hugging
Face and `huggingface-cli login` before running.

**Runs.** `bash run_experiments.sh` regenerates everything under
`results_repro/` (~17.5 GPU-hours on one RTX A6000; stages cache to disk and
completed cells are skipped on re-run). Seed 42 throughout.

**Analysis + tables.** `python analysis/make_tables.py` rebuilds
`analysis/tables/`, `analysis/figures/`, and `analysis/results_summary.json`
from `results_repro/`. Every number in the paper traces to a committed file —
the map is `analysis/provenance.md`.

**Paper.** From `paper/`:
`./prepare_tables.sh && pdflatex main && bibtex main && pdflatex main && pdflatex main`

## Citation

Please cite the original paper alongside this study:

```bibtex
@inproceedings{todd2024function,
  title={Function Vectors in Large Language Models},
  author={Todd, Eric and Li, Millicent L. and Sen Sharma, Arnab and Mueller, Aaron and Wallace, Byron C. and Bau, David},
  booktitle={Proceedings of the Twelfth International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

Citation entry for this study will be added upon publication in the
BlackboxNLP 2026 proceedings.

## License

MIT (see `LICENSE`). The vendored `function_vectors/` directory retains its
own MIT license (© 2023 Eric Todd).
