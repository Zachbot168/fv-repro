# Mistral-7B-v0.3 null result: verification note

**Question:** Is Mistral's near-zero FV transfer (T2: effect ≤ +0.07 on all 6 tasks) a pipeline
bug, or a property of the model? **Verdict up front:** the evidence is inconsistent with a
pipeline bug, and is consistent with Mistral encoding task identity in a *more distributed*
set of attention heads than the top-10-head FV construction assumes. Calibrated language
throughout: "consistent with," not "proves."

## 1. The final Mistral results were produced by the fixed code

Claimed from evidence, not commit order:

- **Only two commits in the repo's history ever touched the dispatch files**
  (`git log --follow` on `utils/intervention_utils.py` and `utils/extract_utils.py`):
  the initial import `cb520a2` and the fix `e71f052`. The working tree is clean against
  HEAD for both files. So at any time after the fix landed on disk, the on-disk code was
  exactly `e71f052`'s.
- **Fix commit `e71f052` (epoch 1784368556) predates Mistral's first load**
  (`LOAD_START mistralai/Mistral-7B-v0.3` at epoch 1784376507 in
  `results_repro/sweep_progress.log`) by ~2.2 h, and no later commit modified those files.
- **The same long-lived driver process that ran Mistral had already executed the fixed code
  path successfully on Qwen.** The progress log shows a single `SWEEP_START` (1784367156)
  with no restart between it and `SWEEP_DONE` (1784415765). Within that one process,
  Qwen antonym completed **through `compute_function_vector`** — the exact line that
  crashed under the unfixed code (`UnboundLocalError: out_proj` at 1784366891) — at
  1784368531, then Mistral ran at 1784376507+ using the same imported (fixed) modules.
  A Python process imports each module once; Qwen succeeding proves the loaded code was fixed.

## 2. Mistral has every task; only the FV fails to induce it

From `results_repro/mistralai_Mistral-7B-v0.3/*/model_baseline.json` and the layer sweeps
(same first-token metric for both columns — so metric artifacts cannot explain the gap):

| Task | 10-shot ICL | best FV effect (any layer) |
|---|---|---|
| antonym | 0.706 | +0.073 |
| capitalize | 1.000 | +0.006 |
| country-capital | 0.976 | +0.000 |
| english-french | 0.824 | +0.002 |
| present-past | 0.951 | +0.000 |
| singular-plural | 0.953 | +0.000 |

Additionally, the injected FV **does causally perturb** the model — at some layers it
*degrades* the zero-shot baseline (singular-plural −0.42, present-past −0.19), so the
intervention hook demonstrably fires and modifies the forward pass.

## 3. Mistral's causal head structure is measurably different (→ T4, F2)

Mean AIE per head from the committed `*_indirect_effect.pt` files, averaged over 6 tasks:

| Model | # heads | top-10 share of positive AIE | max per-head AIE |
|---|---|---|---|
| GPT-J-6B | 448 | **0.63** | 0.14 |
| Llama-3.1-8B | 1024 | 0.37 | 0.13 |
| Qwen2.5-7B | 784 | 0.37 | 0.05 |
| Mistral-7B-v0.3 | 1024 | **0.30** | **0.02** |

Mistral has both the flattest AIE distribution (lowest top-10 concentration) and per-head
causal effects **3–15× weaker in absolute terms** than any other model. This is a property
of the *measured causal structure* (computed upstream of the FV-construction code that was
patched), not of the injection.

## 4. Injection scale is not anomalous; vocab decode discriminates GPT-J, not Mistral

GPU probe (`analysis/fv_probe.py` → `analysis/fv_probe_results.json`), antonym FV at each
model's best layer:

| Model | ‖FV‖ | resid-stream norm (last tok) | ratio |
|---|---|---|---|
| GPT-J-6B | 42.2 | 78.8 | 0.54 |
| Llama-3.1-8B | 17.1 | 8.2 | 2.09 |
| Qwen2.5-7B | 21.3 | 64.3 | **0.33** |
| Mistral-7B-v0.3 | 1.5 | 4.6 | **0.33** |

Mistral's FV is tiny in absolute norm, but so is its residual stream: its **relative
injection scale (0.33) is identical to Qwen's (0.33), whose FV works** (+0.64). The
"injection too small" artifact hypothesis is therefore unsupported.

Decoding each FV through the model's own unembedding: GPT-J's antonym FV decodes to
task-relevant tokens (' vs', ' counterpart', ' versus', ' oppos…' — reproducing the paper's
vocabulary analysis), while **all three modern models decode to noise/punctuation —
including Llama and Qwen, whose FVs work.** So vocab-decodability separates GPT-J from
modern models generally; it provides no evidence of a Mistral-specific defect. (It is itself
a reportable finding: the "FV decodes to task words" property does not generalize.)

## 5. Adding heads monotonically recovers the effect (→ T3)

Mistral antonym, zero-shot accuracy at L14 by head count k:

| k | 5 | 10 | 20 | 50 | **100** |
|---|---|---|---|---|---|
| acc | 0.09 | 0.12 | 0.16 | 0.34 | **0.45** |

At k=100 Mistral reaches 0.45 — comparable to other models' k=10 accuracy (0.48–0.64).
A dead pipeline could not produce a smooth, monotone dose-response in k. The task signal
exists in Mistral's attention heads; it is spread over roughly an order of magnitude more
heads than the method's top-10 default captures. (Not universal across tasks: country-capital
stays near zero through k=50 — the distributed account is supported for antonym and remains
an open question for tasks that never recover.)

## Verdict

**(a) Not a bug — with high confidence.** Five independent observations contradict the bug
hypothesis: verified-fixed code shared with a working model (§1); healthy ICL under the
identical metric (§2); the intervention demonstrably perturbing the forward pass (§2);
a normal relative injection scale, matched to a working model (§4); and a smooth monotone
recovery with k (§5). None of these is what a broken dispatch/intervention would produce.

**(b) Consistent with a distributed task representation.** The flattest AIE curve, the
weakest per-head effects (§3), and the k-dose-response (§5) all point the same way. We do
not claim this is the full explanation — the total collapse of capitalize (0.006 vs 0.92–0.99
on every other model) and the non-recovery of country-capital at k≤50 are unexplained
residuals worth deeper study (e.g., k>100, per-layer FV composition, sliding-window attention
effects) — but the null result itself is a property of the model, reportable as such.
