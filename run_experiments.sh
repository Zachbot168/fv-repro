#!/usr/bin/env bash
# Per-task-CLI sweep for the FV generalizability study.
#
# This is the FALLBACK path. The primary driver is function_vectors/src/run_sweep.py,
# which loads each model once (orchestration-only). Use this script only if the driver's
# mean-activation equivalence check fails; it re-loads the model per task (eats reload cost)
# but is otherwise identical in computation.
#
# Run from the repo root:  bash run_experiments.sh   (it cd's into function_vectors/src itself)
# Safe to re-run: completed (model,task) pairs are skipped; mean-acts/IE are cached per stage.
#
# Canonical layout:
#   results_repro/<model_tag>/<task>/          main run (all-layer sweep, top-10 heads)
#   results_repro/<model_tag>/k<K>/<task>/     k-ablation at the main run's best layer
set -euo pipefail
cd "$(dirname "$0")/function_vectors/src"

MODELS=(
  "meta-llama/Llama-3.1-8B"
  "Qwen/Qwen2.5-7B"
  "mistralai/Mistral-7B-v0.3"
  "EleutherAI/gpt-j-6b"
)
TASKS=(antonym capitalize country-capital english-french present-past singular-plural)
RESULTS_ROOT="../../results_repro"
SEED=42

best_layer() {  # $1 = path to zs_results_layer_sweep.json ; echoes best injection layer
  python - "$1" <<'PY'
import json, sys
zs = json.load(open(sys.argv[1]))
bL, bA = 0, -1.0
for L, e in zs.items():
    a = dict(e.get('intervention_topk') or []).get(1, 0.0)
    if a > bA:
        bA, bL = a, int(L)
print(bL)
PY
}

for MODEL in "${MODELS[@]}"; do
  TAG=$(echo "$MODEL" | tr '/' '_')
  MROOT="$RESULTS_ROOT/$TAG"

  # ---- main: 6 tasks, sweep all layers, top-10 heads ----
  for TASK in "${TASKS[@]}"; do
    if [ -f "$MROOT/$TASK/model_baseline.json" ]; then
      echo "SKIP $TAG/$TASK main (cached)"; continue
    fi
    echo "=== $MODEL / $TASK (main) ==="
    python evaluate_function_vector.py \
      --dataset_name "$TASK" --model_name "$MODEL" \
      --root_data_dir ../dataset_files \
      --save_path_root "$MROOT" --ie_path_root "$MROOT" \
      --seed $SEED --n_top_heads 10 --edit_layer -1
  done

  # ---- k-ablation at the fixed best layer (single layer, NOT a full sweep) ----
  for TASK in antonym country-capital; do
    ZS="$MROOT/$TASK/zs_results_layer_sweep.json"
    if [ ! -f "$ZS" ]; then echo "SKIP_KABL $TAG/$TASK (no main zs results)"; continue; fi
    BL=$(best_layer "$ZS")
    for K in 5 20 50; do
      KROOT="$MROOT/k$K"
      if [ -f "$KROOT/$TASK/zs_results_editlayer_${BL}.json" ]; then
        echo "SKIP $TAG/$TASK k$K (cached)"; continue
      fi
      mkdir -p "$KROOT/$TASK"
      # reuse the main run's filter set so we don't re-run the 10-shot filter eval
      if [ -f "$MROOT/$TASK/fs_results_layer_sweep.json" ]; then
        cp -n "$MROOT/$TASK/fs_results_layer_sweep.json" "$KROOT/$TASK/fs_results_layer_sweep.json" || true
      fi
      echo "=== $MODEL / $TASK  k=$K @ layer $BL ==="
      # --compute_baseline '' -> bool('') == False, skips the redundant baseline recompute
      python evaluate_function_vector.py \
        --dataset_name "$TASK" --model_name "$MODEL" \
        --root_data_dir ../dataset_files \
        --save_path_root "$KROOT" --ie_path_root "$MROOT" \
        --seed $SEED --n_top_heads $K --edit_layer "$BL" --compute_baseline ''
    done
  done
done
echo "ALL DONE — results in $RESULTS_ROOT"
