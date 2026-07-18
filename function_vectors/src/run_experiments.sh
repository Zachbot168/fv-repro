#!/usr/bin/env bash
# Full experiment sweep for the FV generalizability study.
# Run from function_vectors/src/ on a GPU box. Safe to re-run: results are cached per stage.
set -euo pipefail

MODELS=(
  "EleutherAI/gpt-j-6b"
  "meta-llama/Llama-3.1-8B"
  "Qwen/Qwen2.5-7B"
  "mistralai/Mistral-7B-v0.3"
)
TASKS=(antonym capitalize country-capital english-french present-past singular-plural)
RESULTS_ROOT="../results_repro"
SEED=42

for MODEL in "${MODELS[@]}"; do
  MODEL_TAG=$(echo "$MODEL" | tr '/' '_')
  for TASK in "${TASKS[@]}"; do
    OUT="$RESULTS_ROOT/$MODEL_TAG/$TASK"
    mkdir -p "$OUT"
    echo "=== $MODEL / $TASK ==="

    # Stage 1+2: indirect effect (computes+caches mean activations as a side effect)
    if [ ! -f "$OUT/${TASK}_indirect_effect.pt" ]; then
      python compute_indirect_effect.py \
        --dataset_name "$TASK" \
        --model_name "$MODEL" \
        --root_data_dir ../dataset_files \
        --save_path_root "$OUT" \
        --seed $SEED --n_shots 10 --n_trials 25
    fi

    # Stage 3: FV eval, sweep all layers, top-10 heads (paper default for 6-7B scale)
    python evaluate_function_vector.py \
      --dataset_name "$TASK" \
      --model_name "$MODEL" \
      --root_data_dir ../dataset_files \
      --save_path_root "$OUT" \
      --ie_path_root "$OUT" \
      --seed $SEED --n_top_heads 10 --edit_layer -1
  done

  # Head-count ablation on two tasks only (budget control)
  for TASK in antonym country-capital; do
    OUT="$RESULTS_ROOT/$MODEL_TAG/$TASK"
    for K in 5 20 50; do
      python evaluate_function_vector.py \
        --dataset_name "$TASK" --model_name "$MODEL" \
        --root_data_dir ../dataset_files \
        --save_path_root "$OUT/k$K" --ie_path_root "$OUT" \
        --seed $SEED --n_top_heads $K --edit_layer -1
    done
  done
done
echo "ALL DONE — results in $RESULTS_ROOT"
