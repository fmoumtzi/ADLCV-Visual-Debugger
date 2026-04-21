#!/usr/bin/env bash

set -euo pipefail

# Reduce peak RAM during HF model materialization (especially on CPU-only runs).
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-false}"
export HF_PARALLEL_LOADING_WORKERS="${HF_PARALLEL_LOADING_WORKERS:-1}"

NUM_SAMPLES="${NUM_SAMPLES:-50}"
SPLIT="${SPLIT:-val}"
ZERO_SHOT_SPLIT="${ZERO_SHOT_SPLIT:-${SPLIT}}"
MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-VL-3B-Instruct}"
TASK1_OUTPUT="${TASK1_OUTPUT:-results/task1_vqa/generations.jsonl}"
TASK2_DATA="${TASK2_DATA:-results/task2/task2_verification_data.jsonl}"
ZERO_SHOT_OUTPUT="${ZERO_SHOT_OUTPUT:-results/task2/zero_shot_predictions.jsonl}"
ZERO_SHOT_METRICS="${ZERO_SHOT_METRICS:-results/task2/metrics_zero_shot.json}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-results/task2/sft}"
SFT_PREDS="${SFT_PREDS:-results/task2/sft_predictions.jsonl}"
SFT_METRICS="${SFT_METRICS:-results/task2/metrics_sft.json}"
GRPO_OUTPUT_DIR="${GRPO_OUTPUT_DIR:-results/task2/grpo}"
GRPO_PREDS="${GRPO_PREDS:-results/task2/grpo_predictions.jsonl}"
GRPO_METRICS="${GRPO_METRICS:-results/task2/metrics_grpo.json}"

echo "Running simplified Task 2 pipeline"
echo "split=${SPLIT} zero_shot_split=${ZERO_SHOT_SPLIT} num_samples=${NUM_SAMPLES}"

python qwen_vl_vqa.py \
  --split "${SPLIT}" \
  --num_samples "${NUM_SAMPLES}" \
  --model_path "${MODEL_PATH}" \
  --output "${TASK1_OUTPUT}"

python src/task2/prepare_verification_data.py \
  --input_jsonl "${TASK1_OUTPUT}" \
  --output_jsonl "${TASK2_DATA}" \
  --auto_label_mode target_overlap

python src/task2/verify_zero_shot.py \
  --input_jsonl "${TASK2_DATA}" \
  --output_jsonl "${ZERO_SHOT_OUTPUT}" \
  --metrics_output_json "${ZERO_SHOT_METRICS}" \
  --split "${ZERO_SHOT_SPLIT}" \
  --slice_by_question_type \
  --model_path "${MODEL_PATH}"

python src/task2/train_sft_verifier.py \
  --train_jsonl "${TASK2_DATA}" \
  --output_dir "${SFT_OUTPUT_DIR}" \
  --predictions_output_jsonl "${SFT_PREDS}" \
  --metrics_output_json "${SFT_METRICS}" \
  --model_path "${MODEL_PATH}" \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --slice_by_question_type

python src/task2/train_grpo_verifier.py \
  --train_jsonl "${TASK2_DATA}" \
  --model_path "${SFT_OUTPUT_DIR}/best_checkpoint" \
  --output_dir "${GRPO_OUTPUT_DIR}" \
  --predictions_output_jsonl "${GRPO_PREDS}" \
  --metrics_output_json "${GRPO_METRICS}" \
  --num_candidates 4 \
  --eval_every_steps 50 \
  --slice_by_question_type

echo "Done."
echo "Zero-shot metrics: ${ZERO_SHOT_METRICS}"
echo "SFT metrics: ${SFT_METRICS}"
echo "GRPO metrics: ${GRPO_METRICS}"
