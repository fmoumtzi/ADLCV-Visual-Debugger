#!/usr/bin/env bash
#BSUB -J verify_qwen_sft
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 04:00
#BSUB -o logs/verify_qwen_sft_%J.out
#BSUB -e logs/verify_qwen_sft_%J.err

set -euo pipefail

# Activate your environment
source /dtu/blackhole/0d/223821/miniforge3/bin/activate
conda activate adlcv
module load cuda/12.8.1 || true

# Run the Unified Evaluation Script
CUDA_VISIBLE_DEVICES=0 python src/task2/verify_any_model.py \
  --input_jsonl results/task2/task2_verification_data.jsonl \
  --output_jsonl results/task2/qwen_sft_preds.jsonl \
  --metrics_output_json results/task2/qwen_sft_metrics.json \
  --model_path models/Qwen2.5-VL-3B-Instruct \
  --lora_path results/task2/qwen_sft_lora \
  --split test \
  --slice_by_question_type