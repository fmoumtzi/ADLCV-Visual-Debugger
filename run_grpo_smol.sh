#!/usr/bin/env bash
#BSUB -J smol_grpo
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 24:00
#BSUB -o logs/smol_grpo_%J.out
#BSUB -e logs/smol_grpo_%J.err

set -euo pipefail

source /dtu/blackhole/0d/223821/miniforge3/bin/activate
conda activate adlcv
module load cuda/12.8.1 || true

python src/task2/train_grpo_trl_verifier.py \
  --train_jsonl results/task2/smolvlm_5k_verification_data.jsonl \
  --output_dir results/task2/grpo_smolvlm_lora \
  --model smolvlm \
  --model_path models/SmolVLM-Instruct \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_generations 4 \
  --max_completion_length 64 \
  --lora_r 8 \
  --num_train_epochs 1.0 \
  --resume_from_checkpoint results/task2/grpo_smolvlm_lora/checkpoint-2200