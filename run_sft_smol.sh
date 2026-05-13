#!/usr/bin/env bash
#BSUB -J smol_sft
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 12:00
#BSUB -o logs/smol_sft_%J.out
#BSUB -e logs/smol_sft_%J.err

set -euo pipefail

# Activate your environment
source /dtu/blackhole/0d/223821/miniforge3/bin/activate
conda activate adlcv
module load cuda/12.8.1 || true

# Launch SFT Training
python src/task2/train_sft_verifier.py \
  --train_jsonl results/task2/smolvlm_5k_verification_data.jsonl \
  --output_dir results/task2/sft_smolvlm_lora \
  --model_path models/SmolVLM-Instruct \
  --use_lora \
  --lora_r 8 \
  --epochs 1 \
  --batch_size 4 \
  --grad_accum_steps 2 \
  --slice_by_question_type