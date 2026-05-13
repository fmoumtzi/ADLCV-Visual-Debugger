#!/usr/bin/env bash
#BSUB -J llm_judge_5k
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 12:00
#BSUB -o logs/llm_judge_5k_%J.out
#BSUB -e logs/llm_judge_5k_%J.err

set -euo pipefail

# Point directly to your blackhole miniforge installation
source /dtu/blackhole/0d/223821/miniforge3/bin/activate
conda activate adlcv

module load cuda/12.8.1 || true

# Run the LLM Judge evaluation
python src/llm_judge.py \
  --judge_model Patronus-Lynx-8B \
  --exp_name exp_005_smolvlm_vqa_5k \
  --eval_prompt_type taxonomy_json