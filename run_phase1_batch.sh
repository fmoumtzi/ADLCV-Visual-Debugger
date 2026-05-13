#!/usr/bin/env bash
#BSUB -J smolvlm_5k_gen
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -M 24GB
#BSUB -W 12:00
#BSUB -o logs/phase1_gen_%J.out
#BSUB -e logs/phase1_gen_%J.err

set -euo pipefail

# Point directly to your blackhole miniforge installation
source /dtu/blackhole/0d/223821/miniforge3/bin/activate
conda activate adlcv

module load cuda/12.8.1 || true

# Run the generation script
python src/phase1_generation.py \
  --vlm_model SmolVLM-Instruct \
  --dataset vqa \
  --num_samples 5000 \
  --prompt_type structured \
  --exp_name exp_005_smolvlm_vqa_5k