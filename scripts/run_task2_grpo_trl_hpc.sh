#!/usr/bin/env bash
#BSUB -J trl_grpo_vlm
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 24:00
#BSUB -u s253772@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o trl_grpo_vlm_%J.out
#BSUB -e trl_grpo_vlm_%J.err

set -euo pipefail

cd /dtu/blackhole/1a/223866/ADLCV-Visual-Debugger || exit 1
source /dtu/blackhole/1a/223866/miniforge3/bin/activate
conda activate adlcv || exit 1
module load cuda/12.8.1

export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_ENTITY="${WANDB_ENTITY:-fmoumtzi-technical-university-of-denmark}"
export WANDB_PROJECT="${WANDB_PROJECT:-ADLCV}"

MODEL="${MODEL:-qwen2}"
if [ -z "${MODEL_PATH:-}" ]; then
    if [ "${MODEL}" = "smolvlm" ]; then
        MODEL_PATH="models/SmolVLM-Instruct"
    else
        MODEL_PATH="models/Qwen2-VL-2B-Instruct"
    fi
fi

TASK2_DATA="${TASK2_DATA:-results/task2/task2_verification_data.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/task2/trl_grpo_${MODEL}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-trl-grpo-${MODEL}-${LSB_JOBID:-local}}"

if [ "${WANDB_MODE:-online}" != "offline" ]; then
    : "${WANDB_API_KEY:?Set WANDB_API_KEY or run with WANDB_MODE=offline.}"
    wandb login --relogin "${WANDB_API_KEY}"
fi

python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('bf16:', torch.cuda.is_available() and torch.cuda.is_bf16_supported())"
nvidia-smi || true

accelerate launch --num_processes 1 src/task2/train_grpo_trl_verifier.py \
  --train_jsonl "${TASK2_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --model_path "${MODEL_PATH}" \
  --num_train_epochs "${EPOCHS:-1}" \
  --max_steps "${MAX_STEPS:--1}" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-2}" \
  --num_generations "${NUM_GENERATIONS:-4}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH:-128}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --top_p "${TOP_P:-0.95}" \
  --beta "${BETA:-0.0}" \
  --loss_type "${LOSS_TYPE:-dapo}" \
  --logging_steps "${LOGGING_STEPS:-5}" \
  --save_steps "${SAVE_STEPS:-100}" \
  --eval_steps "${EVAL_STEPS:-100}" \
  --lora_r "${LORA_R:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --lora_target_modules "${LORA_TARGET_MODULES:-q_proj,v_proj}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-auto}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --use_4bit
