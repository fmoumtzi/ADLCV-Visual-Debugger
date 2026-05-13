#!/bin/sh
### General options
#BSUB -J vlm_sft_lora
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=26GB]"
#BSUB -M 26GB
#BSUB -W 24:00
#BSUB -o logs/vlm_sft_lora_%J.out
#BSUB -e logs/vlm_sft_lora_%J.err
# all BSUB option comments should be above this line!

cd /dtu/blackhole/12/223890/ADLCV-Visual-Debugger || exit 1
mkdir -p logs

source /work3/s253753/miniforge3/bin/activate
conda activate adlcv || exit 1

export HF_HOME=/work3/s253753/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false

# Optional but often helpful on shared systems
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# W&B
export WANDB_API_KEY="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"
export WANDB_PROJECT="visual-debugger"
wandb login --relogin "$WANDB_API_KEY"

python -c "import torch; print('torch:', torch.__version__)"
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
python -c "import torch; print('device count:', torch.cuda.device_count())"
python -c "import torch; print('bf16 supported:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"
nvidia-smi || true

python src/task2/train_sft_verifier.py \
  --train_jsonl results/task2/alex_converted_data_qwen_balanced.jsonl \
  --output_dir results/task2/sft_lora_balanced_1 \
  --epochs 50 \
  --batch_size 1 \
  --grad_accum_steps 4 \
  --learning_rate 2e-4 \
  --model_path models/Qwen2-VL-2B-Instruct \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  --num_workers 2 \
  --use_lora \
  --save_each_epoch \
  --wandb_project visual-debugger \
  --wandb_run_name qwen2vl_2b_sft_lora_balanced1
