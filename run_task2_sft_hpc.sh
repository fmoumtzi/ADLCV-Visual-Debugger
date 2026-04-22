#!/bin/sh
### General options
### -- specify queue --
#BSUB -q c02516
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### –- specify queue --
### -- set the job Name --
#BSUB -J train_gnn_traj
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need xGB of memory per core/slot --
#BSUB -R "rusage[mem=4GB]"
###BSUB -R "select[gpu80gb]"
###BSUB -R "select[gpu32gb]"
#### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot --
###BSUB -M 128GB
### -- set walltime limit: hh:mm --
#BSUB -W 04:00
### -- set the email address --
#BSUB -u s253819@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- set the job output file --
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o train_gnn_traj_%J.out
#BSUB -e train_gnn_traj_%J.err
# all  BSUB option comments should be above this line!

if [ ! -d "/dtu/blackhole/07/224071/GNNenv" ]; then
    python3 -m venv "/dtu/blackhole/07/224071/GNNenv"
fi

# Activate your venv in blackhole
source "/dtu/blackhole/07/224071/GNNenv/bin/activate"
module load cuda/12.8.1

# go to the repo root on blackhole
cd /dtu/blackhole/07/224071/ADLCV-Visual-Debugger

# Run training from repo root
export PYTHONPATH=src

TASK2_DATA="${TASK2_DATA:-results/task2/task2_verification_data.jsonl}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-results/task2/sft}"
SFT_PREDS="${SFT_PREDS:-results/task2/sft_predictions.jsonl}"
SFT_METRICS="${SFT_METRICS:-results/task2/metrics_sft.json}"
MODEL_PATH="${MODEL_PATH:-models/Qwen2-VL-2B-Instruct}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-sft-verifier-${LSB_JOBID:-local}}"

if [ -n "${WANDB_PROJECT}" ]; then
  if ! python -c "import wandb" >/dev/null 2>&1; then
    echo "wandb is not installed in the active environment. Install requirements or pip install wandb."
    exit 1
  fi
  if [ "${WANDB_MODE:-online}" != "offline" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WANDB_PROJECT is set but WANDB_API_KEY is empty. Set WANDB_API_KEY or run wandb login in this environment."
    exit 1
  fi
fi

python src/task2/train_sft_verifier.py \
  --train_jsonl "${TASK2_DATA}" \
  --output_dir "${SFT_OUTPUT_DIR}" \
  --predictions_output_jsonl "${SFT_PREDS}" \
  --metrics_output_json "${SFT_METRICS}" \
  --model_path "${MODEL_PATH}" \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --slice_by_question_type \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_RUN_NAME}"
