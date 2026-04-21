# Task 2 Run Guide

This guide shows how to run the full Task 2 pipeline in this repo.

## 1) Prerequisites

From repo root:

```bash
pip install -r requirements.txt
```

You need:
- VQA/COCO data under `data/` (same layout used by your existing scripts)
- Qwen model available locally at `models/Qwen2.5-VL-3B-Instruct` (or pass another `--model_path`)
- Task 1 generations JSONL (for example `results/<task1_run>/generations.jsonl`)

## 2) Build Task 2 Verification Data

### Option A: with manual claim labels (recommended)

```bash
python src/task2/prepare_verification_data.py \
  --input_jsonl results/<task1_run>/generations.jsonl \
  --label_jsonl results/task2/manual_labels.jsonl \
  --output_jsonl results/task2/task2_verification_data.jsonl \
  --drop_unlabeled
```

### Option B: weak bootstrap labels (quick start)

```bash
python src/task2/prepare_verification_data.py \
  --input_jsonl results/<task1_run>/generations.jsonl \
  --output_jsonl results/task2/task2_verification_data.jsonl \
  --auto_label_mode target_overlap
```

This writes rows with:
- `claims` (sentence-level splits)
- `labels` (`CORRECT` / `HALLUCINATED`)
- deterministic split (`train` / `val` / `test`)

## 3) Zero-Shot Baseline

```bash
python src/task2/verify_zero_shot.py \
  --input_jsonl results/task2/task2_verification_data.jsonl \
  --output_jsonl results/task2/zero_shot_predictions.jsonl \
  --metrics_output_json results/task2/metrics_zero_shot.json \
  --split test \
  --slice_by_question_type
```

## 4) Train SFT Verifier

```bash
python src/task2/train_sft_verifier.py \
  --train_jsonl results/task2/task2_verification_data.jsonl \
  --output_dir results/task2/sft \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --slice_by_question_type
```

Main SFT artifacts:
- `results/task2/sft/best_checkpoint/`
- `results/task2/sft/final_checkpoint/`
- `results/task2/sft_predictions.jsonl`
- `results/task2/metrics_sft.json`

## 5) Train GRPO (from SFT checkpoint)

```bash
python src/task2/train_grpo_verifier.py \
  --train_jsonl results/task2/task2_verification_data.jsonl \
  --model_path results/task2/sft/best_checkpoint \
  --output_dir results/task2/grpo \
  --num_candidates 4 \
  --eval_every_steps 50 \
  --slice_by_question_type
```

Main GRPO artifacts:
- `results/task2/grpo/best_checkpoint/`
- `results/task2/grpo/final_checkpoint/`
- `results/task2/grpo_predictions.jsonl`
- `results/task2/metrics_grpo.json`

## 6) Evaluate Any Prediction File

```bash
python src/task2/evaluate_verifier.py \
  --gold_jsonl results/task2/task2_verification_data.jsonl \
  --pred_jsonl results/task2/zero_shot_predictions.jsonl \
  --output_json results/task2/metrics_zero_shot.json \
  --slice_by_question_type
```

Swap `--pred_jsonl` to compare methods:
- `results/task2/zero_shot_predictions.jsonl`
- `results/task2/sft_predictions.jsonl`
- `results/task2/grpo_predictions.jsonl`

## 7) Optional: Use Config Values

Reference defaults:
- `configs/task2_sft.yaml`
- `configs/task2_grpo.yaml`

These files are templates. Apply values as CLI flags in your run commands.

## 8) Quick End-to-End Minimal Run

```bash
python src/task2/prepare_verification_data.py \
  --input_jsonl results/<task1_run>/generations.jsonl \
  --output_jsonl results/task2/task2_verification_data.jsonl \
  --auto_label_mode target_overlap

python src/task2/verify_zero_shot.py \
  --input_jsonl results/task2/task2_verification_data.jsonl \
  --split test

python src/task2/train_sft_verifier.py \
  --train_jsonl results/task2/task2_verification_data.jsonl

python src/task2/train_grpo_verifier.py \
  --train_jsonl results/task2/task2_verification_data.jsonl \
  --model_path results/task2/sft/best_checkpoint
```

## 9) Common Issues

- `ModuleNotFoundError: transformers`
  - Install deps: `pip install -r requirements.txt`
- Missing image/file errors
  - Verify `data/` and model paths
- No training rows found
  - Ensure rows have `labels` and your split filter matches (`train` by default)
