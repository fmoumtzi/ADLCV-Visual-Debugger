# ADLCV-Visual-Debugger

## Task 2: RL Self-Verification Pipeline

This repo now includes a full Task 2 pipeline under `src/task2` for claim-level hallucination verification using `Qwen2.5-VL-3B-Instruct`.

### Added scripts

- `src/task2/prepare_verification_data.py`
- `src/task2/verify_zero_shot.py`
- `src/task2/evaluate_verifier.py`
- `src/task2/train_sft_verifier.py`
- `src/task2/train_grpo_verifier.py`
- `src/task2/prompts.py`
- `src/task2/reward.py`
- `configs/task2_sft.yaml`
- `configs/task2_grpo.yaml`

### 1) Prepare Task 2 data contract

```bash
python src/task2/prepare_verification_data.py \
  --input_jsonl results/<task1_run>/generations.jsonl \
  --output_jsonl results/task2/task2_verification_data.jsonl \
  --auto_label_mode target_overlap
```

Notes:
- For real experiments, pass manual labels via `--label_jsonl`.
- Use `--drop_unlabeled` to enforce only claim-labeled rows.

### 2) Zero-shot verifier baseline

```bash
python src/task2/verify_zero_shot.py \
  --input_jsonl results/task2/task2_verification_data.jsonl \
  --output_jsonl results/task2/zero_shot_predictions.jsonl \
  --metrics_output_json results/task2/metrics_zero_shot.json \
  --split test \
  --slice_by_question_type
```

### 3) SFT verifier training

```bash
python src/task2/train_sft_verifier.py \
  --train_jsonl results/task2/task2_verification_data.jsonl \
  --output_dir results/task2/sft \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 1
```

### 4) GRPO fine-tuning from SFT checkpoint

```bash
python src/task2/train_grpo_verifier.py \
  --train_jsonl results/task2/task2_verification_data.jsonl \
  --model_path results/task2/sft/best_checkpoint \
  --output_dir results/task2/grpo \
  --num_candidates 4 \
  --eval_every_steps 50
```

### 5) Evaluate predictions

```bash
python src/task2/evaluate_verifier.py \
  --gold_jsonl results/task2/task2_verification_data.jsonl \
  --pred_jsonl results/task2/zero_shot_predictions.jsonl \
  --output_json results/task2/metrics_zero_shot.json \
  --slice_by_question_type
```
