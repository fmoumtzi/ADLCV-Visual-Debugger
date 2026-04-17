# Task 2: RL-Trained Self-Verification (GRPO) Implementation Plan

## Objective
Build a **verification model behavior** for `Qwen2.5-VL-3B-Instruct` that, given:
- an image, and
- the model's own prior response (from Task 1 generation),

outputs sentence-level labels:
- `CORRECT`
- `HALLUCINATED`

Then compare:
- `Zero-shot` verification prompting
- `SFT` on verification labels
- `SFT + GRPO`

with precision/recall/F1 for hallucination detection, with VQA-specific slices by:
- `answer_type` (`yes/no`, `number`, `other`)
- optional `question_type`

## What Needs To Be Done
1. Standardize a Task 2 data format that links:
- image path / image id
- VQA question (and choices when available)
- original model response
- claim-level splits of that response
- claim-level ground-truth labels (`CORRECT`/`HALLUCINATED`)
- VQA metadata (`answer_type`, `question_type`)

2. Build a **verification inference pipeline** (no training) for the zero-shot baseline:
- prompt model with `(image + prior response)`
- require structured output: one line per claim with label
- parse outputs robustly and align with claim ids

3. Build an **evaluation module**:
- precision/recall/F1 for `HALLUCINATED` detection overall
- precision/recall/F1 by VQA `answer_type`
- optional precision/recall/F1 by VQA `question_type`
- track false alarms on true claims and misses on hallucinated claims

4. Build SFT training data + trainer:
- input: `(image + prior response + claim list)`
- target: labels for each claim
- train supervised verifier checkpoint

5. Add GRPO fine-tuning stage on top of SFT checkpoint:
- sample `K=4` candidate verification outputs per example
- reward +1 for each correct claim judgment
- reward -1 for each incorrect claim judgment
- compute relative rewards among candidates (GRPO style)
- update policy to prefer higher-reward candidates

6. Add experiment tracking and reproducibility using w&b:
- config files for model/training/eval
- saved checkpoints
- per-step or per-epoch eval curves (overall and by VQA slice)
- single summary table comparing zero-shot vs SFT vs SFT+GRPO

## Recommended File Additions
- `src/task2/prepare_verification_data.py`
- `src/task2/verify_zero_shot.py`
- `src/task2/train_sft_verifier.py`
- `src/task2/train_grpo_verifier.py`
- `src/task2/evaluate_verifier.py`
- `src/task2/prompts.py`
- `src/task2/reward.py`
- `configs/task2_sft.yaml`
- `configs/task2_grpo.yaml`
- `results/task2/...` (metrics, predictions, checkpoints metadata)

## Data Contract (JSONL)
Use one row per image-response pair:

```json
{
  "image_id": 391895,
  "image_path": "data/coco/val2014/COCO_val2014_000000391895.jpg",
  "source_dataset": "vqa",
  "question": "Is the man wearing a helmet?",
  "choices": ["yes", "no"],
  "target_answer": "no",
  "answer_type": "yes/no",
  "question_type": "is this",
  "model_response": "Yes, he is wearing a helmet.",
  "claims": [
    {"claim_id": 0, "text": "He is wearing a helmet."}
  ],
  "labels": [
    {"claim_id": 0, "verdict": "HALLUCINATED"}
  ]
}
```

## Baseline to Final Training Flow
1. **Zero-shot verifier**
- run on full Task 2 eval split
- save predictions and metrics

2. **SFT verifier**
- train on claim-labeled training split
- evaluate on same held-out eval split

3. **SFT + GRPO verifier**
- initialize from SFT checkpoint
- run GRPO with `K=4` candidates per sample
- evaluate at checkpoints; keep best overall F1 model

## GRPO Reward Definition (Task-Specific)
For each claim:
- `+1` if predicted verdict matches ground truth
- `-1` otherwise

Total candidate reward:
- sum over all claims in the sample
- optional normalization by number of claims for stability

GRPO update uses relative candidate rewards inside each `K=4` group.

## Evaluation Protocol
Compute and log:
- overall precision/recall/F1 on `HALLUCINATED` class
- precision/recall/F1 by VQA `answer_type`
- optional precision/recall/F1 by VQA `question_type`
- confusion matrix on claim verdicts
- calibration-like stats: fraction of claims predicted hallucinated

Track over time:
- for SFT: per epoch
- for GRPO: every N update steps

## Output Artifacts
- `results/task2/zero_shot_predictions.jsonl`
- `results/task2/sft_predictions.jsonl`
- `results/task2/grpo_predictions.jsonl`
- `results/task2/metrics_zero_shot.json`
- `results/task2/metrics_sft.json`
- `results/task2/metrics_grpo.json`
- `results/task2/curves/*.csv` or `*.json` for plotting training trajectories

## Concrete Execution Plan
1. Finalize Task 2 labeled data schema and train/val/test split.
2. Implement claim parser + output parser with strict formatting and fallback recovery.
3. Implement zero-shot verifier script and baseline metrics.
4. Implement SFT dataset builder + SFT trainer.
5. Implement GRPO trainer with `K=4` candidate sampling and relative-reward updates.
6. Add unified evaluator for overall and VQA-sliced metrics plus checkpoint selection.
7. Run full comparison experiments and export final report table/plots.

## Acceptance Criteria
- All 3 methods run end-to-end on same eval split.
- Metrics reported overall and by VQA `answer_type` (`yes/no`, `number`, `other`).
- GRPO training logs show metric trajectory over time.
- Final markdown/table clearly states whether SFT+GRPO improves over zero-shot and SFT.
