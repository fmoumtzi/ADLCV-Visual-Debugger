# SFT and GRPO for Task 2 (What They Do)

## SFT (Supervised Fine-Tuning)
SFT teaches the model to imitate labeled targets.

For this project, SFT means:
- Input: image + prior model response (and optionally claim list)
- Target: per-claim labels (`CORRECT` or `HALLUCINATED`)

What SFT does well:
- Quickly teaches output format and task behavior
- Uses direct ground truth supervision
- Usually gives a strong, stable baseline

SFT limitation:
- It learns from fixed labels only; it does not explicitly optimize a task reward over multiple sampled outputs.

## GRPO (Group Relative Policy Optimization)
GRPO is RL fine-tuning that compares multiple sampled outputs for the same input and pushes the model toward relatively better ones.

For this project:
- For each training sample, generate `K=4` candidate verification outputs
- Score each candidate with your reward function:
  - `+1` per correct claim judgment
  - `-1` per incorrect claim judgment
- Compute relative advantage inside the candidate group
- Update the model so higher-reward candidates become more likely

What GRPO does well:
- Optimizes directly for your task reward
- Can improve decision quality beyond imitation learning
- Helps when multiple valid output styles exist but some are more accurate

GRPO limitation:
- More complex and less stable than SFT if reward and parsing are noisy.

## Why Use SFT Before GRPO
SFT first gives the model:
- the correct verification format
- a reasonable initial classifier behavior

Then GRPO refines it by optimizing end-task reward.

In practice for this task:
- `Zero-shot`: no task-specific training
- `SFT`: learns labeled verification behavior
- `SFT + GRPO`: starts from SFT and improves using reward-based policy updates

## Practical Difference in One Line
- SFT: "learn to match labels."
- GRPO: "learn to maximize reward relative to alternative sampled outputs."

