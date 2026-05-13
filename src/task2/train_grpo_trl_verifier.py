import argparse
import os
import re
import sys
from typing import Dict, Optional

import torch

try:
    from .io_utils import parse_label_entries, read_jsonl
    from .prompts import build_verification_prompt, normalize_hallucination_label
except ImportError:
    from io_utils import parse_label_entries, read_jsonl
    from prompts import build_verification_prompt, normalize_hallucination_label


MODEL_DEFAULTS = {
    "qwen2": ("models/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-2B-Instruct"),
    "smolvlm": ("models/SmolVLM-Instruct", "HuggingFaceTB/SmolVLM-Instruct"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="TRL GRPO training for Task 2 verifier.")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", choices=sorted(MODEL_DEFAULTS), default="qwen2")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--min_claims", type=int, default=1)

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--loss_type", default="dapo")
    parser.add_argument("--format_reward_weight", type=float, default=0.2)
    parser.add_argument("--format_instruction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj")
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--attn_implementation", default="auto")

    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--log_completions", action="store_true")
    parser.add_argument("--allow_vllm_import", action="store_true")
    return parser.parse_args()


def resolve_model_path(model_key: str, model_path: str) -> str:
    if model_path:
        return model_path
    local_path, hf_id = MODEL_DEFAULTS[model_key]
    return local_path if os.path.isdir(local_path) else hf_id


def resolve_attn_implementation(value: str) -> Optional[str]:
    if not value or value == "none":
        return None
    if value != "auto":
        return value
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def disable_vllm_import() -> None:
    sys.modules.setdefault("vllm", None)


def add_format_example(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "Example output:\n"
        "0\tFALSE\n"
        "1\tTRUE\n\n"
        "Now output only the verification lines for the actual claims above."
    )


def build_dataset(jsonl_path: str, split: str, min_claims: int, use_format_instruction: bool):
    from datasets import Dataset, Image

    rows = []
    for row in read_jsonl(jsonl_path):
        if split and row.get("split") != split:
            continue
        gold = parse_label_entries(row.get("labels", []))
        claims = row.get("claims", [])
        claim_ids = [int(claim["claim_id"]) for claim in claims if int(claim["claim_id"]) in gold]
        if len(claim_ids) < min_claims:
            continue

        prompt = build_verification_prompt(
            question=row.get("question", ""),
            prior_response=row.get("model_response", ""),
            claims=claims,
            choices=row.get("choices", []),
            target_answer=row.get("target_answer", ""),
        )
        if use_format_instruction:
            prompt = add_format_example(prompt)
        rows.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "image": row["image_path"],
                "claim_ids": claim_ids,
                "gold_hallucinations": [bool(gold[claim_id]) for claim_id in claim_ids],
            }
        )

    if not rows:
        return None
    return Dataset.from_list(rows).cast_column("image", Image(decode=True))


def completion_to_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        content = completion[0].get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(part.get("text", "")) for part in content if part.get("type") == "text")
    return str(completion)


def parse_completion(text: str) -> Dict[int, bool]:
    predictions = {}
    for line in str(text).splitlines():
        match = re.match(r"^\s*(\d+)\s*[:\-|,\t ]+\s*([A-Za-z_ ]+)\s*$", line.strip())
        if not match:
            continue
        value = normalize_hallucination_label(match.group(2))
        if value is not None:
            predictions[int(match.group(1))] = value
    return predictions


def claim_reward(completions, claim_ids, gold_hallucinations, **kwargs):
    rewards = []
    for completion, ids, golds in zip(completions, claim_ids, gold_hallucinations):
        pred = parse_completion(completion_to_text(completion))
        scores = [
            1.0 if claim_id in pred and bool(pred[claim_id]) == bool(gold) else -1.0
            for claim_id, gold in zip(ids, golds)
        ]
        rewards.append(sum(scores) / max(len(scores), 1))
    return rewards


def combined_reward(completions, claim_ids, gold_hallucinations, format_reward_weight=0.2, **kwargs):
    rewards = []
    for completion, ids, golds in zip(completions, claim_ids, gold_hallucinations):
        pred = parse_completion(completion_to_text(completion))
        matched = sum(1 for claim_id in ids if claim_id in pred)
        format_score = matched / max(len(ids), 1)
        claim_scores = [
            1.0 if claim_id in pred and bool(pred[claim_id]) == bool(gold) else -1.0
            for claim_id, gold in zip(ids, golds)
        ]
        claim_score = sum(claim_scores) / max(len(claim_scores), 1)
        rewards.append(claim_score + format_reward_weight * format_score)
    return rewards


def main():
    args = parse_args()
    if not args.allow_vllm_import:
        disable_vllm_import()

    from peft import LoraConfig
    from transformers import AutoProcessor, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    model_path = resolve_model_path(args.model, args.model_path)
    train_dataset = build_dataset(
        args.train_jsonl,
        args.train_split,
        min_claims=args.min_claims,
        use_format_instruction=args.format_instruction,
    )
    eval_dataset = (
        build_dataset(
            args.train_jsonl,
            args.val_split,
            min_claims=1,
            use_format_instruction=args.format_instruction,
        )
        if args.eval_steps > 0
        else None
    )
    if train_dataset is None:
        raise RuntimeError(f"No labeled rows found for split={args.train_split!r}.")

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    model_init_kwargs = {"dtype": torch.bfloat16}
    attn = resolve_attn_implementation(args.attn_implementation)
    if attn:
        model_init_kwargs["attn_implementation"] = attn
    if args.use_4bit:
        model_init_kwargs["device_map"] = {"": 0}
        model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    report_to = "wandb" if args.wandb_project else "none"
    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        model_init_kwargs=model_init_kwargs,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        epsilon=args.epsilon,
        loss_type=args.loss_type,
        scale_rewards="group",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        use_cache=False,
        optim=args.optim,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        report_to=report_to,
        run_name=args.wandb_run_name or None,
        remove_unused_columns=False,
        log_completions=args.log_completions,
        seed=args.seed,
    )

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
    )
    def reward_func(completions, claim_ids, gold_hallucinations, **kwargs):
        return combined_reward(
            completions,
            claim_ids,
            gold_hallucinations,
            format_reward_weight=args.format_reward_weight,
            **kwargs,
        )

    trainer = GRPOTrainer(
        model=model_path,
        args=grpo_args,
        processing_class=processor,
        reward_funcs=reward_func,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    #trainer.train()
    trainer.train(resume_from_checkpoint="results/task2/grpo_smolvlm_sft_merged_lora_v2/checkpoint-300")
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
