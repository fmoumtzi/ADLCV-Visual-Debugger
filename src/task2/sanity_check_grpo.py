import argparse
import os
import sys
import torch

try:
    from .io_utils import read_jsonl, parse_label_entries
    from .prompts import build_verification_prompt
except ImportError:
    from io_utils import read_jsonl, parse_label_entries
    from prompts import build_verification_prompt


def disable_vllm_import():
    sys.modules.setdefault("vllm", None)


def build_dataset(jsonl_path, split, max_examples, explicit_image_token=True):
    from datasets import Dataset, Image

    rows = []
    for row in read_jsonl(jsonl_path):
        if split and row.get("split") != split:
            continue
        if not row.get("image_path") or not row.get("claims") or not row.get("labels"):
            continue

        gold = parse_label_entries(row["labels"])
        claim_ids = [
            int(claim["claim_id"])
            for claim in row["claims"]
            if int(claim["claim_id"]) in gold
        ]
        if not claim_ids:
            continue

        prompt = (
            "Describe this image in one short sentence. "
            "Mention the main objects and scene. Do not output TRUE or FALSE."
        )

        if explicit_image_token:
            content = [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt

        rows.append(
            {
                "prompt": [{"role": "user", "content": content}],
                "image": row["image_path"],
                "claim_ids": claim_ids,
                "gold_hallucinations": [bool(gold[cid]) for cid in claim_ids],
            }
        )

        if len(rows) >= max_examples:
            break

    if not rows:
        raise RuntimeError("No rows found.")

    return Dataset.from_list(rows).cast_column("image", Image(decode=True))


def debug_reward(completions, **kwargs):
    print("\n" + "=" * 80)
    print("REWARD FUNCTION CALLED")
    print("Number of completions:", len(completions))

    for i, c in enumerate(completions[:8]):
        print("-" * 80)
        print(f"Completion {i}:")
        print(repr(c))

    print("=" * 80 + "\n")

    # dummy reward so GRPO can run one step
    return [0.0 for _ in completions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--model_path", default="models/Qwen2-VL-2B-Instruct")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output_dir", default="results/debug_grpo_image_sanity")
    parser.add_argument("--max_examples", type=int, default=4)
    parser.add_argument("--explicit_image_token", action="store_true")
    parser.add_argument("--no_explicit_image_token", action="store_true")
    args = parser.parse_args()

    disable_vllm_import()

    from transformers import AutoProcessor, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig

    os.makedirs(args.output_dir, exist_ok=True)

    explicit = args.explicit_image_token and not args.no_explicit_image_token

    print("Building dataset")
    print("Explicit image token:", explicit)

    dataset = build_dataset(
        args.jsonl,
        args.split,
        args.max_examples,
        explicit_image_token=explicit,
    )

    print("Dataset item 0:")
    print(dataset[0]["prompt"])
    print("Image type:", type(dataset[0]["image"]))
    print("Image:", dataset[0]["image"])

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    model_init_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }

    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        model_init_kwargs=model_init_kwargs,
        max_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_completion_length=64,
        temperature=0.8,
        top_p=0.95,
        learning_rate=1e-6,
        beta=0.0,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        log_completions=True,
        bf16=True,
        tf32=True,
    )

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )

    trainer = GRPOTrainer(
        model=args.model_path,
        args=grpo_args,
        processing_class=processor,
        reward_funcs=debug_reward,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("Starting 1-step GRPO sanity run...")
    trainer.train()
    print("Finished sanity run.")


if __name__ == "__main__":
    main()