import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

try:
    from .evaluate_verifier import evaluate_rows
    from .io_utils import ensure_parent_dir, read_jsonl, write_jsonl
    from .prompts import (
        build_verification_prompt,
        format_target_labels,
        parse_verifier_output,
    )
    from .runtime import load_qwen_vl, open_rgb
except ImportError:
    from evaluate_verifier import evaluate_rows
    from io_utils import ensure_parent_dir, read_jsonl, write_jsonl
    from prompts import build_verification_prompt, format_target_labels, parse_verifier_output
    from runtime import load_qwen_vl, open_rgb


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float


class Task2Dataset(Dataset):
    def __init__(self, rows: List[Dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training for Task 2 verifier")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", default="")
    parser.add_argument("--output_dir", default="results/task2/sft")
    parser.add_argument(
        "--predictions_output_jsonl",
        default="results/task2/sft_predictions.jsonl",
    )
    parser.add_argument(
        "--metrics_output_json",
        default="results/task2/metrics_sft.json",
    )
    parser.add_argument("--model_path", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--max_new_tokens_eval", type=int, default=128)
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--save_each_epoch", action="store_true")
    parser.add_argument("--slice_by_question_type", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_run_name", default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filter_rows(rows: List[Dict], split: str, require_labels: bool = True) -> List[Dict]:
    filtered = [row for row in rows if (not split or row.get("split") == split)]
    if require_labels:
        filtered = [row for row in filtered if row.get("labels")]
    return filtered


def build_text_pair(row: Dict) -> Dict[str, str]:
    claims = row.get("claims", [])
    labels = row.get("labels", [])
    prompt = build_verification_prompt(
        question=row.get("question", ""),
        prior_response=row.get("model_response", ""),
        claims=claims,
        choices=row.get("choices", []),
        target_answer=row.get("target_answer", ""),
    )
    target = format_target_labels(labels)
    return {"prompt": prompt, "target": target}


def build_collate_fn(processor):
    def collate(batch_rows: List[Dict]) -> Dict[str, torch.Tensor]:
        images = []
        prompt_texts = []
        full_texts = []

        for row in batch_rows:
            pair = build_text_pair(row)
            image = open_rgb(row["image_path"])
            images.append(image)

            user_msg = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": pair["prompt"]},
                    ],
                }
            ]
            full_msgs = user_msg + [{"role": "assistant", "content": pair["target"]}]

            prompt_texts.append(
                processor.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=True)
            )
            full_texts.append(
                processor.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
            )

        try:
            prompt_enc = processor(
                text=prompt_texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            full_enc = processor(
                text=full_texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
        finally:
            for image in images:
                image.close()

        prompt_lengths = prompt_enc["attention_mask"].sum(dim=1)
        labels = full_enc["input_ids"].clone()
        for i in range(labels.shape[0]):
            labels[i, : int(prompt_lengths[i].item())] = -100
        labels[full_enc["attention_mask"] == 0] = -100

        batch = {
            "input_ids": full_enc["input_ids"],
            "attention_mask": full_enc["attention_mask"],
            "labels": labels,
        }
        for key in ("pixel_values", "image_grid_thw", "video_grid_thw", "mm_token_type_ids"):
            if key in full_enc:
                batch[key] = full_enc[key]
        return batch

    return collate


def run_validation_generation(
    model,
    processor,
    device,
    val_rows: List[Dict],
    max_new_tokens: int,
    limit: int,
) -> List[Dict]:
    eval_rows = val_rows[: limit] if limit > 0 else val_rows
    preds = []

    model.eval()
    for row in tqdm(eval_rows, desc="Validation generation", leave=False):
        claims = row.get("claims", [])
        expected_claim_ids = [int(claim["claim_id"]) for claim in claims]
        prompt = build_verification_prompt(
            question=row.get("question", ""),
            prior_response=row.get("model_response", ""),
            claims=claims,
            choices=row.get("choices", []),
            target_answer=row.get("target_answer", ""),
        )

        image = open_rgb(row["image_path"])
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            chat_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(
                text=[chat_text],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        finally:
            image.close()

        parsed = parse_verifier_output(output_text, expected_claim_ids)
        preds.append(
            {
                "image_id": row.get("image_id"),
                "question_id": row.get("question_id"),
                "question": row.get("question", ""),
                "predictions": [
                    {"claim_id": claim_id, "hallucination": parsed[claim_id]}
                    for claim_id in expected_claim_ids
                ],
            }
        )

    model.train()
    return preds


def evaluate_checkpoint(
    model,
    processor,
    device,
    val_rows,
    max_new_tokens,
    eval_limit,
    include_question_type,
):
    if not val_rows:
        return {}, []
    preds = run_validation_generation(
        model=model,
        processor=processor,
        device=device,
        val_rows=val_rows,
        max_new_tokens=max_new_tokens,
        limit=eval_limit,
    )
    val_subset = val_rows[: len(preds)]
    metrics = evaluate_rows(
        gold_rows=val_subset,
        pred_rows=preds,
        include_question_type=include_question_type,
    )
    return metrics, preds


def maybe_init_wandb(args, train_cfg: TrainConfig):
    if not args.wandb_project:
        return None
    if wandb is None:
        raise RuntimeError("wandb is not installed. Add wandb to requirements or disable logging.")

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or None,
        config={
            "train_jsonl": args.train_jsonl,
            "val_jsonl": args.val_jsonl,
            "output_dir": args.output_dir,
            "model_path": args.model_path,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "epochs": train_cfg.epochs,
            "batch_size": train_cfg.batch_size,
            "grad_accum_steps": train_cfg.grad_accum_steps,
            "learning_rate": train_cfg.learning_rate,
            "weight_decay": train_cfg.weight_decay,
            "warmup_ratio": train_cfg.warmup_ratio,
            "max_grad_norm": train_cfg.max_grad_norm,
        },
    )
    return run


def main():
    args = parse_args()
    set_seed(args.seed)

    ensure_parent_dir(os.path.join(args.output_dir, "dummy"))
    os.makedirs(args.output_dir, exist_ok=True)

    train_rows_all = read_jsonl(args.train_jsonl)
    train_rows = filter_rows(train_rows_all, split=args.train_split, require_labels=True)
    if not train_rows and not args.train_split:
        train_rows = [row for row in train_rows_all if row.get("labels")]
    if not train_rows:
        raise RuntimeError("No labeled training rows found.")

    val_rows = []
    if args.val_jsonl:
        val_rows = filter_rows(read_jsonl(args.val_jsonl), split="", require_labels=True)
    else:
        val_rows = filter_rows(train_rows_all, split=args.val_split, require_labels=True)

    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
    )

    model, processor, device, resolved_model_path = load_qwen_vl(
        args.model_path,
        for_training=True,
    )
    print(f"Loaded base model from {resolved_model_path} on {device}")
    print(f"Train samples: {len(train_rows)} | Validation samples: {len(val_rows)}")

    train_loader = DataLoader(
        Task2Dataset(train_rows),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=build_collate_fn(processor),
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(train_cfg.grad_accum_steps, 1))
    total_steps = max(steps_per_epoch * train_cfg.epochs, 1)
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    wb_run = maybe_init_wandb(args, train_cfg)

    training_curve = []
    best_f1 = -1.0
    best_epoch = -1
    global_step = 0

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"SFT epoch {epoch}/{train_cfg.epochs}")
        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / train_cfg.grad_accum_steps
            loss.backward()

            loss_item = float(outputs.loss.detach().cpu().item())
            running_loss += loss_item
            pbar.set_postfix({"loss": f"{loss_item:.4f}"})

            should_step = (
                step % train_cfg.grad_accum_steps == 0 or step == len(train_loader)
            )
            if should_step:
                if train_cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        avg_train_loss = running_loss / max(len(train_loader), 1)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": avg_train_loss,
            "learning_rate": scheduler.get_last_lr()[0],
        }

        val_metrics = {}
        val_preds = []
        if val_rows:
            val_metrics, val_preds = evaluate_checkpoint(
                model=model,
                processor=processor,
                device=device,
                val_rows=val_rows,
                max_new_tokens=args.max_new_tokens_eval,
                eval_limit=args.eval_max_samples,
                include_question_type=args.slice_by_question_type,
            )
            record["val_precision"] = val_metrics["overall"]["precision"]
            record["val_recall"] = val_metrics["overall"]["recall"]
            record["val_f1"] = val_metrics["overall"]["f1"]

            if record["val_f1"] > best_f1:
                best_f1 = record["val_f1"]
                best_epoch = epoch
                best_dir = os.path.join(args.output_dir, "best_checkpoint")
                os.makedirs(best_dir, exist_ok=True)
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)
                with open(os.path.join(best_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(val_metrics, f, indent=2)
                write_jsonl(os.path.join(best_dir, "predictions.jsonl"), val_preds)
                ensure_parent_dir(args.metrics_output_json)
                with open(args.metrics_output_json, "w", encoding="utf-8") as f:
                    json.dump(val_metrics, f, indent=2)
                write_jsonl(args.predictions_output_jsonl, val_preds)

        training_curve.append(record)

        if args.save_each_epoch:
            epoch_dir = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            model.save_pretrained(epoch_dir)
            processor.save_pretrained(epoch_dir)
            if val_metrics:
                with open(os.path.join(epoch_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(val_metrics, f, indent=2)
                write_jsonl(os.path.join(epoch_dir, "predictions.jsonl"), val_preds)

        print(
            f"Epoch {epoch}: train_loss={avg_train_loss:.4f}",
            (
                f", val_p/r/f1={record['val_precision']:.4f}/"
                f"{record['val_recall']:.4f}/{record['val_f1']:.4f}"
                if "val_f1" in record
                else ""
            ),
        )

        if wb_run is not None:
            wandb.log(record)

    final_dir = os.path.join(args.output_dir, "final_checkpoint")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    curve_path = os.path.join(args.output_dir, "curves_sft.json")
    with open(curve_path, "w", encoding="utf-8") as f:
        json.dump(training_curve, f, indent=2)

    summary = {
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "best_val_f1": best_f1 if best_f1 >= 0 else None,
        "best_epoch": best_epoch if best_epoch > 0 else None,
        "final_checkpoint": final_dir,
        "curve_path": curve_path,
    }
    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved final checkpoint to {final_dir}")
    print(f"Saved training curve to {curve_path}")

    if wb_run is not None:
        wandb.summary.update(summary)
        wandb.finish()


if __name__ == "__main__":
    main()
