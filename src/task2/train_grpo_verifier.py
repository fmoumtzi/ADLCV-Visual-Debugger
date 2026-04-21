import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import torch
from torch.optim import AdamW
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

try:
    from .evaluate_verifier import evaluate_rows
    from .io_utils import ensure_parent_dir, parse_label_entries, read_jsonl, write_jsonl
    from .prompts import build_verification_prompt, parse_verifier_output
    from .reward import relative_group_advantages, score_predictions
    from .runtime import load_qwen_vl, open_rgb
except ImportError:
    from evaluate_verifier import evaluate_rows
    from io_utils import ensure_parent_dir, parse_label_entries, read_jsonl, write_jsonl
    from prompts import build_verification_prompt, parse_verifier_output
    from reward import relative_group_advantages, score_predictions
    from runtime import load_qwen_vl, open_rgb


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning for Task 2 verifier")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", default="")
    parser.add_argument("--model_path", required=True, help="SFT checkpoint path (policy init)")
    parser.add_argument("--output_dir", default="results/task2/grpo")
    parser.add_argument(
        "--predictions_output_jsonl",
        default="results/task2/grpo_predictions.jsonl",
    )
    parser.add_argument(
        "--metrics_output_json",
        default="results/task2/metrics_grpo.json",
    )
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)

    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--reward_normalized", action="store_true")
    parser.add_argument("--kl_coef", type=float, default=0.02)

    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--eval_every_steps", type=int, default=50)
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--slice_by_question_type", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
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


def build_prompt(row: Dict) -> Tuple[str, List[int], Dict[int, str]]:
    claims = row.get("claims", [])
    claim_ids = [int(claim["claim_id"]) for claim in claims]
    gold = parse_label_entries(row.get("labels", []))
    prompt = build_verification_prompt(
        question=row.get("question", ""),
        prior_response=row.get("model_response", ""),
        claims=claims,
        choices=row.get("choices", []),
        target_answer=row.get("target_answer", ""),
    )
    return prompt, claim_ids, gold


def generate_candidate_texts(
    model,
    processor,
    device,
    image,
    prompt: str,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(device)

    outputs = []
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
    }
    with torch.inference_mode():
        for _ in range(num_candidates):
            generated_ids = model.generate(**inputs, **generation_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            outputs.append(text)
    return outputs


def compute_candidate_logprob(
    model,
    processor,
    device,
    image,
    prompt: str,
    candidate_text: str,
    requires_grad: bool,
) -> torch.Tensor:
    user_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    full_messages = user_messages + [{"role": "assistant", "content": candidate_text}]

    prompt_chat = processor.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_chat = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_inputs = processor(
        text=[prompt_chat],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(device)
    full_inputs = processor(
        text=[full_chat],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(device)

    prompt_len = int(prompt_inputs["attention_mask"].sum().item())
    input_ids = full_inputs["input_ids"]
    if input_ids.shape[1] <= prompt_len:
        return torch.zeros((), device=device, dtype=torch.float32)

    def _forward():
        outputs = model(**full_inputs)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]

        start = max(prompt_len - 1, 0)
        cand_logits = logits[:, start:, :]
        cand_labels = labels[:, start:]
        if cand_labels.numel() == 0:
            return torch.zeros((), device=device, dtype=logits.dtype)

        token_logprobs = torch.log_softmax(cand_logits, dim=-1).gather(
            dim=-1,
            index=cand_labels.unsqueeze(-1),
        ).squeeze(-1)
        return token_logprobs.mean()

    if requires_grad:
        return _forward()
    with torch.no_grad():
        return _forward().detach()


def run_validation(
    model,
    processor,
    device,
    val_rows: List[Dict],
    max_new_tokens: int,
    limit: int,
    include_question_type: bool,
):
    if not val_rows:
        return {}, []

    subset = val_rows[:limit] if limit > 0 else val_rows
    preds = []

    model.eval()
    for row in tqdm(subset, desc="GRPO validation", leave=False):
        prompt, claim_ids, _ = build_prompt(row)
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
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(
                text=[prompt_text],
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

        parsed = parse_verifier_output(output_text, expected_claim_ids=claim_ids)
        preds.append(
            {
                "image_id": row.get("image_id"),
                "question_id": row.get("question_id"),
                "question": row.get("question", ""),
                "predictions": [
                    {"claim_id": claim_id, "verdict": parsed[claim_id]}
                    for claim_id in claim_ids
                ],
            }
        )

    metrics = evaluate_rows(
        gold_rows=subset,
        pred_rows=preds,
        include_question_type=include_question_type,
    )
    model.train()
    return metrics, preds


def maybe_eval_and_save(
    args,
    policy_model,
    processor,
    device,
    val_rows,
    curve,
    global_step,
    best,
):
    if not val_rows:
        return best

    metrics, preds = run_validation(
        model=policy_model,
        processor=processor,
        device=device,
        val_rows=val_rows,
        max_new_tokens=args.max_new_tokens,
        limit=args.eval_max_samples,
        include_question_type=args.slice_by_question_type,
    )
    f1 = metrics["overall"]["f1"]
    curve.append(
        {
            "step": global_step,
            "val_precision": metrics["overall"]["precision"],
            "val_recall": metrics["overall"]["recall"],
            "val_f1": f1,
        }
    )
    print(
        f"Step {global_step}: val_p/r/f1={metrics['overall']['precision']:.4f}/"
        f"{metrics['overall']['recall']:.4f}/{f1:.4f}"
    )

    if f1 > best["f1"]:
        best["f1"] = f1
        best["step"] = global_step
        best_dir = os.path.join(args.output_dir, "best_checkpoint")
        os.makedirs(best_dir, exist_ok=True)
        policy_model.save_pretrained(best_dir)
        processor.save_pretrained(best_dir)
        with open(os.path.join(best_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        write_jsonl(os.path.join(best_dir, "predictions.jsonl"), preds)
        ensure_parent_dir(args.metrics_output_json)
        with open(args.metrics_output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        write_jsonl(args.predictions_output_jsonl, preds)
    return best


def maybe_init_wandb(args):
    if not args.wandb_project:
        return None
    if wandb is None:
        raise RuntimeError("wandb is not installed. Add wandb to requirements or disable logging.")

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or None,
        config=vars(args),
    )
    return run


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    train_all = read_jsonl(args.train_jsonl)
    train_rows = filter_rows(train_all, split=args.train_split, require_labels=True)
    if not train_rows and not args.train_split:
        train_rows = [row for row in train_all if row.get("labels")]
    if not train_rows:
        raise RuntimeError("No labeled training rows found for GRPO.")

    if args.val_jsonl:
        val_rows = filter_rows(read_jsonl(args.val_jsonl), split="", require_labels=True)
    else:
        val_rows = filter_rows(train_all, split=args.val_split, require_labels=True)

    policy_model, processor, device, resolved = load_qwen_vl(args.model_path, for_training=True)
    ref_model = None
    if args.kl_coef > 0:
        ref_model, _, _, _ = load_qwen_vl(args.model_path, for_training=False)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    print(f"Policy init checkpoint: {resolved}")
    print(f"Train rows: {len(train_rows)} | Val rows: {len(val_rows)}")

    optimizer = AdamW(
        policy_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    wb_run = maybe_init_wandb(args)

    curve = []
    best = {"f1": -1.0, "step": -1}
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_rows)
        policy_model.train()
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_rows, desc=f"GRPO epoch {epoch}/{args.epochs}")
        for row_idx, row in enumerate(pbar, start=1):
            image = open_rgb(row["image_path"])
            try:
                prompt, claim_ids, gold = build_prompt(row)
                candidate_texts = generate_candidate_texts(
                    model=policy_model,
                    processor=processor,
                    device=device,
                    image=image,
                    prompt=prompt,
                    num_candidates=args.num_candidates,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                rewards = []
                parsed_candidates = []
                for text in candidate_texts:
                    pred_map = parse_verifier_output(text, expected_claim_ids=claim_ids)
                    reward_info = score_predictions(pred_map, gold)
                    reward_value = (
                        reward_info["normalized_reward"]
                        if args.reward_normalized
                        else reward_info["total_reward"]
                    )
                    rewards.append(float(reward_value))
                    parsed_candidates.append((text, pred_map))

                advantages = relative_group_advantages(rewards)
                losses = []
                for (candidate_text, _), advantage in zip(parsed_candidates, advantages):
                    lp_policy = compute_candidate_logprob(
                        model=policy_model,
                        processor=processor,
                        device=device,
                        image=image,
                        prompt=prompt,
                        candidate_text=candidate_text,
                        requires_grad=True,
                    )
                    if ref_model is not None:
                        lp_ref = compute_candidate_logprob(
                            model=ref_model,
                            processor=processor,
                            device=device,
                            image=image,
                            prompt=prompt,
                            candidate_text=candidate_text,
                            requires_grad=False,
                        )
                    else:
                        lp_ref = lp_policy.detach()

                    advantage_t = torch.tensor(advantage, device=device, dtype=lp_policy.dtype)
                    pg_loss = -(advantage_t * lp_policy)
                    kl_loss = args.kl_coef * (lp_policy - lp_ref).pow(2)
                    losses.append(pg_loss + kl_loss)

                loss = torch.stack(losses).mean() / max(args.grad_accum_steps, 1)
                loss.backward()

                raw_reward_mean = sum(rewards) / max(len(rewards), 1)
                pbar.set_postfix(
                    {
                        "loss": f"{loss.detach().cpu().item() * args.grad_accum_steps:.4f}",
                        "reward": f"{raw_reward_mean:.3f}",
                    }
                )

                if row_idx % args.grad_accum_steps == 0 or row_idx == len(train_rows):
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if wb_run is not None:
                        wandb.log(
                            {
                                "train/loss": float(loss.detach().cpu().item() * args.grad_accum_steps),
                                "train/reward_mean": raw_reward_mean,
                                "step": global_step,
                            }
                        )

                    if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                        best = maybe_eval_and_save(
                            args=args,
                            policy_model=policy_model,
                            processor=processor,
                            device=device,
                            val_rows=val_rows,
                            curve=curve,
                            global_step=global_step,
                            best=best,
                        )
                        if wb_run is not None and curve:
                            wandb.log(
                                {
                                    "eval/precision": curve[-1]["val_precision"],
                                    "eval/recall": curve[-1]["val_recall"],
                                    "eval/f1": curve[-1]["val_f1"],
                                    "step": global_step,
                                }
                            )
            finally:
                image.close()

    best = maybe_eval_and_save(
        args=args,
        policy_model=policy_model,
        processor=processor,
        device=device,
        val_rows=val_rows,
        curve=curve,
        global_step=global_step,
        best=best,
    )

    final_dir = os.path.join(args.output_dir, "final_checkpoint")
    os.makedirs(final_dir, exist_ok=True)
    policy_model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    curve_path = os.path.join(args.output_dir, "curves_grpo.json")
    with open(curve_path, "w", encoding="utf-8") as f:
        json.dump(curve, f, indent=2)

    summary = {
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "best_val_f1": best["f1"] if best["f1"] >= 0 else None,
        "best_step": best["step"] if best["step"] >= 0 else None,
        "final_checkpoint": final_dir,
        "curve_path": curve_path,
    }
    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved final GRPO checkpoint to {final_dir}")
    print(f"Saved GRPO curve to {curve_path}")

    if wb_run is not None:
        wandb.summary.update(summary)
        wandb.finish()


if __name__ == "__main__":
    main()
