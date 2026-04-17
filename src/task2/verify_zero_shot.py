import argparse
import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm

try:
    from .evaluate_verifier import evaluate_rows
    from .io_utils import ensure_parent_dir, read_jsonl, write_jsonl
    from .prompts import build_verification_prompt, parse_verifier_output
    from .runtime import load_qwen_vl, open_rgb
except ImportError:
    from evaluate_verifier import evaluate_rows
    from io_utils import ensure_parent_dir, read_jsonl, write_jsonl
    from prompts import build_verification_prompt, parse_verifier_output
    from runtime import load_qwen_vl, open_rgb


DEFAULT_OUTPUT = "results/task2/zero_shot_predictions.jsonl"
DEFAULT_METRICS = "results/task2/metrics_zero_shot.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Run zero-shot Task 2 self-verification.")
    parser.add_argument("--input_jsonl", required=True, help="Task 2 JSONL")
    parser.add_argument("--output_jsonl", default=DEFAULT_OUTPUT, help="Prediction JSONL")
    parser.add_argument(
        "--metrics_output_json",
        default=DEFAULT_METRICS,
        help="Where to store aggregated metrics JSON",
    )
    parser.add_argument(
        "--model_path",
        default="models/Qwen2.5-VL-3B-Instruct",
        help="Local model path or HF model id.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--split",
        default="",
        help="Optional split filter (train/val/test).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of rows")
    parser.add_argument("--skip_missing_images", action="store_true")
    parser.add_argument("--slice_by_question_type", action="store_true")
    return parser.parse_args()


def ask_verifier(
    model,
    processor,
    device,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    image = open_rgb(image_path)
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

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **generation_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text.strip()
    finally:
        image.close()


def maybe_filter_rows(rows: List[Dict], split: str, limit: int) -> List[Dict]:
    filtered = rows
    if split:
        filtered = [row for row in filtered if row.get("split") == split]
    if limit > 0:
        filtered = filtered[:limit]
    return filtered


def main():
    args = parse_args()

    if not os.path.isfile(args.input_jsonl):
        raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")

    rows = read_jsonl(args.input_jsonl)
    rows = maybe_filter_rows(rows, split=args.split, limit=args.limit)
    if not rows:
        raise RuntimeError("No rows matched the provided filters.")

    model, processor, device, resolved_model_path = load_qwen_vl(args.model_path, for_training=False)
    print(f"Loaded verifier model from {resolved_model_path} on {device}")

    predictions = []
    skipped = 0

    for row in tqdm(rows, desc="Zero-shot verification"):
        image_path = row.get("image_path", "")
        if not os.path.isfile(image_path):
            if args.skip_missing_images:
                skipped += 1
                continue
            raise FileNotFoundError(f"Missing image: {image_path}")

        claims = row.get("claims", [])
        expected_claim_ids = [int(claim["claim_id"]) for claim in claims]
        prompt = build_verification_prompt(
            question=row.get("question", ""),
            prior_response=row.get("model_response", ""),
            claims=claims,
            choices=row.get("choices", []),
            target_answer=row.get("target_answer", ""),
        )

        raw_output = ask_verifier(
            model=model,
            processor=processor,
            device=device,
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        pred_map = parse_verifier_output(raw_output, expected_claim_ids=expected_claim_ids)
        pred_list = [{"claim_id": cid, "verdict": pred_map[cid]} for cid in expected_claim_ids]

        predictions.append(
            {
                "image_id": row.get("image_id"),
                "question_id": row.get("question_id"),
                "question": row.get("question", ""),
                "answer_type": row.get("answer_type", ""),
                "question_type": row.get("question_type", ""),
                "predictions": pred_list,
                "raw_output": raw_output,
            }
        )

    write_jsonl(args.output_jsonl, predictions)
    print(f"Saved {len(predictions)} prediction rows to {args.output_jsonl}")
    if skipped:
        print(f"Skipped rows with missing images: {skipped}")

    metrics = evaluate_rows(
        gold_rows=rows,
        pred_rows=predictions,
        include_question_type=args.slice_by_question_type,
    )
    ensure_parent_dir(args.metrics_output_json)
    with open(args.metrics_output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON to {args.metrics_output_json}")
    print(
        "Overall HALLUCINATED precision/recall/F1:",
        f"{metrics['overall']['precision']:.4f}/"
        f"{metrics['overall']['recall']:.4f}/"
        f"{metrics['overall']['f1']:.4f}",
    )


if __name__ == "__main__":
    main()
