import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from evaluate_verifier import evaluate_rows
from io_utils import ensure_parent_dir, parse_label_entries, read_jsonl, write_jsonl
from prompts import add_verification_format_example, build_verification_prompt, parse_verifier_output
from runtime import build_base_load_kwargs, load_qwen_vl, open_rgb, resolve_runtime

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Task 2 self-verification and evaluation script.")
    parser.add_argument("--input_jsonl", required=True, help="Task 2 Verification JSONL")
    parser.add_argument("--output_jsonl", required=True, help="Where to save predictions")
    parser.add_argument("--metrics_output_json", required=True, help="Where to save metrics JSON")
    
    # Model Loading Arguments
    parser.add_argument("--model_path", required=True, help="Local path or HF ID for the BASE model")
    parser.add_argument("--lora_path", default=None, help="Optional path to the trained LoRA adapter directory")
    
    # Generation Arguments
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    
    # Slicing and Formatting
    parser.add_argument(
        "--split",
        default="test",
        help="Split filter to evaluate. Use 'all' or empty string to evaluate every row.",
    )
    parser.add_argument(
        "--format_instruction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the concrete output-format example used during TRL GRPO training.",
    )
    parser.add_argument(
        "--strict_output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Report whether outputs contain exactly one TRUE/FALSE line for each expected claim_id.",
    )
    parser.add_argument(
        "--invalid_output_policy",
        choices=["legacy", "keep", "wrong"],
        default="legacy",
        help=(
            "How to score rows that fail strict output validation: "
            "'legacy' uses parse_verifier_output, 'keep' uses strict parsed labels with missing labels as FALSE, "
            "and 'wrong' counts invalid rows as wrong."
        ),
    )
    parser.add_argument("--slice_by_question_type", action="store_true")
    
    return parser.parse_args()


def maybe_filter_rows(rows: List[Dict], split: str) -> List[Dict]:
    split = (split or "").strip()
    if not split or split.lower() == "all":
        return rows
    return [row for row in rows if row.get("split") == split]


def parse_strict_verifier_output(raw_text: str, expected_claim_ids: List[int]) -> Tuple[Dict[int, bool], Dict]:
    expected_ids = [int(claim_id) for claim_id in expected_claim_ids]
    expected_set = set(expected_ids)
    lines = [line.strip() for line in str(raw_text or "").splitlines() if line.strip()]
    predictions: Dict[int, bool] = {}
    reasons = []
    malformed_lines = []
    duplicate_claim_ids = []

    if not lines:
        reasons.append("empty_output")

    if len(expected_ids) == 1 and len(lines) == 1:
        line_norm = lines[0].upper()
        if line_norm in {"TRUE", "FALSE"}:
            claim_id = expected_ids[0]
            predictions[claim_id] = line_norm == "TRUE"
            diagnostics = {
                "valid": True,
                "invalid_reasons": [],
                "parsed_claim_ids": [claim_id],
                "missing_claim_ids": [],
                "extra_claim_ids": [],
                "duplicate_claim_ids": [],
                "malformed_lines": [],
                "implicit_single_claim_label": True,
            }
            return predictions, diagnostics

    for line in lines:
        line_norm = line.upper()
        if "TRUE_OR_FALSE" in line_norm or "FALSE_OR_TRUE" in line_norm:
            reasons.append("placeholder_output")
            malformed_lines.append(line)
            continue

        match = re.fullmatch(r"(\d+)\s*[:\-|,\t ]+\s*(TRUE|FALSE)", line, flags=re.IGNORECASE)
        if not match:
            malformed_lines.append(line)
            continue

        claim_id = int(match.group(1))
        if claim_id in predictions:
            duplicate_claim_ids.append(claim_id)
            continue
        predictions[claim_id] = match.group(2).upper() == "TRUE"

    if malformed_lines and "placeholder_output" not in reasons:
        reasons.append("malformed_line")

    missing_claim_ids = sorted(expected_set - set(predictions))
    extra_claim_ids = sorted(set(predictions) - expected_set)
    if missing_claim_ids:
        reasons.append("missing_claim_id")
    if extra_claim_ids:
        reasons.append("extra_claim_id")
    if duplicate_claim_ids:
        reasons.append("duplicate_claim_id")

    diagnostics = {
        "valid": not reasons,
        "invalid_reasons": sorted(set(reasons)),
        "parsed_claim_ids": sorted(predictions),
        "missing_claim_ids": missing_claim_ids,
        "extra_claim_ids": extra_claim_ids,
        "duplicate_claim_ids": sorted(set(duplicate_claim_ids)),
        "malformed_lines": malformed_lines[:5],
        "implicit_single_claim_label": False,
    }
    return {claim_id: predictions.get(claim_id, False) for claim_id in expected_ids}, diagnostics


def count_invalid_as_wrong(pred_map: Dict[int, bool], expected_claim_ids: List[int], row: Dict) -> Dict[int, bool]:
    gold_labels = parse_label_entries(row.get("labels", []))
    strict_wrong = {}
    for claim_id in expected_claim_ids:
        if claim_id in gold_labels:
            strict_wrong[claim_id] = not bool(gold_labels[claim_id])
        else:
            strict_wrong[claim_id] = pred_map.get(claim_id, False)
    return strict_wrong


def is_qwen_vl_model(model_path: str) -> bool:
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return str(getattr(config, "model_type", "")).lower() in {"qwen2_vl", "qwen2_5_vl"}
    except Exception:
        path_lower = str(model_path).lower()
        return "qwen" in path_lower and "smol" not in path_lower


def load_vlm(model_path: str):
    if is_qwen_vl_model(model_path):
        model, processor, device, resolved_model_path = load_qwen_vl(model_path, for_training=False)
        return model, processor, device, resolved_model_path, "qwen"

    device = resolve_runtime()
    load_kwargs = build_base_load_kwargs(device, for_training=False)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        **load_kwargs,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if "device_map" not in load_kwargs:
        model.to(device)
    model.eval()
    return model, processor, device, model_path, "generic"


def build_inputs(processor, family, image, prompt, device):
    if family == "qwen":
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
        ]
    else:
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(
        text=[text] if isinstance(text, str) else text,
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(device)


def ask_verifier(model, processor, device, family, image_path, prompt, max_new_tokens, temperature, top_p):
    """Handles the exact inference logic using the loaded VLM."""
    image = open_rgb(image_path)
    try:
        inputs = build_inputs(processor, family, image, prompt, device)

        generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **generation_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text.strip()
    finally:
        image.close()

def main():
    args = parse_args()
    
    print(f"\n=== Initializing Unified Evaluator ===")
    rows = read_jsonl(args.input_jsonl)
    total_rows = len(rows)
    rows = maybe_filter_rows(rows, split=args.split)
    if not rows:
        raise RuntimeError(f"No verification tasks matched split={args.split!r}.")
    split_label = args.split if args.split and args.split.lower() != "all" else "all"
    print(f"Loaded {len(rows)} verification tasks for split={split_label!r} ({total_rows} total rows).")

    # 1. Load the Base Model
    print(f"Loading base model from: {args.model_path}")
    model, processor, device, resolved_model_path, family = load_vlm(args.model_path)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
    
    # 2. Attach LoRA Adapter if provided
    if args.lora_path:
        if not os.path.isdir(args.lora_path):
            raise FileNotFoundError(f"LoRA path not found: {args.lora_path}")
        print(f"Attaching LoRA weights from: {args.lora_path} ...")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model.eval()
        print("LoRA successfully injected.")
    else:
        print("No LoRA path provided. Running in ZERO-SHOT mode.")

    predictions = []
    invalid_output_rows = 0
    invalid_output_claims = 0
    total_output_claims = 0
    implicit_single_claim_label_rows = 0
    invalid_reason_counts = Counter()
    
    # 3. Inference Loop
    for row in tqdm(rows, desc="Verifying Claims"):
        image_path = row.get("image_path", "")
        claims = row.get("claims", [])
        expected_claim_ids = [int(c["claim_id"]) for c in claims]
        
        prompt = build_verification_prompt(
            question=row.get("question", ""),
            prior_response=row.get("model_response", ""),
            claims=claims,
            choices=row.get("choices", []),
            target_answer=row.get("target_answer", "")
        )
        if args.format_instruction:
            prompt = add_verification_format_example(prompt)

        raw_output = ask_verifier(
            model,
            processor,
            device,
            family,
            image_path,
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
        )
        if args.strict_output:
            pred_map, output_diagnostics = parse_strict_verifier_output(
                raw_output,
                expected_claim_ids=expected_claim_ids,
            )
            if not output_diagnostics["valid"]:
                invalid_output_rows += 1
                invalid_output_claims += len(expected_claim_ids)
                invalid_reason_counts.update(output_diagnostics["invalid_reasons"])
                if args.invalid_output_policy == "legacy":
                    pred_map = parse_verifier_output(raw_output, expected_claim_ids=expected_claim_ids)
                elif args.invalid_output_policy == "wrong":
                    pred_map = count_invalid_as_wrong(pred_map, expected_claim_ids, row)
        else:
            pred_map = parse_verifier_output(raw_output, expected_claim_ids=expected_claim_ids)
            output_diagnostics = {
                "valid": True,
                "invalid_reasons": [],
                "parsed_claim_ids": expected_claim_ids,
                "missing_claim_ids": [],
                "extra_claim_ids": [],
                "duplicate_claim_ids": [],
                "malformed_lines": [],
                "implicit_single_claim_label": False,
            }
        total_output_claims += len(expected_claim_ids)
        if output_diagnostics["implicit_single_claim_label"]:
            implicit_single_claim_label_rows += 1
        
        predictions.append({
            "image_id": row.get("image_id"),
            "question_id": row.get("question_id"),
            "question": row.get("question", ""),
            "answer_type": row.get("answer_type", ""),
            "question_type": row.get("question_type", ""),
            "predictions": [{"claim_id": cid, "hallucination": pred_map[cid]} for cid in expected_claim_ids],
            "valid_output": output_diagnostics["valid"],
            "invalid_reasons": output_diagnostics["invalid_reasons"],
            "parsed_claim_ids": output_diagnostics["parsed_claim_ids"],
            "missing_claim_ids": output_diagnostics["missing_claim_ids"],
            "extra_claim_ids": output_diagnostics["extra_claim_ids"],
            "duplicate_claim_ids": output_diagnostics["duplicate_claim_ids"],
            "malformed_lines": output_diagnostics["malformed_lines"],
            "implicit_single_claim_label": output_diagnostics["implicit_single_claim_label"],
            "raw_output": raw_output,
        })

    # Save Predictions
    ensure_parent_dir(args.output_jsonl)
    write_jsonl(args.output_jsonl, predictions)

    # 4. Evaluation Module
    print("\n=== Calculating Metrics ===")
    metrics = evaluate_rows(gold_rows=rows, pred_rows=predictions, include_question_type=args.slice_by_question_type)
    metrics["output_format"] = {
        "strict_output": args.strict_output,
        "invalid_output_policy": args.invalid_output_policy,
        "total_rows": len(rows),
        "valid_output_rows": len(rows) - invalid_output_rows if args.strict_output else len(rows),
        "invalid_output_rows": invalid_output_rows if args.strict_output else 0,
        "invalid_output_rate": invalid_output_rows / max(len(rows), 1) if args.strict_output else 0.0,
        "total_claims": total_output_claims,
        "invalid_output_claims": invalid_output_claims if args.strict_output else 0,
        "invalid_output_claim_rate": invalid_output_claims / max(total_output_claims, 1) if args.strict_output else 0.0,
        "implicit_single_claim_label_rows": implicit_single_claim_label_rows if args.strict_output else 0,
        "implicit_single_claim_label_rate": implicit_single_claim_label_rows / max(len(rows), 1) if args.strict_output else 0.0,
        "invalid_reasons": dict(sorted(invalid_reason_counts.items())),
    }
    
    ensure_parent_dir(args.metrics_output_json)
    with open(args.metrics_output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    overall = metrics['overall']
    print("\n[ FINAL RESULTS ]")
    print(f"Precision: {overall['precision']:.4f}")
    print(f"Recall:    {overall['recall']:.4f}")
    print(f"F1 Score:  {overall['f1']:.4f}")
    if args.strict_output:
        output_format = metrics["output_format"]
        print(f"Invalid output rate: {output_format['invalid_output_rate']:.4f} ({invalid_output_rows}/{len(rows)} rows)")
    print(f"Metrics saved to: {args.metrics_output_json}\n")

if __name__ == "__main__":
    main()
