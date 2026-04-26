import argparse
import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm
from peft import PeftModel

from .evaluate_verifier import evaluate_rows
from .io_utils import ensure_parent_dir, read_jsonl, write_jsonl
from .prompts import build_verification_prompt, parse_verifier_output
from .runtime import load_qwen_vl, open_rgb

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
    
    # Slicing and Formatting
    parser.add_argument("--slice_by_question_type", action="store_true")
    
    return parser.parse_args()

def ask_verifier(model, processor, device, image_path, prompt, max_new_tokens, temperature):
    """Handles the exact inference logic using the loaded VLM."""
    image = open_rgb(image_path)
    try:
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)

        generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

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
    print(f"Loaded {len(rows)} verification tasks.")

    # 1. Load the Base Model
    print(f"Loading base model from: {args.model_path}")
    model, processor, device, resolved_model_path = load_qwen_vl(args.model_path, for_training=False)
    
    # 2. Attach LoRA Adapter if provided
    if args.lora_path:
        if not os.path.isdir(args.lora_path):
            raise FileNotFoundError(f"LoRA path not found: {args.lora_path}")
        print(f"Attaching LoRA weights from: {args.lora_path} ...")
        model = PeftModel.from_pretrained(model, args.lora_path)
        print("LoRA successfully injected.")
    else:
        print("No LoRA path provided. Running in ZERO-SHOT mode.")

    predictions = []
    
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

        raw_output = ask_verifier(model, processor, device, image_path, prompt, args.max_new_tokens, args.temperature)
        pred_map = parse_verifier_output(raw_output, expected_claim_ids=expected_claim_ids)
        
        predictions.append({
            "image_id": row.get("image_id"),
            "question_id": row.get("question_id"),
            "question": row.get("question", ""),
            "answer_type": row.get("answer_type", ""),
            "question_type": row.get("question_type", ""),
            "predictions": [{"claim_id": cid, "hallucination": pred_map[cid]} for cid in expected_claim_ids],
            "raw_output": raw_output,
        })

    # Save Predictions
    ensure_parent_dir(args.output_jsonl)
    write_jsonl(args.output_jsonl, predictions)

    # 4. Evaluation Module
    print("\n=== Calculating Metrics ===")
    metrics = evaluate_rows(gold_rows=rows, pred_rows=predictions, include_question_type=args.slice_by_question_type)
    
    ensure_parent_dir(args.metrics_output_json)
    with open(args.metrics_output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    overall = metrics['overall']
    print("\n[ FINAL RESULTS ]")
    print(f"Precision: {overall['precision']:.4f}")
    print(f"Recall:    {overall['recall']:.4f}")
    print(f"F1 Score:  {overall['f1']:.4f}")
    print(f"Metrics saved to: {args.metrics_output_json}\n")

if __name__ == "__main__":
    main()