import argparse
import hashlib
import os
from typing import Dict, List, Optional

from tqdm import tqdm

try:
    from .io_utils import read_jsonl, write_jsonl
    from .prompts import split_into_claims
except ImportError:
    from io_utils import read_jsonl, write_jsonl
    from prompts import split_into_claims


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Task 2 verification JSONL from Task 1 generations."
    )
    parser.add_argument(
        "--input_jsonl",
        required=True,
        help="Task 1 generations JSONL (e.g., results/.../generations.jsonl).",
    )
    parser.add_argument(
        "--output_jsonl",
        required=True,
        help="Output JSONL in Task 2 format.",
    )
    parser.add_argument(
        "--label_jsonl",
        default="",
        help=(
            "Optional JSONL with manual claim labels. Each row should include"
            " image_id + question (or question_id) + labels list."
        ),
    )
    parser.add_argument(
        "--auto_label_mode",
        choices=["none", "target_overlap"],
        default="none",
        help="Weak-label mode if manual labels are unavailable.",
    )
    parser.add_argument(
        "--max_claims",
        type=int,
        default=8,
        help="Maximum claims extracted per sample.",
    )
    parser.add_argument(
        "--drop_unlabeled",
        action="store_true",
        help="Drop rows that still have no labels after label lookup/auto-labeling.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Split ratio for train.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Split ratio for val.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Split ratio for test.",
    )
    parser.add_argument(
        "--seed_key",
        default="task2-default",
        help="Stable seed key for deterministic split hashing.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def sample_key(row: Dict) -> str:
    question_id = row.get("question_id")
    if question_id is not None:
        return f"{row.get('image_id')}::{question_id}"
    return f"{row.get('image_id')}::{normalize_text(row.get('question', ''))}"


def build_label_lookup(label_rows: List[Dict]) -> Dict[str, List[Dict]]:
    out = {}
    for row in label_rows:
        key = sample_key(row)
        labels = row.get("labels", [])
        if labels:
            out[key] = labels
    return out


def assign_split(key: str, train_ratio: float, val_ratio: float, seed_key: str) -> str:
    hashed = hashlib.sha1(f"{seed_key}::{key}".encode("utf-8")).hexdigest()
    frac = int(hashed[:8], 16) / 0xFFFFFFFF
    if frac < train_ratio:
        return "train"
    if frac < train_ratio + val_ratio:
        return "val"
    return "test"


def auto_label_target_overlap(claims: List[Dict], target_answer: str) -> List[Dict]:
    target_norm = normalize_text(target_answer)
    labels = []
    for claim in claims:
        claim_norm = normalize_text(claim["text"])
        if target_norm and target_norm in claim_norm:
            verdict = "CORRECT"
        else:
            verdict = "HALLUCINATED"
        labels.append({"claim_id": claim["claim_id"], "verdict": verdict})
    return labels


def standardize_row(row: Dict, max_claims: int) -> Dict:
    question = row.get("question", row.get("vqa_question", ""))
    model_response = row.get("model_response", row.get("vlm_generation", ""))
    raw_claims = row.get("claims", [])

    claims = []
    if raw_claims:
        for idx, claim in enumerate(raw_claims[:max_claims]):
            claims.append({
                "claim_id": int(claim.get("claim_id", idx)),
                "text": str(claim.get("text", "")).strip(),
            })
        claims = [c for c in claims if c["text"]]

    if not claims:
        split_claims = split_into_claims(model_response, max_claims=max_claims)
        claims = [{"claim_id": idx, "text": text} for idx, text in enumerate(split_claims)]

    out = {
        "image_id": row.get("image_id"),
        "image_path": row.get("image_path", ""),
        "source_dataset": row.get("source_dataset", row.get("dataset", "vqa")),
        "question_id": row.get("question_id"),
        "question": question,
        "choices": row.get("choices", []),
        "target_answer": row.get("target_answer", row.get("vqa_ground_truth", "")),
        "answer_type": row.get("answer_type", ""),
        "question_type": row.get("question_type", ""),
        "model_response": model_response,
        "claims": claims,
        "labels": row.get("labels", []),
    }
    return out


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "Split ratios must sum to 1.0. "
            f"Got train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )


def main():
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    if not os.path.isfile(args.input_jsonl):
        raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")

    rows = read_jsonl(args.input_jsonl)
    label_lookup = {}
    if args.label_jsonl:
        label_rows = read_jsonl(args.label_jsonl)
        label_lookup = build_label_lookup(label_rows)

    converted = []
    dropped = 0

    for row in tqdm(rows, desc="Preparing Task 2 rows"):
        std = standardize_row(row, max_claims=args.max_claims)
        key = sample_key(std)

        if key in label_lookup:
            std["labels"] = label_lookup[key]
        elif args.auto_label_mode == "target_overlap":
            std["labels"] = auto_label_target_overlap(std["claims"], std.get("target_answer", ""))

        std["split"] = assign_split(
            key,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed_key=args.seed_key,
        )

        if not std.get("claims"):
            dropped += 1
            continue

        if not std.get("labels") and args.drop_unlabeled:
            dropped += 1
            continue

        converted.append(std)

    write_jsonl(args.output_jsonl, converted)
    print(f"Wrote {len(converted)} rows to {args.output_jsonl}")
    if dropped:
        print(f"Dropped {dropped} rows due to missing claims/labels.")


if __name__ == "__main__":
    main()
