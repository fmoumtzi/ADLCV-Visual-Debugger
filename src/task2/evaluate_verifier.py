import argparse
import json
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

try:
    from .io_utils import ensure_parent_dir, parse_label_entries, read_jsonl
except ImportError:
    from io_utils import ensure_parent_dir, parse_label_entries, read_jsonl

POSITIVE_CLASS = "HALLUCINATED"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Task 2 verifier predictions.")
    parser.add_argument("--gold_jsonl", required=True, help="Gold Task 2 JSONL with labels")
    parser.add_argument(
        "--pred_jsonl",
        required=True,
        help=(
            "Predictions JSONL with fields image_id/question_id/question + "
            "predictions=[{claim_id, verdict}]"
        ),
    )
    parser.add_argument(
        "--output_json",
        default="",
        help="Optional metrics output JSON path",
    )
    parser.add_argument(
        "--slice_by_question_type",
        action="store_true",
        help="Also compute per-question_type metrics.",
    )
    return parser.parse_args()


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def row_key(row: Dict) -> str:
    question_id = row.get("question_id")
    if question_id is not None:
        return f"{row.get('image_id')}::{question_id}"
    question = " ".join(str(row.get("question", "")).strip().lower().split())
    return f"{row.get('image_id')}::{question}"


def normalize_pred_row(row: Dict) -> Dict[int, str]:
    preds = {}
    for item in row.get("predictions", []):
        claim_id = int(item["claim_id"])
        verdict = str(item.get("verdict", "")).upper()
        preds[claim_id] = verdict
    return preds


def evaluate_rows(
    gold_rows: List[Dict],
    pred_rows: List[Dict],
    include_question_type: bool,
) -> Dict:
    pred_lookup = {row_key(row): row for row in pred_rows}

    tp = fp = fn = tn = 0
    false_alarms = 0
    misses = 0

    by_answer_type_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    by_question_type_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

    evaluated_claims = 0
    missing_predictions = 0

    for gold in gold_rows:
        key = row_key(gold)
        pred_row = pred_lookup.get(key)
        if pred_row is None:
            missing_predictions += 1
            continue

        gold_labels = parse_label_entries(gold.get("labels", []))
        pred_labels = normalize_pred_row(pred_row)
        answer_type = gold.get("answer_type", "unknown") or "unknown"
        question_type = gold.get("question_type", "unknown") or "unknown"

        for claim_id, gold_verdict in gold_labels.items():
            pred_verdict = pred_labels.get(claim_id, "CORRECT")
            gold_pos = gold_verdict == POSITIVE_CLASS
            pred_pos = pred_verdict == POSITIVE_CLASS
            evaluated_claims += 1

            if gold_pos and pred_pos:
                tp += 1
                by_answer_type_counts[answer_type]["tp"] += 1
                if include_question_type:
                    by_question_type_counts[question_type]["tp"] += 1
            elif (not gold_pos) and pred_pos:
                fp += 1
                false_alarms += 1
                by_answer_type_counts[answer_type]["fp"] += 1
                if include_question_type:
                    by_question_type_counts[question_type]["fp"] += 1
            elif gold_pos and (not pred_pos):
                fn += 1
                misses += 1
                by_answer_type_counts[answer_type]["fn"] += 1
                if include_question_type:
                    by_question_type_counts[question_type]["fn"] += 1
            else:
                tn += 1
                by_answer_type_counts[answer_type]["tn"] += 1
                if include_question_type:
                    by_question_type_counts[question_type]["tn"] += 1

    overall = prf(tp, fp, fn)
    overall.update(
        {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "false_alarms": false_alarms,
            "misses": misses,
            "evaluated_claims": evaluated_claims,
            "missing_prediction_rows": missing_predictions,
            "hallucinated_prediction_rate": safe_div(tp + fp, max(tp + fp + fn + tn, 1)),
        }
    )

    by_answer_type = {}
    for answer_type, counts in sorted(by_answer_type_counts.items()):
        metrics = prf(counts["tp"], counts["fp"], counts["fn"])
        metrics.update(counts)
        by_answer_type[answer_type] = metrics

    by_question_type = {}
    if include_question_type:
        for question_type, counts in sorted(by_question_type_counts.items()):
            metrics = prf(counts["tp"], counts["fp"], counts["fn"])
            metrics.update(counts)
            by_question_type[question_type] = metrics

    results = {
        "positive_class": POSITIVE_CLASS,
        "overall": overall,
        "by_answer_type": by_answer_type,
    }
    if include_question_type:
        results["by_question_type"] = by_question_type
    return results


def print_summary(metrics: Dict) -> None:
    overall = metrics["overall"]
    print("=== Overall (HALLUCINATED) ===")
    print(
        f"precision={overall['precision']:.4f}, "
        f"recall={overall['recall']:.4f}, f1={overall['f1']:.4f}"
    )
    print(
        f"tp={overall['tp']}, fp={overall['fp']}, fn={overall['fn']}, tn={overall['tn']}, "
        f"false_alarms={overall['false_alarms']}, misses={overall['misses']}"
    )

    print("\n=== By answer_type ===")
    for answer_type, row in metrics["by_answer_type"].items():
        print(
            f"{answer_type}: precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, f1={row['f1']:.4f}, "
            f"tp={row['tp']}, fp={row['fp']}, fn={row['fn']}, tn={row['tn']}"
        )

    if "by_question_type" in metrics:
        print("\n=== By question_type ===")
        for question_type, row in metrics["by_question_type"].items():
            print(
                f"{question_type}: precision={row['precision']:.4f}, "
                f"recall={row['recall']:.4f}, f1={row['f1']:.4f}, "
                f"tp={row['tp']}, fp={row['fp']}, fn={row['fn']}, tn={row['tn']}"
            )


def main():
    args = parse_args()
    gold_rows = read_jsonl(args.gold_jsonl)
    pred_rows = read_jsonl(args.pred_jsonl)

    metrics = evaluate_rows(
        gold_rows=gold_rows,
        pred_rows=pred_rows,
        include_question_type=args.slice_by_question_type,
    )
    print_summary(metrics)

    if args.output_json:
        ensure_parent_dir(args.output_json)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics JSON to {args.output_json}")


if __name__ == "__main__":
    main()
