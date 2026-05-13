import argparse
import json
import random
import re
from collections import Counter


def clean_generation(text):
    text = str(text or "").strip()
    text = re.sub(r"\s*(\(\d+,\s*\d+\)\s*,?)+\s*$", "", text).strip()
    return text


def split_balanced(rows, train_ratio=0.8, val_ratio=0.1):
    hallucinated = [r for r in rows if r["hallucination"]]
    factual = [r for r in rows if not r["hallucination"]]

    random.shuffle(hallucinated)
    random.shuffle(factual)

    n = min(len(hallucinated), len(factual))
    hallucinated = hallucinated[:n]
    factual = factual[:n]

    def split_class(class_rows):
        n_total = len(class_rows)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train = class_rows[:n_train]
        val = class_rows[n_train:n_train + n_val]
        test = class_rows[n_train + n_val:]

        return train, val, test

    h_train, h_val, h_test = split_class(hallucinated)
    f_train, f_val, f_test = split_class(factual)

    train = h_train + f_train
    val = h_val + f_val
    test = h_test + f_test

    for r in train:
        r["split"] = "train"
    for r in val:
        r["split"] = "val"
    for r in test:
        r["split"] = "test"

    final_rows = train + val + test
    random.shuffle(final_rows)
    return final_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    converted = []

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)

            generation = generation = str(row.get("vlm_generation", "")).strip()
            if not generation:
                continue

            is_correct = bool(row.get("evaluation", {}).get("is_correct", False))
            hallucination = not is_correct

            full_gt = row.get("vqa_ground_truth", [])
            if not isinstance(full_gt, list):
                full_gt = [full_gt]

            converted.append(
                {
                    "image_id": row.get("image_id"),
                    "image_path": row.get("image_path", ""),
                    "source_dataset": row.get("dataset", "vqa"),
                    "question_id": row.get("trace_id", f"new_{i}"),
                    "question": row.get("vqa_question", ""),
                    "choices": [],
                    "target_answer": full_gt,
                    "vqa_ground_truth": full_gt,
                    "answer_type": "",
                    "question_type": "",
                    "model_response": generation,
                    "hallucination": hallucination,
                    "claims": [{"claim_id": 0, "text": generation}],
                    "labels": [{"claim_id": 0, "hallucination": hallucination}],
                    "split": "",
                    "prompt_type": row.get("prompt_type", ""),
                    "judge_model": row.get("judge_model", ""),
                    "evaluation_type": row.get("evaluation_type", ""),
                    "judge_reasoning": row.get("evaluation", {}).get("reasoning", ""),
                    "trace_id": row.get("trace_id", ""),
                }
            )

    balanced = split_balanced(converted)

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in balanced:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Original rows: {len(converted)}")
    print(f"Balanced rows: {len(balanced)}")
    print("Overall:", Counter(r["hallucination"] for r in balanced))

    for split in ["train", "val", "test"]:
        split_rows = [r for r in balanced if r["split"] == split]
        print(split, len(split_rows), Counter(r["hallucination"] for r in split_rows))


if __name__ == "__main__":
    main()