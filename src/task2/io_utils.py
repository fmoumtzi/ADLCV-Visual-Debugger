import json
import os
from typing import Dict, Iterable, Iterator, List, Optional


REQUIRED_TASK2_KEYS = {
    "image_id",
    "image_path",
    "question",
    "model_response",
    "claims",
    "labels",
}


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stream_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc


def parse_claim_entries(claims: List[Dict]) -> List[Dict]:
    out = []
    for idx, claim in enumerate(claims):
        claim_id = claim.get("claim_id", idx)
        text = str(claim.get("text", "")).strip()
        if not text:
            continue
        out.append({"claim_id": int(claim_id), "text": text})
    return out


def parse_label_entries(labels: List[Dict]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for entry in labels:
        if "claim_id" not in entry:
            continue
        verdict = str(entry.get("verdict", "")).strip().upper()
        if verdict:
            out[int(entry["claim_id"])] = verdict
    return out


def normalize_task2_row(row: Dict) -> Dict:
    claims = parse_claim_entries(row.get("claims", []))
    labels = parse_label_entries(row.get("labels", []))

    normalized = {
        "image_id": row.get("image_id"),
        "image_path": row.get("image_path", ""),
        "source_dataset": row.get("source_dataset", row.get("dataset", "vqa")),
        "question": row.get("question", row.get("vqa_question", "")),
        "choices": row.get("choices", []),
        "target_answer": row.get("target_answer", row.get("vqa_ground_truth", "")),
        "answer_type": row.get("answer_type", ""),
        "question_type": row.get("question_type", ""),
        "model_response": row.get(
            "model_response",
            row.get("vlm_generation", row.get("vlm_answer", "")),
        ),
        "claims": claims,
        "labels": [{"claim_id": cid, "verdict": verdict} for cid, verdict in sorted(labels.items())],
        "split": row.get("split", ""),
        "question_id": row.get("question_id"),
    }
    return normalized


def validate_task2_row(row: Dict, require_labels: bool = True) -> Optional[str]:
    missing = [key for key in REQUIRED_TASK2_KEYS if key not in row]
    if missing:
        return f"missing keys: {missing}"
    if not row.get("claims"):
        return "claims is empty"
    if require_labels and not row.get("labels"):
        return "labels is empty"
    return None
