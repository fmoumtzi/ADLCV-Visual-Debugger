import re
from typing import Dict, Iterable, List, Optional

VALID_VERDICTS = {"CORRECT", "HALLUCINATED"}


def normalize_verdict(value: str) -> Optional[str]:
    if not value:
        return None
    text = value.strip().upper()
    alias_map = {
        "TRUE": "CORRECT",
        "FALSE": "HALLUCINATED",
        "SUPPORTED": "CORRECT",
        "UNSUPPORTED": "HALLUCINATED",
        "INCORRECT": "HALLUCINATED",
        "WRONG": "HALLUCINATED",
    }
    if text in VALID_VERDICTS:
        return text
    return alias_map.get(text)


def split_into_claims(response: str, max_claims: int = 8) -> List[str]:
    cleaned = re.sub(r"\s+", " ", str(response or "")).strip()
    if not cleaned:
        return []

    raw_parts = re.split(r"(?<=[\.!?])\s+|\n+|\s*;\s*", cleaned)
    claims = []
    for part in raw_parts:
        text = part.strip(" -\t\n\r")
        if not text:
            continue
        if text not in claims:
            claims.append(text)
        if len(claims) >= max_claims:
            break

    if not claims:
        claims = [cleaned]
    return claims


def format_claims(claims: Iterable[Dict]) -> str:
    lines = []
    for claim in claims:
        lines.append(f"{int(claim['claim_id'])}: {claim['text']}")
    return "\n".join(lines)


def format_target_labels(labels: Iterable[Dict]) -> str:
    ordered = sorted(labels, key=lambda x: int(x["claim_id"]))
    return "\n".join(f"{int(row['claim_id'])}\t{row['verdict'].upper()}" for row in ordered)


def build_verification_prompt(
    question: str,
    prior_response: str,
    claims: Iterable[Dict],
    choices: Optional[List[str]] = None,
    target_answer: str = "",
) -> str:
    choices = choices or []
    choices_block = "\n".join(f"- {choice}" for choice in choices)
    claims_block = format_claims(claims)

    pieces = [
        "You are verifying a previous visual answer claim-by-claim.",
        "For each claim_id, output exactly one verdict: CORRECT or HALLUCINATED.",
        "Output format: one line per claim, exactly `claim_id<TAB>VERDICT`.",
        "Do not output extra text.",
        f"Question: {question}",
    ]
    if choices_block:
        pieces.append(f"Options (if applicable):\n{choices_block}")
    if target_answer:
        pieces.append(f"Reference answer: {target_answer}")
    pieces.extend([
        f"Prior model response: {prior_response}",
        f"Claims to verify:\n{claims_block}",
    ])
    return "\n\n".join(pieces)


def parse_verifier_output(raw_text: str, expected_claim_ids: List[int]) -> Dict[int, str]:
    predictions: Dict[int, str] = {}
    lines = [line.strip() for line in str(raw_text).splitlines() if line.strip()]

    for line in lines:
        match = re.match(r"^\s*(\d+)\s*[:\-|,\t ]+\s*([A-Za-z_ ]+)\s*$", line)
        if not match:
            continue
        claim_id = int(match.group(1))
        verdict = normalize_verdict(match.group(2))
        if verdict is not None:
            predictions[claim_id] = verdict

    if len(predictions) < len(expected_claim_ids):
        verdict_hits = []
        for line in lines:
            line_norm = line.upper()
            if "HALLUCINATED" in line_norm:
                verdict_hits.append("HALLUCINATED")
            elif "CORRECT" in line_norm:
                verdict_hits.append("CORRECT")

        for idx, claim_id in enumerate(expected_claim_ids):
            if claim_id in predictions:
                continue
            if idx < len(verdict_hits):
                predictions[claim_id] = verdict_hits[idx]

    for claim_id in expected_claim_ids:
        if claim_id not in predictions:
            predictions[claim_id] = "CORRECT"

    return predictions
