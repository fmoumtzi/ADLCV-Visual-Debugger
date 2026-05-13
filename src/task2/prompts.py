import re
from typing import Dict, Iterable, List, Optional

BOOL_LABELS = {"TRUE", "FALSE"}


def normalize_hallucination_label(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value).strip().upper()
    alias_map = {
        "HALLUCINATED": True,
        "UNSUPPORTED": True,
        "INCORRECT": True,
        "WRONG": True,
        "CORRECT": False,
        "SUPPORTED": False,
    }
    if text in BOOL_LABELS:
        return text == "TRUE"
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
    lines = []
    for row in ordered:
        value = normalize_hallucination_label(row.get("hallucination"))
        if value is None:
            continue
        lines.append(f"{int(row['claim_id'])}\t{'TRUE' if value else 'FALSE'}")
    return "\n".join(lines)


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
        "For each claim_id, output hallucination as TRUE or FALSE.",
        "Output exactly one line per claim: claim_id, a tab, then TRUE or FALSE.",
        "Do not output extra text.",
        f"Question: {question}",
    ]
    if choices_block:
        pieces.append(f"Options (if applicable):\n{choices_block}")
    #if target_answer:
        # don't include target answer in prompt
        #pieces.append(f"Reference answer: {target_answer}")
        
    pieces.extend([
        f"Prior model response: {prior_response}",
        f"Claims to verify:\n{claims_block}",
    ])
    return "\n\n".join(pieces)

def build_reasoned_verification_prompt(
    question: str,
    prior_response: str,
    claims,
    choices=None,
    target_answer: str = "",
) -> str:
    choices = choices or []
    choices_block = "\n".join(f"- {choice}" for choice in choices)
    claims_block = format_claims(claims)

    pieces = [
        "You are verifying a previous visual answer using the image.",
        "First write a short reasoning sentence based only on the image and question.",
        "Then output the final hallucination verdict.",
        "Use exactly this format:",
        "Reasoning: <one short sentence>",
        "Verdict: <claim_id> TRUE_OR_FALSE",
        "",
        "TRUE means the claim is hallucinated or unsupported by the image.",
        "FALSE means the claim is correct/supported by the image.",
        f"Question: {question}",
    ]

    if choices_block:
        pieces.append(f"Options:\n{choices_block}")

    pieces.extend([
        f"Prior model response: {prior_response}",
        f"Claims to verify:\n{claims_block}",
    ])

    return "\n\n".join(pieces)

def format_reasoned_target_verdict_only(labels) -> str:
    ordered = sorted(labels, key=lambda x: int(x["claim_id"]))
    lines = ["Reasoning:"]
    for row in ordered:
        value = normalize_hallucination_label(row.get("hallucination"))
        if value is None:
            continue
        lines.append(f"Verdict: {int(row['claim_id'])} {'TRUE' if value else 'FALSE'}")
    return "\n".join(lines)

def add_verification_format_example(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "Example output:\n"
        "0\tFALSE\n"
        "1\tTRUE\n\n"
        "Now output only the verification lines for the actual claims above."
    )


def parse_verifier_output(raw_text: str, expected_claim_ids: List[int]) -> Dict[int, bool]:
    predictions: Dict[int, bool] = {}
    lines = [line.strip() for line in str(raw_text).splitlines() if line.strip()]

    for line in lines:
        match = re.match(r"^\s*(\d+)\s*[:\-|,\t ]+\s*([A-Za-z_ ]+)\s*$", line)
        if not match:
            continue
        claim_id = int(match.group(1))
        hallucination = normalize_hallucination_label(match.group(2))
        if hallucination is not None:
            predictions[claim_id] = hallucination

    if len(predictions) < len(expected_claim_ids):
        bool_hits = []
        for line in lines:
            line_norm = line.strip().upper()
            if "TRUE_OR_FALSE" in line_norm or "FALSE_OR_TRUE" in line_norm:
                continue
            if re.fullmatch(r"(?:\d+\s*)?(?:TRUE|HALLUCINATED|UNSUPPORTED|INCORRECT|WRONG)", line_norm):
                bool_hits.append(True)
            elif re.fullmatch(r"(?:\d+\s*)?(?:FALSE|CORRECT|SUPPORTED)", line_norm):
                bool_hits.append(False)

        for idx, claim_id in enumerate(expected_claim_ids):
            if claim_id in predictions:
                continue
            if idx < len(bool_hits):
                predictions[claim_id] = bool_hits[idx]

    for claim_id in expected_claim_ids:
        if claim_id not in predictions:
            predictions[claim_id] = False

    return predictions
