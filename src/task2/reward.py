from typing import Dict, Iterable, List


def score_claim(predicted_verdict: str, gold_verdict: str) -> float:
    return 1.0 if str(predicted_verdict).upper() == str(gold_verdict).upper() else -1.0


def score_predictions(predictions: Dict[int, str], gold_labels: Dict[int, str]) -> Dict:
    rewards = []
    for claim_id, gold in sorted(gold_labels.items()):
        pred = predictions.get(claim_id, "CORRECT")
        rewards.append(score_claim(pred, gold))

    total_reward = sum(rewards)
    normalized_reward = total_reward / max(len(rewards), 1)
    return {
        "claim_rewards": rewards,
        "total_reward": total_reward,
        "normalized_reward": normalized_reward,
    }


def relative_group_advantages(rewards: Iterable[float], eps: float = 1e-6) -> List[float]:
    values = list(float(x) for x in rewards)
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    std = var ** 0.5
    if std < eps:
        return [0.0 for _ in values]
    return [(x - mean) / std for x in values]
