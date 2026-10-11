"""Original local scorers extracted without model imports; not official evaluators."""
from __future__ import annotations
import re
import string
from typing import Any, Dict, List, Optional, Iterable

def _normalize_answer(text: Any) -> str:
    value = str(text or "").lower().strip()
    value = value.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    translator = str.maketrans({ch: " " for ch in string.punctuation})
    value = value.translate(translator)
    value = re.sub(r"\s+", " ", value).strip()
    return value

def _tokens(text: Any) -> List[str]:
    normalized = _normalize_answer(text)
    return normalized.split() if normalized else []

def _token_f1(pred: str, answer: str) -> float:
    pred_tokens = _tokens(pred)
    answer_tokens = _tokens(answer)
    if not pred_tokens or not answer_tokens:
        return float(pred_tokens == answer_tokens)
    common: Dict[str, int] = {}
    for tok in answer_tokens:
        common[tok] = common.get(tok, 0) + 1
    overlap = 0
    for tok in pred_tokens:
        if common.get(tok, 0) > 0:
            overlap += 1
            common[tok] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / max(1e-12, precision + recall)

def _float_value(text: Any) -> Optional[float]:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(text or "").replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None

def _relaxed_numeric_match(pred: str, answer: str) -> bool:
    pred_value = _float_value(pred)
    answer_value = _float_value(answer)
    if pred_value is None or answer_value is None:
        return False
    if answer_value == 0:
        return abs(pred_value - answer_value) <= 1e-6
    return abs(pred_value - answer_value) / abs(answer_value) <= 0.05

def _score_prediction(dataset: str, pred: str, answers: List[str]) -> Dict[str, Any]:
    normalized_pred = _normalize_answer(pred)
    normalized_answers = [_normalize_answer(answer) for answer in answers]
    exact_matches = [normalized_pred == answer and bool(answer) for answer in normalized_answers]
    relaxed_matches = [_relaxed_numeric_match(pred, answer) for answer in answers]
    contains_matches = [
        bool(answer) and (answer in normalized_pred or normalized_pred in answer)
        for answer in normalized_answers
    ]
    best_f1 = max([_token_f1(pred, answer) for answer in answers] or [0.0])
    if dataset == "textvqa" and len(answers) > 1:
        vqa_soft = min(1.0, sum(1 for hit in exact_matches if hit) / 3.0)
    else:
        vqa_soft = 1.0 if any(exact_matches) or any(relaxed_matches) else 0.0
    return {
        "exact_match": bool(any(exact_matches)),
        "relaxed_numeric_match": bool(any(relaxed_matches)),
        "contains_match": bool(any(contains_matches)),
        "token_f1": best_f1,
        "vqa_soft_score": vqa_soft,
        "normalized_pred": normalized_pred,
        "normalized_answers": normalized_answers[:10],
    }

def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())

def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]

def anls(prediction: Any, answers: Iterable[Any], threshold: float = 0.5) -> float:
    pred = normalize(prediction)
    best = 0.0
    for answer in answers:
        gold = normalize(answer)
        denominator = max(len(pred), len(gold))
        similarity = 1.0 if denominator == 0 else 1.0 - levenshtein(pred, gold) / denominator
        best = max(best, similarity if similarity >= threshold else 0.0)
    return float(best)
