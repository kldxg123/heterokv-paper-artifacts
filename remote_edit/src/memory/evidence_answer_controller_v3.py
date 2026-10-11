"""Lightweight evidence-unit answer controller for Workflow4.

The controller is intentionally small and auditable.  It never reads gold
answers at inference time; it scores the raw model answer and short candidates
extracted from retrieved evidence units.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


STOPWORDS = {
    "a",
    "an",
    "are",
    "at",
    "be",
    "can",
    "does",
    "for",
    "has",
    "how",
    "in",
    "is",
    "it",
    "kind",
    "long",
    "many",
    "much",
    "of",
    "on",
    "or",
    "shown",
    "spell",
    "that",
    "the",
    "there",
    "this",
    "what",
    "who",
    "with",
}

DRINK_TERMS = {"ale", "lager", "stout", "ipa", "porter", "pilsner", "beer", "wine", "whisky", "whiskey"}

FEATURE_NAMES = [
    "is_raw_pred",
    "is_evidence",
    "source_rank_inv",
    "source_score",
    "candidate_len_norm",
    "candidate_token_count_norm",
    "candidate_in_raw_pred",
    "raw_pred_in_candidate",
    "question_overlap",
    "has_digit",
    "has_percent",
    "has_money",
    "has_duration_unit",
    "is_onoff",
    "is_drink_term",
    "wants_numeric_x_has_digit",
    "wants_duration_x_has_duration",
    "wants_brand_x_brand_note",
    "wants_drink_x_drink_term",
    "wants_onoff_x_is_onoff",
    "wants_text_x_ocr",
    "label_is_ocr",
    "label_is_page",
    "label_is_summary",
    "label_is_image_classes",
    "source_is_generated_note",
]


def normalize(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9@%.$+-]+", str(text or "").lower()))


def _tokens(text: Any) -> List[str]:
    return [token for token in normalize(text).split() if len(token) > 1 and token not in STOPWORDS]


def _numbers(text: Any) -> List[str]:
    return re.findall(r"\d+(?:\.\d+)?", str(text or ""))


def _packet_value(packet: Any, key: str, default: Any = None) -> Any:
    if isinstance(packet, dict):
        return packet.get(key, default)
    return getattr(packet, key, default)


def packet_label(packet: Any) -> str:
    meta = _packet_value(packet, "metadata", {}) or {}
    return str(meta.get("label") or "")


def packet_text(packet: Any) -> str:
    if isinstance(packet, dict):
        return str(packet.get("text_preview") or packet.get("text") or "")
    return str(getattr(packet, "text", "") or "")


def packet_score(packet: Any) -> float:
    try:
        return float(_packet_value(packet, "score", 0.0))
    except Exception:
        return 0.0


def _strip_prefix(text: str) -> str:
    value = str(text or "")
    for prefix in [
        "visible OCR chunk:",
        "visible OCR tokens:",
        "OCR line:",
        "page header/top lines:",
        "page footer/bottom lines:",
        "page line index:",
        "page numeric/date/time/value profile:",
        "memory handle:",
        "image classes:",
    ]:
        value = value.replace(prefix, " ")
    value = re.sub(
        r"\b(?:ocr_token_chunk|ocr_tokens_all|ocr_line_group|ocr_field_group|page_header|page_footer|page_line_index|image_classes)\b",
        " ",
        value,
    )
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ;|")


def question_profile(question: str) -> Dict[str, Any]:
    q = str(question or "").lower()
    return {
        "terms": set(_tokens(q)),
        "wants_numeric": any(cue in q for cue in ["how many", "number", "percent", "price", "time", "aged", "value", "amount"]),
        "wants_duration": any(cue in q for cue in ["how long", "aged", "age of", "years old"]),
        "wants_brand": "brand" in q,
        "wants_drink": any(cue in q for cue in ["kind of beer", "type of beer", "kind of drink"]),
        "wants_onoff": "on or off" in q or "switches on" in q or "switches off" in q,
        "wants_text": any(cue in q for cue in ["text spell", "word spell", "small white text", "what does the text", "what does the sign", "read"]),
    }


def _add_candidate(
    out: List[Dict[str, Any]],
    seen: set[str],
    *,
    text: str,
    source: str,
    label: str,
    score: float,
    rank: int,
) -> None:
    value = re.sub(r"\s+", " ", str(text or "")).strip(" .,:;\"'")
    if not value:
        return
    norm = normalize(value)
    if not norm or norm in seen:
        return
    seen.add(norm)
    out.append({"text": value, "source": source, "label": label, "source_score": float(score), "rank": int(rank)})


def build_answer_candidates(question: str, raw_pred: str, packets: Sequence[Any], *, max_candidates: int = 80) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()
    _add_candidate(candidates, seen, text=raw_pred, source="raw_pred", label="raw_pred", score=1.0, rank=0)

    raw_text = " ".join(packet_text(packet) for packet in packets)
    qprof = question_profile(question)
    if qprof["wants_onoff"]:
        off_count = len(re.findall(r"\boff\b", raw_text, flags=re.IGNORECASE))
        on_count = len(re.findall(r"\bon\b", raw_text, flags=re.IGNORECASE))
        if off_count or on_count:
            _add_candidate(candidates, seen, text="off" if off_count >= on_count else "on", source="rule_onoff", label="rule", score=0.95, rank=1)
    if qprof["wants_duration"]:
        match = re.search(r"\b(\d{1,3})\s*(years?|yrs?|months?|days?|hours?|minutes?)\b", raw_text, flags=re.IGNORECASE)
        if match:
            _add_candidate(candidates, seen, text=f"{match.group(1)} {match.group(2)}", source="rule_duration", label="rule", score=0.92, rank=1)
    if qprof["wants_numeric"]:
        ocr_match = re.search(r"visible OCR tokens:\s*([0-9][0-9A-Za-z.%:-]*)", raw_text, flags=re.IGNORECASE)
        if ocr_match:
            _add_candidate(candidates, seen, text=ocr_match.group(1), source="rule_numeric_ocr", label="rule", score=0.88, rank=1)
        for number in sorted(set(_numbers(raw_text)), key=lambda item: (-len(item), raw_text.find(item)))[:8]:
            _add_candidate(candidates, seen, text=number, source="number_from_units", label="rule", score=0.72, rank=2)
    if qprof["wants_brand"]:
        for match in re.finditer(r"-\s*([A-Za-z0-9][A-Za-z0-9 .&'/+-]{0,32}?)\s*\(\s*\(?\s*Brand name", raw_text, flags=re.IGNORECASE):
            _add_candidate(candidates, seen, text=match.group(1), source="brand_note", label="generated_note", score=0.90, rank=1)
    if qprof["wants_drink"]:
        lower = raw_text.lower()
        for term in DRINK_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", lower):
                _add_candidate(candidates, seen, text=term, source="drink_term", label="rule", score=0.80, rank=1)

    for rank, packet in enumerate(packets, start=1):
        raw = packet_text(packet)
        label = packet_label(packet)
        score = packet_score(packet)
        chunks = [_strip_prefix(raw)]
        if "generated visual notes:" in raw:
            chunks.append(raw.split("generated visual notes:", 1)[1])
        for chunk in chunks:
            for part in re.split(r"\s*(?:\|\||\||;|\n|,| - )\s*", chunk):
                cleaned = re.sub(r"\s+", " ", part).strip(" .:-()[]{}\"'")
                if not cleaned or len(cleaned) > 72:
                    continue
                _add_candidate(candidates, seen, text=cleaned, source="evidence_segment", label=label, score=score, rank=rank)
    return candidates[:max_candidates]


def candidate_features(question: str, raw_pred: str, candidate: Dict[str, Any]) -> List[float]:
    qprof = question_profile(question)
    text = str(candidate.get("text") or "")
    norm_text = normalize(text)
    norm_raw = normalize(raw_pred)
    terms = set(_tokens(text))
    q_terms = qprof["terms"]
    label = str(candidate.get("label") or "")
    source = str(candidate.get("source") or "")
    token_count = len(norm_text.split())
    has_digit = bool(re.search(r"\d", text))
    has_percent = "%" in text
    has_money = "$" in text or any(sym in text.lower() for sym in ["usd", "eur", "£"])
    has_duration = bool(re.search(r"\b(year|years|yr|yrs|month|months|day|days|hour|hours|minute|minutes)\b", text, flags=re.IGNORECASE))
    is_onoff = normalize(text) in {"on", "off"}
    is_drink = normalize(text) in DRINK_TERMS
    is_ocr = label.startswith("ocr") or "ocr" in source
    is_page = label.startswith("page_")
    is_summary = label == "searchable_summary"
    is_image_classes = label == "image_classes"
    is_raw = source == "raw_pred"
    try:
        rank = int(candidate.get("rank", 0))
    except Exception:
        rank = 0
    try:
        source_score = float(candidate.get("source_score", 0.0))
    except Exception:
        source_score = 0.0
    return [
        1.0 if is_raw else 0.0,
        0.0 if is_raw else 1.0,
        1.0 / max(1, rank + 1),
        source_score,
        min(1.0, len(norm_text) / 48.0),
        min(1.0, token_count / 8.0),
        1.0 if norm_text and norm_text in norm_raw else 0.0,
        1.0 if norm_raw and norm_raw in norm_text else 0.0,
        len(q_terms & terms) / max(1, len(q_terms)),
        1.0 if has_digit else 0.0,
        1.0 if has_percent else 0.0,
        1.0 if has_money else 0.0,
        1.0 if has_duration else 0.0,
        1.0 if is_onoff else 0.0,
        1.0 if is_drink else 0.0,
        1.0 if qprof["wants_numeric"] and has_digit else 0.0,
        1.0 if qprof["wants_duration"] and has_duration else 0.0,
        1.0 if qprof["wants_brand"] and source == "brand_note" else 0.0,
        1.0 if qprof["wants_drink"] and is_drink else 0.0,
        1.0 if qprof["wants_onoff"] and is_onoff else 0.0,
        1.0 if qprof["wants_text"] and is_ocr else 0.0,
        1.0 if is_ocr else 0.0,
        1.0 if is_page else 0.0,
        1.0 if is_summary else 0.0,
        1.0 if is_image_classes else 0.0,
        1.0 if source == "brand_note" else 0.0,
    ]


def load_controller(path: str | Path | None) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    model = json.loads(p.read_text(encoding="utf-8"))
    if model.get("feature_names") != FEATURE_NAMES:
        raise ValueError("unit answer controller feature mismatch")
    return model


def _linear_score(weights: Sequence[float], features: Sequence[float], bias: float) -> float:
    return float(bias) + sum(float(w) * float(x) for w, x in zip(weights, features))


def apply_controller(
    *,
    question: str,
    raw_pred: str,
    packets: Sequence[Any],
    model: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    if not model:
        return raw_pred, {"changed": False, "reason": "model_missing"}
    candidates = build_answer_candidates(question, raw_pred, packets)
    if not candidates:
        return raw_pred, {"changed": False, "reason": "no_candidates"}
    weights = model.get("weights") or [0.0] * len(FEATURE_NAMES)
    bias = float(model.get("bias", 0.0))
    margin_threshold = float(model.get("margin_threshold", 0.0))
    scored = []
    raw_score = None
    for candidate in candidates:
        features = candidate_features(question, raw_pred, candidate)
        score = _linear_score(weights, features, bias)
        item = dict(candidate)
        item["model_score"] = score
        scored.append(item)
        if candidate.get("source") == "raw_pred":
            raw_score = score
    scored.sort(key=lambda item: (-float(item["model_score"]), item.get("rank", 99), len(str(item.get("text") or ""))))
    top = scored[0]
    raw_score = float(raw_score if raw_score is not None else -1e9)
    margin = float(top["model_score"]) - raw_score
    if top.get("source") == "raw_pred":
        return raw_pred, {
            "changed": False,
            "reason": "raw_pred_selected",
            "top": top,
            "raw_score": raw_score,
            "margin": margin,
        }
    if margin < margin_threshold:
        return raw_pred, {
            "changed": False,
            "reason": "margin_below_threshold",
            "top": top,
            "raw_score": raw_score,
            "margin": margin,
            "margin_threshold": margin_threshold,
        }
    return str(top.get("text") or raw_pred), {
        "changed": True,
        "reason": "learned_controller_override",
        "top": top,
        "raw_pred": raw_pred,
        "raw_score": raw_score,
        "margin": margin,
        "margin_threshold": margin_threshold,
    }

