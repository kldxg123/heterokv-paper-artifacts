"""Workflow4 / HeteroKV-v3 multimodal context memory primitives.

This module is intentionally independent from the HeteroKV-v2 cache manager.
It provides a default-off, CPU-testable interface for storing and retrieving
text, KV-page, OCR, and image-region evidence.  The v3 path is evidence
grounded: generated answer control may only use retrieved ``EvidencePacket``
objects, never hidden gold spans or a decode-time full-prompt scan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TEXT = "text"
KV_PAGE = "kv_page"
IMAGE = "image"
IMAGE_REGION = "image_region"
OCR_BLOCK = "ocr_block"
CHART_REGION = "chart_region"
TABLE_REGION = "table_region"

SUPPORTED_MODALITIES = {
    TEXT,
    KV_PAGE,
    IMAGE,
    IMAGE_REGION,
    OCR_BLOCK,
    CHART_REGION,
    TABLE_REGION,
}

EXACT_COPY = "exact_copy"
EXTRACTIVE_QA = "extractive_qa"
ABSTRACTIVE_QA = "abstractive_qa"
VISUAL_QA = "visual_qa"
CHART_COUNT = "chart_count"
NO_ANSWER = "no_answer"

ANSWER_POLICIES = {
    EXACT_COPY,
    EXTRACTIVE_QA,
    ABSTRACTIVE_QA,
    VISUAL_QA,
    CHART_COUNT,
    NO_ANSWER,
}

_ORACLE_KEYS = {
    "gold",
    "gold_answer",
    "gold_answers",
    "answer_start",
    "answer_end",
    "oracle",
    "oracle_chunk",
    "oracle_span",
    "target",
    "target_answer",
}


def _stable_hash(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", str(text or "").lower())


def _safe_float_list(values: Optional[Sequence[float]]) -> Optional[List[float]]:
    if values is None:
        return None
    return [float(v) for v in values]


def _cosine(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _check_no_oracle_metadata(metadata: Mapping[str, Any]) -> None:
    lowered = {str(key).lower() for key in metadata.keys()}
    blocked = sorted(lowered & _ORACLE_KEYS)
    if blocked:
        raise ValueError(f"v3 memory metadata contains oracle-only keys: {blocked}")


@dataclass(frozen=True)
class BBox:
    """Image/document bounding box.

    Coordinates may be pixels or normalized values.  The provenance fields on
    ``MemoryUnit`` specify the source image/page; this class only stores the
    rectangle itself.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    normalized: bool = False

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("BBox must satisfy x1 >= x0 and y1 >= y0")
        if self.normalized:
            for value in (self.x0, self.y0, self.x1, self.y1):
                if value < 0.0 or value > 1.0:
                    raise ValueError("Normalized BBox coordinates must be in [0, 1]")

    def to_list(self) -> List[float]:
        return [float(self.x0), float(self.y0), float(self.x1), float(self.y1)]


@dataclass(frozen=True)
class MemoryUnit:
    """A modality-agnostic external context memory unit."""

    unit_id: str
    modality: str
    source_id: str
    text: str = ""
    token_span: Optional[Tuple[int, int]] = None
    bbox: Optional[BBox] = None
    embedding: Optional[List[float]] = None
    kv_ref: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_step: int = 0
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.unit_id:
            raise ValueError("MemoryUnit requires a non-empty unit_id")
        if self.modality not in SUPPORTED_MODALITIES:
            raise ValueError(f"Unsupported modality: {self.modality}")
        _check_no_oracle_metadata(self.metadata)

    def searchable_text(self) -> str:
        parts = [
            self.text,
            str(self.metadata.get("caption", "")),
            str(self.metadata.get("ocr_text", "")),
            str(self.metadata.get("label", "")),
            str(self.metadata.get("table_text", "")),
        ]
        return _normalize_text(" ".join(part for part in parts if part))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["bbox"] = None if self.bbox is None else {
            "x0": self.bbox.x0,
            "y0": self.bbox.y0,
            "x1": self.bbox.x1,
            "y1": self.bbox.y1,
            "normalized": self.bbox.normalized,
        }
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryUnit":
        bbox_data = data.get("bbox")
        bbox = None
        if isinstance(bbox_data, Mapping):
            bbox = BBox(
                x0=float(bbox_data["x0"]),
                y0=float(bbox_data["y0"]),
                x1=float(bbox_data["x1"]),
                y1=float(bbox_data["y1"]),
                normalized=bool(bbox_data.get("normalized", False)),
            )
        span = data.get("token_span")
        token_span = None if span is None else (int(span[0]), int(span[1]))
        return cls(
            unit_id=str(data["unit_id"]),
            modality=str(data["modality"]),
            source_id=str(data["source_id"]),
            text=str(data.get("text", "")),
            token_span=token_span,
            bbox=bbox,
            embedding=_safe_float_list(data.get("embedding")),
            kv_ref=dict(data.get("kv_ref") or {}) or None,
            metadata=dict(data.get("metadata") or {}),
            created_step=int(data.get("created_step", 0)),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass(frozen=True)
class RetrievalQuery:
    query_text: str
    query_embedding: Optional[List[float]] = None
    allowed_modalities: Optional[List[str]] = None
    policy_hint: str = ABSTRACTIVE_QA
    top_k: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.policy_hint not in ANSWER_POLICIES:
            raise ValueError(f"Unsupported answer policy: {self.policy_hint}")
        if self.allowed_modalities is not None:
            unknown = sorted(set(self.allowed_modalities) - SUPPORTED_MODALITIES)
            if unknown:
                raise ValueError(f"Unsupported modalities in query: {unknown}")
        _check_no_oracle_metadata(self.metadata)


@dataclass(frozen=True)
class EvidencePacket:
    packet_id: str
    unit_id: str
    modality: str
    source_id: str
    score: float
    score_breakdown: Dict[str, float]
    text: str
    token_span: Optional[Tuple[int, int]]
    bbox: Optional[BBox]
    kv_ref: Optional[Dict[str, Any]]
    provenance: Dict[str, Any]
    policy_hint: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_block(self, max_chars: int = 900) -> str:
        label = self.provenance.get("label") or self.source_id
        clipped = _normalize_text(self.text)[:max_chars]
        bbox = ""
        if self.bbox is not None:
            bbox = f" bbox={self.bbox.to_list()}"
        return (
            f"[Evidence {self.packet_id} | {self.modality} | source={label}{bbox} | "
            f"score={self.score:.3f}]\n{clipped}"
        )


@dataclass(frozen=True)
class FusionConfig:
    lexical_weight: float = 0.38
    dense_weight: float = 0.34
    metadata_weight: float = 0.12
    history_weight: float = 0.08
    confidence_weight: float = 0.08


class ContextMemoryIndex:
    """Small deterministic multimodal index for v3 development.

    It is not a FAISS/CLIP replacement.  It defines the repo-local interface
    that later dense, visual, OCR, and Qwen-VL adapters can implement against.
    """

    def __init__(self, *, fusion: Optional[FusionConfig] = None) -> None:
        self.fusion = fusion or FusionConfig()
        self._units: Dict[str, MemoryUnit] = {}
        self._inverted: Dict[str, set[str]] = {}

    @property
    def units(self) -> Dict[str, MemoryUnit]:
        return dict(self._units)

    def add_unit(self, unit: MemoryUnit) -> MemoryUnit:
        if unit.unit_id in self._units:
            raise ValueError(f"Duplicate MemoryUnit id: {unit.unit_id}")
        self._units[unit.unit_id] = unit
        for token in set(_tokens(unit.searchable_text())):
            self._inverted.setdefault(token, set()).add(unit.unit_id)
        return unit

    def add_text_span(
        self,
        *,
        source_id: str,
        text: str,
        token_span: Optional[Tuple[int, int]] = None,
        embedding: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryUnit:
        unit_id = f"text:{source_id}:{token_span or _stable_hash(text)[:12]}"
        return self.add_unit(
            MemoryUnit(
                unit_id=unit_id,
                modality=TEXT,
                source_id=source_id,
                text=text,
                token_span=token_span,
                embedding=_safe_float_list(embedding),
                metadata=dict(metadata or {}),
            )
        )

    def add_kv_page(
        self,
        *,
        source_id: str,
        page_id: str,
        text_preview: str,
        token_span: Tuple[int, int],
        kv_ref: Mapping[str, Any],
        embedding: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryUnit:
        return self.add_unit(
            MemoryUnit(
                unit_id=f"kv:{source_id}:{page_id}",
                modality=KV_PAGE,
                source_id=source_id,
                text=text_preview,
                token_span=token_span,
                embedding=_safe_float_list(embedding),
                kv_ref=dict(kv_ref),
                metadata=dict(metadata or {}),
            )
        )

    def add_image(
        self,
        *,
        source_id: str,
        image_id: str,
        caption: str = "",
        embedding: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryUnit:
        md = dict(metadata or {})
        if caption:
            md.setdefault("caption", caption)
        return self.add_unit(
            MemoryUnit(
                unit_id=f"image:{source_id}:{image_id}",
                modality=IMAGE,
                source_id=source_id,
                text=caption,
                embedding=_safe_float_list(embedding),
                metadata=md,
            )
        )

    def add_image_region(
        self,
        *,
        source_id: str,
        image_id: str,
        region_id: str,
        bbox: BBox,
        caption: str = "",
        embedding: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryUnit:
        md = dict(metadata or {})
        if caption:
            md.setdefault("caption", caption)
        md.setdefault("image_id", image_id)
        return self.add_unit(
            MemoryUnit(
                unit_id=f"region:{source_id}:{image_id}:{region_id}",
                modality=IMAGE_REGION,
                source_id=source_id,
                text=caption,
                bbox=bbox,
                embedding=_safe_float_list(embedding),
                metadata=md,
            )
        )

    def add_chart_region(
        self,
        *,
        source_id: str,
        image_id: str,
        region_id: str,
        bbox: BBox,
        caption: str = "",
        embedding: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryUnit:
        md = dict(metadata or {})
        if caption:
            md.setdefault("caption", caption)
        md.setdefault("image_id", image_id)
        md.setdefault("region_kind", "chart")
        return self.add_unit(
            MemoryUnit(
                unit_id=f"chart:{source_id}:{image_id}:{region_id}",
                modality=CHART_REGION,
                source_id=source_id,
                text=caption,
                bbox=bbox,
                embedding=_safe_float_list(embedding),
                metadata=md,
            )
        )

    def add_table_region(
        self,
        *,
        source_id: str,
        image_id: str,
        region_id: str,
        bbox: BBox,
        table_text: str = "",
        caption: str = "",
        embedding: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryUnit:
        md = dict(metadata or {})
        if caption:
            md.setdefault("caption", caption)
        if table_text:
            md.setdefault("table_text", table_text)
        md.setdefault("image_id", image_id)
        md.setdefault("region_kind", "table")
        return self.add_unit(
            MemoryUnit(
                unit_id=f"table:{source_id}:{image_id}:{region_id}",
                modality=TABLE_REGION,
                source_id=source_id,
                text=table_text or caption,
                bbox=bbox,
                embedding=_safe_float_list(embedding),
                metadata=md,
            )
        )

    def add_ocr_block(
        self,
        *,
        source_id: str,
        image_id: str,
        block_id: str,
        text: str,
        bbox: Optional[BBox] = None,
        embedding: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryUnit:
        md = dict(metadata or {})
        md.setdefault("image_id", image_id)
        md.setdefault("ocr_text", text)
        return self.add_unit(
            MemoryUnit(
                unit_id=f"ocr:{source_id}:{image_id}:{block_id}",
                modality=OCR_BLOCK,
                source_id=source_id,
                text=text,
                bbox=bbox,
                embedding=_safe_float_list(embedding),
                metadata=md,
            )
        )

    def search(self, query: RetrievalQuery) -> List[EvidencePacket]:
        q_tokens = set(_tokens(query.query_text))
        allowed = None if query.allowed_modalities is None else set(query.allowed_modalities)
        candidate_ids = set()
        for token in q_tokens:
            candidate_ids.update(self._inverted.get(token, set()))
        if not candidate_ids:
            candidate_ids = set(self._units.keys())

        packets: List[EvidencePacket] = []
        for unit_id in candidate_ids:
            unit = self._units[unit_id]
            if allowed is not None and unit.modality not in allowed:
                continue
            packet = self._score_unit(unit, query, q_tokens)
            if packet.score > 0.0:
                packets.append(packet)
        packets.sort(key=lambda item: (-item.score, item.unit_id))
        return packets[: max(1, int(query.top_k))]

    def _score_unit(
        self,
        unit: MemoryUnit,
        query: RetrievalQuery,
        q_tokens: set[str],
    ) -> EvidencePacket:
        unit_tokens = set(_tokens(unit.searchable_text()))
        lexical = 0.0
        if q_tokens:
            lexical = len(q_tokens & unit_tokens) / max(1, len(q_tokens))
        dense = max(0.0, _cosine(query.query_embedding, unit.embedding))
        metadata = self._metadata_score(unit, query)
        history = float(unit.metadata.get("historical_attention", 0.0) or 0.0)
        history = max(0.0, min(1.0, history))
        confidence = max(0.0, min(1.0, float(unit.confidence)))
        breakdown = {
            "lexical": lexical,
            "dense": dense,
            "metadata": metadata,
            "history": history,
            "confidence": confidence,
        }
        score = (
            self.fusion.lexical_weight * lexical
            + self.fusion.dense_weight * dense
            + self.fusion.metadata_weight * metadata
            + self.fusion.history_weight * history
            + self.fusion.confidence_weight * confidence
        )
        packet_id = f"ev:{_stable_hash(unit.unit_id + query.query_text)[:12]}"
        provenance = {
            "unit_id": unit.unit_id,
            "source_id": unit.source_id,
            "label": unit.metadata.get("label"),
            "created_step": unit.created_step,
            "from_retrieved_unit": True,
        }
        return EvidencePacket(
            packet_id=packet_id,
            unit_id=unit.unit_id,
            modality=unit.modality,
            source_id=unit.source_id,
            score=float(score),
            score_breakdown=breakdown,
            text=unit.searchable_text(),
            token_span=unit.token_span,
            bbox=unit.bbox,
            kv_ref=unit.kv_ref,
            provenance=provenance,
            policy_hint=query.policy_hint,
            metadata=dict(unit.metadata),
        )

    @staticmethod
    def _metadata_score(unit: MemoryUnit, query: RetrievalQuery) -> float:
        if not query.metadata:
            return 0.0
        hits = 0
        total = 0
        for key, value in query.metadata.items():
            if key in _ORACLE_KEYS:
                continue
            total += 1
            if str(unit.metadata.get(key, "")).lower() == str(value).lower():
                hits += 1
        return 0.0 if total == 0 else hits / total


@dataclass(frozen=True)
class AnswerControlDecision:
    policy: str
    should_override: bool
    answer_text: Optional[str]
    generation_context: str
    evidence_ids: List[str]
    reason: str


class AnswerControllerV3:
    """Conservative answer controller over retrieved v3 evidence packets."""

    def __init__(
        self,
        *,
        override_threshold: float = 0.72,
        max_context_chars: int = 2800,
    ) -> None:
        self.override_threshold = float(override_threshold)
        self.max_context_chars = int(max_context_chars)

    def infer_policy(
        self,
        *,
        dataset: Optional[str] = None,
        task_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> str:
        joined = " ".join(
            str(item or "").lower()
            for item in (dataset, task_type, prompt[:512] if prompt else "")
        )
        if any(token in joined for token in ["docvqa", "textvqa", "image", "figure"]):
            return VISUAL_QA
        if any(token in joined for token in ["chartqa", "chart", "count", "how many"]):
            return CHART_COUNT
        if any(token in joined for token in ["niah", "needle", "passkey", "exact"]):
            return EXACT_COPY
        if any(token in joined for token in ["passage_retrieval", "paragraph"]):
            return EXTRACTIVE_QA
        return ABSTRACTIVE_QA

    def decide(
        self,
        *,
        policy: str,
        evidence: Sequence[EvidencePacket],
        lm_answer: Optional[str] = None,
    ) -> AnswerControlDecision:
        if policy not in ANSWER_POLICIES:
            raise ValueError(f"Unsupported v3 answer policy: {policy}")
        packets = sorted(evidence, key=lambda item: (-item.score, item.unit_id))
        context = self.build_generation_context(packets)
        evidence_ids = [packet.packet_id for packet in packets]
        if not packets:
            return AnswerControlDecision(
                policy=NO_ANSWER,
                should_override=False,
                answer_text=lm_answer,
                generation_context="",
                evidence_ids=[],
                reason="no_retrieved_evidence",
            )
        top = packets[0]
        if policy == EXACT_COPY and top.score >= self.override_threshold:
            answer = _normalize_text(top.metadata.get("answer_text") or top.text)
            return AnswerControlDecision(
                policy=policy,
                should_override=True,
                answer_text=answer,
                generation_context=context,
                evidence_ids=evidence_ids,
                reason="exact_copy_high_confidence_retrieved_evidence",
            )
        return AnswerControlDecision(
            policy=policy,
            should_override=False,
            answer_text=lm_answer,
            generation_context=context,
            evidence_ids=evidence_ids,
            reason="evidence_context_only_no_pointer_override",
        )

    def build_generation_context(self, evidence: Sequence[EvidencePacket]) -> str:
        blocks: List[str] = []
        used = 0
        for packet in evidence:
            block = packet.to_prompt_block()
            if used + len(block) > self.max_context_chars and blocks:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    def build_augmented_prompt(
        self,
        *,
        question: str,
        evidence: Sequence[EvidencePacket],
        policy: str = ABSTRACTIVE_QA,
        base_instruction: Optional[str] = None,
    ) -> str:
        """Build a retrieval-grounded prompt for downstream LLM/VLM generation.

        The method only consumes already retrieved evidence packets.  It does
        not inspect a full prompt, labels, or hidden answer spans.
        """

        if policy not in ANSWER_POLICIES:
            raise ValueError(f"Unsupported v3 answer policy: {policy}")
        instruction = base_instruction or (
            "Answer using only the retrieved evidence when it is sufficient. "
            "If the evidence is insufficient, say that the evidence is insufficient."
        )
        context = self.build_generation_context(evidence)
        return (
            f"{instruction}\n\n"
            f"Policy: {policy}\n\n"
            f"Retrieved evidence:\n{context or '[none]'}\n\n"
            f"Question:\n{_normalize_text(question)}\n\n"
            "Answer:"
        )
