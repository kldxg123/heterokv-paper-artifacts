"""Workflow4 document-image memory adapters.

The functions in this module turn visible document-image annotations such as
OCR blocks, captions, image regions, chart/table regions, and optional text
snippets into v3 `MemoryUnit` objects.  They do not load a vision model and they
do not read gold answers.  Real DocVQA/ChartQA/TextVQA runners can use this as
the evidence-indexing layer before handing retrieved packets to Qwen2.5-VL or a
text-only fallback generator.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.memory.context_memory_v3 import (
    ABSTRACTIVE_QA,
    CHART_COUNT,
    CHART_REGION,
    IMAGE,
    IMAGE_REGION,
    OCR_BLOCK,
    TABLE_REGION,
    VISUAL_QA,
    BBox,
    ContextMemoryIndex,
    EvidencePacket,
    RetrievalQuery,
)


DOCVQA = "docvqa"
CHARTQA = "chartqa"
TEXTVQA = "textvqa"

DOCUMENT_DATASETS = {DOCVQA, CHARTQA, TEXTVQA}
VISUAL_MODALITIES = [OCR_BLOCK, IMAGE_REGION, CHART_REGION, TABLE_REGION]


def hashed_multimodal_embedding(text: str, *, dim: int = 96) -> List[float]:
    """Process-stable development embedding over OCR/caption text."""

    vec = [0.0 for _ in range(max(1, int(dim)))]
    for token in re.findall(r"[A-Za-z0-9_\u4e00-\u9fff$€£.%/-]+", str(text or "").lower()):
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:8], byteorder="big", signed=False) % len(vec)
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [v / norm for v in vec]
    return vec


def bbox_from(value: Any) -> Optional[BBox]:
    """Parse a bbox from common list/dict schemas."""

    if value is None:
        return None
    if isinstance(value, BBox):
        return value
    if isinstance(value, Mapping):
        if {"x0", "y0", "x1", "y1"}.issubset(value.keys()):
            return BBox(
                float(value["x0"]),
                float(value["y0"]),
                float(value["x1"]),
                float(value["y1"]),
                normalized=bool(value.get("normalized", False)),
            )
        if {"left", "top", "right", "bottom"}.issubset(value.keys()):
            return BBox(
                float(value["left"]),
                float(value["top"]),
                float(value["right"]),
                float(value["bottom"]),
                normalized=bool(value.get("normalized", False)),
            )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 4:
        return BBox(float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    raise ValueError(f"unsupported bbox schema: {value!r}")


def infer_document_policy(*, dataset: str, question: str = "") -> str:
    ds = str(dataset or "").lower()
    q = str(question or "").lower()
    if "chart" in ds or any(word in q for word in ["how many", "count", "number of"]):
        return CHART_COUNT if "chart" in ds else VISUAL_QA
    if "docvqa" in ds or "textvqa" in ds or "vqa" in ds:
        return VISUAL_QA
    return ABSTRACTIVE_QA


def _visible_metadata(row: Mapping[str, Any], *, dataset: str, kind: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "dataset": dataset,
        "source_kind": kind,
    }
    for key in [
        "label",
        "page",
        "page_id",
        "image_id",
        "region_id",
        "region_kind",
        "crop_path",
        "ocr_engine",
        "ocr_confidence",
        "visible_value",
    ]:
        if key in row and row[key] is not None:
            metadata[key] = row[key]
    return metadata


def build_document_memory_index(
    *,
    record: Mapping[str, Any],
    dataset: str,
    embedding_dim: int = 96,
) -> Tuple[ContextMemoryIndex, Dict[str, Any]]:
    """Build a v3 visual/document memory index from visible fields only."""

    index = ContextMemoryIndex()
    dataset = str(dataset).lower()
    doc_id = str(record.get("document_id") or record.get("doc_id") or record.get("id") or "doc")
    image_id = str(record.get("image_id") or record.get("image") or record.get("image_path") or "image")
    caption = str(record.get("caption") or record.get("image_caption") or "")

    unit_counts = {IMAGE: 0, OCR_BLOCK: 0, IMAGE_REGION: 0, CHART_REGION: 0, TABLE_REGION: 0}
    if caption or image_id:
        index.add_image(
            source_id=doc_id,
            image_id=image_id,
            caption=caption,
            embedding=hashed_multimodal_embedding(caption or image_id, dim=embedding_dim),
            metadata={"dataset": dataset, "label": record.get("label", "image"), "image_id": image_id},
        )
        unit_counts[IMAGE] += 1

    for idx, block in enumerate(record.get("ocr_blocks") or record.get("ocr") or []):
        text = str(block.get("text") or block.get("ocr_text") or "")
        if not text.strip():
            continue
        block_id = str(block.get("block_id") or block.get("id") or idx)
        index.add_ocr_block(
            source_id=doc_id,
            image_id=str(block.get("image_id") or image_id),
            block_id=block_id,
            text=text,
            bbox=bbox_from(block.get("bbox")),
            embedding=hashed_multimodal_embedding(text, dim=embedding_dim),
            metadata=_visible_metadata(block, dataset=dataset, kind="ocr_block"),
        )
        unit_counts[OCR_BLOCK] += 1

    for idx, region in enumerate(record.get("regions") or record.get("image_regions") or []):
        bbox = bbox_from(region.get("bbox"))
        if bbox is None:
            continue
        caption_text = str(region.get("caption") or region.get("text") or "")
        region_id = str(region.get("region_id") or region.get("id") or idx)
        region_kind = str(region.get("region_kind") or region.get("type") or "image").lower()
        metadata = _visible_metadata(region, dataset=dataset, kind=f"{region_kind}_region")
        if region_kind == "chart":
            index.add_chart_region(
                source_id=doc_id,
                image_id=str(region.get("image_id") or image_id),
                region_id=region_id,
                bbox=bbox,
                caption=caption_text,
                embedding=hashed_multimodal_embedding(caption_text, dim=embedding_dim),
                metadata=metadata,
            )
            unit_counts[CHART_REGION] += 1
        elif region_kind == "table":
            table_text = str(region.get("table_text") or region.get("text") or caption_text)
            index.add_table_region(
                source_id=doc_id,
                image_id=str(region.get("image_id") or image_id),
                region_id=region_id,
                bbox=bbox,
                table_text=table_text,
                caption=caption_text,
                embedding=hashed_multimodal_embedding(f"{caption_text} {table_text}", dim=embedding_dim),
                metadata=metadata,
            )
            unit_counts[TABLE_REGION] += 1
        else:
            index.add_image_region(
                source_id=doc_id,
                image_id=str(region.get("image_id") or image_id),
                region_id=region_id,
                bbox=bbox,
                caption=caption_text,
                embedding=hashed_multimodal_embedding(caption_text, dim=embedding_dim),
                metadata=metadata,
            )
            unit_counts[IMAGE_REGION] += 1

    summary = {
        "dataset": dataset,
        "document_id": doc_id,
        "image_id": image_id,
        "embedding": f"hashed_multimodal_dim_{embedding_dim}",
        "unit_counts": unit_counts,
        "unit_count": sum(unit_counts.values()),
    }
    return index, summary


def retrieve_document_evidence(
    *,
    index: ContextMemoryIndex,
    question: str,
    dataset: str,
    top_k: int = 5,
    embedding_dim: int = 96,
) -> List[EvidencePacket]:
    packets = index.search(
        RetrievalQuery(
            query_text=question,
            query_embedding=hashed_multimodal_embedding(question, dim=embedding_dim),
            allowed_modalities=VISUAL_MODALITIES,
            policy_hint=infer_document_policy(dataset=dataset, question=question),
            top_k=max(int(top_k) * 3, int(top_k), 1),
        )
    )
    packets.sort(key=lambda packet: (-_document_rerank_score(question, packet), packet.unit_id))
    return packets[: max(1, int(top_k))]


def _document_rerank_score(question: str, packet: EvidencePacket) -> float:
    q_tokens = set(re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", str(question or "").lower()))
    p_tokens = set(re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", str(packet.text or "").lower()))
    score = float(packet.score)
    label = str(packet.metadata.get("label") or "")
    label_bonus = {
        "page_header": 0.11,
        "page_field_index": 0.10,
        "page_line_index": 0.08,
        "page_numeric_profile": 0.08,
        "page_footer": 0.04,
        "page_identity": -0.04,
    }.get(label, 0.0)
    score += label_bonus
    if q_tokens:
        score += 0.18 * len(q_tokens & p_tokens) / max(1, len(q_tokens))
    cue_groups = [
        ({"total", "amount", "price", "cost"}, {"total", "amount", "price", "cost", "$", "usd"}),
        ({"revenue", "sales", "profit"}, {"revenue", "sales", "profit", "m", "million"}),
        ({"name", "brand", "sign", "cafe"}, {"name", "brand", "sign", "cafe"}),
        ({"date", "due"}, {"date", "due"}),
    ]
    text_lower = str(packet.text or "").lower()
    for q_cues, p_cues in cue_groups:
        if q_tokens & q_cues and any(cue in text_lower for cue in p_cues):
            score += 0.16
    for number in re.findall(r"\d{2,4}", str(question or "")):
        if number in text_lower:
            score += 0.18
    if any(cue in q_tokens for cue in {"name", "company", "foundation", "university", "address", "located", "sent"}):
        if label in {"page_header", "page_field_index", "page_line_index"}:
            score += 0.14
    if any(cue in q_tokens for cue in {"committee", "meeting", "session", "attended", "strength", "total", "amount"}):
        if label in {"page_field_index", "page_line_index", "page_numeric_profile"}:
            score += 0.12
    if any(ch.isdigit() for ch in str(question or "")) and label == "page_numeric_profile":
        score += 0.10
    if packet.metadata.get("visible_value"):
        score += 0.08
    if packet.modality == OCR_BLOCK:
        score += 0.05
    if packet.modality in {CHART_REGION, TABLE_REGION}:
        score += 0.04
    return score


def evidence_source_recall(
    *,
    expected_source_ids: Iterable[str],
    evidence: Sequence[EvidencePacket],
    k: int,
) -> bool:
    expected = {str(item) for item in expected_source_ids}
    if not expected:
        return False
    for packet in evidence[: max(1, int(k))]:
        if packet.unit_id in expected or packet.source_id in expected:
            return True
    return False


def visible_answer_from_evidence(
    *,
    question: str,
    evidence: Sequence[EvidencePacket],
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Conservative visible-text extractor for synthetic smoke tests.

    This is not a benchmark scorer and not a replacement for Qwen/Qwen-VL
    generation.  It returns a visible value only when the retrieved packet
    already contains a local value candidate such as a money amount or chart
    number.
    """

    if not evidence:
        return None, {"source": "no_evidence", "override": False}
    top = evidence[0]
    visible_value = top.metadata.get("visible_value")
    if visible_value:
        return str(visible_value), {
            "source": "visible_value_metadata",
            "override": True,
            "unit_id": top.unit_id,
            "score": top.score,
        }
    text = top.text
    q = str(question or "").lower()
    patterns = []
    if any(word in q for word in ["total", "price", "cost", "amount"]):
        patterns.extend([r"[$€£]\s*[0-9][0-9,]*(?:\.[0-9]+)?", r"[0-9][0-9,]*(?:\.[0-9]+)?\s*(?:usd|dollars)"])
    if any(word in q for word in ["revenue", "sales", "profit", "value"]):
        patterns.extend([r"[0-9][0-9,]*(?:\.[0-9]+)?\s*(?:m|million|b|billion|%)?", r"[$€£]\s*[0-9][0-9,]*(?:\.[0-9]+)?"])
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip(), {
                "source": "regex_from_retrieved_visible_text",
                "override": True,
                "unit_id": top.unit_id,
                "score": top.score,
            }
    return None, {"source": "no_visible_value_candidate", "override": False, "unit_id": top.unit_id}
