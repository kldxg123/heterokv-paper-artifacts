import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.memory.context_memory_v3 import CHART_REGION, OCR_BLOCK
from src.memory.document_context_v3 import (
    CHARTQA,
    DOCVQA,
    build_document_memory_index,
    evidence_source_recall,
    retrieve_document_evidence,
    visible_answer_from_evidence,
)


def test_v3_document_index_retrieves_invoice_ocr_block():
    record = {
        "document_id": "invoice-7",
        "image_id": "invoice-page",
        "ocr_blocks": [
            {"block_id": "vendor", "text": "Vendor: Northwind", "bbox": [0, 0, 100, 20]},
            {"block_id": "total", "text": "Total due: $42.50", "bbox": [0, 40, 140, 70]},
        ],
    }
    index, summary = build_document_memory_index(record=record, dataset=DOCVQA)

    evidence = retrieve_document_evidence(
        index=index,
        question="What is the total due on the invoice?",
        dataset=DOCVQA,
        top_k=2,
    )
    answer, info = visible_answer_from_evidence(
        question="What is the total due on the invoice?",
        evidence=evidence,
    )

    assert summary["unit_counts"][OCR_BLOCK] == 2
    assert evidence[0].unit_id.endswith(":total")
    assert evidence[0].bbox is not None
    assert answer == "$42.50"
    assert info["override"] is True


def test_v3_document_index_retrieves_chart_region_and_ocr():
    record = {
        "document_id": "chart-doc",
        "image_id": "chart-page",
        "regions": [
            {
                "region_id": "bar-2025",
                "region_kind": "chart",
                "caption": "2025 revenue bar",
                "bbox": [0.1, 0.1, 0.4, 0.8],
                "visible_value": "42M",
            }
        ],
        "ocr_blocks": [
            {"block_id": "value-2025", "text": "2025 revenue 42M", "bbox": [200, 120, 280, 150]},
        ],
    }
    index, summary = build_document_memory_index(record=record, dataset=CHARTQA)

    evidence = retrieve_document_evidence(
        index=index,
        question="What is the revenue for 2025?",
        dataset=CHARTQA,
        top_k=3,
    )
    answer, _ = visible_answer_from_evidence(
        question="What is the revenue for 2025?",
        evidence=evidence,
    )

    assert summary["unit_counts"][CHART_REGION] == 1
    assert evidence_source_recall(
        expected_source_ids=["chart:chart-doc:chart-page:bar-2025"],
        evidence=evidence,
        k=3,
    )
    assert answer in {"42M", "2025"}


def test_v3_document_index_ignores_gold_fields():
    record = {
        "document_id": "doc",
        "image_id": "img",
        "gold_answer": "secret",
        "ocr_blocks": [{"block_id": "visible", "text": "Visible text only", "bbox": [0, 0, 1, 1]}],
    }
    index, _ = build_document_memory_index(record=record, dataset=DOCVQA)

    units = index.units

    assert len(units) == 2
    assert all("gold_answer" not in unit.metadata for unit in units.values())
