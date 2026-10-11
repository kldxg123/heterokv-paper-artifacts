"""CPU-only Workflow4.3 document-image memory smoke.

This runner validates the v3 OCR/image-region evidence path on synthetic
DocVQA/TextVQA/ChartQA-style records.  It does not load Qwen-VL or a benchmark
dataset and must not be reported as real multimodal QA performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.context_memory_v3 import AnswerControllerV3  # noqa: E402
from src.memory.document_context_v3 import (  # noqa: E402
    CHARTQA,
    DOCVQA,
    TEXTVQA,
    build_document_memory_index,
    evidence_source_recall,
    infer_document_policy,
    retrieve_document_evidence,
    visible_answer_from_evidence,
)


def _synthetic_records() -> List[Dict[str, Any]]:
    return [
        {
            "dataset": DOCVQA,
            "document_id": "invoice-7",
            "image_id": "invoice-page",
            "question": "What is the total due on the invoice?",
            "expected_answer": "$42.50",
            "expected_source_ids": ["ocr:invoice-7:invoice-page:total"],
            "record": {
                "document_id": "invoice-7",
                "image_id": "invoice-page",
                "caption": "invoice page with total due",
                "ocr_blocks": [
                    {"block_id": "vendor", "text": "Vendor: Northwind", "bbox": [0, 0, 100, 20]},
                    {"block_id": "total", "text": "Total due: $42.50", "bbox": [0, 40, 140, 70]},
                    {"block_id": "date", "text": "Due date: 2026-06-30", "bbox": [0, 80, 170, 105]},
                ],
            },
        },
        {
            "dataset": TEXTVQA,
            "document_id": "street-sign-3",
            "image_id": "sign-image",
            "question": "What cafe name is written on the sign?",
            "expected_answer": "Cafe Azul",
            "expected_source_ids": ["ocr:street-sign-3:sign-image:cafe"],
            "record": {
                "document_id": "street-sign-3",
                "image_id": "sign-image",
                "caption": "storefront with a cafe sign",
                "ocr_blocks": [
                    {"block_id": "cafe", "text": "Cafe Azul", "bbox": [50, 20, 220, 80], "visible_value": "Cafe Azul"},
                    {"block_id": "hours", "text": "Open 8 AM - 6 PM", "bbox": [60, 100, 220, 130]},
                ],
            },
        },
        {
            "dataset": CHARTQA,
            "document_id": "revenue-chart",
            "image_id": "chart-page",
            "question": "What is the revenue for 2025?",
            "expected_answer": "42M",
            "expected_source_ids": [
                "chart:revenue-chart:chart-page:bar-2025",
                "ocr:revenue-chart:chart-page:value-2025",
            ],
            "record": {
                "document_id": "revenue-chart",
                "image_id": "chart-page",
                "caption": "bar chart showing yearly revenue",
                "regions": [
                    {
                        "region_id": "bar-2025",
                        "region_kind": "chart",
                        "caption": "2025 revenue bar",
                        "bbox": [0.58, 0.20, 0.72, 0.84],
                        "visible_value": "42M",
                    }
                ],
                "ocr_blocks": [
                    {"block_id": "value-2025", "text": "2025 revenue 42M", "bbox": [280, 120, 360, 150]},
                    {"block_id": "value-2024", "text": "2024 revenue 31M", "bbox": [190, 155, 270, 180]},
                ],
            },
        },
    ]


def run_smoke() -> Dict[str, Any]:
    controller = AnswerControllerV3()
    rows: List[Dict[str, Any]] = []
    source_hits_at_1 = 0
    source_hits_at_3 = 0
    exact_answers = 0

    for item in _synthetic_records():
        dataset = item["dataset"]
        question = item["question"]
        index, summary = build_document_memory_index(record=item["record"], dataset=dataset)
        evidence = retrieve_document_evidence(
            index=index,
            question=question,
            dataset=dataset,
            top_k=3,
        )
        hit1 = evidence_source_recall(
            expected_source_ids=item["expected_source_ids"],
            evidence=evidence,
            k=1,
        )
        hit3 = evidence_source_recall(
            expected_source_ids=item["expected_source_ids"],
            evidence=evidence,
            k=3,
        )
        answer, answer_info = visible_answer_from_evidence(question=question, evidence=evidence)
        exact = str(answer or "").lower() == str(item["expected_answer"]).lower()
        policy = infer_document_policy(dataset=dataset, question=question)
        prompt = controller.build_augmented_prompt(
            question=question,
            evidence=evidence,
            policy=policy,
            base_instruction="Answer using only retrieved document-image evidence.",
        )
        source_hits_at_1 += int(hit1)
        source_hits_at_3 += int(hit3)
        exact_answers += int(exact)
        rows.append(
            {
                "dataset": dataset,
                "question": question,
                "expected_answer": item["expected_answer"],
                "predicted_visible_answer": answer,
                "answer_info": answer_info,
                "source_hit_at_1": hit1,
                "source_hit_at_3": hit3,
                "policy": policy,
                "index_summary": summary,
                "top_evidence": [
                    {
                        "unit_id": packet.unit_id,
                        "modality": packet.modality,
                        "source_id": packet.source_id,
                        "score": packet.score,
                        "label": packet.metadata.get("label"),
                        "bbox": None if packet.bbox is None else packet.bbox.to_list(),
                        "text_preview": packet.text[:180],
                    }
                    for packet in evidence
                ],
                "prompt_preview": prompt[:900],
            }
        )

    total = len(rows)
    status = "passed"
    if source_hits_at_3 != total or exact_answers < 2:
        status = "failed"
    return {
        "status": status,
        "scope": "cpu_only_synthetic_document_image_memory_smoke_not_benchmark_evidence",
        "total": total,
        "source_recall_at_1": source_hits_at_1 / max(1, total),
        "source_recall_at_3": source_hits_at_3 / max(1, total),
        "visible_answer_exact": exact_answers / max(1, total),
        "rows": rows,
    }


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Workflow4.3 Document-Image Memory Smoke",
        "",
        "This is a CPU-only synthetic DocVQA/TextVQA/ChartQA-style interface smoke.",
        "It is not real multimodal benchmark evidence and does not load Qwen-VL.",
        "",
        f"- status: `{result['status']}`",
        f"- rows: `{result['total']}`",
        f"- source recall@1: `{result['source_recall_at_1']}`",
        f"- source recall@3: `{result['source_recall_at_3']}`",
        f"- visible answer exact: `{result['visible_answer_exact']}`",
        "",
        "| Dataset | Question | Predicted visible answer | Recall@1 | Recall@3 | Top evidence |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in result["rows"]:
        top = row["top_evidence"][0] if row["top_evidence"] else {}
        lines.append(
            "| `{dataset}` | {question} | `{answer}` | `{hit1}` | `{hit3}` | `{unit}` / `{modality}` |".format(
                dataset=row["dataset"],
                question=str(row["question"]).replace("|", "/"),
                answer=row.get("predicted_visible_answer"),
                hit1=row["source_hit_at_1"],
                hit3=row["source_hit_at_3"],
                unit=top.get("unit_id", ""),
                modality=top.get("modality", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This validates only the v3 evidence interface for OCR and image regions.",
            "- It does not prove DocVQA, ChartQA, TextVQA, or approximate full-context QA.",
            "- Real multimodal claims require dataset-backed Qwen2.5-VL or vision-encoder runs.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("review-stage/WORKFLOW4_DOCUMENT_MEMORY_SMOKE_20260608.json"),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("review-stage/WORKFLOW4_DOCUMENT_MEMORY_SMOKE_20260608.md"),
    )
    args = parser.parse_args()

    result = run_smoke()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(result, args.md_out)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
