"""CPU-only Workflow4 / HeteroKV-v3 context-memory smoke runner.

This script validates the v3 memory/index/controller interface without loading
Qwen, Qwen-VL, OCR models, CLIP/SigLIP, or benchmark datasets.  It is a
development gate, not paper evidence.
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

from src.memory.context_memory_v3 import (  # noqa: E402
    ABSTRACTIVE_QA,
    EXACT_COPY,
    IMAGE_REGION,
    OCR_BLOCK,
    VISUAL_QA,
    AnswerControllerV3,
    BBox,
    ContextMemoryIndex,
    RetrievalQuery,
)


def _build_index() -> ContextMemoryIndex:
    index = ContextMemoryIndex()
    index.add_text_span(
        source_id="long-doc-a",
        text=(
            "The HeteroKV residency manager keeps active HBM KV bounded while "
            "compressed external records grow in DRAM."
        ),
        token_span=(120, 144),
        embedding=[1.0, 0.0, 0.0],
        metadata={"label": "text-evidence"},
    )
    index.add_text_span(
        source_id="long-doc-b",
        text="The unrelated paragraph discusses training recipes and optimizer schedules.",
        token_span=(240, 258),
        embedding=[0.0, 1.0, 0.0],
        metadata={"label": "distractor-text"},
    )
    index.add_image_region(
        source_id="chart-page-1",
        image_id="chart-1",
        region_id="bar-2025",
        bbox=BBox(0.10, 0.20, 0.45, 0.82, normalized=True),
        caption="bar chart revenue for 2025",
        embedding=[0.0, 0.1, 0.9],
        metadata={"label": "chart-region-2025"},
    )
    index.add_ocr_block(
        source_id="chart-page-1",
        image_id="chart-1",
        block_id="ocr-value-2025",
        text="2025 revenue: 42M",
        bbox=BBox(320, 140, 520, 190),
        embedding=[0.0, 0.1, 0.9],
        metadata={"label": "ocr-value-2025"},
    )
    index.add_ocr_block(
        source_id="chart-page-2",
        image_id="chart-2",
        block_id="ocr-distractor",
        text="2022 revenue: 18M",
        bbox=BBox(10, 20, 200, 60),
        embedding=[0.0, 0.8, 0.2],
        metadata={"label": "ocr-distractor"},
    )
    return index


def _hit_at_k(packets: List[Any], expected_source: str, k: int) -> bool:
    return any(packet.source_id == expected_source for packet in packets[:k])


def run_smoke() -> Dict[str, Any]:
    index = _build_index()
    controller = AnswerControllerV3(override_threshold=0.50)

    text_packets = index.search(
        RetrievalQuery(
            query_text="How does HeteroKV keep GPU KV memory bounded?",
            query_embedding=[1.0, 0.0, 0.0],
            policy_hint=ABSTRACTIVE_QA,
            top_k=3,
        )
    )
    visual_packets = index.search(
        RetrievalQuery(
            query_text="What revenue does the 2025 bar show?",
            query_embedding=[0.0, 0.1, 0.9],
            allowed_modalities=[IMAGE_REGION, OCR_BLOCK],
            policy_hint=VISUAL_QA,
            top_k=3,
        )
    )
    exact_index = ContextMemoryIndex()
    exact_index.add_text_span(
        source_id="passkey-doc",
        text="The passkey is ZX-913.",
        embedding=[1.0],
        metadata={"answer_text": "ZX-913", "label": "literal"},
    )
    exact_packets = exact_index.search(
        RetrievalQuery(
            query_text="What is the passkey?",
            query_embedding=[1.0],
            policy_hint=EXACT_COPY,
            top_k=1,
        )
    )

    text_decision = controller.decide(
        policy=ABSTRACTIVE_QA,
        evidence=text_packets,
        lm_answer="draft answer",
    )
    text_augmented_prompt = controller.build_augmented_prompt(
        question="How does HeteroKV keep GPU KV memory bounded?",
        evidence=text_packets,
        policy=ABSTRACTIVE_QA,
    )
    visual_decision = controller.decide(
        policy=VISUAL_QA,
        evidence=visual_packets,
        lm_answer="draft visual answer",
    )
    exact_decision = controller.decide(
        policy=EXACT_COPY,
        evidence=exact_packets,
        lm_answer="wrong",
    )

    result = {
        "status": "passed",
        "scope": "cpu_only_synthetic_interface_smoke_not_paper_evidence",
        "text_recall_at_1": _hit_at_k(text_packets, "long-doc-a", 1),
        "text_recall_at_3": _hit_at_k(text_packets, "long-doc-a", 3),
        "visual_recall_at_1": _hit_at_k(visual_packets, "chart-page-1", 1),
        "visual_recall_at_3": _hit_at_k(visual_packets, "chart-page-1", 3),
        "text_controller_override": text_decision.should_override,
        "visual_controller_override": visual_decision.should_override,
        "exact_controller_override": exact_decision.should_override,
        "exact_answer": exact_decision.answer_text,
        "text_augmented_prompt_preview": text_augmented_prompt[:600],
        "top_text_packet": text_packets[0].to_prompt_block(max_chars=180) if text_packets else "",
        "top_visual_packet": visual_packets[0].to_prompt_block(max_chars=180) if visual_packets else "",
    }
    if not result["text_recall_at_1"]:
        result["status"] = "failed_text_recall"
    if not result["visual_recall_at_3"]:
        result["status"] = "failed_visual_recall"
    if result["text_controller_override"] or result["visual_controller_override"]:
        result["status"] = "failed_non_exact_override"
    if not result["exact_controller_override"] or result["exact_answer"] != "ZX-913":
        result["status"] = "failed_exact_override"
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Workflow4 V3 Memory Smoke",
        "",
        "This is a CPU-only synthetic interface smoke. It is not benchmark or paper evidence.",
        "",
        f"- status: `{result['status']}`",
        f"- text recall@1: `{result['text_recall_at_1']}`",
        f"- text recall@3: `{result['text_recall_at_3']}`",
        f"- visual recall@1: `{result['visual_recall_at_1']}`",
        f"- visual recall@3: `{result['visual_recall_at_3']}`",
        f"- text controller override: `{result['text_controller_override']}`",
        f"- visual controller override: `{result['visual_controller_override']}`",
        f"- exact controller override: `{result['exact_controller_override']}`",
        f"- exact answer: `{result['exact_answer']}`",
        "",
        "## Top Evidence Blocks",
        "",
        "```text",
        str(result.get("top_text_packet", "")),
        "",
        str(result.get("top_visual_packet", "")),
        "```",
        "",
        "## Augmented Prompt Preview",
        "",
        "```text",
        str(result.get("text_augmented_prompt_preview", "")),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=Path("review-stage/WORKFLOW4_MEMORY_SMOKE_20260608.json"))
    parser.add_argument("--md-out", type=Path, default=Path("review-stage/WORKFLOW4_MEMORY_SMOKE_20260608.md"))
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
