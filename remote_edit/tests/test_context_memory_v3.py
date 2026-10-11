import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.memory.context_memory_v3 import (
    ABSTRACTIVE_QA,
    EXACT_COPY,
    IMAGE_REGION,
    OCR_BLOCK,
    TEXT,
    AnswerControllerV3,
    BBox,
    ContextMemoryIndex,
    MemoryUnit,
    RetrievalQuery,
)


def test_v3_memory_unit_roundtrip_is_deterministic():
    unit = MemoryUnit(
        unit_id="text:doc:1",
        modality=TEXT,
        source_id="doc",
        text="The quarterly revenue was 42 million.",
        token_span=(10, 18),
        embedding=[0.1, 0.2, 0.3],
        metadata={"label": "paragraph-1"},
    )

    restored = MemoryUnit.from_dict(unit.to_dict())

    assert restored == unit
    assert "quarterly revenue" in restored.searchable_text()


def test_v3_rejects_oracle_metadata_at_index_time():
    index = ContextMemoryIndex()

    with pytest.raises(ValueError):
        index.add_text_span(
            source_id="doc",
            text="visible text",
            metadata={"gold_answer": "hidden"},
        )


def test_v3_text_search_uses_lexical_and_dense_signals():
    index = ContextMemoryIndex()
    index.add_text_span(
        source_id="doc-a",
        text="The storage controller evicts old KV pages to DRAM.",
        embedding=[1.0, 0.0, 0.0],
    )
    index.add_text_span(
        source_id="doc-b",
        text="The unrelated image shows a city skyline.",
        embedding=[0.0, 1.0, 0.0],
    )

    packets = index.search(
        RetrievalQuery(
            query_text="Which controller stores KV pages in DRAM?",
            query_embedding=[1.0, 0.0, 0.0],
            allowed_modalities=[TEXT],
            top_k=2,
        )
    )

    assert packets[0].source_id == "doc-a"
    assert packets[0].score_breakdown["lexical"] > 0.0
    assert packets[0].score_breakdown["dense"] > 0.99
    assert packets[0].provenance["from_retrieved_unit"] is True


def test_v3_multimodal_search_returns_ocr_and_region_provenance():
    index = ContextMemoryIndex()
    index.add_image_region(
        source_id="chart-page-7",
        image_id="img-1",
        region_id="bar-2025",
        bbox=BBox(0.1, 0.2, 0.4, 0.7, normalized=True),
        caption="bar chart revenue 2025",
        embedding=[0.2, 0.8],
        metadata={"label": "chart-region"},
    )
    index.add_ocr_block(
        source_id="chart-page-7",
        image_id="img-1",
        block_id="ocr-title",
        text="Revenue increased to 42M in 2025",
        bbox=BBox(10, 20, 200, 60),
        embedding=[0.3, 0.7],
    )

    packets = index.search(
        RetrievalQuery(
            query_text="What revenue is shown for 2025?",
            query_embedding=[0.25, 0.75],
            allowed_modalities=[IMAGE_REGION, OCR_BLOCK],
            top_k=2,
        )
    )

    assert {packet.modality for packet in packets} == {IMAGE_REGION, OCR_BLOCK}
    assert all(packet.bbox is not None for packet in packets)
    assert all(packet.source_id == "chart-page-7" for packet in packets)


def test_v3_answer_controller_only_overrides_exact_copy():
    index = ContextMemoryIndex()
    index.add_text_span(
        source_id="passkey-doc",
        text="passkey: ZX-913",
        embedding=[1.0],
        metadata={"answer_text": "ZX-913"},
    )
    evidence = index.search(
        RetrievalQuery(
            query_text="What is the passkey?",
            query_embedding=[1.0],
            policy_hint=EXACT_COPY,
            top_k=1,
        )
    )
    controller = AnswerControllerV3(override_threshold=0.5)

    exact = controller.decide(policy=EXACT_COPY, evidence=evidence, lm_answer="wrong")
    abstractive = controller.decide(policy=ABSTRACTIVE_QA, evidence=evidence, lm_answer="draft")

    assert exact.should_override is True
    assert exact.answer_text == "ZX-913"
    assert abstractive.should_override is False
    assert abstractive.answer_text == "draft"
    assert "Evidence" in abstractive.generation_context


def test_v3_augmented_prompt_is_retrieval_grounded():
    index = ContextMemoryIndex()
    index.add_text_span(
        source_id="doc",
        text="The active HBM cache remains bounded while external memory grows.",
        embedding=[1.0],
    )
    evidence = index.search(
        RetrievalQuery(
            query_text="What remains bounded?",
            query_embedding=[1.0],
            top_k=1,
        )
    )
    controller = AnswerControllerV3()

    prompt = controller.build_augmented_prompt(
        question="What remains bounded?",
        evidence=evidence,
        policy=ABSTRACTIVE_QA,
    )

    assert "Retrieved evidence:" in prompt
    assert "active HBM cache remains bounded" in prompt
    assert "Question:" in prompt
    assert "What remains bounded?" in prompt


def test_v3_controller_infers_visual_policy_from_dataset():
    controller = AnswerControllerV3()

    assert controller.infer_policy(dataset="DocVQA") == "visual_qa"
    assert controller.infer_policy(dataset="passage_retrieval_en") == "extractive_qa"
