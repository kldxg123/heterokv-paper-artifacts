# Architecture grounding and invariants

This revision expands scientific detail without adding method functionality.
The implementation hashes are in `audit/academic_architecture_build.json`.

| Diagram element | Implementation basis | Interpretation |
| --- | --- | --- |
| MemoryUnits and source provenance | `context_memory_v3.py`: `MemoryUnit`, `EvidencePacket` | Typed external records; provenance fields and references, not a security proof. |
| Query-independent visible ingest | `document_context_v3.py`: `build_document_memory_index`; runner `_build_memory_record` | OCR/layout/image metadata; answer annotations are not ranking inputs. |
| Source-handle filtering | Runner source-bound branch and `_retrieve_units_for_source` | The application supplies the source. This is not no-source discovery. |
| Token inverted index and candidate union | `ContextMemoryIndex.add_unit` / `search` | Query-token hits are unioned; no hits falls back within the source-local index. Image-only units are excluded from the text index. |
| Fusion score | `FusionConfig`; `_score_unit` | Default weights 0.38/0.34/0.12/0.08/0.08 for lexical/hash/metadata/history/confidence. Hand-set, not learned or ablated-optimal. |
| Hash cosine H | `hashed_multimodal_embedding` | A deterministic SHA-1 histogram over visible text, not a visual encoder, CLIP, or learned dense embedding. |
| History A and confidence C | `_score_unit` | History is metadata if recorded; it is not a new live attention computation. Confidence is stored and clipped, not a calibrated correctness probability. |
| Two-stage reranking | `retrieve_document_evidence`; `_unit_generation_score` | Base top-3k, document top-k, then generation-oriented reordering of the retained packets; stable unit-ID ties. |
| B0 text-only | `_packet_text_context`; `_unit_text_messages` | At most k selected units. E3 clips each text block to 520 chars and combined serialized evidence to 2,200 chars; instruction/question are additional. |
| B2 source image | `source_full_image` branch; `_build_messages` | One routed image plus question. Pixel policy is handled by the processor. B2 does not depend on the text-unit ranker and is not equal-payload with B0. |
| Optional conservative hybrid | `_unit_answer_decision`; `unit_hybrid` branch | A heuristic sufficiency decision, with source-image fallback. Not a trained/calibrated gate or established crop-only success. |
| VLM request boundary | `_unit_text_messages`; `_build_messages`; `_decode_with_qwenvl` | Only packed evidence is tensorized/served; model weights remain unchanged. |
| Accounting | Saved row-level traces and paper cost model | Active-input caps do not prove constant token count, GPU peak, or total memory. External CPU/disk state can grow. |

## Diagram-specific acceptance checks

- Supplied source handle is a dashed control input, not evidence flowing into the question.
- B2 has a direct source-filter path; no candidate/score/rerank edge selects its image.
- The optional hybrid is visibly separate and marked heuristic.
- No learned retriever, end-to-end training, Qwen weight change, or v2 KV-cache mechanism is depicted as an evaluated v3 component.
- Panels A/B are conceptual/control families, not a claim of empirical dominance.
- The legacy `bm25_ocr` control is labeled OCR lexical, not standard BM25.
- Native PDF/SVG contain no bitmap architecture; all delivered labels are selectable.
- Experimental inputs, plots, tables, and all non-architecture paper content remain unchanged.
