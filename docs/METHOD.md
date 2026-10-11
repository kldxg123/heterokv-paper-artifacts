# Data-to-answer execution order

## 1. Visible source ingestion

Read document/image references and visible OCR blocks/layout spans. Ingest does
not receive the current question, answer annotation or hidden answer span.
Create `MemoryUnit` records with unit/source IDs, modality, text, token span or
bbox, optional KV reference, and transformation provenance.

The implementation snapshot is in `remote_edit/src/memory/context_memory_v3.py`
and `document_context_v3.py`. The evaluated v3 is an external evidence layer;
its VLM requests use ordinary model generation, not the earlier v2 KV manager.

## 2. External source-local storage

The Python `ContextMemoryIndex` holds records and a token inverted index on CPU.
Images remain referenced on disk. External state can grow with sources/units;
no constant-total-memory claim follows from a cap on model input.

## 3. Application supplies question and source handle

Filter to the indicated source. This is source/document-cued access, not automatic
discovery across an unconstrained pool. Answer labels are used only offline for
evaluation, not for the source-local scoring step.

## 4. B0 text evidence selection

Token hits form a candidate union; an empty hit set falls back within the local
source. The default score is `0.38 L + 0.34 H + 0.12 M + 0.08 A + 0.08 C`:
lexical overlap, nonnegative cosine of deterministic hashed visible text,
metadata match, recorded history metadata, and stored confidence.

`H` is a stable SHA-1 token histogram, not a learned visual/dense encoder.
History does not compute new attention, and confidence is not calibrated QA
correctness. The document route reranks base top-3k to top-k; a generation
reordering operates on retained packets. Unit IDs provide deterministic ties.

E3 controls use stable random ordering, OCR lexical or layout-aware scoring.
The old string `bm25_ocr` is retained for trace matching, not claimed as BM25.

## 5. Two distinct payload families

- **B0:** no active image. E1 uses top-4 units; E3 uses top-12, clips each block
  to 520 characters and combined evidence to 2,200 characters. The common prompt
  and question are additional; character/unit caps are not identical token counts.
- **B2:** one directly routed source image plus question/prompt; the processor
  enforces a 1,003,520-pixel maximum in E1/E3. B2 bypasses text ranking and is not
  an equal-payload B0 control.
- **Optional hybrid:** hand-set sufficiency checks choose a unit route or image
  fallback. E2 falls back in 99.9%/99.1% of DocVQA/TextVQA questions. This does
  not demonstrate successful crop-only QA or a learned/calibrated selector.

## 6. EvidencePacket and VLM boundary

Packets carry selected IDs/references, scores where applicable, source provenance
and active-input accounting. Only assembled text/image inputs enter Qwen2-VL.
There is no weight update, implicit retrieval of excluded source content, or
provider-side memory-saving assertion.

## 7. Answer and measurements

Record model answer, numeric metrics, measured prompt/generated tokens and
timings when available. E1/E3 PyTorch peaks differ from E4 device-level NVML
measurements. E4 is a localhost shared service over preselected packets;
retrieval/index time is excluded. Provenance is an audit mechanism, not an
access-control or security proof.
