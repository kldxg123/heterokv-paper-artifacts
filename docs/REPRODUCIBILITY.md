# Reproduction guide

## Offline audit (no GPU, no network)

```bash
python -m pip install -r requirements-public.txt
python scripts/validate_public_release.py
python -m pytest -q
python scripts/recompute_public_results.py --output outputs/public_reanalysis.json
```

The first command installs local-analysis libraries, not model weights. The
subsequent commands verify release hashes, run synthetic unit tests and check
per-request numeric values against paper CSVs. Annotation-independent aggregation
is possible with public data; rescoring predictions from original labels requires
the separately obtained dataset and valid slice signatures.

## Compile the unchanged manuscript

With an IEEEtran-enabled TeX installation, from `paper/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The supplied PDF is six pages. `paper/figures/` includes the editable draw.io,
SVG/PDF architecture and all three measured data plots. The exported architecture
is native vector artwork, not a raster trace.

`python scripts/plot_public_tables.py` rebuilds the three measured curves from
the public CSVs into `outputs/rebuilt_figures/`. Font/layout differences may
change rendering but not the plotted numeric points; supplied paper assets
are not overwritten.

## Optional GPU reproduction

Install `requirements-gpu.txt` in a separate environment appropriate for your
GPU. The measured E1/E3 setup used Qwen2-VL-7B, Transformers 4.57.6, bfloat16,
eager attention, batch one, max 24 answer tokens, max 1,003,520 image pixels,
on A100 80GB PCIe. Model revision and historical PyTorch/CUDA/OCR versions are
unknown, not fabricated. Select a checkpoint and record its revision for new runs.

After obtaining/validating datasets, an explicit source-cued B0 run is:

```bash
python remote_edit/scripts/run_workflow4_multimodal_memory_retrieval.py \
  --dataset docvqa --start 0 --limit 500 --source-pool-limit 500 \
  --model-path models/Qwen2-VL-7B --query-binding source_handle \
  --ingest-mode metadata --ocr-mode auto --region-mode none \
  --source-selection-mode source_handle_fill_topk --unit-selection-mode heterokv \
  --answer-mode unit_text_only --unit-top-k 12 --top-k 5 --seed 9001 \
  --source-pool-multiplier 4 --source-granularity image \
  --max-pixels 1003520 --answer-max-new-tokens 24 --cap-gib 34 \
  --output remote_edit/experiments/docvqa_b0.json \
  --md-output remote_edit/experiments/docvqa_b0.md
```

For the source-image family set `--answer-mode source_full_image`; retain the
same questions/source handle, but report the extra image/pixel payload.
For controls set `--unit-selection-mode random`, `bm25_ocr`, or `layout_aware`.
Do not rename `bm25_ocr` into standard BM25 when reporting these results.
Repeat for TextVQA using the same fixed-prefix slice and distinguish local metrics.
The generic runner includes historical exploratory modes; their presence is not
evidence of successful or evaluated functionality.

For E1 the wrapper `remote_edit/scripts/run_e1_source_local_scaling_cached.py`
adds `--e1-query-indices`, `--e1-candidate-scale`, `--e1-method`,
`--e1-repeat-index`; use `data/E1_SAMPLE_MANIFEST.json` for fixed lists.
For E2 retain the recorded `unit_hybrid` route; do not call its score crop-only QA.

E4 uses `remote_edit/scripts/run_e4_shared_serving.py` with **locally regenerated
unsanitized E2 artifacts** and a separately installed vLLM 0.18.0 environment.
`--help` describes required workload and model arguments. Public derived rows
deliberately lack OCR/images and are not an E4 generation workload.
Use concurrency 1/2/4/8, three fresh services, 10 warm-ups and 100 measured
requests/repeat, max 24 output tokens. Keep the service localhost-only.

GPU experiments are not started by any release validator or CI job. Check GPU
availability before explicit execution; process cap alone does not prevent
competition with other users. Do not terminate unrelated jobs.

## Known limits

- Post-clipping complete prompts and original OCR caches are not published or
  completely recoverable; independently regenerated evidence may differ.
- Derived score aggregation is reproducible, but exact model-output replay is
  not guaranteed without the missing checkpoint/environment state.
- `remote_edit/scripts/runtime_support.py` extracts only the original seed/cap
  helpers. Its imports replace the old v2 smoke-runner import; scored inference
  logic is otherwise preserved and source hashes are in the manifest.
- Public records may redact contact/credential-like prediction strings while
  retaining the original prediction hash and measured numeric scores.
