# Dataset access and data availability

## What is public here

`results/E1` through `results/E4` contain **derived evaluation data**, not a new
benchmark or mirror of third-party datasets. Numeric scores, timings, selected
unit IDs and model-generated predictions are retained. Questions/annotation
answers and OCR/page previews are omitted; signatures allow authorized users to
verify that their local dataset slice matches. Absolute source/model paths,
HTTP errors, resource process IDs and credentials are omitted.

`data/slices/` identifies the exact ordered E1/E2/E3/E4 question prefixes or fixed
indices with source/question identifiers when recorded and SHA-256 signatures.
The upstream dataset revision was not recorded and is explicitly `unknown`.
Changed upstream ordering/content must not silently count as the same slice.

## DocVQA

Original **Single Document VQA (2020 Task 1)** images, OCR and annotations must
be obtained under the original terms. [Official dataset page](https://site.docvqa.org/datasets/docvqa)
links to the RRC challenge download portal; [general dataset instructions](https://site.docvqa.org/datasets)
require login and acceptance of download terms. This release does **not** assert
permission to redistribute those files, and does not include them.

The historical runner used `HuggingFaceM4/DocumentVQA`, validation split.
[The mirror](https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA) does not
replace original terms. The user must verify access/authorization before use.
For offline recovery, use the recorded `questionId`/`docId` and signed question
prefix; official and mirror ordering may differ.

## TextVQA

[Official TextVQA dataset](https://textvqa.org/dataset/) states CC BY 4.0 for the
dataset, and provides v0.5.1/v0.5 annotations, Rosetta OCR and OpenImages links.
Credit **Amanpreet Singh et al., Towards VQA Models That Can Read, CVPR 2019**,
the TextVQA project, and any applicable image-specific attribution. Images derive
from OpenImages; check original image rights rather than apply our code notice.

The historical runner used `lmms-lab/TextVQA`, validation split. We publish sample
IDs/signatures and derived results, not multi-gigabyte source images. The runner
also used its own OCR/layout processing; raw Rosetta tokens are not a substitute
for the unrecorded historical OCR package/environment.

## Obtain and verify locally (explicit opt-in)

```bash
python -m pip install datasets
python scripts/prepare_datasets.py --dataset textvqa --accept-original-terms \
  --output data/downloaded/textvqa
python scripts/prepare_datasets.py --dataset docvqa --accept-original-terms \
  --output data/downloaded/docvqa
```

These commands use the historical mirror IDs and check questions/answers against
the release slice. `--endpoint https://hf-mirror.com` is an optional transport
mirror, not new licensing permission. Existing output directories are refused.
Benchmark answers are written into a separate local evaluation file and are
never passed into `MemoryUnit` creation or ranking.

All downloading is opt-in; no source datasets or weights were newly downloaded
to prepare this release. Original data is not subject to an MIT license here.
