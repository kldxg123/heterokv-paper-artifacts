# Evidence and claim boundaries

The manuscript, all 19 table files and three data plots are copied from the
2026-10-11 academically expanded architecture edition without changing data.
The release manifest records their source and public checksums.

| Evidence | Public representation | Claim scope |
| --- | --- | --- |
| E1 | 32 sanitized runs; all candidate points and endpoint mean/SD | Source-local 5x/3x workload growth; deterministic setup only |
| E2 | 2 sanitized runs, 3,000 questions | Image-dominated hybrid QA; ANLS >=0.5 cutoff; local soft VQA |
| E3 | 10 paired first-500 runs | Quality/resource frontier; B0 not consistently superior |
| E4 | 24 serving runs, each 100 measured and 10 warm-up requests | Three fresh services; localhost; no online retrieval/index |

Numbers omitted from raw logs are not invented. Index-only CPU/disk size,
post-clipping complete prompts, isolated route/pack latency, checkpoint revision
and historical PyTorch/CUDA/OCR versions were not reliably recorded.
Full bitwise reproduction of those unrecorded properties is not promised.

The inherited experimental audit contains 23,450 successful checks. In this
release we reverify its input hashes and recompute core public statistics; we do
not describe that as a new GPU run. Historical pilots, no-source v134/v141,
DeepSeek n=20 and v2 exact recall are not promoted into the E1--E4 main table.

Unsafe extensions: solved source discovery, FullKV/full-context equivalence,
broad reasoning, total memory O(1), learned encoder superiority, B2 equal text
payload, raw-image commercial API support, provider GPU savings, or E4 within
a 24GB memory envelope. E4 used an A100 80GB and approximately 35--36 GiB NVML.
