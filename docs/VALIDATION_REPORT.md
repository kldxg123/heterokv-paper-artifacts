# Public release validation

Performed locally on 2026-10-11, with no new model/API/SSH experiments.

| Check | Result |
| --- | --- |
| Public synthetic unit tests | 16 passed |
| Core offline result reaggregation | 341 checks passed; no mismatches |
| E1--E4 derived records | 32 / 2 / 10 / 24 runs, respectively |
| Original input hashes | Reverified against inherited audit before projection |
| Inherited experimental audit | 23,450 successful checks; not a new generation run |
| Main inference and E4 CLI imports | `--help` passed; no model/service started |
| Independent LaTeX compile | pdflatex + bibtex + two pdflatex passes; six pages |
| Final compile diagnostics | No fatal, missing figure, undefined reference/citation, overfull or rerun warning |
| Published paper | Byte-identical copy of the latest manuscript PDF/source |
| Public secret/source-content scan | See `PUBLIC_VALIDATION.json` |
| Original frozen artifacts | Not modified |

The initial local Python runtime lacked pytest. It was installed into a separate
packaging-tools directory outside the public snapshot. Temporary installation
and TeX-build files are not uploaded. No missing resource measurements are filled
in and no historical pilot score replaces the audited E1--E4 table.

`scripts/recompute_public_results.py` reproduces numeric aggregate checks from
published derived records. It does not claim to re-score unavailable original
labels, recreate unpublished OCR content, or rerun GPU inference.
