"""Paired bootstrap analysis for stable-hash E3 first-500 QA rows."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any


DATASETS = ("textvqa", "docvqa")
METHODS = ("random", "bm25_ocr", "layout_aware", "heterokv_b0", "b2_source_image")
COMPARISONS = (
    ("heterokv_b0", "random"),
    ("heterokv_b0", "bm25_ocr"),
    ("heterokv_b0", "layout_aware"),
    ("b2_source_image", "heterokv_b0"),
)


def artifact_name(dataset: str, method: str) -> str:
    return f"cscloud_e3_{dataset}_{method}_metadata_n500_gpu0_stablehash_20260814.json"


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap(left: list[float], right: list[float], repeats: int, seed: int) -> dict[str, float]:
    differences = [a - b for a, b in zip(left, right)]
    observed = statistics.mean(differences)
    rng = random.Random(seed)
    n = len(differences)
    samples: list[float] = []
    for _ in range(repeats):
        samples.append(sum(differences[rng.randrange(n)] for _ in range(n)) / n)
    return {
        "observed_delta": observed,
        "ci95_low": percentile(samples, 0.025),
        "ci95_high": percentile(samples, 0.975),
        "probability_delta_gt_zero": sum(value > 0.0 for value in samples) / repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for dataset in DATASETS:
        scores: dict[str, list[float]] = {}
        indices: dict[str, list[int]] = {}
        for method in METHODS:
            path = args.artifact_dir / artifact_name(dataset, method)
            if not path.exists():
                errors.append(f"missing: {path.name}")
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("rows") or []
            if payload.get("status") != "ok" or payload.get("embedding_hash") != "sha1_v1" or len(rows) != 500:
                errors.append(f"invalid artifact: {path.name}")
                continue
            indices[method] = [int(row.get("index", -1)) for row in rows]
            scores[method] = [float((row.get("scores") or {}).get("vqa_soft_score", 0.0) or 0.0) for row in rows]
        if len(scores) != len(METHODS):
            continue
        reference_indices = indices[METHODS[0]]
        for method in METHODS[1:]:
            if indices[method] != reference_indices:
                errors.append(f"{dataset}: question order differs for {method}")
        if any(error.startswith(f"{dataset}:") for error in errors):
            continue
        for comparison_index, (left, right) in enumerate(COMPARISONS):
            stats = paired_bootstrap(
                scores[left],
                scores[right],
                repeats=args.bootstrap_repeats,
                seed=args.seed + comparison_index,
            )
            results.append(
                {
                    "dataset": dataset,
                    "left": left,
                    "right": right,
                    "sample_count": len(scores[left]),
                    "left_mean": statistics.mean(scores[left]),
                    "right_mean": statistics.mean(scores[right]),
                    **stats,
                }
            )

    status = "complete" if len(results) == len(DATASETS) * len(COMPARISONS) and not errors else "incomplete_or_invalid"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "status": status,
        "bootstrap_repeats": args.bootstrap_repeats,
        "seed": args.seed,
        "validation_errors": errors,
        "comparisons": results,
    }
    (args.output_dir / "E3_PAIRED_BOOTSTRAP.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# E3 Paired Bootstrap on First-500 Questions",
        "",
        f"Status: **{status}**. Bootstrap resamples: `{args.bootstrap_repeats}`; fixed seed: `{args.seed}`.",
        "",
        "| Dataset | Left - right | n | Left VQA | Right VQA | Delta | Paired 95% CI | P(delta>0) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['dataset']} | {row['left']} - {row['right']} | {row['sample_count']} | "
            f"{row['left_mean']:.4f} | {row['right_mean']:.4f} | {row['observed_delta']:+.4f} | "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['probability_delta_gt_zero']:.4f} |"
        )
    lines += ["", "## Validation", ""]
    lines += [f"- {item}" for item in errors] if errors else ["- All method rows use the same 500 question indices and stable-hash provenance."]
    lines += [
        "",
        "Intervals that include zero must be described as competitive/indistinguishable on this slice, not as a significant win.",
        "",
    ]
    (args.output_dir / "E3_PAIRED_BOOTSTRAP_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "comparisons": len(results), "errors": len(errors)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
