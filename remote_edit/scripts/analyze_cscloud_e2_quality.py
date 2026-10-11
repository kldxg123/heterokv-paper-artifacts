from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_anls(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def anls(prediction: Any, answers: Iterable[Any], threshold: float = 0.5) -> float:
    pred = normalize_anls(prediction)
    best = 0.0
    for answer in answers:
        gold = normalize_anls(answer)
        denominator = max(len(pred), len(gold))
        similarity = 1.0 if denominator == 0 else 1.0 - levenshtein(pred, gold) / denominator
        best = max(best, similarity if similarity >= threshold else 0.0)
    return float(best)


def bootstrap_ci(values: list[float], repeats: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    chunk = 500
    for start in range(0, repeats, chunk):
        count = min(chunk, repeats - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        means[start : start + count] = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def analyze(dataset: str, path: Path, repeats: int, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = load_json(path)
    rows = payload.get("rows") or []
    if not rows:
        raise ValueError(f"no rows in {path}")
    per_row: list[dict[str, Any]] = []
    main_values: list[float] = []
    vqa_values: list[float] = []
    for row in rows:
        scores = row.get("scores") or {}
        vqa = float(scores.get("vqa_soft_score", 0.0) or 0.0)
        doc_anls = anls(row.get("pred"), row.get("answers") or []) if dataset == "docvqa" else math.nan
        main = doc_anls if dataset == "docvqa" else vqa
        failed = bool(row.get("oom")) or bool(row.get("error"))
        fallback = "source_full_image" in str(row.get("answer_mode_effective") or "")
        main_values.append(main)
        vqa_values.append(vqa)
        per_row.append(
            {
                "dataset": dataset,
                "sample_index": row.get("index"),
                "source_id": row.get("source_id"),
                "main_metric": "ANLS" if dataset == "docvqa" else "VQA accuracy",
                "main_score": main,
                "vqa_soft_score": vqa,
                "success": not failed,
                "fallback_used": fallback,
            }
        )
    main_low, main_high = bootstrap_ci(main_values, repeats, seed)
    vqa_low, vqa_high = bootstrap_ci(vqa_values, repeats, seed + 1)
    summary = {
        "dataset": dataset,
        "artifact": str(path),
        "status": payload.get("status"),
        "sample_count": len(rows),
        "main_metric": "ANLS" if dataset == "docvqa" else "VQA accuracy",
        "main_mean": float(np.mean(main_values)),
        "main_bootstrap_95_ci": [main_low, main_high],
        "vqa_soft_mean": float(np.mean(vqa_values)),
        "vqa_bootstrap_95_ci": [vqa_low, vqa_high],
        "success_rate": float(np.mean([row["success"] for row in per_row])),
        "generation_failure_rate": float(np.mean([not row["success"] for row in per_row])),
        "source_image_fallback_rate": float(np.mean([row["fallback_used"] for row in per_row])),
        "reported_metrics": payload.get("metrics"),
        "bootstrap_repeats": repeats,
        "bootstrap_seed": seed,
        "canary_included": False,
        "boundary": "Source/document-cued unit_hybrid_conservative path; fallback rate must be interpreted explicitly.",
    }
    return summary, per_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docvqa", type=Path, required=True)
    parser.add_argument("--textvqa", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_rows = []
    for offset, (dataset, path) in enumerate((("docvqa", args.docvqa), ("textvqa", args.textvqa))):
        summary, rows = analyze(dataset, path, args.bootstrap_repeats, args.seed + offset * 10)
        summaries.append(summary)
        all_rows.extend(rows)
    (args.output_dir / "E2_QUALITY_SUMMARY.json").write_text(json.dumps({"status": "ok", "datasets": summaries}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.output_dir / "E2_QUALITY_ROWS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    lines = ["# E2 Expanded Quality Evaluation", "", "Canary rows are excluded. Confidence intervals use 10,000 CPU bootstrap resamples.", "", "| Dataset | n | Main metric | Mean | Bootstrap 95% CI | Success | Source-image fallback |", "| --- | ---: | --- | ---: | --- | ---: | ---: |"]
    for item in summaries:
        low, high = item["main_bootstrap_95_ci"]
        lines.append(f"| {item['dataset']} | {item['sample_count']} | {item['main_metric']} | {item['main_mean']:.4f} | [{low:.4f}, {high:.4f}] | {item['success_rate']:.4f} | {item['source_image_fallback_rate']:.4f} |")
    lines += ["", "## Boundary", "", "These are source/document-cued results. The conservative hybrid falls back to the full source image on most rows, so this evidence supports the bounded direct source-image quality path, not crop-only or no-source retrieval.", ""]
    (args.output_dir / "E2_QUALITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "datasets": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()