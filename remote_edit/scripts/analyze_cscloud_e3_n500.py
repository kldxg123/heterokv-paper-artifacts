"""Analyze strict stable-hash E3 first-500 same-source baselines."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DATASETS = ("textvqa", "docvqa")
METHODS = ("random", "bm25_ocr", "layout_aware", "heterokv_b0", "b2_source_image")


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for dataset in DATASETS:
        for method in METHODS:
            name = f"cscloud_e3_{dataset}_{method}_metadata_n500_gpu0_stablehash_20260814.json"
            path = args.artifact_dir / name
            if not path.exists():
                errors.append(f"missing: {name}")
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            result_rows = payload.get("rows") or []
            if payload.get("status") != "ok":
                errors.append(f"{name}: status={payload.get('status')}")
            if payload.get("embedding_hash") != "sha1_v1":
                errors.append(f"{name}: embedding_hash={payload.get('embedding_hash')}")
            if len(result_rows) != 500:
                errors.append(f"{name}: expected 500 rows, found {len(result_rows)}")
            if payload.get("query_binding") != "source_handle":
                errors.append(f"{name}: query_binding={payload.get('query_binding')}")

            metrics = payload.get("metrics") or {}
            memory = payload.get("memory_summary") or {}
            timing = payload.get("timing") or {}
            prompt_tokens = [
                float((row.get("answer_timing") or {}).get("actual_input_tokens", 0.0) or 0.0)
                for row in result_rows
            ]
            generation_sec = [
                float((row.get("answer_timing") or {}).get("elapsed_sec", 0.0) or 0.0)
                for row in result_rows
            ]
            payload_family = "bounded_source_image" if method == "b2_source_image" else "unit_text_top12"
            expected_answer_mode = "source_full_image" if method == "b2_source_image" else "unit_text_only"
            answer_modes = {str(row.get("answer_mode_effective") or row.get("answer_mode") or "") for row in result_rows}
            if answer_modes != {expected_answer_mode}:
                errors.append(f"{name}: expected answer mode {expected_answer_mode}, found {sorted(answer_modes)}")
            # selected_image_paths is a retrieval-candidate trace even in text-only
            # mode. Count only images actually consumed by the answer backend.
            active_images = [1.0 if expected_answer_mode == "source_full_image" else 0.0 for _ in result_rows]
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "payload_family": payload_family,
                    "answer_mode": expected_answer_mode,
                    "sample_count": len(result_rows),
                    "vqa": float(metrics.get("vqa_soft_score", 0.0) or 0.0),
                    "exact_or_relaxed": float(metrics.get("exact_or_relaxed_accuracy", 0.0) or 0.0),
                    "token_f1": float(metrics.get("avg_token_f1", 0.0) or 0.0),
                    "source_recall_at_1": float(metrics.get("source_recall_at_1", 0.0) or 0.0),
                    "unit_recall_at_k": float(metrics.get("topk_unit_answer_recall_at_k", 0.0) or 0.0),
                    "prompt_tokens_mean": mean(prompt_tokens),
                    "active_images_mean": mean([float(value) for value in active_images]),
                    "generation_sec_mean": mean(generation_sec),
                    "ingest_sec": float(timing.get("ingest_elapsed_sec", 0.0) or 0.0),
                    "answer_sec": float(timing.get("answer_elapsed_sec", 0.0) or 0.0),
                    "peak_allocated_gib": float(memory.get("max_allocated_gib", 0.0) or 0.0),
                    "peak_reserved_gib": float(memory.get("max_reserved_gib", 0.0) or 0.0),
                    "max_rss_gib": float(memory.get("max_rss_gib", 0.0) or 0.0),
                    "artifact": str(path),
                }
            )

    expected = len(DATASETS) * len(METHODS)
    status = "complete" if len(rows) == expected and not errors else "incomplete_or_invalid"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "status": status,
        "scope": "same-source first-500; unit-text top-12 methods and bounded source-image B2 are separate payload families",
        "row_count": len(rows),
        "expected_row_count": expected,
        "validation_errors": errors,
        "rows": rows,
    }
    (args.output_dir / "E3_BASELINE_N500.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if rows:
        with (args.output_dir / "E3_BASELINE_N500.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# E3 Stable-Hash Same-Source First-500 Baselines",
        "",
        f"Status: **{status}** ({len(rows)}/{expected} method-dataset rows).",
        "",
        "Random, BM25/OCR, layout-aware, and HeteroKV B0 use top-12 unit text. B2 uses one bounded source image and is a separate payload family, not an equal-payload selector baseline.",
        "",
        "| Dataset | Method | Family | n | VQA | Unit recall@k | Mean prompt tokens | Active images | Generation s | Peak reserved GiB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['payload_family']} | {row['sample_count']} | "
            f"{row['vqa']:.4f} | {row['unit_recall_at_k']:.4f} | {row['prompt_tokens_mean']:.1f} | "
            f"{row['active_images_mean']:.2f} | {row['generation_sec_mean']:.3f} | {row['peak_reserved_gib']:.3f} |"
        )
    lines += ["", "## Validation", ""]
    lines += [f"- {item}" for item in errors] if errors else ["- All ten artifacts passed status, stable-hash, n=500, and source-handle checks."]
    lines += [
        "",
        "## Claim Boundary",
        "",
        "This is a source/document-cued comparison. It does not demonstrate no-source discovery. B2 is the quality-oriented bounded active-image path; B0 and lexical/layout methods are lower-payload unit-text selectors.",
        "",
    ]
    (args.output_dir / "E3_BASELINE_N500_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.8))
        colors = {"unit_text_top12": "#2f6f9f", "bounded_source_image": "#d66b2c"}
        markers = {"docvqa": "o", "textvqa": "s"}
        for row in rows:
            label = f"{row['dataset']} {row['method']}"
            axes[0].scatter(row["peak_reserved_gib"], row["vqa"], color=colors[row["payload_family"]], marker=markers[row["dataset"]], label=label, s=48)
            axes[1].scatter(row["generation_sec_mean"], row["vqa"], color=colors[row["payload_family"]], marker=markers[row["dataset"]], s=48)
        axes[0].set(xlabel="Peak reserved GPU (GiB)", ylabel="VQA", title="Quality-memory")
        axes[1].set(xlabel="Mean generation latency (s)", ylabel="VQA", title="Quality-latency")
        for axis in axes:
            axis.grid(alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0.2, 1, 1))
        fig.savefig(args.output_dir / "E3_PARETO_N500.png", dpi=220)
        fig.savefig(args.output_dir / "E3_PARETO_N500.pdf")
        plt.close(fig)
    print(json.dumps({"status": status, "rows": len(rows), "errors": len(errors)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
