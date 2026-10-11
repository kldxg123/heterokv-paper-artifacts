from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


NAME_RE = re.compile(
    r"cscloud_e1_(?P<dataset>docvqa|textvqa)_(?P<method>fixed_b0|b2_source_image)"
    r"_scale(?P<scale>\d+)_r(?P<repeat>\d+)_n(?P<n>\d+)_gpu0_[0-9_]+\.json$"
)


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _row_mean(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = (value or {}).get(key) if isinstance(value, dict) else None
        if value is not None:
            values.append(float(value))
    return mean(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected: dict[tuple[str, str, int], int] = {}
    for dataset in ("docvqa", "textvqa"):
        scales = [int(value) for value in manifest[dataset]["scales"]]
        for method in ("fixed_b0", "b2_source_image"):
            for scale in scales:
                expected[(dataset, method, scale)] = 3 if scale in {min(scales), max(scales)} else 1

    run_rows: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    seen_runs: set[tuple[str, str, int, int]] = set()
    for path in sorted(args.artifact_dir.glob("cscloud_e1_*.json")):
        match = NAME_RE.match(path.name)
        if not match:
            continue
        dataset = match.group("dataset")
        method = match.group("method")
        scale = int(match.group("scale"))
        repeat = int(match.group("repeat"))
        n = int(match.group("n"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("e1_embedding_hash") != "sha1_v1":
            continue
        key = (dataset, method, scale, repeat)
        if key in seen_runs:
            validation_errors.append(f"duplicate run key: {key}")
            continue
        seen_runs.add(key)
        rows = payload.get("rows") or []
        expected_indices = [int(value) for value in manifest[dataset]["indices"]]
        actual_indices = [int(value) for value in payload.get("e1_query_original_indices") or []]
        if actual_indices != expected_indices:
            validation_errors.append(f"{path.name}: fixed query indices differ from manifest")
        if n != len(expected_indices) or len(rows) != n:
            validation_errors.append(f"{path.name}: expected n={len(expected_indices)}, filename n={n}, rows={len(rows)}")
        if payload.get("e1_gold_used_by_selector") is not False:
            validation_errors.append(f"{path.name}: selector gold-use guard is not false")
        if int(payload.get("e1_candidate_scale_requested", -1)) != scale:
            validation_errors.append(f"{path.name}: candidate scale mismatch")
        available = [int(value) for value in (payload.get("e1_available_units_by_source") or {}).values()]
        if not available or min(available) < scale:
            validation_errors.append(f"{path.name}: one or more sources have fewer real units than requested")
        for row in rows:
            if int(row.get("e1_candidate_scale_requested", -1)) != scale:
                validation_errors.append(f"{path.name}: row candidate scale mismatch")
                break
            unit_ids = [str(item.get("unit_id")) for item in (row.get("topk_unit_evidence") or [])]
            if len(unit_ids) != len(set(unit_ids)):
                validation_errors.append(f"{path.name}: duplicate selected unit in row {row.get('index')}")
                break
        metrics = payload.get("metrics") or {}
        memory = payload.get("memory_summary") or {}
        timing = payload.get("timing") or {}
        run_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "payload_family": "unit_text_top4" if method == "fixed_b0" else "bounded_source_image",
                "candidate_scale": scale,
                "repeat": repeat,
                "sample_count": len(rows),
                "vqa": float(metrics.get("vqa_soft_score", 0.0) or 0.0),
                "unit_recall_at_4": float(metrics.get("unit_answer_recall_at_4", 0.0) or 0.0),
                "source_recall_at_1": float(metrics.get("source_recall_at_1", 0.0) or 0.0),
                "peak_allocated_gib": float(memory.get("max_allocated_gib", 0.0) or 0.0),
                "peak_reserved_gib": float(memory.get("max_reserved_gib", 0.0) or 0.0),
                "max_rss_gib": float(memory.get("max_rss_gib", 0.0) or 0.0),
                "ingest_sec": float(timing.get("ingest_elapsed_sec", 0.0) or 0.0),
                "answer_sec": float(timing.get("answer_elapsed_sec", 0.0) or 0.0),
                "prompt_tokens_mean": _row_mean(rows, ("answer_timing", "actual_input_tokens")),
                "generation_sec_mean": _row_mean(rows, ("answer_timing", "elapsed_sec")),
                "min_real_units_available": min(available) if available else 0,
                "artifact": str(path),
            }
        )

    for (dataset, method, scale), repeats in expected.items():
        found = sorted(row["repeat"] for row in run_rows if row["dataset"] == dataset and row["method"] == method and row["candidate_scale"] == scale)
        if found != list(range(repeats)):
            validation_errors.append(f"missing/extra repeats for {(dataset, method, scale)}: expected {list(range(repeats))}, found {found}")

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["dataset"], row["method"], row["candidate_scale"])].append(row)
    summary_rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        dataset, method, scale = key
        summary: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "payload_family": values[0]["payload_family"],
            "candidate_scale": scale,
            "repeat_count": len(values),
            "sample_count_per_run": values[0]["sample_count"],
        }
        for field in ("vqa", "unit_recall_at_4", "source_recall_at_1", "peak_allocated_gib", "peak_reserved_gib", "max_rss_gib", "ingest_sec", "answer_sec", "prompt_tokens_mean", "generation_sec_mean"):
            numbers = [float(row[field]) for row in values]
            summary[f"{field}_mean"] = mean(numbers)
            summary[f"{field}_std"] = stdev(numbers)
        summary["artifacts"] = ";".join(row["artifact"] for row in values)
        summary_rows.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = "complete" if not validation_errors and len(run_rows) == sum(expected.values()) else "incomplete_or_invalid"
    output = {
        "status": status,
        "scope": "A100 supporting rerun; fixed source/document-cued questions; real source-local non-image MemoryUnits; B0 and B2 are different payload families",
        "run_count": len(run_rows),
        "expected_run_count": sum(expected.values()),
        "validation_errors": validation_errors,
        "run_rows": run_rows,
        "summary_rows": summary_rows,
    }
    (args.output_dir / "E1_SCALING_SUMMARY.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if summary_rows:
        with (args.output_dir / "E1_SCALING_SUMMARY.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)

    lines = [
        "# E1 True Source-Local Candidate/Unit Scaling",
        "",
        f"Status: **{status}** ({len(run_rows)}/{sum(expected.values())} formal runs).",
        "",
        "The same fixed questions are evaluated at every scale. Each scale is a nested prefix of real source-local non-image MemoryUnits sorted by stable unit id; no units are duplicated or synthesized, and the selector cannot access gold answers. Endpoint memory settings are repeated three times.",
        "",
        "B0 uses four selected text units. B2 sends one bounded source image. These are separate payload families and are not equal-payload baselines.",
        "",
        "| Dataset | Method | Scale | Repeats | VQA mean +/- std | Unit recall@4 | Peak allocated GiB | Peak reserved GiB | Mean prompt tokens | Mean generation s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['candidate_scale']} | {row['repeat_count']} | "
            f"{row['vqa_mean']:.4f} +/- {row['vqa_std']:.4f} | {row['unit_recall_at_4_mean']:.4f} | "
            f"{row['peak_allocated_gib_mean']:.3f} | {row['peak_reserved_gib_mean']:.3f} | "
            f"{row['prompt_tokens_mean_mean']:.1f} | {row['generation_sec_mean_mean']:.3f} |"
        )
    lines += ["", "## Validation", ""]
    lines += [f"- {item}" for item in validation_errors] if validation_errors else ["- All fixed-sample, real-unit, gold-use, scale, duplicate-unit, and repeat-count checks passed."]
    lines += ["", "## Claim Boundary", "", "This experiment evaluates source-local candidate/unit scaling after a source/document handle is given. It does not establish unconstrained no-source discovery, total-memory O(1), or full-context equivalence.", ""]
    (args.output_dir / "E1_SCALING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    if summary_rows:
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
        colors = {"fixed_b0": "#2f6f9f", "b2_source_image": "#d66b2c"}
        for (dataset, method), values in sorted(defaultdict(list, {k: [] for k in []}).items()):
            pass
        for dataset in ("docvqa", "textvqa"):
            for method in ("fixed_b0", "b2_source_image"):
                values = sorted((row for row in summary_rows if row["dataset"] == dataset and row["method"] == method), key=lambda row: row["candidate_scale"])
                if not values:
                    continue
                label = f"{dataset} {method}"
                style = "-" if dataset == "docvqa" else "--"
                x = [row["candidate_scale"] for row in values]
                axes[0].plot(x, [row["vqa_mean"] for row in values], marker="o", linestyle=style, color=colors[method], label=label)
                axes[1].plot(x, [row["peak_reserved_gib_mean"] for row in values], marker="o", linestyle=style, color=colors[method])
                axes[2].plot(x, [row["unit_recall_at_4_mean"] for row in values], marker="o", linestyle=style, color=colors[method])
        axes[0].set(xlabel="Available source-local units", ylabel="VQA", title="Quality")
        axes[1].set(xlabel="Available source-local units", ylabel="Peak reserved GPU (GiB)", title="Observed GPU peak")
        axes[2].set(xlabel="Available source-local units", ylabel="Answer unit recall@4", title="Evidence recall")
        for axis in axes:
            axis.grid(alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.15, 1, 1))
        fig.savefig(args.output_dir / "E1_SCALING.png", dpi=220)
        fig.savefig(args.output_dir / "E1_SCALING.pdf")
        plt.close(fig)
    print(json.dumps({"status": status, "runs": len(run_rows), "errors": len(validation_errors)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
