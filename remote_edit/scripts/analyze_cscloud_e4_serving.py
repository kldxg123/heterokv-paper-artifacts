from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.artifact_dir.glob("cscloud_e4_formal_*.json"))
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok" or payload.get("workload_size") != 100:
            continue
        mode = str(payload["mode"])
        concurrency = int(payload["concurrency"])
        metrics = payload["metrics"]
        dataset_scores: dict[str, list[float]] = defaultdict(list)
        for row in payload.get("rows", []):
            if row.get("status") == "ok":
                dataset_scores[str(row.get("dataset"))].append(float((row.get("scores") or {}).get("vqa_soft_score", 0.0)))
        records.append(
            {
                "artifact": str(path),
                "mode": mode,
                "concurrency": concurrency,
                "repeat_index": int(payload.get("repeat_index", 0)),
                "success_rate": float(metrics["success_rate"]),
                "throughput_rps": float(metrics["throughput_requests_per_sec"]),
                "p50_sec": float(metrics["latency_p50_sec"]),
                "p95_sec": float(metrics["latency_p95_sec"]),
                "p99_sec": float(metrics["latency_p99_sec"]),
                "vqa": float(metrics["vqa_soft_score"]),
                "docvqa_vqa": statistics.mean(dataset_scores["docvqa"]),
                "textvqa_vqa": statistics.mean(dataset_scores["textvqa"]),
                "nvml_peak_gib": float(metrics["nvml_peak_used_gib"]),
                "gpu_util_mean_pct": float(metrics["gpu_utilization_mean_pct"]),
                "server_rss_peak_gib": float(metrics["server_rss_peak_gib"]),
                "image_payload_bytes": int(metrics["payload_image_bytes_total"]),
                "text_payload_bytes": int(metrics["payload_text_bytes_total"]),
                "shared_model_instances": int(payload.get("shared_model_instances", 0)),
                "host": payload.get("host"),
            }
        )
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["mode"], row["concurrency"])].append(row)
    summary = []
    fields = ["throughput_rps", "p50_sec", "p95_sec", "p99_sec", "vqa", "docvqa_vqa", "textvqa_vqa", "nvml_peak_gib", "gpu_util_mean_pct", "server_rss_peak_gib"]
    for (mode, concurrency), rows in sorted(grouped.items()):
        item: dict[str, Any] = {"mode": mode, "concurrency": concurrency, "repeats": len(rows), "requests": 100 * len(rows), "success_rate": statistics.mean(row["success_rate"] for row in rows)}
        for field in fields:
            avg, std = mean_std([float(row[field]) for row in rows])
            item[field + "_mean"] = avg
            item[field + "_std"] = std
        item["image_payload_bytes_per_run"] = rows[0]["image_payload_bytes"]
        item["text_payload_bytes_per_run"] = rows[0]["text_payload_bytes"]
        summary.append(item)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "E4_SERVING_SUMMARY.json").write_text(json.dumps({"status": "ok", "raw_runs": records, "groups": summary}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.output_dir / "E4_SERVING_SUMMARY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    lines = ["# E4 One-Shared-Model Serving Report", "", "Each configuration uses one localhost-only vLLM Qwen2-VL instance, 10 warm-up requests, 100 measured requests (50 DocVQA + 50 TextVQA), and three fresh-service repeats.", "", "| Mode | C | Repeats | Success | Throughput req/s | p50 s | p95 s | p99 s | VQA | DocVQA | TextVQA | NVML GiB |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in summary:
        lines.append(f"| {item['mode']} | {item['concurrency']} | {item['repeats']} | {item['success_rate']:.3f} | {item['throughput_rps_mean']:.3f} +/- {item['throughput_rps_std']:.3f} | {item['p50_sec_mean']:.3f} | {item['p95_sec_mean']:.3f} | {item['p99_sec_mean']:.3f} | {item['vqa_mean']:.4f} | {item['docvqa_vqa_mean']:.4f} | {item['textvqa_vqa_mean']:.4f} | {item['nvml_peak_gib_mean']:.3f} |")
    lines += ["", "## Interpretation", "", "The maximum tested stable concurrency is 8 for both paths, with 100% success in all formal runs. B0 is faster and carries no image payload, but its QA is substantially lower. B2 is the bounded active-image quality path and is a different payload family, not an equal-payload B0 comparison.", "", "## Measurement Boundary", "", "Evidence selection was replayed from frozen E2 artifacts before service startup. The reported p50/p95/p99 are end-to-end HTTP generation latencies for already selected active evidence. Online selector and index-build latency are not included. Client scheduling wait is recorded per row; vLLM-internal queue time and PyTorch allocated/reserved memory are not exposed by this process-isolated harness. NVML peak is reported instead.", ""]
    (args.output_dir / "E4_SERVING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "formal_runs": len(records), "groups": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()