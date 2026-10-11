"""Recompute core paper aggregates from sanitized numeric records, offline only."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def table(name):
    with (ROOT / "paper/tables" / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/public_reanalysis.json")
    args = parser.parse_args()
    checks = []

    def equal(label, actual, expected):
        good = math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9)
        checks.append({"check": label, "actual": actual, "expected": float(expected), "ok": good})

    manifest = json.loads((ROOT / "results/run_manifest.json").read_text(encoding="utf-8"))
    runs = {}
    e1_endpoints = table("e1_endpoint_mean_sd.csv")
    e3_costs = table("e3_cost_and_budget.csv")
    e3_coverage = table("e3_prepack_coverage_diagnostics.csv")
    e4_groups = table("e4_serving_mean_sd.csv")
    e2_rows = table("e2_large_sample_quality.csv")
    for item in manifest["runs"]:
        data = load(ROOT / item["public_artifact"])
        runs[item["run_id"]] = data
        equal(item["run_id"] + "/row_count", len(data["rows"]), item["rows"])
        if data["evidence_group"] == "E3":
            saved = next(r for r in e3_costs if r["dataset"] == data["dataset"] and r["artifact_method"] == data["method"])
            values = [r["scores"]["vqa_soft_score"] for r in data["rows"]]
            equal(item["run_id"] + "/qa", statistics.mean(values), saved["qa"])
            tokens = [r["answer_timing"]["actual_input_tokens"] for r in data["rows"]]
            equal(item["run_id"] + "/tokens", statistics.mean(tokens), saved["input_tokens_mean"])
            times = [r["answer_timing"]["elapsed_sec"] * 1000 for r in data["rows"]]
            equal(item["run_id"] + "/generation_ms", statistics.mean(times), saved["generation_ms_mean"])
            for public_key, raw_key in (("peak_allocated_gib", "max_allocated_gib"),
                                        ("peak_reserved_gib", "max_reserved_gib"),
                                        ("host_process_peak_rss_gib", "max_rss_gib")):
                equal(item["run_id"] + "/" + public_key, data["memory_summary"][raw_key], saved[public_key])
            cov = next(r for r in e3_coverage if r["dataset"] == data["dataset"] and r["artifact_method"] == data["method"])
            flags = [r["unit_answer_recall_at_12"] for r in data["rows"]]
            equal(item["run_id"] + "/prepack_coverage", statistics.mean(flags), cov["prepack_answer_coverage"])
            if data["dataset"] == "docvqa":
                anls_row = next(r for r in table("e3_docvqa_recomputed_anls.csv") if r["artifact_method"] == data["method"])
                equal(item["run_id"] + "/anls", statistics.mean(r["derived_anls_ge_0_5"] for r in data["rows"]), anls_row["recomputed_anls"])
        elif data["evidence_group"] == "E2":
            # The CSV includes multiple named metrics per dataset; inspect its schema.
            score = statistics.mean(r["derived_anls_ge_0_5"] if data["dataset"] == "docvqa"
                                    else r["scores"]["vqa_soft_score"] for r in data["rows"])
            candidates = [r for r in e2_rows if r["dataset"] == data["dataset"]]
            saved = candidates[0]
            value = saved.get("score", saved.get("mean", saved.get("main_score")))
            if value is not None:
                equal(item["run_id"] + "/main_score", score, value)
            equal(item["run_id"] + "/n", len(data["rows"]), saved.get("n", saved.get("sample_count")))
            expected_rate = .999 if data["dataset"] == "docvqa" else .991
            equal(item["run_id"] + "/fallback", statistics.mean(not r["unit_decision"]["use_unit"] for r in data["rows"]), expected_rate)
    for endpoint in e1_endpoints:
        group = [p for p in runs.values() if p["evidence_group"] == "E1" and
                 p["dataset"] == endpoint["dataset"] and p["method"] == endpoint["method"] and
                 p["recorded_config"]["e1_candidate_scale_requested"] == int(endpoint["candidate_units"])]
        equal("E1/repeats", len(group), endpoint["repeats"])
        for name, raw in (("allocated", "max_allocated_gib"), ("reserved", "max_reserved_gib")):
            values = [p["memory_summary"][raw] for p in group]
            equal("E1/" + name + "_mean", statistics.mean(values), endpoint[name + "_mean_gib"])
            equal("E1/" + name + "_sd", statistics.stdev(values), endpoint[name + "_sd_gib"])
    for saved in e4_groups:
        group = [p for p in runs.values() if p["evidence_group"] == "E4" and p["method"] == saved["mode"] and
                 p["recorded_config"]["concurrency"] == int(saved["concurrency"])]
        equal("E4/repeats", len(group), saved["repeats"])
        for name, metric in (("throughput_rps", "throughput_requests_per_sec"),
                              ("p95_sec", "latency_p95_sec"), ("nvml_peak_gib", "nvml_peak_used_gib")):
            values = [p["metrics"][metric] for p in group]
            equal("E4/" + name + "_mean", statistics.mean(values), saved[name + "_mean"])
            equal("E4/" + name + "_sd", statistics.stdev(values), saved[name + "_std"])
        for p in group:
            latencies = [r["end_to_end_latency_sec"] for r in p["rows"] if r["status"] == "ok"]
            equal("E4/request_success", len(latencies), 100)
            equal("E4/throughput", len(latencies) / p["metrics"]["elapsed_sec"], p["metrics"]["throughput_requests_per_sec"])
            equal("E4/text_payload", sum(r["text_payload_bytes"] for r in p["rows"]), p["metrics"]["payload_text_bytes_total"])
            equal("E4/image_payload", sum(r["image_payload_bytes"] for r in p["rows"]), p["metrics"]["payload_image_bytes_total"])
    failed = [c for c in checks if not c["ok"]]
    output = {"status": "passed" if not failed else "failed", "checks": checks,
              "checks_count": len(checks), "failed_count": len(failed), "new_experiments": 0,
              "scope": "numeric reaggregation; omitted original labels are not silently reconstructed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "checks": len(checks), "failed": failed}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
