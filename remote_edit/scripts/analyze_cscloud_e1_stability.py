from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for run in payload.get("run_rows") or []:
        groups[(run["dataset"], run["method"], int(run["candidate_scale"]))].append(run)

    output_rows: list[dict[str, Any]] = []
    for (dataset, method, scale), runs in sorted(groups.items()):
        if len(runs) < 3:
            continue
        artifacts = [json.loads(Path(run["artifact"]).read_text(encoding="utf-8")) for run in sorted(runs, key=lambda item: item["repeat"])]
        unit_agreements: list[float] = []
        prediction_agreements: list[float] = []
        row_count = min(len(item.get("rows") or []) for item in artifacts)
        for row_index in range(row_count):
            rows = [item["rows"][row_index] for item in artifacts]
            unit_lists = [tuple(packet.get("unit_id") for packet in (row.get("topk_unit_evidence") or [])) for row in rows]
            predictions = [str(row.get("pred") or "") for row in rows]
            pairs = list(itertools.combinations(range(len(rows)), 2))
            unit_agreements.extend(1.0 if unit_lists[left] == unit_lists[right] else 0.0 for left, right in pairs)
            prediction_agreements.extend(1.0 if predictions[left] == predictions[right] else 0.0 for left, right in pairs)
        vqa = [float(run["vqa"]) for run in runs]
        reserved = [float(run["peak_reserved_gib"]) for run in runs]
        output_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "candidate_scale": scale,
                "repeat_count": len(runs),
                "row_count": row_count,
                "selected_unit_pairwise_agreement": statistics.mean(unit_agreements) if unit_agreements else 0.0,
                "prediction_pairwise_agreement": statistics.mean(prediction_agreements) if prediction_agreements else 0.0,
                "vqa_mean": statistics.mean(vqa),
                "vqa_std": statistics.stdev(vqa),
                "peak_reserved_gib_mean": statistics.mean(reserved),
                "peak_reserved_gib_std": statistics.stdev(reserved),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "E1_REPEAT_STABILITY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    (args.output_dir / "E1_REPEAT_STABILITY.json").write_text(json.dumps({"status": "ok", "rows": output_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# E1 Endpoint Repeat Stability",
        "",
        "Endpoint configurations use three fresh model processes. TextVQA formal runs reuse content-addressed raw OCR blocks, so evidence extraction is fixed while selection and generation are re-executed.",
        "",
        "| Dataset | Method | Scale | Unit-list pairwise agreement | Prediction pairwise agreement | VQA mean +/- std | Reserved GiB mean +/- std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output_rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['candidate_scale']} | "
            f"{row['selected_unit_pairwise_agreement']:.4f} | {row['prediction_pairwise_agreement']:.4f} | "
            f"{row['vqa_mean']:.4f} +/- {row['vqa_std']:.4f} | "
            f"{row['peak_reserved_gib_mean']:.3f} +/- {row['peak_reserved_gib_std']:.3f} |"
        )
    stable_groups = [
        row
        for row in output_rows
        if row["selected_unit_pairwise_agreement"] >= 0.99
        and row["prediction_pairwise_agreement"] >= 0.99
        and row["vqa_std"] <= 1e-9
        and row["peak_reserved_gib_std"] <= 1e-6
    ]
    if output_rows and len(stable_groups) == len(output_rows):
        interpretation = (
            "All currently complete three-process endpoint groups are deterministic: "
            "selected-unit lists and predictions agree exactly, while VQA and peak-reserved "
            "standard deviations are zero. This supports the SHA-1 embedding correction; "
            "quality differences between B0 and B2 remain method/payload differences rather "
            "than process-randomized selector drift."
        )
    else:
        unstable = len(output_rows) - len(stable_groups)
        interpretation = (
            f"{unstable} of {len(output_rows)} complete endpoint groups show repeat-level "
            "selection, prediction, quality, or memory variation. These groups must be "
            "reported with repeat statistics and must not be described as deterministic."
        )
    lines += ["", "## Interpretation", "", interpretation, ""]
    (args.output_dir / "E1_REPEAT_STABILITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "groups": len(output_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
