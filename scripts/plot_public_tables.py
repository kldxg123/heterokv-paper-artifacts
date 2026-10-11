"""Rebuild data plots from public CSVs into an optional output folder, not paper assets."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT / "paper/tables" / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/rebuilt_figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"random": "#777777", "bm25_ocr": "#368466", "layout_aware": "#b08638",
              "heterokv_b0": "#356d99", "b2_source_image": "#b55a52"}
    names = {"random": "Random text", "bm25_ocr": "OCR lexical", "layout_aware": "Layout",
             "heterokv_b0": "B0 text", "b2_source_image": "B2 image"}

    def save(fig, name):
        for suffix in ("pdf", "png"):
            fig.savefig(args.output_dir / (name + "." + suffix), dpi=240, bbox_inches="tight")
        plt.close(fig)

    data = rows("e1_source_local_scaling.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), constrained_layout=True)
    for ax, dataset in zip(axes, ("docvqa", "textvqa")):
        for method, color, label in (("fixed_b0", "#356d99", "B0 text"), ("b2_source_image", "#b55a52", "B2 image")):
            points = sorted((r for r in data if r["dataset"] == dataset and r["path"] == method),
                            key=lambda r: int(r["candidate_units"]))
            ax.plot([int(r["candidate_units"]) for r in points], [float(r["peak_reserved_gib"]) for r in points],
                    marker="o", color=color, linestyle="--" if method == "b2_source_image" else "-", label=label)
        ax.set(title=dataset, xlabel="Source-local candidate units", ylabel="Peak reserved memory (GiB)")
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    save(fig, "e1_source_local_scaling")
    data = rows("e3_cost_and_budget.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, dataset, title in zip(axes, ("docvqa", "textvqa"), ("DocVQA local relaxed QA", "TextVQA local soft VQA")):
        for method in names:
            point = next(r for r in data if r["dataset"] == dataset and r["artifact_method"] == method)
            ax.scatter(float(point["peak_reserved_gib"]), float(point["qa"]), color=colors[method],
                       marker="D" if method == "b2_source_image" else "o", label=names[method])
        ax.set(title=title, xlabel="Peak reserved memory (GiB)", ylabel="Local QA score")
        ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False)
    fig.subplots_adjust(bottom=.3, wspace=.35)
    save(fig, "e3_quality_resource_pareto")
    data = rows("e4_serving_mean_sd.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), constrained_layout=True)
    for mode, color, label in (("b0_text_only", "#356d99", "B0 text"), ("b2_source_image", "#b55a52", "B2 image")):
        points = sorted((r for r in data if r["mode"] == mode), key=lambda r: int(r["concurrency"]))
        for ax, metric in zip(axes, ("throughput_rps", "p95_sec")):
            ax.errorbar([int(r["concurrency"]) for r in points], [float(r[metric + "_mean"]) for r in points],
                        yerr=[float(r[metric + "_std"]) for r in points], marker="o", capsize=3, color=color, label=label)
    for ax, ylabel in zip(axes, ("Throughput (requests/s)", "p95 latency (s)")):
        ax.set(xlabel="Client concurrency", ylabel=ylabel, xticks=[1, 2, 4, 8])
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    save(fig, "e4_shared_serving")


if __name__ == "__main__":
    main()
