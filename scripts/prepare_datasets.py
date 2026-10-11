"""Opt-in dataset acquisition with original-terms confirmation and slice verification."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def signature(question, answers):
    value = {"question": str(question or ""), "answers": [str(x) for x in answers or []]}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("docvqa", "textvqa"))
    parser.add_argument("--accept-original-terms", action="store_true")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.accept_original_terms:
        parser.error("Obtain authorization and accept original dataset terms before download")
    if args.output.exists():
        parser.error("Output exists; refusing to overwrite local data")
    os.environ["HF_ENDPOINT"] = args.endpoint
    from datasets import load_dataset
    repo = "HuggingFaceM4/DocumentVQA" if args.dataset == "docvqa" else "lmms-lab/TextVQA"
    n = 1000 if args.dataset == "docvqa" else 2000
    path = ROOT / "data/slices" / f"{args.dataset}_validation_first{n}.jsonl"
    expected = {row["index"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())}
    dataset = load_dataset(repo, split="validation", streaming=True, revision=args.revision)
    args.output.mkdir(parents=True)
    (args.output / "images").mkdir()
    checks = []
    records, labels = [], []
    for i, item in enumerate(dataset):
        if i >= n:
            break
        answers = [str(x) for x in item.get("answers") or []]
        actual = signature(item.get("question"), answers)
        if i not in expected or actual != expected[i]["question_annotation_signature_sha256"]:
            raise RuntimeError(f"Dataset slice mismatch at index {i}; partial output retained, do not evaluate as original")
        source_id = expected[i]["source_id"]
        image = item.get("image")
        if image is None:
            raise RuntimeError(f"Missing image at index {i}")
        image_path = Path("images") / f"{i:06d}.png"
        image.save(args.output / image_path)
        ids = {k: item[k] for k in ("questionId", "question_id", "image_id", "docId") if k in item}
        # Keep annotations separate; only visible source data is ingestible.
        records.append({"index": i, "source_id": source_id, "identifiers": ids,
                        "image_path": image_path.as_posix(), "question": item.get("question", ""),
                        "ocr_tokens": item.get("ocr_tokens") or []})
        labels.append({"index": i, "answers": answers})
        checks.append({"index": i, "signature_sha256": actual})
    if len(records) != n:
        raise RuntimeError("Dataset ended before the expected fixed prefix")
    for name, rows in (("questions_and_visible_sources.jsonl", records), ("evaluation_annotations.jsonl", labels)):
        (args.output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (args.output / "dataset_manifest.json").write_text(json.dumps({"dataset_id": repo, "split": "validation",
        "requested_revision": args.revision, "historical_revision": "unknown_not_recorded",
        "verified_count": n, "checks": checks}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "verified", "dataset": args.dataset, "count": n}))


if __name__ == "__main__":
    main()
