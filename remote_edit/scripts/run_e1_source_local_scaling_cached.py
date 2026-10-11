from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import run_workflow4_multimodal_memory_retrieval as base


def _parse_wrapper() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--e1-query-indices", required=True)
    parser.add_argument("--e1-candidate-scale", type=int, required=True)
    parser.add_argument("--e1-method", choices=["fixed_b0", "b2_source_image"], required=True)
    parser.add_argument("--e1-repeat-index", type=int, default=0)
    parser.add_argument("--e1-manifest-id", default="cscloud_e1_fixed_v1")
    return parser.parse_known_args()


def _indices(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or len(values) != len(set(values)) or min(values) < 0:
        raise ValueError("e1 query indices must be unique non-negative integers")
    return values


def main() -> None:
    wrapper, remaining = _parse_wrapper()
    query_indices = _indices(wrapper.e1_query_indices)
    if wrapper.e1_candidate_scale <= 0:
        raise ValueError("e1 candidate scale must be positive")

    sys.argv = [sys.argv[0], *remaining]
    args = base.parse_args()
    args.start = 0
    args.limit = len(query_indices)
    args.source_pool_limit = len(query_indices)
    args.query_binding = "source_handle"
    args.source_selection_mode = "packet_first"
    args.top_k = 1
    if wrapper.e1_method == "fixed_b0":
        args.answer_mode = "unit_text_only"
        args.unit_top_k = 4
    else:
        args.answer_mode = "source_full_image"
        args.unit_top_k = 4

    original_loader = base._load_dataset_slice
    original_retrieve = base._retrieve_units_for_source
    original_rapidocr = base._rapidocr_blocks
    available_by_source: dict[str, int] = {}
    ocr_cache_root = base.PROJECT_ROOT / "experiments" / "cscloud_e1_ocr_cache_v1"

    def cached_rapidocr(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        image_path = Path(kwargs["image_path"])
        digest = hashlib.sha256()
        digest.update(image_path.read_bytes())
        digest.update(str(kwargs.get("dataset") or "").encode("utf-8"))
        digest.update(str(kwargs.get("source_id") or "").encode("utf-8"))
        digest.update(str(kwargs.get("image_id") or "").encode("utf-8"))
        digest.update(str(kwargs.get("max_blocks", 80)).encode("ascii"))
        cache_path = ocr_cache_root / f"{digest.hexdigest()}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            info = dict(cached["info"])
            info["e1_cache_hit"] = True
            return list(cached["blocks"]), info
        blocks, info = original_rapidocr(**kwargs)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"blocks": blocks, "info": info}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(cache_path)
        result_info = dict(info)
        result_info["e1_cache_hit"] = False
        return blocks, result_info

    def fixed_loader(load_args: argparse.Namespace) -> list[dict[str, Any]]:
        expanded = argparse.Namespace(**vars(load_args))
        expanded.start = 0
        expanded.limit = max(query_indices) + 1
        expanded.source_pool_limit = expanded.limit
        rows = original_loader(expanded)
        if len(rows) <= max(query_indices):
            raise RuntimeError(f"dataset returned {len(rows)} rows; need index {max(query_indices)}")
        return [rows[index] for index in query_indices]

    def capped_retrieve(
        *,
        source_units: Sequence[Any],
        question: str,
        dataset: str,
        top_k: int,
        embedding_dim: int,
        selection_mode: str = "heterokv",
        seed: int = 0,
    ) -> list[Any]:
        ordered = sorted(
            (unit for unit in source_units if getattr(unit, "modality", None) != "image"),
            key=lambda unit: str(getattr(unit, "unit_id", "")),
        )
        source = str(getattr(ordered[0], "source_id", "unknown")) if ordered else "unknown"
        available_by_source[source] = max(available_by_source.get(source, 0), len(ordered))
        capped = ordered[: int(wrapper.e1_candidate_scale)]
        return original_retrieve(
            source_units=capped,
            question=question,
            dataset=dataset,
            top_k=top_k,
            embedding_dim=embedding_dim,
            selection_mode=selection_mode,
            seed=seed,
        )

    base._load_dataset_slice = fixed_loader
    base._retrieve_units_for_source = capped_retrieve
    base._rapidocr_blocks = cached_rapidocr
    result = base.run_eval(args)
    result.update(
        {
            "experiment": "E1_source_local_candidate_unit_scaling",
            "e1_manifest_id": wrapper.e1_manifest_id,
            "e1_query_original_indices": query_indices,
            "e1_candidate_scale_requested": int(wrapper.e1_candidate_scale),
            "e1_candidate_cap_policy": "real source-local non-image units sorted by stable unit_id; no duplication",
            "e1_method": wrapper.e1_method,
            "e1_repeat_index": int(wrapper.e1_repeat_index),
            "e1_available_units_by_source": available_by_source,
            "e1_gold_used_by_selector": False,
            "e1_ocr_cache": "content-addressed raw OCR blocks only; no query, answer, ranking, or generation output",
            "e1_embedding_hash": "sha1_v1",
        }
    )
    for position, row in enumerate(result.get("rows", [])):
        if position < len(query_indices):
            row["e1_original_dataset_index"] = query_indices[position]
        row["e1_candidate_scale_requested"] = int(wrapper.e1_candidate_scale)
        row["e1_method"] = wrapper.e1_method
        row["e1_repeat_index"] = int(wrapper.e1_repeat_index)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.md_output:
        base.write_markdown(result, Path(args.md_output))
    print(json.dumps({"status": result.get("status"), "rows": len(result.get("rows", [])), "metrics": result.get("metrics")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
