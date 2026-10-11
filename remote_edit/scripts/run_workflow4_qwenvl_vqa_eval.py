"""Workflow4 real multimodal VQA smoke with Qwen-VL.

This runner evaluates a small real DocVQA/TextVQA/ChartQA slice with a local
Qwen2-VL/Qwen2.5-VL-style model.  It is a dataset-backed multimodal reference
path for Workflow4, not yet a full HeteroKV-v3 external-memory result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import string
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import resource  # type: ignore
except Exception:  # pragma: no cover - Windows local smoke fallback.
    resource = None  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.runtime_support import set_cap, set_seed  # noqa: E402


DATASET_DEFAULTS: Dict[str, Dict[str, Optional[str]]] = {
    "textvqa": {
        "dataset_id": "lmms-lab/TextVQA",
        "config": None,
        "split": "validation",
        "question_key": "question",
        "answer_key": "answers",
        "image_key": "image",
    },
    "chartqa": {
        "dataset_id": "HuggingFaceM4/ChartQA",
        "config": None,
        "split": "val",
        "question_key": "query",
        "answer_key": "label",
        "image_key": "image",
    },
    "docvqa": {
        "dataset_id": "HuggingFaceM4/DocumentVQA",
        "config": None,
        "split": "validation",
        "question_key": "question",
        "answer_key": "answers",
        "image_key": "image",
    },
}


def _rss_gib() -> float:
    if resource is None:
        return 0.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _normalize_answer(text: Any) -> str:
    value = str(text or "").lower().strip()
    value = value.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    translator = str.maketrans({ch: " " for ch in string.punctuation})
    value = value.translate(translator)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _tokens(text: Any) -> List[str]:
    normalized = _normalize_answer(text)
    return normalized.split() if normalized else []


def _token_f1(pred: str, answer: str) -> float:
    pred_tokens = _tokens(pred)
    answer_tokens = _tokens(answer)
    if not pred_tokens or not answer_tokens:
        return float(pred_tokens == answer_tokens)
    common: Dict[str, int] = {}
    for tok in answer_tokens:
        common[tok] = common.get(tok, 0) + 1
    overlap = 0
    for tok in pred_tokens:
        if common.get(tok, 0) > 0:
            overlap += 1
            common[tok] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / max(1e-12, precision + recall)


def _float_value(text: Any) -> Optional[float]:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(text or "").replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _relaxed_numeric_match(pred: str, answer: str) -> bool:
    pred_value = _float_value(pred)
    answer_value = _float_value(answer)
    if pred_value is None or answer_value is None:
        return False
    if answer_value == 0:
        return abs(pred_value - answer_value) <= 1e-6
    return abs(pred_value - answer_value) / abs(answer_value) <= 0.05


def _as_answer_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, Iterable):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _score_prediction(dataset: str, pred: str, answers: List[str]) -> Dict[str, Any]:
    normalized_pred = _normalize_answer(pred)
    normalized_answers = [_normalize_answer(answer) for answer in answers]
    exact_matches = [normalized_pred == answer and bool(answer) for answer in normalized_answers]
    relaxed_matches = [_relaxed_numeric_match(pred, answer) for answer in answers]
    contains_matches = [
        bool(answer) and (answer in normalized_pred or normalized_pred in answer)
        for answer in normalized_answers
    ]
    best_f1 = max([_token_f1(pred, answer) for answer in answers] or [0.0])
    if dataset == "textvqa" and len(answers) > 1:
        vqa_soft = min(1.0, sum(1 for hit in exact_matches if hit) / 3.0)
    else:
        vqa_soft = 1.0 if any(exact_matches) or any(relaxed_matches) else 0.0
    return {
        "exact_match": bool(any(exact_matches)),
        "relaxed_numeric_match": bool(any(relaxed_matches)),
        "contains_match": bool(any(contains_matches)),
        "token_f1": best_f1,
        "vqa_soft_score": vqa_soft,
        "normalized_pred": normalized_pred,
        "normalized_answers": normalized_answers[:10],
    }


def _load_dataset_slice(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    defaults = DATASET_DEFAULTS[args.dataset]
    dataset_id = args.dataset_id or defaults["dataset_id"]
    config = args.dataset_config if args.dataset_config is not None else defaults["config"]
    split = args.split or defaults["split"]
    load_kwargs: Dict[str, Any] = {"split": split, "streaming": args.streaming}
    if config:
        ds = load_dataset(dataset_id, config, **load_kwargs)
    else:
        ds = load_dataset(dataset_id, **load_kwargs)
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        if idx < args.start:
            continue
        rows.append(dict(item))
        if len(rows) >= args.limit:
            break
    return rows


def _save_image(image: Any, path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        image.convert("RGB").save(path)
        return path
    if isinstance(image, dict):
        if image.get("path"):
            src = Path(str(image["path"]))
            if src.exists():
                return src
        if image.get("bytes"):
            import io

            Image.open(io.BytesIO(image["bytes"])).convert("RGB").save(path)
            return path
    raise ValueError(f"unsupported image value: {type(image).__name__}")


def _extract_fields(args: argparse.Namespace, item: Dict[str, Any]) -> Tuple[str, List[str], Any, Dict[str, Any]]:
    defaults = DATASET_DEFAULTS[args.dataset]
    question_key = args.question_key or defaults["question_key"] or "question"
    answer_key = args.answer_key or defaults["answer_key"] or "answers"
    image_key = args.image_key or defaults["image_key"] or "image"
    question = str(item.get(question_key, ""))
    answers = _as_answer_list(item.get(answer_key))
    image = item.get(image_key)
    meta = {
        "question_key": question_key,
        "answer_key": answer_key,
        "image_key": image_key,
        "image_id": item.get("image_id") or item.get("questionId") or item.get("question_id"),
        "question_id": item.get("question_id") or item.get("questionId"),
        "ocr_tokens": item.get("ocr_tokens"),
        "question_types": item.get("question_types"),
        "raw_keys": sorted(str(key) for key in item.keys()),
    }
    return question, answers, image, meta


def _build_messages(args: argparse.Namespace, question: str, image_path: Optional[Path]) -> List[Dict[str, Any]]:
    if args.dataset == "chartqa":
        instruction = "Answer the chart question with a short phrase or number. Do not explain."
    elif args.dataset == "docvqa":
        instruction = "Answer the document image question with the exact short answer. Do not explain."
    else:
        instruction = "Answer the image question with the exact short text visible in the image. Do not explain."
    text = f"{instruction}\nQuestion: {question}"
    content: List[Dict[str, Any]] = []
    if args.generation_mode == "vl_full_image":
        if image_path is None:
            raise ValueError("vl_full_image mode requires an image")
        content.append({"type": "image", "image": str(image_path)})
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def _decode_with_qwenvl(
    *,
    model: Any,
    processor: Any,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    import torch
    from qwen_vl_utils import process_vision_info

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to("cuda:0") if hasattr(value, "to") else value for key, value in inputs.items()}
    before = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.time() - before
    input_len = int(inputs["input_ids"].shape[-1])
    generated = output_ids[:, input_len:]
    pred = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return pred, {
        "actual_input_tokens": input_len,
        "generated_tokens": int(generated.shape[-1]),
        "elapsed_sec": elapsed,
        "decode_ms_per_step": elapsed * 1000.0 / max(1, int(generated.shape[-1])),
    }


def run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    set_seed(args.seed)
    set_cap(args.cap_gib)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.manual_seed_all(args.seed)

    model_path = Path(args.model_path) if args.model_path else PROJECT_ROOT / "models" / "Qwen2-VL-7B"
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    processor_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if args.min_pixels > 0:
        processor_kwargs["min_pixels"] = int(args.min_pixels)
    if args.max_pixels > 0:
        processor_kwargs["max_pixels"] = int(args.max_pixels)
    processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    rows_in = _load_dataset_slice(args)
    rows: List[Dict[str, Any]] = []
    score_sum = 0.0
    exact_sum = 0
    contains_sum = 0
    f1_sum = 0.0
    started = time.time()

    image_dir = PROJECT_ROOT / "experiments" / "workflow4_multimodal_images" / args.dataset
    for offset, item in enumerate(rows_in):
        index = args.start + offset
        question, answers, image, meta = _extract_fields(args, item)
        image_path: Optional[Path] = None
        if args.generation_mode == "vl_full_image":
            suffix = "jpg" if args.dataset == "textvqa" else "png"
            image_path = _save_image(image, image_dir / f"{args.dataset}_{index:05d}.{suffix}")
        messages = _build_messages(args, question, image_path)
        row: Dict[str, Any] = {
            "dataset": args.dataset,
            "index": index,
            "question": question,
            "answers": answers[:10],
            "generation_mode": args.generation_mode,
            "image_path": None if image_path is None else str(image_path),
            "metadata": meta,
            "prompt_preview": str(messages)[: args.prompt_preview_chars],
        }
        try:
            pred, timing = _decode_with_qwenvl(
                model=model,
                processor=processor,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
            )
            scores = _score_prediction(args.dataset, pred, answers)
            row.update(
                {
                    "pred": pred,
                    "scores": scores,
                    "oom": False,
                    "error": None,
                    **timing,
                    "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                    "max_rss_gib": _rss_gib(),
                }
            )
            score_sum += float(scores["vqa_soft_score"])
            exact_sum += int(scores["exact_match"] or scores["relaxed_numeric_match"])
            contains_sum += int(scores["contains_match"])
            f1_sum += float(scores["token_f1"])
        except torch.cuda.OutOfMemoryError as exc:
            row.update(
                {
                    "pred": "",
                    "scores": {},
                    "oom": True,
                    "error": str(exc),
                    "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                    "max_rss_gib": _rss_gib(),
                }
            )
        rows.append(row)
        torch.cuda.empty_cache()

    total = max(1, len(rows))
    status = "ok"
    if any(bool(row.get("oom")) for row in rows):
        status = "partial_oom"
    return {
        "status": status,
        "workflow": "workflow4_real_multimodal_vqa_reference",
        "scope": "real_dataset_qwen_vl_small_slice_not_yet_heterokv_v3_external_memory",
        "dataset": args.dataset,
        "dataset_id": args.dataset_id or DATASET_DEFAULTS[args.dataset]["dataset_id"],
        "dataset_config": args.dataset_config or DATASET_DEFAULTS[args.dataset]["config"],
        "split": args.split or DATASET_DEFAULTS[args.dataset]["split"],
        "generation_mode": args.generation_mode,
        "model_path": str(model_path),
        "min_pixels": int(args.min_pixels),
        "max_pixels": int(args.max_pixels),
        "seed": args.seed,
        "start": args.start,
        "limit": args.limit,
        "num_rows": len(rows),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_index_arg": args.gpu_index,
        "transformers_version": transformers.__version__,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "metrics": {
            "vqa_soft_score": score_sum / total,
            "exact_or_relaxed_accuracy": exact_sum / total,
            "contains_accuracy": contains_sum / total,
            "avg_token_f1": f1_sum / total,
        },
        "elapsed_sec": time.time() - started,
        "rows": rows,
        "boundary": (
            "This is real multimodal VQA generation using a local Qwen-VL model and "
            "real benchmark slices. It is a Workflow4 reference/entry point; it does "
            "not yet evaluate HeteroKV-v3 multi-unit retrieval or approximate FullKV."
        ),
    }


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    metrics = result["metrics"]
    lines = [
        "# Workflow4 Real Multimodal VQA Reference",
        "",
        f"- status: `{result['status']}`",
        f"- dataset: `{result['dataset']}` / `{result['dataset_id']}`",
        f"- split: `{result['split']}`",
        f"- model: `{result['model_path']}`",
        f"- mode: `{result['generation_mode']}`",
        f"- rows: `{result['num_rows']}`",
        f"- VQA soft score: `{metrics['vqa_soft_score']:.4f}`",
        f"- exact/relaxed accuracy: `{metrics['exact_or_relaxed_accuracy']:.4f}`",
        f"- contains accuracy: `{metrics['contains_accuracy']:.4f}`",
        f"- avg token F1: `{metrics['avg_token_f1']:.4f}`",
        "",
        "## Boundary",
        "",
        result["boundary"],
        "",
        "| Idx | Question | Prediction | Gold | Score | F1 | Peak reserved GiB |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in result["rows"][:30]:
        scores = row.get("scores") or {}
        lines.append(
            "| {idx} | {question} | `{pred}` | `{gold}` | {score:.2f} | {f1:.2f} | {peak:.2f} |".format(
                idx=row.get("index"),
                question=str(row.get("question", "")).replace("|", "/")[:140],
                pred=str(row.get("pred", "")).replace("|", "/")[:80],
                gold=", ".join(str(answer) for answer in row.get("answers", [])[:3]).replace("|", "/")[:80],
                score=float(scores.get("vqa_soft_score", 0.0)),
                f1=float(scores.get("token_f1", 0.0)),
                peak=float(row.get("max_reserved_gib", 0.0) or 0.0),
            )
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="textvqa")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--question-key", default=None)
    parser.add_argument("--answer-key", default=None)
    parser.add_argument("--image-key", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--cap-gib", type=float, default=34.0)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--min-pixels", type=int, default=0)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--prompt-preview-chars", type=int, default=900)
    parser.add_argument("--generation-mode", choices=["vl_full_image", "no_image"], default="vl_full_image")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="eager")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--md-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_eval(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(result, args.md_output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"],
                "mode": result["generation_mode"],
                "metrics": result["metrics"],
                "output": str(args.output),
                "md_output": str(args.md_output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
