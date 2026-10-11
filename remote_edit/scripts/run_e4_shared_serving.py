"""E4 shared-model serving benchmark for source-cued multimodal QA.

The benchmark starts one localhost-only vLLM OpenAI-compatible server, submits
multiple concurrent clients to that shared model instance, and records
end-to-end latency, throughput, success, GPU, and CPU resource measurements.
It never starts one model per request.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import math
import mimetypes
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import threading
import time
from typing import Any
from urllib import error, request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_workflow4_qwenvl_vqa_eval import _score_prediction  # noqa: E402


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _unit_text(row: dict[str, Any], max_units: int) -> str:
    packets = row.get("top_unit_evidence") or row.get("topk_unit_evidence") or []
    chunks: list[str] = []
    for idx, packet in enumerate(packets[: max(1, max_units)], start=1):
        if not isinstance(packet, dict):
            continue
        text = str(packet.get("text") or packet.get("searchable_text") or packet.get("text_preview") or "").strip()
        if text:
            chunks.append(f"[Evidence {idx}]\n{text}")
    if not chunks:
        preview = str(row.get("selected_searchable_text_preview") or "").strip()
        if preview:
            chunks.append(f"[Evidence 1]\n{preview}")
    return "\n\n".join(chunks)


def _image_path(row: dict[str, Any]) -> str:
    value = row.get("selected_image_path")
    if value:
        return str(value)
    values = row.get("selected_image_paths") or []
    return str(values[0]) if values else ""


def build_workload(
    docvqa_artifact: Path,
    textvqa_artifact: Path,
    per_dataset: int,
    mode: str,
    max_units: int,
) -> list[dict[str, Any]]:
    workload: list[dict[str, Any]] = []
    for dataset, artifact in (("docvqa", docvqa_artifact), ("textvqa", textvqa_artifact)):
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        rows = list(payload.get("rows") or [])[:per_dataset]
        if len(rows) < per_dataset:
            raise ValueError(f"{artifact} contains {len(rows)} rows, expected {per_dataset}")
        for row in rows:
            item = {
                "request_id": f"{dataset}:{row.get('index')}",
                "dataset": dataset,
                "question": str(row.get("question") or ""),
                "answers": [str(x) for x in (row.get("answers") or [])],
                "mode": mode,
                "image_path": _image_path(row),
                "evidence_text": _unit_text(row, max_units),
            }
            if mode == "b2_source_image" and not item["image_path"]:
                raise ValueError(f"missing source image for {item['request_id']}")
            if mode == "b0_text_only" and not item["evidence_text"]:
                raise ValueError(f"missing unit evidence for {item['request_id']}")
            workload.append(item)
    return workload


def _data_url(path: str) -> str:
    image_path = Path(path)
    data = image_path.read_bytes()
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    question = item["question"]
    if item["mode"] == "b2_source_image":
        content: Any = [
            {"type": "image_url", "image_url": {"url": _data_url(item["image_path"])}},
            {
                "type": "text",
                "text": (
                    "Answer the question using only the provided source image. "
                    "Return only the short answer.\nQuestion: " + question
                ),
            },
        ]
    else:
        content = (
            "Answer the question using only the retrieved evidence below. "
            "Return only the short answer.\n\n"
            + item["evidence_text"]
            + "\n\nQuestion: "
            + question
        )
    return [{"role": "user", "content": content}]


def _http_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_server(base_url: str, timeout: float, process: subprocess.Popen[str]) -> None:
    deadline = time.time() + timeout
    last_error = "not_started"
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited with rc={process.returncode}: {last_error}")
        try:
            result = _http_json(f"{base_url}/v1/models", None, min(5.0, timeout))
            if result.get("data"):
                return
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}:{exc}"
        time.sleep(1.0)
    raise TimeoutError(f"vLLM server did not become ready: {last_error}")


class ResourceMonitor:
    def __init__(self, gpu_index: int, server_pid_getter: Any, interval: float = 0.25) -> None:
        self.gpu_index = gpu_index
        self.server_pid_getter = server_pid_getter
        self.interval = interval
        self.rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="e4-resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            row: dict[str, Any] = {"timestamp": time.time()}
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.gpu_index}",
                        "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5,
                )
                parts = [part.strip() for part in proc.stdout.strip().split(",")]
                if len(parts) >= 4:
                    row.update(
                        {
                            "gpu_used_mib": float(parts[0]),
                            "gpu_utilization_pct": float(parts[1]),
                            "gpu_temperature_c": float(parts[2]),
                            "gpu_power_w": float(parts[3]),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                row["gpu_error"] = f"{type(exc).__name__}:{exc}"
            try:
                pid = int(self.server_pid_getter() or 0)
                if pid > 0:
                    proc = subprocess.run(
                        ["ps", "-o", "rss=", "-p", str(pid)],
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=5,
                    )
                    if proc.stdout.strip():
                        row["server_rss_mib"] = float(proc.stdout.strip()) / 1024.0
            except Exception as exc:  # noqa: BLE001
                row["rss_error"] = f"{type(exc).__name__}:{exc}"
            self.rows.append(row)
            self._stop.wait(self.interval)


def _run_one(
    item: dict[str, Any],
    base_url: str,
    served_model_name: str,
    max_new_tokens: int,
    timeout: float,
    scheduled_at: float,
) -> dict[str, Any]:
    worker_started = time.perf_counter()
    payload = {
        "model": served_model_name,
        "messages": _messages(item),
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
    }
    started = time.perf_counter()
    status = "ok"
    error_text = None
    prediction = ""
    usage: dict[str, Any] = {}
    try:
        response = _http_json(f"{base_url}/v1/chat/completions", payload, timeout)
        choices = response.get("choices") or []
        if choices:
            prediction = str((choices[0].get("message") or {}).get("content") or "").strip()
        usage = dict(response.get("usage") or {})
    except error.HTTPError as exc:
        status = "http_error"
        error_text = f"HTTPError:{exc.code}:{exc.reason}"
    except Exception as exc:  # noqa: BLE001
        status = "timeout" if isinstance(exc, TimeoutError) else "error"
        error_text = f"{type(exc).__name__}:{exc}"
    ended = time.perf_counter()
    scores = _score_prediction(item["dataset"], prediction, item["answers"]) if status == "ok" else {}
    return {
        "request_id": item["request_id"],
        "dataset": item["dataset"],
        "mode": item["mode"],
        "status": status,
        "error": error_text,
        "prediction": prediction,
        "answers": item["answers"][:10],
        "scores": scores,
        "client_queue_wait_sec": max(0.0, worker_started - scheduled_at),
        "end_to_end_latency_sec": ended - started,
        "usage": usage,
        "active_images": 1 if item["mode"] == "b2_source_image" else 0,
        "active_units": 0 if item["mode"] == "b2_source_image" else len(
            [x for x in item["evidence_text"].split("[Evidence ") if x.strip()]
        ),
        "image_payload_bytes": Path(item["image_path"]).stat().st_size
        if item["mode"] == "b2_source_image" and Path(item["image_path"]).exists()
        else 0,
        "text_payload_bytes": len(item["evidence_text"].encode("utf-8")),
    }


def _run_batch(
    workload: list[dict[str, Any]],
    concurrency: int,
    base_url: str,
    served_model_name: str,
    max_new_tokens: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], float]:
    scheduled_at = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _run_one,
                item,
                base_url,
                served_model_name,
                max_new_tokens,
                timeout,
                scheduled_at,
            )
            for item in workload
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    elapsed = time.perf_counter() - scheduled_at
    result_order = {item["request_id"]: idx for idx, item in enumerate(workload)}
    results.sort(key=lambda row: result_order.get(str(row["request_id"]), 10**9))
    return results, elapsed


def _aggregate(rows: list[dict[str, Any]], elapsed: float, monitor_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    latencies = [float(row["end_to_end_latency_sec"]) for row in ok_rows]
    score_values = [float((row.get("scores") or {}).get("vqa_soft_score", 0.0)) for row in ok_rows]
    gpu_used = [float(row["gpu_used_mib"]) for row in monitor_rows if row.get("gpu_used_mib") is not None]
    gpu_util = [float(row["gpu_utilization_pct"]) for row in monitor_rows if row.get("gpu_utilization_pct") is not None]
    server_rss = [float(row["server_rss_mib"]) for row in monitor_rows if row.get("server_rss_mib") is not None]
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    return {
        "request_count": len(rows),
        "success_count": len(ok_rows),
        "success_rate": len(ok_rows) / max(1, len(rows)),
        "status_counts": status_counts,
        "elapsed_sec": elapsed,
        "throughput_requests_per_sec": len(ok_rows) / max(elapsed, 1e-9),
        "latency_mean_sec": statistics.mean(latencies) if latencies else None,
        "latency_p50_sec": _percentile(latencies, 0.50),
        "latency_p95_sec": _percentile(latencies, 0.95),
        "latency_p99_sec": _percentile(latencies, 0.99),
        "vqa_soft_score": statistics.mean(score_values) if score_values else None,
        "nvml_peak_used_gib": max(gpu_used) / 1024.0 if gpu_used else None,
        "gpu_utilization_mean_pct": statistics.mean(gpu_util) if gpu_util else None,
        "gpu_utilization_peak_pct": max(gpu_util) if gpu_util else None,
        "server_rss_peak_gib": max(server_rss) / 1024.0 if server_rss else None,
        "payload_image_bytes_total": sum(int(row.get("image_payload_bytes") or 0) for row in rows),
        "payload_text_bytes_total": sum(int(row.get("text_payload_bytes") or 0) for row in rows),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("E4 service must remain localhost-only")
    workload = build_workload(
        Path(args.docvqa_artifact),
        Path(args.textvqa_artifact),
        args.per_dataset,
        args.mode,
        args.max_units,
    )
    if args.request_limit > 0:
        workload = workload[: args.request_limit]
    if not workload:
        raise ValueError("empty workload")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(Path(args.model_path)),
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--limit-mm-per-prompt",
        '{"image": 1}',
        "--trust-remote-code",
    ]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    server_log_path = output_path.with_suffix(".server.log")
    server_log = server_log_path.open("w", encoding="utf-8")
    server_process: subprocess.Popen[str] | None = None
    monitor: ResourceMonitor | None = None
    result: dict[str, Any] = {
        "status": "initializing",
        "mode": args.mode,
        "concurrency": args.concurrency,
        "repeat_index": args.repeat_index,
        "server_command_redacted": [
            token if token != str(Path(args.model_path)) else "<MODEL_PATH>" for token in command
        ],
        "host": args.host,
        "port": args.port,
        "shared_model_instances": 1,
        "workload_size": len(workload),
        "warmup_size": min(args.warmup_requests, len(workload)),
        "model_path": args.model_path,
        "gpu_index": args.gpu_index,
    }
    try:
        server_process = subprocess.Popen(
            command,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        base_url = f"http://{args.host}:{args.port}"
        _wait_server(base_url, args.server_start_timeout, server_process)

        warmup = workload[: min(args.warmup_requests, len(workload))]
        warmup_rows, warmup_elapsed = _run_batch(
            warmup,
            args.concurrency,
            base_url,
            args.served_model_name,
            args.max_new_tokens,
            args.request_timeout,
        )
        if any(row["status"] != "ok" for row in warmup_rows):
            raise RuntimeError("warmup contained failed requests")

        monitor = ResourceMonitor(args.gpu_index, lambda: server_process.pid if server_process else 0)
        monitor.start()
        rows, elapsed = _run_batch(
            workload,
            args.concurrency,
            base_url,
            args.served_model_name,
            args.max_new_tokens,
            args.request_timeout,
        )
        monitor.stop()
        result.update(
            {
                "status": "ok" if all(row["status"] == "ok" for row in rows) else "partial_failure",
                "warmup_elapsed_sec": warmup_elapsed,
                "warmup_rows": warmup_rows,
                "rows": rows,
                "metrics": _aggregate(rows, elapsed, monitor.rows),
                "resource_samples": monitor.rows,
                "server_pid": server_process.pid,
                "server_log": str(server_log_path),
                "boundary": (
                    "One localhost-only shared vLLM model instance. Client concurrency is real HTTP "
                    "concurrency. Queue waiting is client-side scheduling wait; vLLM internal queue "
                    "time is not exposed by this harness."
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001
        if monitor is not None:
            monitor.stop()
        result.update({"status": "error", "error": f"{type(exc).__name__}:{exc}"})
    finally:
        if server_process is not None and server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait(timeout=10)
        server_log.close()

    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docvqa-artifact", required=True)
    parser.add_argument("--textvqa-artifact", required=True)
    parser.add_argument("--mode", choices=["b0_text_only", "b2_source_image"], required=True)
    parser.add_argument("--per-dataset", type=int, default=50)
    parser.add_argument("--request-limit", type=int, default=0)
    parser.add_argument("--max-units", type=int, default=12)
    parser.add_argument("--concurrency", type=int, choices=[1, 2, 4, 8], required=True)
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--warmup-requests", type=int, default=10)
    parser.add_argument("--model-path", default="models/Qwen2-VL-7B")
    parser.add_argument("--served-model-name", default="heterokv-e4")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--server-start-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result.get("status"), "metrics": result.get("metrics")}, ensure_ascii=False))
    raise SystemExit(0 if result.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()
