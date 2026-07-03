"""
benchmark_vllm.py

Benchmark runner for OpenAI-compatible vLLM backends.

Protocol:
- Starts a fresh vLLM server per run, per dataset, and per model
- Uses a separate warm-up workload before each measured run
- Supports warm-up/eval split separation and offsets
- Writes summary CSVs and per-sample CSVs
"""

import argparse
import csv
import json
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import yaml
from openai import OpenAI

from dataset_loader_vllm import (
    format_prompt,
    get_tokenizer,
    load_dolly_workload,
    load_tiny_gsm8k,
    load_tiny_gsm8k_few_shot,
    load_tiny_hellaswag,
    load_tiny_hellaswag_few_shot,
    load_tiny_mmlu,
    load_tiny_mmlu_few_shot,
)
from system_metrics_vllm import get_system_metrics


RESULTS_ROOT = Path("results")


# -----------------------------
# CLI / config
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vllm.yaml",
        help="Path to benchmark config YAML",
    )
    return parser.parse_args()


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dataset_cfg(config: dict, dataset_name: str) -> dict:
    datasets = config.get("benchmark", {}).get("datasets", [])
    for ds in datasets:
        if ds.get("name") == dataset_name:
            return ds
    raise KeyError(f"Missing benchmark.datasets entry for {dataset_name!r}")


def is_dataset_enabled(config: dict, dataset_name: str) -> bool:
    ds = get_dataset_cfg(config, dataset_name)
    return ds.get("enabled", True)


def get_dataset_samples_runs_max_tokens(config: dict, dataset_name: str):
    defaults = config.get("benchmark", {}).get("defaults", {})
    ds = get_dataset_cfg(config, dataset_name)

    samples = ds.get("samples", defaults.get("samples", 20))
    runs = ds.get("runs", defaults.get("runs", 5))

    if "max_tokens" not in ds:
        raise KeyError(f"Missing max_tokens for dataset {dataset_name!r}")

    max_tokens = ds["max_tokens"]
    return samples, runs, max_tokens


def get_dataset_warmup_samples(config: dict, dataset_name: str, default_value: int = 10):
    defaults = config.get("benchmark", {}).get("defaults", {})
    ds = get_dataset_cfg(config, dataset_name)
    return ds.get("warmup_samples", defaults.get("warmup_samples", default_value))


def get_dataset_offset(config: dict, dataset_name: str, default_value: int = 0):
    defaults = config.get("benchmark", {}).get("defaults", {})
    ds = get_dataset_cfg(config, dataset_name)
    return ds.get("offset", defaults.get("offset", default_value))


# -----------------------------
# Basic utilities
# -----------------------------

def mean_sd(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def safe_round(x, digits=4):
    return round(x, digits) if x is not None else None


def write_single_row_csv(path: Path, row: dict):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)


def write_rows_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def add_mean_sd_columns(row: dict, label: str, values, digits=4):
    mean_value, sd_value = mean_sd(values)
    row[f"{label} mean"] = safe_round(mean_value, digits)
    row[f"{label} sd"] = safe_round(sd_value, digits)


def sanitize_model_dirname(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


def parse_host_port_from_base_url(base_url: str):
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def base_origin(base_url: str):
    parsed = urlparse(base_url)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if port is None:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def capture_metric_point(start: float, phase: str, server_pid: int | None = None) -> dict:
    now = time.perf_counter()
    metrics = get_system_metrics(server_pid=server_pid)

    return {
        "t_s": now - start,
        "phase": phase,
        "cpu_percent": metrics.get("cpu_percent"),
        "ram_used_mb": metrics.get("ram_used_mb"),
        "ram_percent": metrics.get("ram_percent"),
        "process_ram_mb": metrics.get("process_ram_mb"),
        "temperature_c": metrics.get("temperature_c"),
        "gpu_util_percent": metrics.get("gpu_util_percent"),
        "vram_used_mb": metrics.get("vram_used_mb"),
        "vram_total_mb": metrics.get("vram_total_mb"),
        "power_w": metrics.get("power_w"),
    }


def round_metric_points(points: list[dict], digits: int = 4) -> list[dict]:
    rounded = []
    for p in points:
        rounded.append({
            "t_s": safe_round(p.get("t_s"), 6),
            "phase": p.get("phase"),
            "cpu_percent": safe_round(p.get("cpu_percent"), digits),
            "ram_used_mb": safe_round(p.get("ram_used_mb"), digits),
            "ram_percent": safe_round(p.get("ram_percent"), digits),
            "process_ram_mb": safe_round(p.get("process_ram_mb"), digits),
            "temperature_c": safe_round(p.get("temperature_c"), digits),
            "gpu_util_percent": safe_round(p.get("gpu_util_percent"), digits),
            "vram_used_mb": safe_round(p.get("vram_used_mb"), digits),
            "vram_total_mb": safe_round(p.get("vram_total_mb"), digits),
            "power_w": safe_round(p.get("power_w"), digits),
        })
    return rounded


# -----------------------------
# Output parsing
# -----------------------------

def extract_letter(text: str):
    if not text:
        return None

    text = text.strip().upper()

    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1)

    match = re.search(r"^([A-D])[.):\-]?\s*$", text)
    if match:
        return match.group(1)

    return None


def extract_last_number(text: str):
    if not text:
        return None

    cleaned = text.replace(",", "").replace("$", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not matches:
        return None

    return matches[-1]


def estimate_token_count(text: str):
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


# -----------------------------
# Accuracy scorers
# -----------------------------

def score_mmlu(sample, output):
    pred = extract_letter(output)
    gold = chr(65 + sample["answer"])
    return pred == gold


def score_hellaswag(sample, output):
    pred = extract_letter(output)
    gold = chr(65 + sample["answer"])
    return pred == gold


def score_gsm8k(sample, output):
    pred = extract_last_number(output)
    gold = sample["answer"]
    return pred == gold


# -----------------------------
# Server lifecycle
# -----------------------------

def tail_file(path: Path, n_lines: int = 40) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n_lines:])
    except Exception:
        return ""


def wait_for_port_closed(host: str, port: int, timeout_s: float = 30.0):
    deadline = time.time() + timeout_s
    last_connect_rc = None

    while time.time() < deadline:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        try:
            rc = s.connect_ex((host, port))
            last_connect_rc = rc
            if rc != 0:
                return
        finally:
            s.close()

        time.sleep(0.5)

    raise TimeoutError(
        f"Port {host}:{port} did not close within {timeout_s}s "
        f"(last connect_ex rc={last_connect_rc})"
    )


def wait_for_server(
    base_url: str,
    timeout_s: float,
    expected_model_id: str | None = None,
    proc=None,
    log_path: Path | None = None,
):
    deadline = time.time() + timeout_s
    models_url = base_url.rstrip("/") + "/models"
    health_url = base_origin(base_url) + "/health"

    last_error = None
    last_models_payload = None

    while time.time() < deadline:
        if proc is not None:
            ret = proc.poll()
            if ret is not None:
                log_tail = tail_file(log_path) if log_path else ""
                raise RuntimeError(
                    f"vLLM server exited early with code {ret}.\n\n"
                    f"Last log lines:\n{log_tail}"
                )

        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status != 200:
                    last_error = RuntimeError(f"Health check returned status {resp.status}")
                    time.sleep(1.0)
                    continue
        except Exception as e:
            last_error = e
            time.sleep(1.0)
            continue

        try:
            with urllib.request.urlopen(models_url, timeout=2) as resp:
                if resp.status != 200:
                    last_error = RuntimeError(f"/models returned status {resp.status}")
                    time.sleep(1.0)
                    continue

                payload = json.loads(resp.read().decode("utf-8"))
                last_models_payload = payload

                data = payload.get("data", [])
                model_ids = [m.get("id") for m in data if m.get("id")]

                if expected_model_id is None:
                    if model_ids:
                        return
                else:
                    if expected_model_id in model_ids:
                        return
                    last_error = RuntimeError(
                        f"/models responded but expected model {expected_model_id!r} "
                        f"not present. Found: {model_ids!r}"
                    )

        except Exception as e:
            last_error = e

        time.sleep(1.0)

    log_tail = tail_file(log_path) if log_path else ""
    raise TimeoutError(
        f"Server did not become ready within {timeout_s}s.\n"
        f"Expected model: {expected_model_id!r}\n"
        f"Tried {models_url} and {health_url}.\n"
        f"Last connection error: {last_error}\n"
        f"Last /models payload: {last_models_payload}\n\n"
        f"Last log lines:\n{log_tail}"
    )


def get_served_model_id(base_url: str) -> str:
    models_url = base_url.rstrip("/") + "/models"

    with urllib.request.urlopen(models_url, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    data = payload.get("data", [])
    if not data:
        raise RuntimeError(f"No models returned by {models_url}")

    served_model_id = data[0].get("id")
    if not served_model_id:
        raise RuntimeError(f"Missing model id in /models response: {payload}")

    return served_model_id


def start_vllm_server(server_cfg: dict, model_cfg: dict, model_id: str, log_path: Path):
    binary = server_cfg.get("binary", "vllm")
    base_url = server_cfg["base_url"]
    startup_timeout_s = server_cfg.get("startup_timeout_s", 300)

    host, port = parse_host_port_from_base_url(base_url)

    try:
        wait_for_port_closed(host, port, timeout_s=5.0)
    except TimeoutError:
        pass

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")

    cmd = [
        binary,
        "serve",
        model_id,
        "--host",
        host,
        "--port",
        str(port),
    ]

    gpu_memory_utilization = model_cfg.get(
        "gpu_memory_utilization",
        server_cfg.get("gpu_memory_utilization"),
    )
    if gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]

    max_model_len = model_cfg.get(
        "max_model_len",
        server_cfg.get("max_model_len"),
    )
    if max_model_len is not None:
        cmd += ["--max-model-len", str(max_model_len)]

    max_num_seqs = model_cfg.get(
        "max_num_seqs",
        server_cfg.get("max_num_seqs"),
    )
    if max_num_seqs is not None:
        cmd += ["--max-num-seqs", str(max_num_seqs)]

    trust_remote_code = model_cfg.get(
        "trust_remote_code",
        server_cfg.get("trust_remote_code", False),
    )
    if trust_remote_code:
        cmd += ["--trust-remote-code"]

    enforce_eager = model_cfg.get(
        "enforce_eager",
        server_cfg.get("enforce_eager", False),
    )
    if enforce_eager:
        cmd += ["--enforce-eager"]

    dtype = model_cfg.get("dtype", server_cfg.get("dtype"))
    if dtype is not None:
        cmd += ["--dtype", str(dtype)]

    tensor_parallel_size = model_cfg.get(
        "tensor_parallel_size",
        server_cfg.get("tensor_parallel_size"),
    )
    if tensor_parallel_size is not None:
        cmd += ["--tensor-parallel-size", str(tensor_parallel_size)]

    env = os.environ.copy()

    pytorch_expandable_segments = model_cfg.get(
        "pytorch_expandable_segments",
        server_cfg.get("pytorch_expandable_segments", False),
    )
    if pytorch_expandable_segments:
        existing = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
        prefix = "expandable_segments:True"
        env["PYTORCH_CUDA_ALLOC_CONF"] = (
            f"{prefix},{existing}" if existing else prefix
        )

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
        cwd=str(Path.cwd()),
        env=env,
    )

    try:
        wait_for_server(
            base_url=base_url,
            timeout_s=startup_timeout_s,
            expected_model_id=model_id,
            proc=proc,
            log_path=log_path,
        )
        return proc, log_file
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
        log_file.close()
        raise


def stop_vllm_server(proc, log_file, base_url: str | None = None):
    if proc is None:
        return

    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None

    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        pass

    try:
        proc.wait(timeout=20)
    except Exception:
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            pass

    try:
        log_file.close()
    except Exception:
        pass

    if base_url is not None:
        host, port = parse_host_port_from_base_url(base_url)
        wait_for_port_closed(host, port, timeout_s=30.0)


# -----------------------------
# Streaming + inline system metrics
# -----------------------------

def stream_model_with_inline_metrics(
    client: OpenAI,
    served_model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    metrics_interval_s: float,
    server_pid=None,
):
    start = time.perf_counter()
    first_token_time = None
    last_metric_time = start

    chunks = []
    metric_points = []

    metric_points.append(
        capture_metric_point(
            start=start,
            phase="start",
            server_pid=server_pid,
        )
    )

    stream = client.chat.completions.create(
        model=served_model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in stream:
        now = time.perf_counter()
        delta = chunk.choices[0].delta.content or ""

        if delta:
            if first_token_time is None:
                first_token_time = now
                metric_points.append(
                    capture_metric_point(
                        start=start,
                        phase="first_token",
                        server_pid=server_pid,
                    )
                )
                last_metric_time = now

            chunks.append(delta)

            if (now - last_metric_time) >= metrics_interval_s:
                metric_points.append(
                    capture_metric_point(
                        start=start,
                        phase="during_stream",
                        server_pid=server_pid,
                    )
                )
                last_metric_time = now

    end = time.perf_counter()

    metric_points.append(
        capture_metric_point(
            start=start,
            phase="end",
            server_pid=server_pid,
        )
    )

    output = "".join(chunks).strip()
    output_tokens = estimate_token_count(output)

    ttft_ms = (first_token_time - start) * 1000.0 if first_token_time is not None else None
    end_to_end_latency_ms = (end - start) * 1000.0

    generation_time_s = (end - first_token_time) if first_token_time is not None else None
    tokens_per_sec = (
        output_tokens / generation_time_s
        if generation_time_s is not None and generation_time_s > 0 and output_tokens > 0
        else None
    )

    cpu_vals = [m["cpu_percent"] for m in metric_points if m.get("cpu_percent") is not None]
    ram_vals = [m["ram_used_mb"] for m in metric_points if m.get("ram_used_mb") is not None]
    proc_ram_vals = [m["process_ram_mb"] for m in metric_points if m.get("process_ram_mb") is not None]
    temp_vals = [m["temperature_c"] for m in metric_points if m.get("temperature_c") is not None]
    gpu_util_vals = [m["gpu_util_percent"] for m in metric_points if m.get("gpu_util_percent") is not None]
    vram_used_vals = [m["vram_used_mb"] for m in metric_points if m.get("vram_used_mb") is not None]
    power_vals = [m["power_w"] for m in metric_points if m.get("power_w") is not None]

    cpu_avg, _ = mean_sd(cpu_vals)
    temp_avg, _ = mean_sd(temp_vals)
    gpu_util_avg, _ = mean_sd(gpu_util_vals)
    power_avg, _ = mean_sd(power_vals)

    return {
        "raw_output": output,
        "output_tokens": output_tokens,
        "ttft_ms": ttft_ms,
        "end_to_end_latency_ms": end_to_end_latency_ms,
        "tokens_per_sec": tokens_per_sec,
        "cpu_percent_avg": cpu_avg,
        "cpu_percent_peak": max(cpu_vals) if cpu_vals else None,
        "ram_used_mb_peak": max(ram_vals) if ram_vals else None,
        "process_ram_mb_peak": max(proc_ram_vals) if proc_ram_vals else None,
        "temp_c_avg": temp_avg,
        "temp_c_peak": max(temp_vals) if temp_vals else None,
        "gpu_util_percent_avg": gpu_util_avg,
        "gpu_util_percent_peak": max(gpu_util_vals) if gpu_util_vals else None,
        "vram_used_mb_peak": max(vram_used_vals) if vram_used_vals else None,
        "power_w_avg": power_avg,
        "power_w_peak": max(power_vals) if power_vals else None,
        "metric_points": metric_points,
    }


# -----------------------------
# Run aggregation
# -----------------------------

def summarize_run(sample_rows, wall_time_s, include_accuracy: bool):
    ttft_vals = [r["ttft_ms"] for r in sample_rows if r["ttft_ms"] is not None]
    latency_vals = [r["end_to_end_latency_ms"] for r in sample_rows if r["end_to_end_latency_ms"] is not None]
    tps_vals = [r["tokens_per_sec"] for r in sample_rows if r["tokens_per_sec"] is not None]
    cpu_avg_vals = [r["cpu_percent_avg"] for r in sample_rows if r["cpu_percent_avg"] is not None]
    temp_avg_vals = [r["temp_c_avg"] for r in sample_rows if r["temp_c_avg"] is not None]
    gpu_avg_vals = [r["gpu_util_percent_avg"] for r in sample_rows if r["gpu_util_percent_avg"] is not None]
    power_avg_vals = [r["power_w_avg"] for r in sample_rows if r["power_w_avg"] is not None]

    cpu_peaks = [r["cpu_percent_peak"] for r in sample_rows if r["cpu_percent_peak"] is not None]
    ram_peaks = [r["ram_used_mb_peak"] for r in sample_rows if r["ram_used_mb_peak"] is not None]
    proc_ram_peaks = [r["process_ram_mb_peak"] for r in sample_rows if r["process_ram_mb_peak"] is not None]
    temp_peaks = [r["temp_c_peak"] for r in sample_rows if r["temp_c_peak"] is not None]
    gpu_peaks = [r["gpu_util_percent_peak"] for r in sample_rows if r["gpu_util_percent_peak"] is not None]
    vram_peaks = [r["vram_used_mb_peak"] for r in sample_rows if r["vram_used_mb_peak"] is not None]
    power_peaks = [r["power_w_peak"] for r in sample_rows if r["power_w_peak"] is not None]

    total_tokens = sum(r["output_tokens"] for r in sample_rows)

    run_summary = {
        "ttft_mean": mean_sd(ttft_vals)[0],
        "latency_mean": mean_sd(latency_vals)[0],
        "tps_mean": mean_sd(tps_vals)[0],
        "cpu_avg_mean": mean_sd(cpu_avg_vals)[0],
        "temp_avg_mean": mean_sd(temp_avg_vals)[0],
        "gpu_avg_mean": mean_sd(gpu_avg_vals)[0],
        "power_avg_mean": mean_sd(power_avg_vals)[0],
        "cpu_peak": max(cpu_peaks) if cpu_peaks else None,
        "ram_peak": max(ram_peaks) if ram_peaks else None,
        "proc_ram_peak": max(proc_ram_peaks) if proc_ram_peaks else None,
        "temp_peak": max(temp_peaks) if temp_peaks else None,
        "gpu_peak": max(gpu_peaks) if gpu_peaks else None,
        "vram_peak": max(vram_peaks) if vram_peaks else None,
        "power_peak": max(power_peaks) if power_peaks else None,
        "sustained_tps": (total_tokens / wall_time_s) if wall_time_s > 0 else None,
        "sustained_sps": (len(sample_rows) / wall_time_s) if wall_time_s > 0 else None,
    }

    if include_accuracy:
        correct_vals = [bool(r["correct"]) for r in sample_rows]
        run_summary["accuracy"] = sum(correct_vals) / len(correct_vals) if correct_vals else None

    return run_summary


# -----------------------------
# Per-sample logging helpers
# -----------------------------

def get_sample_text_fields(dataset_name: str, sample: dict):
    if dataset_name in {"mmlu", "hellaswag"}:
        return {
            "question": sample.get("question"),
            "choices": " || ".join(sample.get("choices", [])),
            "gold_answer": sample.get("answer"),
            "instruction": None,
            "context": None,
        }

    if dataset_name == "gsm8k":
        return {
            "question": sample.get("question"),
            "choices": None,
            "gold_answer": sample.get("answer"),
            "instruction": None,
            "context": None,
        }

    if dataset_name == "dolly":
        return {
            "question": None,
            "choices": None,
            "gold_answer": None,
            "instruction": sample.get("instruction"),
            "context": sample.get("context"),
        }

    return {
        "question": None,
        "choices": None,
        "gold_answer": None,
        "instruction": None,
        "context": None,
    }


def parse_prediction(dataset_name: str, raw_output: str):
    if dataset_name in {"mmlu", "hellaswag"}:
        return extract_letter(raw_output)
    if dataset_name == "gsm8k":
        return extract_last_number(raw_output)
    return None


# -----------------------------
# Main dataset benchmark
# -----------------------------

def run_dataset_benchmark(
    config: dict,
    repo_id: str,
    model_alias: str,
    model_cfg: dict,
    dataset_name: str,
    eval_loader_fn,
    max_tokens: int,
    n_samples: int,
    n_runs: int,
    temperature: float,
    metrics_interval_s: float,
    ctx_size: int,
    tokenizer,
    output_dir: Path,
    scorer_fn=None,
    warmup_loader_fn=None,
    warmup_samples: int = 10,
    eval_offset: int = 0,
    warmup_offset: int = 0,
):
    include_accuracy = scorer_fn is not None
    measured_run_rows = []
    all_sample_log_rows = []

    safety_margin = 128
    max_prompt_tokens = max(1, ctx_size - max_tokens - safety_margin)

    if warmup_loader_fn is None:
        warmup_loader_fn = eval_loader_fn

    served_model_id_for_summary = None
    served_model_ids_seen = []

    for run_idx in range(1, n_runs + 1):
        server_proc = None
        server_log_file = None

        try:
            if config["server"].get("launch", True):
                server_log_path = output_dir / f"server_{dataset_name}_run_{run_idx}.log"
                print(f"\nStarting fresh vLLM server for dataset: {dataset_name}, run {run_idx}")
                server_proc, server_log_file = start_vllm_server(
                    server_cfg=config["server"],
                    model_cfg=model_cfg,
                    model_id=repo_id,
                    log_path=server_log_path,
                )
                print(f"vLLM server ready for {dataset_name}, run {run_idx}")
            else:
                print(f"\nUsing already-running vLLM server for dataset: {dataset_name}, run {run_idx}")
                wait_for_server(
                    base_url=config["server"]["base_url"],
                    timeout_s=config["server"].get("startup_timeout_s", 60),
                    expected_model_id=repo_id,
                    proc=None,
                    log_path=None,
                )

            client = OpenAI(
                base_url=config["server"]["base_url"],
                api_key=config["server"].get("api_key", "dummy"),
            )

            served_model_id = get_served_model_id(config["server"]["base_url"])
            if served_model_id != repo_id:
                raise RuntimeError(
                    f"Expected server to serve model {repo_id!r}, "
                    f"but /models returned {served_model_id!r}"
                )

            served_model_ids_seen.append(served_model_id)

            if served_model_id_for_summary is None:
                served_model_id_for_summary = served_model_id

            server_pid = server_proc.pid if server_proc is not None else None

            warmup_dataset = warmup_loader_fn(
                n=warmup_samples,
                seed=0,
                offset=warmup_offset,
            )
            print(f"\n===== {dataset_name.upper()} WARMUP ({len(warmup_dataset)} samples) RUN {run_idx}/{n_runs} =====")

            for i, sample in enumerate(warmup_dataset, start=1):
                prompt = format_prompt(
                    sample,
                    dataset_name,
                    max_prompt_tokens=max_prompt_tokens,
                    tokenizer=tokenizer,
                )

                result = stream_model_with_inline_metrics(
                    client=client,
                    served_model_id=served_model_id,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    metrics_interval_s=metrics_interval_s,
                    server_pid=server_pid,
                )

                print(
                    f"[{dataset_name} warmup {i}/{len(warmup_dataset)}] "
                    f"ttft_ms={safe_round(result['ttft_ms'], 2)} "
                    f"lat_ms={safe_round(result['end_to_end_latency_ms'], 2)} "
                    f"tok/s={safe_round(result['tokens_per_sec'], 4)} "
                    f"temp_avg_c={safe_round(result['temp_c_avg'], 2)} "
                    f"temp_peak_c={safe_round(result['temp_c_peak'], 2)} "
                    f"gpu_avg={safe_round(result['gpu_util_percent_avg'], 2)} "
                    f"power_avg_w={safe_round(result['power_w_avg'], 2)} "
                    f"warmup=True"
                )

            dataset = eval_loader_fn(
                n=n_samples,
                seed=run_idx,
                offset=eval_offset,
            )
            sample_rows = []

            print(f"\n===== {dataset_name.upper()} MEASURED RUN {run_idx}/{n_runs} =====")
            dataset_start = time.perf_counter()

            for i, sample in enumerate(dataset, start=1):
                prompt = format_prompt(
                    sample,
                    dataset_name,
                    max_prompt_tokens=max_prompt_tokens,
                    tokenizer=tokenizer,
                )

                result = stream_model_with_inline_metrics(
                    client=client,
                    served_model_id=served_model_id,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    metrics_interval_s=metrics_interval_s,
                    server_pid=server_pid,
                )

                row = {
                    "ttft_ms": result["ttft_ms"],
                    "end_to_end_latency_ms": result["end_to_end_latency_ms"],
                    "tokens_per_sec": result["tokens_per_sec"],
                    "output_tokens": result["output_tokens"],
                    "cpu_percent_avg": result["cpu_percent_avg"],
                    "cpu_percent_peak": result["cpu_percent_peak"],
                    "ram_used_mb_peak": result["ram_used_mb_peak"],
                    "process_ram_mb_peak": result["process_ram_mb_peak"],
                    "temp_c_avg": result["temp_c_avg"],
                    "temp_c_peak": result["temp_c_peak"],
                    "gpu_util_percent_avg": result["gpu_util_percent_avg"],
                    "gpu_util_percent_peak": result["gpu_util_percent_peak"],
                    "vram_used_mb_peak": result["vram_used_mb_peak"],
                    "power_w_avg": result["power_w_avg"],
                    "power_w_peak": result["power_w_peak"],
                    "raw_output": result["raw_output"],
                }

                if include_accuracy:
                    row["correct"] = scorer_fn(sample, result["raw_output"])

                sample_rows.append(row)

                sample_fields = get_sample_text_fields(dataset_name, sample)
                parsed_prediction = parse_prediction(dataset_name, result["raw_output"])

                log_row = {
                    "repo_id": repo_id,
                    "served_model_id": served_model_id,
                    "model_alias": model_alias,
                    "dataset_name": dataset_name,
                    "run_idx": run_idx,
                    "sample_idx": i,
                    "prompt": prompt,
                    "question": sample_fields["question"],
                    "choices": sample_fields["choices"],
                    "instruction": sample_fields["instruction"],
                    "context": sample_fields["context"],
                    "gold_answer": sample_fields["gold_answer"],
                    "parsed_prediction": parsed_prediction,
                    "correct": row.get("correct"),
                    "raw_output": result["raw_output"],
                    "output_tokens": result["output_tokens"],
                    "ttft_ms": safe_round(result["ttft_ms"], 4),
                    "end_to_end_latency_ms": safe_round(result["end_to_end_latency_ms"], 4),
                    "tokens_per_sec": safe_round(result["tokens_per_sec"], 4),
                    "cpu_percent_avg": safe_round(result["cpu_percent_avg"], 4),
                    "cpu_percent_peak": safe_round(result["cpu_percent_peak"], 4),
                    "ram_used_mb_peak": safe_round(result["ram_used_mb_peak"], 4),
                    "process_ram_mb_peak": safe_round(result["process_ram_mb_peak"], 4),
                    "temp_c_avg": safe_round(result["temp_c_avg"], 4),
                    "temp_c_peak": safe_round(result["temp_c_peak"], 4),
                    "gpu_util_percent_avg": safe_round(result["gpu_util_percent_avg"], 4),
                    "gpu_util_percent_peak": safe_round(result["gpu_util_percent_peak"], 4),
                    "vram_used_mb_peak": safe_round(result["vram_used_mb_peak"], 4),
                    "power_w_avg": safe_round(result["power_w_avg"], 4),
                    "power_w_peak": safe_round(result["power_w_peak"], 4),
                    "system_metrics_trace_json": json.dumps(
                        round_metric_points(result["metric_points"]),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
                all_sample_log_rows.append(log_row)

                msg = (
                    f"[{dataset_name} {i}/{len(dataset)}] "
                    f"ttft_ms={safe_round(row['ttft_ms'], 2)} "
                    f"lat_ms={safe_round(row['end_to_end_latency_ms'], 2)} "
                    f"tok/s={safe_round(row['tokens_per_sec'], 4)} "
                    f"temp_avg_c={safe_round(row['temp_c_avg'], 2)} "
                    f"temp_peak_c={safe_round(row['temp_c_peak'], 2)} "
                    f"gpu_avg={safe_round(row['gpu_util_percent_avg'], 2)} "
                    f"power_avg_w={safe_round(row['power_w_avg'], 2)}"
                )
                if include_accuracy:
                    msg += f" correct={row['correct']}"
                print(msg)

            dataset_end = time.perf_counter()
            wall_time_s = dataset_end - dataset_start

            measured_run_rows.append(
                summarize_run(
                    sample_rows=sample_rows,
                    wall_time_s=wall_time_s,
                    include_accuracy=include_accuracy,
                )
            )

        finally:
            if server_proc is not None:
                print(f"Stopping vLLM server for dataset: {dataset_name}, run {run_idx}")
                stop_vllm_server(
                    server_proc,
                    server_log_file,
                    base_url=config["server"]["base_url"],
                )
                time.sleep(1.0)

    summary_row = {
        "Model alias": model_alias,
        "Repo id": repo_id,
        "Served model id": served_model_id_for_summary,
        "All served model ids": " | ".join(served_model_ids_seen),
    }

    cpu_peaks = [r["cpu_peak"] for r in measured_run_rows if r["cpu_peak"] is not None]
    ram_peaks = [r["ram_peak"] for r in measured_run_rows if r["ram_peak"] is not None]
    proc_ram_peaks = [r["proc_ram_peak"] for r in measured_run_rows if r["proc_ram_peak"] is not None]
    temp_peaks = [r["temp_peak"] for r in measured_run_rows if r["temp_peak"] is not None]
    gpu_peaks = [r["gpu_peak"] for r in measured_run_rows if r["gpu_peak"] is not None]
    vram_peaks = [r["vram_peak"] for r in measured_run_rows if r["vram_peak"] is not None]
    power_peaks = [r["power_peak"] for r in measured_run_rows if r["power_peak"] is not None]

    if include_accuracy:
        add_mean_sd_columns(summary_row, "Accuracy", [r["accuracy"] for r in measured_run_rows], 4)

    add_mean_sd_columns(summary_row, "TTFT ms", [r["ttft_mean"] for r in measured_run_rows], 4)
    add_mean_sd_columns(summary_row, "End-to-end latency ms", [r["latency_mean"] for r in measured_run_rows], 4)
    add_mean_sd_columns(summary_row, "Tokens/sec", [r["tps_mean"] for r in measured_run_rows], 4)
    add_mean_sd_columns(summary_row, "CPU percent avg", [r["cpu_avg_mean"] for r in measured_run_rows], 4)
    add_mean_sd_columns(summary_row, "Temp C avg", [r["temp_avg_mean"] for r in measured_run_rows], 4)
    add_mean_sd_columns(summary_row, "GPU percent avg", [r["gpu_avg_mean"] for r in measured_run_rows], 4)
    add_mean_sd_columns(summary_row, "Power W avg", [r["power_avg_mean"] for r in measured_run_rows], 4)
    add_mean_sd_columns(
        summary_row,
        "Sequential sustained tokens/sec",
        [r["sustained_tps"] for r in measured_run_rows],
        4,
    )
    add_mean_sd_columns(
        summary_row,
        "Sequential sustained samples/sec",
        [r["sustained_sps"] for r in measured_run_rows],
        4,
    )

    summary_row["CPU percent peak"] = safe_round(max(cpu_peaks), 4) if cpu_peaks else None
    summary_row["RAM used MB peak"] = safe_round(max(ram_peaks), 4) if ram_peaks else None
    summary_row["Process RAM MB peak"] = safe_round(max(proc_ram_peaks), 4) if proc_ram_peaks else None
    summary_row["Temp C peak"] = safe_round(max(temp_peaks), 4) if temp_peaks else None
    summary_row["GPU percent peak"] = safe_round(max(gpu_peaks), 4) if gpu_peaks else None
    summary_row["VRAM used MB peak"] = safe_round(max(vram_peaks), 4) if vram_peaks else None
    summary_row["Power W peak"] = safe_round(max(power_peaks), 4) if power_peaks else None

    return summary_row, all_sample_log_rows


# -----------------------------
# Per-model benchmark
# -----------------------------

def benchmark_one_model(
    config: dict,
    config_path: str,
    model_cfg: dict,
    run_id: str,
):
    if "id" not in model_cfg:
        raise KeyError("Each model entry must include 'id'")

    repo_id = model_cfg["id"]
    model_alias = model_cfg.get("alias", sanitize_model_dirname(repo_id))

    metrics_interval_s = config.get("benchmark", {}).get("metrics_interval_s", 0.2)
    temperature = config.get("generation", {}).get("temperature", 0)
    ctx_size = model_cfg.get("max_model_len", config["server"]["max_model_len"])

    output_dir = RESULTS_ROOT / model_alias / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, output_dir / "config.yaml")

    print("\n==============================")
    print(f"Benchmarking repo id: {repo_id}")
    print(f"Benchmarking model alias: {model_alias}")
    print(f"Results dir: {output_dir}")
    print("==============================")

    tokenizer = get_tokenizer(repo_id)

    if is_dataset_enabled(config, "mmlu"):
        mmlu_samples, mmlu_runs, mmlu_max_tokens = get_dataset_samples_runs_max_tokens(config, "mmlu")
        mmlu_warmup_samples = get_dataset_warmup_samples(config, "mmlu")
        mmlu_offset = get_dataset_offset(config, "mmlu")

        mmlu_row, mmlu_samples_rows = run_dataset_benchmark(
            config=config,
            repo_id=repo_id,
            model_alias=model_alias,
            model_cfg=model_cfg,
            dataset_name="mmlu",
            eval_loader_fn=load_tiny_mmlu,
            warmup_loader_fn=load_tiny_mmlu_few_shot,
            max_tokens=mmlu_max_tokens,
            n_samples=mmlu_samples,
            n_runs=mmlu_runs,
            warmup_samples=mmlu_warmup_samples,
            temperature=temperature,
            metrics_interval_s=metrics_interval_s,
            ctx_size=ctx_size,
            tokenizer=tokenizer,
            scorer_fn=score_mmlu,
            eval_offset=mmlu_offset,
            warmup_offset=0,
            output_dir=output_dir,
        )

        write_single_row_csv(output_dir / "mmlu.csv", mmlu_row)
        write_rows_csv(output_dir / "mmlu_samples.csv", mmlu_samples_rows)

    if is_dataset_enabled(config, "hellaswag"):
        hellaswag_samples, hellaswag_runs, hellaswag_max_tokens = get_dataset_samples_runs_max_tokens(config, "hellaswag")
        hellaswag_warmup_samples = get_dataset_warmup_samples(config, "hellaswag")
        hellaswag_offset = get_dataset_offset(config, "hellaswag")

        hellaswag_row, hellaswag_samples_rows = run_dataset_benchmark(
            config=config,
            repo_id=repo_id,
            model_alias=model_alias,
            model_cfg=model_cfg,
            dataset_name="hellaswag",
            eval_loader_fn=load_tiny_hellaswag,
            warmup_loader_fn=load_tiny_hellaswag_few_shot,
            max_tokens=hellaswag_max_tokens,
            n_samples=hellaswag_samples,
            n_runs=hellaswag_runs,
            warmup_samples=hellaswag_warmup_samples,
            temperature=temperature,
            metrics_interval_s=metrics_interval_s,
            ctx_size=ctx_size,
            tokenizer=tokenizer,
            scorer_fn=score_hellaswag,
            eval_offset=hellaswag_offset,
            warmup_offset=0,
            output_dir=output_dir,
        )

        write_single_row_csv(output_dir / "hellaswag.csv", hellaswag_row)
        write_rows_csv(output_dir / "hellaswag_samples.csv", hellaswag_samples_rows)

    if is_dataset_enabled(config, "gsm8k"):
        gsm8k_samples, gsm8k_runs, gsm8k_max_tokens = get_dataset_samples_runs_max_tokens(config, "gsm8k")
        gsm8k_warmup_samples = get_dataset_warmup_samples(config, "gsm8k")
        gsm8k_offset = get_dataset_offset(config, "gsm8k")

        gsm8k_row, gsm8k_samples_rows = run_dataset_benchmark(
            config=config,
            repo_id=repo_id,
            model_alias=model_alias,
            model_cfg=model_cfg,
            dataset_name="gsm8k",
            eval_loader_fn=load_tiny_gsm8k,
            warmup_loader_fn=load_tiny_gsm8k_few_shot,
            max_tokens=gsm8k_max_tokens,
            n_samples=gsm8k_samples,
            n_runs=gsm8k_runs,
            warmup_samples=gsm8k_warmup_samples,
            temperature=temperature,
            metrics_interval_s=metrics_interval_s,
            ctx_size=ctx_size,
            tokenizer=tokenizer,
            scorer_fn=score_gsm8k,
            eval_offset=gsm8k_offset,
            warmup_offset=0,
            output_dir=output_dir,
        )

        write_single_row_csv(output_dir / "gsm8k.csv", gsm8k_row)
        write_rows_csv(output_dir / "gsm8k_samples.csv", gsm8k_samples_rows)

    if is_dataset_enabled(config, "dolly"):
        dolly_samples, dolly_runs, dolly_max_tokens = get_dataset_samples_runs_max_tokens(config, "dolly")
        dolly_warmup_samples = get_dataset_warmup_samples(config, "dolly")
        dolly_offset = get_dataset_offset(config, "dolly")

        def _dolly_warmup_loader(n, seed=None, offset=0):
            warmup_seed = 999_000 if seed is None else 999_000 + int(seed)
            return load_dolly_workload(
                n=n,
                split="train",
                seed=warmup_seed,
                offset=offset,
            )

        dolly_row, dolly_samples_rows = run_dataset_benchmark(
            config=config,
            repo_id=repo_id,
            model_alias=model_alias,
            model_cfg=model_cfg,
            dataset_name="dolly",
            eval_loader_fn=load_dolly_workload,
            warmup_loader_fn=_dolly_warmup_loader,
            max_tokens=dolly_max_tokens,
            n_samples=dolly_samples,
            n_runs=dolly_runs,
            warmup_samples=dolly_warmup_samples,
            temperature=temperature,
            metrics_interval_s=metrics_interval_s,
            ctx_size=ctx_size,
            tokenizer=tokenizer,
            scorer_fn=None,
            eval_offset=dolly_offset,
            warmup_offset=0,
            output_dir=output_dir,
        )

        write_single_row_csv(output_dir / "dolly.csv", dolly_row)
        write_rows_csv(output_dir / "dolly_samples.csv", dolly_samples_rows)

    print(f"\nSaved results to: {output_dir}")


# -----------------------------
# Entry point
# -----------------------------

def main():
    args = parse_args()
    config = load_config(args.config)

    backend_type = config.get("backend", {}).get("type")
    if backend_type != "vllm":
        raise ValueError(f"Expected backend.type='vllm', got {backend_type!r}")

    models = config.get("models", [])
    if not models:
        raise KeyError("Missing 'models' list in config")

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for model_cfg in models:
        try:
            benchmark_one_model(
                config=config,
                config_path=args.config,
                model_cfg=model_cfg,
                run_id=run_id,
            )
        except Exception as e:
            print(f"\n[SKIP] Model failed: {model_cfg.get('id')}")
            print(e)
            continue


if __name__ == "__main__":
    main()