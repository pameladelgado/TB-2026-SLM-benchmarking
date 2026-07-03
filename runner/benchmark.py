"""
benchmark.py

Unified benchmark runner for OpenAI-compatible llama.cpp backends.

Protocol:
- Starts a fresh llama-server per run, per dataset, and per model
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

from dataset_loader import (
    format_prompt,
    load_dolly_workload,
    load_tiny_gsm8k,
    load_tiny_gsm8k_few_shot,
    load_tiny_hellaswag,
    load_tiny_hellaswag_few_shot,
    load_tiny_mmlu,
    load_tiny_mmlu_few_shot,
)
from system_metrics import get_system_metrics


RESULTS_ROOT = Path("results")


# -----------------------------
# CLI / config
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/llama_cpp.yaml",
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


def prefix_keys(d: dict, prefix: str):
    return {f"{prefix}{k}": v for k, v in d.items()}


def add_mean_sd_columns(row: dict, label: str, values, digits=4):
    mean_value, sd_value = mean_sd(values)
    row[f"{label} mean"] = safe_round(mean_value, digits)
    row[f"{label} sd"] = safe_round(sd_value, digits)


def sanitize_model_dirname(name: str) -> str:
    stem = Path(name).stem
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", stem)
    return cleaned.strip("_")


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

def tail_file(path: Path, n_lines: int = 200) -> str:
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


def get_model_ids_from_base_url(base_url: str) -> list[str]:
    models_url = base_url.rstrip("/") + "/models"
    with urllib.request.urlopen(models_url, timeout=10) as resp:
        if resp.status != 200:
            raise RuntimeError(f"/models returned HTTP {resp.status}")
        payload = json.loads(resp.read().decode("utf-8"))

    data = payload.get("data", [])
    return [m.get("id") for m in data if m.get("id")]


def get_served_model_id(base_url: str) -> str:
    model_ids = get_model_ids_from_base_url(base_url)
    if not model_ids:
        raise RuntimeError(f"No models returned by {base_url.rstrip('/') + '/models'}")
    return model_ids[0]


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
    last_model_ids = None

    while time.time() < deadline:
        if proc is not None:
            ret = proc.poll()
            if ret is not None:
                log_tail = tail_file(log_path) if log_path else ""
                raise RuntimeError(
                    f"llama-server exited early with code {ret}.\n\n"
                    f"Last log lines:\n{log_tail}"
                )

        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status != 200:
                    last_error = RuntimeError(f"/health returned HTTP {resp.status}")
                    time.sleep(1.0)
                    continue
        except Exception as e:
            last_error = e
            time.sleep(1.0)
            continue

        try:
            with urllib.request.urlopen(models_url, timeout=2) as resp:
                if resp.status != 200:
                    last_error = RuntimeError(f"/models returned HTTP {resp.status}")
                    time.sleep(1.0)
                    continue

                payload = json.loads(resp.read().decode("utf-8"))
                data = payload.get("data", [])
                model_ids = [m.get("id") for m in data if m.get("id")]
                last_model_ids = model_ids

                if expected_model_id is None:
                    if model_ids:
                        return
                else:
                    if expected_model_id in model_ids:
                        return
                    last_error = RuntimeError(
                        f"/models responded but expected model alias {expected_model_id!r} "
                        f"was not present. Found: {model_ids!r}"
                    )

        except Exception as e:
            last_error = e

        time.sleep(1.0)

    log_tail = tail_file(log_path) if log_path else ""
    raise TimeoutError(
        f"Server did not become ready within {timeout_s}s.\n"
        f"Expected model alias: {expected_model_id!r}\n"
        f"Tried {models_url} and {health_url}.\n"
        f"Last model ids: {last_model_ids!r}\n"
        f"Last error: {last_error}\n\n"
        f"Last log lines:\n{log_tail}"
    )


def cfg_get(model_cfg: dict, server_cfg: dict, key: str, default=None):
    return model_cfg.get(key, server_cfg.get(key, default))


def add_optional_arg(cmd: list[str], model_cfg: dict, server_cfg: dict, cfg_key: str, cli_arg: str):
    value = cfg_get(model_cfg, server_cfg, cfg_key)
    if value is not None:
        cmd += [cli_arg, str(value)]


def add_optional_bool_arg(cmd: list[str], model_cfg: dict, server_cfg: dict, cfg_key: str, cli_arg: str):
    value = cfg_get(model_cfg, server_cfg, cfg_key, False)
    if value:
        cmd += [cli_arg]


def start_llama_server(server_cfg: dict, model_cfg: dict, model_path: str, model_alias: str, log_path: Path):
    binary = str(Path(server_cfg["binary"]).resolve())
    model_path = str(Path(model_path).resolve())
    ctx_size = cfg_get(model_cfg, server_cfg, "ctx_size", 4096)
    startup_timeout_s = server_cfg.get("startup_timeout_s", 180)
    base_url = server_cfg["base_url"]

    host, port = parse_host_port_from_base_url(base_url)

    try:
        wait_for_port_closed(host, port, timeout_s=5.0)
    except TimeoutError:
        existing_models = []
        try:
            existing_models = get_model_ids_from_base_url(base_url)
        except Exception:
            pass

        raise RuntimeError(
            f"Port {host}:{port} is already in use before starting llama-server. "
            f"Existing server models: {existing_models!r}. "
            f"Stop the existing server or change server.base_url/port."
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")

    env = os.environ.copy()

    llama_bin_dir = str(Path("engines/llama.cpp/build/bin").resolve())
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{llama_bin_dir}:{existing_ld}" if existing_ld else llama_bin_dir
    )

    cmd = [
        binary,
        "-m", model_path,
        "-c", str(ctx_size),
        "--host", host,
        "--port", str(port),
        "--alias", model_alias,
    ]

    add_optional_arg(cmd, model_cfg, server_cfg, "n_gpu_layers", "-ngl")
    add_optional_arg(cmd, model_cfg, server_cfg, "threads", "-t")
    add_optional_arg(cmd, model_cfg, server_cfg, "batch_size", "-b")
    add_optional_arg(cmd, model_cfg, server_cfg, "ubatch_size", "-ub")
    add_optional_arg(cmd, model_cfg, server_cfg, "parallel", "-np")
    add_optional_arg(cmd, model_cfg, server_cfg, "ctx_size", "-c")

    add_optional_bool_arg(cmd, model_cfg, server_cfg, "flash_attn", "-fa")
    add_optional_bool_arg(cmd, model_cfg, server_cfg, "cont_batching", "-cb")

    extra_args = []
    extra_args.extend(server_cfg.get("extra_args", []) or [])
    extra_args.extend(model_cfg.get("extra_args", []) or [])
    cmd += [str(x) for x in extra_args]

    # Remove duplicated "-c" if ctx_size was also added by add_optional_arg.
    # The first "-c" is canonical.
    cleaned_cmd = []
    seen_ctx = False
    skip_next = False
    for i, item in enumerate(cmd):
        if skip_next:
            skip_next = False
            continue
        if item == "-c":
            if seen_ctx:
                skip_next = True
                continue
            seen_ctx = True
        cleaned_cmd.append(item)
    cmd = cleaned_cmd

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
            expected_model_id=model_alias,
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
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        log_file.close()
        raise


def stop_llama_server(proc, log_file, base_url: str | None = None):
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
    model_name: str,
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
    metric_samples = []

    metric_samples.append(get_system_metrics(server_pid=server_pid))

    stream = client.chat.completions.create(
        model=model_name,
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
                metric_samples.append(get_system_metrics(server_pid=server_pid))
                last_metric_time = now

            chunks.append(delta)

            if (now - last_metric_time) >= metrics_interval_s:
                metric_samples.append(get_system_metrics(server_pid=server_pid))
                last_metric_time = now

    end = time.perf_counter()

    metric_samples.append(get_system_metrics(server_pid=server_pid))

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

    cpu_vals = [m["cpu_percent"] for m in metric_samples if m.get("cpu_percent") is not None]
    ram_vals = [m["ram_used_mb"] for m in metric_samples if m.get("ram_used_mb") is not None]
    proc_ram_vals = [m["process_ram_mb"] for m in metric_samples if m.get("process_ram_mb") is not None]
    temp_vals = [m["temperature_c"] for m in metric_samples if m.get("temperature_c") is not None]

    cpu_avg, _ = mean_sd(cpu_vals)
    temp_avg, _ = mean_sd(temp_vals)

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
        "system_metrics_trace": metric_samples,
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

    cpu_peaks = [r["cpu_percent_peak"] for r in sample_rows if r["cpu_percent_peak"] is not None]
    ram_peaks = [r["ram_used_mb_peak"] for r in sample_rows if r["ram_used_mb_peak"] is not None]
    proc_ram_peaks = [r["process_ram_mb_peak"] for r in sample_rows if r["process_ram_mb_peak"] is not None]
    temp_peaks = [r["temp_c_peak"] for r in sample_rows if r["temp_c_peak"] is not None]

    total_tokens = sum(r["output_tokens"] for r in sample_rows)

    run_summary = {
        "ttft_mean": mean_sd(ttft_vals)[0],
        "latency_mean": mean_sd(latency_vals)[0],
        "tps_mean": mean_sd(tps_vals)[0],
        "cpu_avg_mean": mean_sd(cpu_avg_vals)[0],
        "temp_avg_mean": mean_sd(temp_avg_vals)[0],
        "cpu_peak": max(cpu_peaks) if cpu_peaks else None,
        "ram_peak": max(ram_peaks) if ram_peaks else None,
        "proc_ram_peak": max(proc_ram_peaks) if proc_ram_peaks else None,
        "temp_peak": max(temp_peaks) if temp_peaks else None,
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
    model_cfg: dict,
    model_name: str,
    model_path: str,
    dataset_name: str,
    eval_loader_fn,
    max_tokens: int,
    n_samples: int,
    n_runs: int,
    temperature: float,
    metrics_interval_s: float,
    ctx_size: int,
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
    served_model_ids_seen = []

    safety_margin = 2048
    max_prompt_tokens = max(1, ctx_size - max_tokens - safety_margin)

    if warmup_loader_fn is None:
        warmup_loader_fn = eval_loader_fn

    for run_idx in range(1, n_runs + 1):
        server_proc = None
        server_log_file = None

        try:
            if config["server"].get("launch", True):
                server_log_path = output_dir / f"server_{dataset_name}_run_{run_idx}.log"
                print(f"\nStarting fresh llama-server for dataset: {dataset_name}, run {run_idx}")
                server_proc, server_log_file = start_llama_server(
                    server_cfg=config["server"],
                    model_cfg=model_cfg,
                    model_path=model_path,
                    model_alias=model_name,
                    log_path=server_log_path,
                )
                print(f"llama-server ready for {dataset_name}, run {run_idx}")
            else:
                print(f"\nUsing already-running llama-server for dataset: {dataset_name}, run {run_idx}")
                wait_for_server(
                    base_url=config["server"]["base_url"],
                    timeout_s=config["server"].get("startup_timeout_s", 60),
                    expected_model_id=model_name,
                    proc=None,
                    log_path=None,
                )

            client = OpenAI(
                base_url=config["server"]["base_url"],
                api_key=config["server"].get("api_key", "dummy"),
            )

            served_model_id = get_served_model_id(config["server"]["base_url"])
            if served_model_id != model_name:
                raise RuntimeError(
                    f"Expected llama-server to serve model alias {model_name!r}, "
                    f"but /models returned {served_model_id!r}"
                )
            served_model_ids_seen.append(served_model_id)

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
                )

                result = stream_model_with_inline_metrics(
                    client=client,
                    model_name=model_name,
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
                )

                result = stream_model_with_inline_metrics(
                    client=client,
                    model_name=model_name,
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
                    "raw_output": result["raw_output"],
                }

                if include_accuracy:
                    row["correct"] = scorer_fn(sample, result["raw_output"])

                sample_rows.append(row)

                sample_fields = get_sample_text_fields(dataset_name, sample)
                parsed_prediction = parse_prediction(dataset_name, result["raw_output"])

                log_row = {
                    "model_name": model_name,
                    "served_model_id": served_model_id,
                    "model_path": model_path,
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
                    "system_metrics_trace_json": json.dumps(
                        result["system_metrics_trace"],
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
                    f"temp_peak_c={safe_round(row['temp_c_peak'], 2)}"
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
                print(f"Stopping llama-server for dataset: {dataset_name}, run {run_idx}")
                stop_llama_server(
                    server_proc,
                    server_log_file,
                    base_url=config["server"]["base_url"],
                )
                time.sleep(1.0)

    summary_row = {
        "Model name": model_name,
        "Model path": model_path,
        "Served model id": served_model_ids_seen[0] if served_model_ids_seen else None,
        "All served model ids": " | ".join(served_model_ids_seen),
    }

    cpu_peaks = [r["cpu_peak"] for r in measured_run_rows if r["cpu_peak"] is not None]
    ram_peaks = [r["ram_peak"] for r in measured_run_rows if r["ram_peak"] is not None]
    proc_ram_peaks = [r["proc_ram_peak"] for r in measured_run_rows if r["proc_ram_peak"] is not None]
    temp_peaks = [r["temp_peak"] for r in measured_run_rows if r["temp_peak"] is not None]

    if include_accuracy:
        add_mean_sd_columns(summary_row, "Accuracy", [r["accuracy"] for r in measured_run_rows], 4)

    add_mean_sd_columns(summary_row, "TTFT ms", [r["ttft_mean"] for r in measured_run_rows], 4)
    add_mean_sd_columns(summary_row, "End-to-end latency ms", [r["latency_mean"] for r in measured_run_rows], 4)
    add_mean_sd_columns(summary_row, "Tokens/sec", [r["tps_mean"] for r in measured_run_rows], 4)
    add_mean_sd_columns(summary_row, "CPU percent avg", [r["cpu_avg_mean"] for r in measured_run_rows], 4)
    add_mean_sd_columns(summary_row, "Temp C avg", [r["temp_avg_mean"] for r in measured_run_rows], 4)
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

    return summary_row, all_sample_log_rows


# -----------------------------
# llama-bench integration
# -----------------------------

def run_llama_bench(llama_bench_cfg: dict):
    if not llama_bench_cfg.get("enabled", False):
        return {}

    required_keys = ["binary", "model", "n_prompt", "n_gen"]
    for key in required_keys:
        if key not in llama_bench_cfg:
            raise KeyError(f"Missing llama_bench config key: {key}")

    cmd = [
        llama_bench_cfg["binary"],
        "-m", str(llama_bench_cfg["model"]),
        "-p", str(llama_bench_cfg["n_prompt"]),
        "-n", str(llama_bench_cfg["n_gen"]),
        "-r", str(llama_bench_cfg.get("repetitions", 1)),
    ]

    if llama_bench_cfg.get("output_format", "csv") == "csv":
        cmd += ["--output", "csv"]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout.strip()
    if not stdout:
        return {}

    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) >= 2:
        try:
            reader = csv.DictReader(lines)
            rows = list(reader)
            if rows:
                row = rows[-1]
                return {
                    "llama_bench model": row.get("model") or llama_bench_cfg["model"],
                    "llama_bench n_prompt": row.get("n_prompt") or llama_bench_cfg["n_prompt"],
                    "llama_bench n_gen": row.get("n_gen") or llama_bench_cfg["n_gen"],
                    "llama_bench prompt_tps": row.get("t_pp") or row.get("prompt_tps"),
                    "llama_bench gen_tps": row.get("t_tg") or row.get("gen_tps"),
                    "llama_bench test_time_s": row.get("test_time") or row.get("avg_ts"),
                }
        except Exception:
            pass

    return {
        "llama_bench model": llama_bench_cfg["model"],
        "llama_bench n_prompt": llama_bench_cfg["n_prompt"],
        "llama_bench n_gen": llama_bench_cfg["n_gen"],
        "llama_bench raw_output": stdout,
    }


# -----------------------------
# Per-model benchmark
# -----------------------------

def benchmark_one_model(
    config: dict,
    config_path: str,
    model_cfg: dict,
    run_id: str,
):
    model_name = model_cfg["name"]
    model_path = model_cfg["path"]

    metrics_interval_s = config.get("benchmark", {}).get("metrics_interval_s", 0.2)
    temperature = config.get("generation", {}).get("temperature", 0)
    ctx_size = model_cfg.get("ctx_size", config["server"]["ctx_size"])

    output_dir = RESULTS_ROOT / sanitize_model_dirname(model_name) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, output_dir / "config.yaml")

    print("\n==============================")
    print(f"Benchmarking model: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Server launch mode: {'local' if config['server'].get('launch', True) else 'external'}")
    print(f"Results dir: {output_dir}")
    print("==============================")

    if is_dataset_enabled(config, "mmlu"):
        mmlu_samples, mmlu_runs, mmlu_max_tokens = get_dataset_samples_runs_max_tokens(config, "mmlu")
        mmlu_warmup_samples = get_dataset_warmup_samples(config, "mmlu")
        mmlu_offset = get_dataset_offset(config, "mmlu")

        mmlu_row, mmlu_samples_rows = run_dataset_benchmark(
            config=config,
            model_cfg=model_cfg,
            model_name=model_name,
            model_path=model_path,
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
            model_cfg=model_cfg,
            model_name=model_name,
            model_path=model_path,
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
            model_cfg=model_cfg,
            model_name=model_name,
            model_path=model_path,
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
            model_cfg=model_cfg,
            model_name=model_name,
            model_path=model_path,
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
            scorer_fn=None,
            eval_offset=dolly_offset,
            warmup_offset=0,
            output_dir=output_dir,
        )

        llama_bench_cfg = dict(config.get("llama_bench", {}))
        if llama_bench_cfg.get("enabled", False):
            llama_bench_cfg.setdefault("model", model_path)

        llama_bench_row = run_llama_bench(llama_bench_cfg)

        combined_dolly_row = {"Model name": model_name, "Model path": model_path}
        dolly_prefixed = prefix_keys(
            {k: v for k, v in dolly_row.items() if k not in {"Model name", "Model path"}},
            "Dolly ",
        )
        combined_dolly_row.update(dolly_prefixed)
        combined_dolly_row.update(llama_bench_row)

        write_single_row_csv(output_dir / "dolly.csv", combined_dolly_row)
        write_rows_csv(output_dir / "dolly_samples.csv", dolly_samples_rows)

    print(f"\nSaved results to: {output_dir}")


# -----------------------------
# Entry point
# -----------------------------

def main():
    args = parse_args()
    config = load_config(args.config)

    models = config.get("models", [])
    if not models:
        raise KeyError("Missing 'models' list in config")

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for model_cfg in models:
        try:
            if "name" not in model_cfg:
                raise KeyError("Each model entry must include 'name'")
            if "path" not in model_cfg:
                raise KeyError("Each model entry must include 'path'")

            benchmark_one_model(
                config=config,
                config_path=args.config,
                model_cfg=model_cfg,
                run_id=run_id,
            )
        except Exception as e:
            print(f"\n[SKIP] Model failed: {model_cfg.get('name') or model_cfg.get('path')}")
            print(e)
            continue


if __name__ == "__main__":
    main()
