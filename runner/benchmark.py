"""
benchmark.py

Unified benchmark runner for llama.cpp-backed OpenAI-compatible inference.

What it does:
- Runs one benchmark pass per dataset
- Uses streaming for all datasets
- Computes accuracy where applicable
- Collects inline system metrics without extra threads
- Writes one summary CSV per dataset
- Writes one sample-level CSV per dataset
- Adds llama-bench results next to Dolly results in dolly.csv

Expected outputs:
- results/<run_id>/config.yaml
- results/<run_id>/mmlu.csv
- results/<run_id>/mmlu_samples.csv
- results/<run_id>/hellaswag.csv
- results/<run_id>/hellaswag_samples.csv
- results/<run_id>/gsm8k.csv
- results/<run_id>/gsm8k_samples.csv
- results/<run_id>/dolly.csv
- results/<run_id>/dolly_samples.csv
"""

import argparse
import csv
import re
import statistics
import subprocess
import time
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import yaml
from openai import OpenAI

from dataset_loader import (
    load_tiny_mmlu,
    load_tiny_hellaswag,
    load_tiny_gsm8k,
    load_dolly_workload,
    format_mcq,
    format_gsm8k,
    format_dolly,
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
        help="Path to llama.cpp benchmark config YAML",
    )
    return parser.parse_args()


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return normalize_config(config)


def normalize_config(config: dict):
    model_cfg = config.setdefault("model", {})
    server_cfg = config.setdefault("server", {})
    bench_cfg = config.setdefault("benchmark", {})
    gen_cfg = config.setdefault("generation", {})
    llama_bench_cfg = config.setdefault("llama_bench", {})

    model_path = model_cfg.get("path")
    if not model_path:
        raise ValueError("config.model.path is required")

    if not model_cfg.get("name"):
        model_cfg["name"] = Path(model_path).name

    base_url = server_cfg.get("base_url", "http://127.0.0.1:8000/v1")
    server_cfg["base_url"] = base_url

    parsed = urlparse(base_url)
    server_cfg.setdefault("host", parsed.hostname or "127.0.0.1")
    server_cfg.setdefault("port", parsed.port or 8000)
    server_cfg.setdefault("ctx_size", 8192)
    server_cfg.setdefault("api_key", "dummy")

    bench_cfg.setdefault("n_samples", 20)
    bench_cfg.setdefault("n_runs", 5)
    bench_cfg.setdefault("dolly_samples", 50)
    bench_cfg.setdefault("dolly_runs", 5)
    bench_cfg.setdefault("metrics_interval_s", 0.2)

    gen_cfg.setdefault("temperature", 0)
    gen_cfg.setdefault("max_tokens", {})

    llama_bench_cfg.setdefault("enabled", False)
    llama_bench_cfg.setdefault("binary", "engines/llama.cpp/build/bin/llama-bench")
    llama_bench_cfg.setdefault("model", model_path)
    llama_bench_cfg.setdefault("n_prompt", 512)
    llama_bench_cfg.setdefault("n_gen", 128)
    llama_bench_cfg.setdefault("repetitions", 1)
    llama_bench_cfg.setdefault("output_format", "csv")

    return config


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


# -----------------------------
# Output parsing
# -----------------------------

def extract_letter(text: str):
    """Extract A/B/C/D from model output."""
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
    """Extract the final numeric value from text."""
    if not text:
        return None

    cleaned = text.replace(",", "").replace("$", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not matches:
        return None

    return matches[-1]


def estimate_token_count(text: str):
    """
    Approximate output token count.

    This is not tokenizer-accurate. It is only a consistent proxy
    until server-reported or tokenizer-based counts are added.
    """
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
# Streaming + inline system metrics
# -----------------------------

def stream_model_with_inline_metrics(
    client: OpenAI,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    metrics_interval_s: float,
):
    """
    Run streaming inference and collect:
    - TTFT
    - end-to-end latency
    - estimated tokens/sec
    - inline system metrics

    Metrics are sampled inline with best-effort cadence.
    No background thread is used.
    """
    start = time.perf_counter()
    first_token_time = None
    last_metric_time = start

    chunks = []
    metric_samples = []

    metric_samples.append(get_system_metrics())

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
                metric_samples.append(get_system_metrics())
                last_metric_time = now

            chunks.append(delta)

            if (now - last_metric_time) >= metrics_interval_s:
                metric_samples.append(get_system_metrics())
                last_metric_time = now

    end = time.perf_counter()

    metric_samples.append(get_system_metrics())

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
    """
    Normalize human-readable sample fields for sample-level CSV logging.
    """
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
    client: OpenAI,
    model_name: str,
    dataset_name: str,
    loader_fn,
    formatter_fn,
    max_tokens: int,
    n_samples: int,
    n_runs: int,
    temperature: float,
    metrics_interval_s: float,
    scorer_fn=None,
):
    """
    Benchmark one dataset across repeated runs.

    Returns:
    - summary_row
    - sample_log_rows
    """
    include_accuracy = scorer_fn is not None
    run_rows = []
    all_sample_log_rows = []

    for run_idx in range(1, n_runs + 1):
        dataset = loader_fn(n=n_samples, seed=run_idx)
        sample_rows = []

        print(f"\n===== {dataset_name.upper()} RUN {run_idx}/{n_runs} =====")
        dataset_start = time.perf_counter()

        for i, sample in enumerate(dataset, start=1):
            prompt = formatter_fn(sample)

            result = stream_model_with_inline_metrics(
                client=client,
                model_name=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                metrics_interval_s=metrics_interval_s,
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
            }
            all_sample_log_rows.append(log_row)

            msg = (
                f"[{dataset_name} {i}/{len(dataset)}] "
                f"ttft_ms={safe_round(row['ttft_ms'], 2)} "
                f"lat_ms={safe_round(row['end_to_end_latency_ms'], 2)} "
                f"tok/s={safe_round(row['tokens_per_sec'], 4)}"
            )
            if include_accuracy:
                msg += f" correct={row['correct']}"
            print(msg)

        dataset_end = time.perf_counter()
        wall_time_s = dataset_end - dataset_start

        run_rows.append(
            summarize_run(
                sample_rows=sample_rows,
                wall_time_s=wall_time_s,
                include_accuracy=include_accuracy,
            )
        )

    summary_row = {"Model name": model_name}

    cpu_peaks = [r["cpu_peak"] for r in run_rows if r["cpu_peak"] is not None]
    ram_peaks = [r["ram_peak"] for r in run_rows if r["ram_peak"] is not None]
    proc_ram_peaks = [r["proc_ram_peak"] for r in run_rows if r["proc_ram_peak"] is not None]
    temp_peaks = [r["temp_peak"] for r in run_rows if r["temp_peak"] is not None]

    if include_accuracy:
        add_mean_sd_columns(summary_row, "Accuracy", [r["accuracy"] for r in run_rows], 4)

    add_mean_sd_columns(summary_row, "TTFT ms", [r["ttft_mean"] for r in run_rows], 4)
    add_mean_sd_columns(summary_row, "End-to-end latency ms", [r["latency_mean"] for r in run_rows], 4)
    add_mean_sd_columns(summary_row, "Tokens/sec", [r["tps_mean"] for r in run_rows], 4)
    add_mean_sd_columns(summary_row, "CPU percent avg", [r["cpu_avg_mean"] for r in run_rows], 4)
    add_mean_sd_columns(summary_row, "Temp C avg", [r["temp_avg_mean"] for r in run_rows], 4)
    add_mean_sd_columns(
        summary_row,
        "Sequential sustained tokens/sec",
        [r["sustained_tps"] for r in run_rows],
        4,
    )
    add_mean_sd_columns(
        summary_row,
        "Sequential sustained samples/sec",
        [r["sustained_sps"] for r in run_rows],
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
    """
    Run llama-bench and return a flat dict of microbenchmark metrics.

    Tries CSV output first. If parsing fails, falls back to returning raw output.
    """
    if not llama_bench_cfg.get("enabled", False):
        return {}

    cmd = [
        llama_bench_cfg["binary"],
        "-m", llama_bench_cfg["model"],
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
# Entry point
# -----------------------------

def main():
    args = parse_args()
    config = load_config(args.config)

    model_name = config["model"]["name"]

    base_url = config["server"]["base_url"]
    api_key = config["server"]["api_key"]

    n_samples = config["benchmark"].get("n_samples", 20)
    n_runs = config["benchmark"].get("n_runs", 5)
    dolly_samples = config["benchmark"].get("dolly_samples", 50)
    dolly_runs = config["benchmark"].get("dolly_runs", 5)
    metrics_interval_s = config["benchmark"].get("metrics_interval_s", 0.2)

    temperature = config["generation"].get("temperature", 0)
    max_tokens = config["generation"]["max_tokens"]

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = RESULTS_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config, output_dir / "config.yaml")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    mmlu_row, mmlu_samples = run_dataset_benchmark(
        client=client,
        model_name=model_name,
        dataset_name="mmlu",
        loader_fn=load_tiny_mmlu,
        formatter_fn=format_mcq,
        max_tokens=max_tokens["mmlu"],
        n_samples=n_samples,
        n_runs=n_runs,
        temperature=temperature,
        metrics_interval_s=metrics_interval_s,
        scorer_fn=score_mmlu,
    )
    write_single_row_csv(output_dir / "mmlu.csv", mmlu_row)
    write_rows_csv(output_dir / "mmlu_samples.csv", mmlu_samples)
    print(f"\nSaved: {output_dir / 'mmlu.csv'}")
    print(f"Saved: {output_dir / 'mmlu_samples.csv'}")

    hellaswag_row, hellaswag_samples = run_dataset_benchmark(
        client=client,
        model_name=model_name,
        dataset_name="hellaswag",
        loader_fn=load_tiny_hellaswag,
        formatter_fn=format_mcq,
        max_tokens=max_tokens["hellaswag"],
        n_samples=n_samples,
        n_runs=n_runs,
        temperature=temperature,
        metrics_interval_s=metrics_interval_s,
        scorer_fn=score_hellaswag,
    )
    write_single_row_csv(output_dir / "hellaswag.csv", hellaswag_row)
    write_rows_csv(output_dir / "hellaswag_samples.csv", hellaswag_samples)
    print(f"\nSaved: {output_dir / 'hellaswag.csv'}")
    print(f"Saved: {output_dir / 'hellaswag_samples.csv'}")

    gsm8k_row, gsm8k_samples = run_dataset_benchmark(
        client=client,
        model_name=model_name,
        dataset_name="gsm8k",
        loader_fn=load_tiny_gsm8k,
        formatter_fn=format_gsm8k,
        max_tokens=max_tokens["gsm8k"],
        n_samples=n_samples,
        n_runs=n_runs,
        temperature=temperature,
        metrics_interval_s=metrics_interval_s,
        scorer_fn=score_gsm8k,
    )
    write_single_row_csv(output_dir / "gsm8k.csv", gsm8k_row)
    write_rows_csv(output_dir / "gsm8k_samples.csv", gsm8k_samples)
    print(f"\nSaved: {output_dir / 'gsm8k.csv'}")
    print(f"Saved: {output_dir / 'gsm8k_samples.csv'}")

    dolly_row, dolly_samples_rows = run_dataset_benchmark(
        client=client,
        model_name=model_name,
        dataset_name="dolly",
        loader_fn=load_dolly_workload,
        formatter_fn=format_dolly,
        max_tokens=max_tokens["dolly"],
        n_samples=dolly_samples,
        n_runs=dolly_runs,
        temperature=temperature,
        metrics_interval_s=metrics_interval_s,
        scorer_fn=None,
    )

    llama_bench_row = run_llama_bench(config.get("llama_bench", {}))

    combined_dolly_row = {"Model name": model_name}
    dolly_prefixed = prefix_keys(
        {k: v for k, v in dolly_row.items() if k != "Model name"},
        "Dolly ",
    )
    combined_dolly_row.update(dolly_prefixed)
    combined_dolly_row.update(llama_bench_row)

    write_single_row_csv(output_dir / "dolly.csv", combined_dolly_row)
    write_rows_csv(output_dir / "dolly_samples.csv", dolly_samples_rows)
    print(f"\nSaved: {output_dir / 'dolly.csv'}")
    print(f"Saved: {output_dir / 'dolly_samples.csv'}")
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()