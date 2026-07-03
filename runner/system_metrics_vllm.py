import subprocess
from typing import Optional, Dict, Any

import psutil


psutil.cpu_percent(interval=None)


def _safe_float(value: str) -> Optional[float]:
    value = value.strip()
    if value in ("", "N/A", "[N/A]"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def get_cpu_ram_metrics(server_pid: Optional[int] = None) -> Dict[str, Any]:
    vm = psutil.virtual_memory()

    process_ram_mb = None
    if server_pid is not None:
        try:
            process_ram_mb = round(
                psutil.Process(server_pid).memory_info().rss / 1024 / 1024,
                2,
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            process_ram_mb = None
    else:
        process_ram_mb = round(psutil.Process().memory_info().rss / 1024 / 1024, 2)

    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_mb": round(vm.used / 1024 / 1024, 2),
        "ram_percent": vm.percent,
        "process_ram_mb": process_ram_mb,
    }


def get_gpu_metrics() -> Dict[str, Any]:
    default = {
        "gpu_util_percent": None,
        "vram_used_mb": None,
        "vram_total_mb": None,
        "power_w": None,
    }

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return default

        parts = [x.strip() for x in lines[0].split(",")]
        if len(parts) != 4:
            return default

        util, mem_used, mem_total, power = parts

        return {
            "gpu_util_percent": _safe_float(util),
            "vram_used_mb": _safe_float(mem_used),
            "vram_total_mb": _safe_float(mem_total),
            "power_w": _safe_float(power),
        }

    except (subprocess.SubprocessError, FileNotFoundError):
        return default


def get_temperature() -> Optional[float]:
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True,
            text=True,
            check=True,
        )
        value = result.stdout.strip().replace("temp=", "").replace("'C", "")
        return float(value)
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass

    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None

        for key in ("coretemp", "cpu-thermal", "k10temp"):
            if key in temps and temps[key]:
                return float(temps[key][0].current)

        for entries in temps.values():
            if entries:
                return float(entries[0].current)

    except Exception:
        pass

    return None


def get_system_metrics(
    include_gpu: bool = True,
    server_pid: Optional[int] = None,
) -> Dict[str, Any]:
    metrics = get_cpu_ram_metrics(server_pid=server_pid)
    metrics["temperature_c"] = get_temperature()

    if include_gpu:
        metrics.update(get_gpu_metrics())
    else:
        metrics.update(
            {
                "gpu_util_percent": None,
                "vram_used_mb": None,
                "vram_total_mb": None,
                "power_w": None,
            }
        )

    return metrics