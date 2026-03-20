import subprocess
import psutil


def get_cpu_ram_metrics():
    vm = psutil.virtual_memory()

    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_mb": round(vm.used / 1024 / 1024, 2),
        "ram_percent": vm.percent,
        "process_ram_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 2),
    }


def get_temperature():
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip().replace("temp=", "").replace("'C", ""))
    except Exception:
        return None


def get_system_metrics():
    metrics = get_cpu_ram_metrics()
    metrics["temperature_c"] = get_temperature()
    return metrics