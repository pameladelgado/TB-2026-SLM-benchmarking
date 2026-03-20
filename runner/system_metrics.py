import subprocess
import psutil


# Prime CPU measurement (important for non-blocking usage)
psutil.cpu_percent(interval=None)


def get_cpu_ram_metrics():
    vm = psutil.virtual_memory()

    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_mb": round(vm.used / 1024 / 1024, 2),
        "ram_percent": vm.percent,
        "process_ram_mb": round(
            psutil.Process().memory_info().rss / 1024 / 1024, 2
        ),
    }


def get_temperature():
    """
    Cross-platform best-effort temperature retrieval.

    Priority:
    1. Raspberry Pi (vcgencmd)
    2. psutil sensors (Linux laptops/desktops)
    3. None (fallback)
    """

    # --- Raspberry Pi ---
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(
            result.stdout.strip().replace("temp=", "").replace("'C", "")
        )
    except Exception:
        pass

    # --- Linux (psutil sensors) ---
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None

        # Common sensor groups
        for key in ("coretemp", "cpu-thermal", "k10temp"):
            if key in temps and temps[key]:
                return float(temps[key][0].current)

        # fallback: take first available
        for entries in temps.values():
            if entries:
                return float(entries[0].current)

    except Exception:
        pass

    # --- Fallback ---
    return None


def get_system_metrics():
    metrics = get_cpu_ram_metrics()
    metrics["temperature_c"] = get_temperature()
    return metrics