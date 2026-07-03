import subprocess
import psutil


# Prime CPU measurement for non-blocking usage
psutil.cpu_percent(interval=None)


def get_cpu_ram_metrics(server_pid=None):
    vm = psutil.virtual_memory()

    process_ram_mb = None
    if server_pid is not None:
        try:
            process_ram_mb = round(
                psutil.Process(server_pid).memory_info().rss / 1024 / 1024,
                2,
            )
        except Exception:
            process_ram_mb = None

    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_mb": round(vm.used / 1024 / 1024, 2),
        "ram_percent": vm.percent,
        "process_ram_mb": process_ram_mb,
    }


def get_temperature():
    """
    Cross-platform best-effort temperature retrieval.

    Priority:
    1. Raspberry Pi (vcgencmd)
    2. psutil sensors (Linux laptops/desktops)
    3. None (fallback)
    """

    # Raspberry Pi
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

    # Linux sensors
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


def get_system_metrics(server_pid=None):
    metrics = get_cpu_ram_metrics(server_pid=server_pid)
    metrics["temperature_c"] = get_temperature()
    return metrics