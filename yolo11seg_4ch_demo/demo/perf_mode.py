"""Pin CPU/GPU clocks high so the demo is smooth from the very first frame."""

from __future__ import annotations

import glob as _glob
import subprocess
from typing import Callable, Dict, List, Optional

CPU_GOVERNOR_GLOB = "/sys/devices/system/cpu/cpufreq/policy*/scaling_governor"
GPU_DEVFREQ_GLOB = "/sys/class/devfreq/*.gpu"

PERFORMANCE = "performance"

ReadFn = Callable[[str], Optional[str]]
WriteFn = Callable[[str, str], bool]
GlobFn = Callable[[str], List[str]]
LogFn = Callable[[str], None]


def _default_read(path: str) -> Optional[str]:
    try:
        with open(path, "r") as fh:
            return fh.read().strip()
    except OSError:
        return None


def _default_write(path: str, value: str) -> bool:
    """Write ``value`` to ``path``; try direct write then ``sudo -n tee``."""

    try:
        with open(path, "w") as fh:
            fh.write(value)
        return True
    except OSError:
        pass
    try:
        subprocess.run(
            ["sudo", "-n", "tee", path],
            input=value.encode(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=True,
        )
        return True
    except Exception:
        return False


def plan_performance(read: ReadFn, glob_fn: GlobFn) -> Dict[str, str]:
    """Return a ``{sysfs_path: target_value}`` plan to pin clocks high."""

    plan: Dict[str, str] = {}
    for path in sorted(glob_fn(CPU_GOVERNOR_GLOB)):
        cur = read(path)
        if cur is not None and cur != PERFORMANCE:
            plan[path] = PERFORMANCE

    for gpu_dir in sorted(glob_fn(GPU_DEVFREQ_GLOB)):
        max_freq = read(f"{gpu_dir}/max_freq")
        min_path = f"{gpu_dir}/min_freq"
        cur_min = read(min_path)
        if max_freq is not None and cur_min is not None and cur_min != max_freq:
            plan[min_path] = max_freq
    return plan


def enable_performance(
    read: ReadFn = _default_read,
    write: WriteFn = _default_write,
    glob_fn: GlobFn = _glob.glob,
    log: Optional[LogFn] = None,
) -> Dict[str, str]:
    """Pin CPU/GPU clocks high; return originals for :func:`restore`."""

    log = log or (lambda msg: print(msg, flush=True))
    plan = plan_performance(read, glob_fn)

    saved: Dict[str, str] = {}
    changed: List[str] = []
    for path, target in plan.items():
        original = read(path)
        if original is None:
            continue
        if write(path, target):
            saved[path] = original
            changed.append(path)

    if changed:
        log(
            f"[INFO] perf-mode: pinned {len(changed)} clock node(s) to max for "
            f"smooth startup"
        )
    elif plan:
        log(
            "[INFO] perf-mode: could not pin clocks (needs permission); demo will "
            "still run but may take a few seconds to smooth out"
        )
    else:
        log("[INFO] perf-mode: clocks already at performance settings")
    return saved


def restore(saved: Dict[str, str], write: WriteFn = _default_write) -> None:
    """Best-effort restore of values captured by :func:`enable_performance`."""

    for path, value in saved.items():
        try:
            write(path, value)
        except Exception:
            pass

