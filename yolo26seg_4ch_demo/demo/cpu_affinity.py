"""Auto-detect CPU clusters and pin the process to the desired one.

On big.LITTLE SoCs like the RK3588 the cores are split into a slower
power-efficient cluster (A55, e.g. cpu0-3 at ~1.8 GHz) and a faster performance
cluster (A76, e.g. cpu4-7 at ~2.3 GHz). The clusters are detected at runtime
from each core's maximum frequency, so this works without hard-coding a specific
board's core numbering.

The sysfs read and the affinity syscall are injected so the classification logic
is unit-testable on any host.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

_CPUFREQ_GLOB = "/sys/devices/system/cpu/cpu{n}/cpufreq/cpuinfo_max_freq"

# Cores whose max frequency is within this fraction of the cluster's frequency
# are treated as the same cluster (guards against tiny per-core differences).
_FREQ_TOLERANCE = 0.02

VALID_CLUSTERS = ("performance", "efficiency", "none")


def read_core_max_freqs(max_cpus: int = 256) -> Dict[int, int]:
    """Read each online CPU's max frequency (kHz) from sysfs.

    Returns an empty dict when cpufreq is unavailable (e.g. non-Linux hosts).
    """

    freqs: Dict[int, int] = {}
    for n in range(max_cpus):
        path = _CPUFREQ_GLOB.format(n=n)
        try:
            with open(path, "r") as fh:
                freqs[n] = int(fh.read().strip())
        except (OSError, ValueError):
            # First missing core typically means we've passed the last CPU, but
            # keep scanning a little in case of holes from offline cores.
            if n > 0 and not freqs:
                break
            continue
    return freqs


def classify_clusters(max_freqs: Dict[int, int]) -> Dict[str, List[int]]:
    """Group CPUs into 'efficiency' and 'performance' clusters by max frequency.

    The efficiency cluster is the set of cores at (within tolerance of) the
    lowest max frequency; the performance cluster is every remaining (faster)
    core. When all cores share one frequency there is a single cluster, so both
    'efficiency' and 'performance' resolve to the full CPU set. Returns empty
    lists when no data is available.
    """

    if not max_freqs:
        return {"efficiency": [], "performance": []}

    lowest = min(max_freqs.values())

    def _within(freq: int, target: int) -> bool:
        if target == 0:
            return freq == 0
        return abs(freq - target) <= target * _FREQ_TOLERANCE

    efficiency = sorted(c for c, f in max_freqs.items() if _within(f, lowest))
    performance = sorted(c for c in max_freqs if c not in set(efficiency))
    # Single-cluster CPU (all cores same speed): performance == whole CPU.
    if not performance:
        performance = sorted(max_freqs)
    return {"efficiency": efficiency, "performance": performance}


def select_cpus(cluster: str, max_freqs: Optional[Dict[int, int]] = None) -> List[int]:
    """Return the CPU indices for the requested cluster ('performance'/'efficiency')."""

    if cluster == "none":
        return []
    if cluster not in ("performance", "efficiency"):
        raise ValueError(
            f"cluster must be one of {VALID_CLUSTERS}, got {cluster!r}"
        )
    if max_freqs is None:
        max_freqs = read_core_max_freqs()
    clusters = classify_clusters(max_freqs)
    return clusters.get(cluster, [])


def apply_affinity(
    cluster: str,
    max_freqs: Optional[Dict[int, int]] = None,
    set_affinity: Optional[Callable[[int, set], None]] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> List[int]:
    """Pin the current process to the CPUs of ``cluster``.

    Returns the list of CPUs the process was pinned to (empty if no affinity was
    applied, e.g. cluster='none', no data, or the platform lacks
    ``sched_setaffinity``). Never raises on failure.
    """

    log = logger or (lambda msg: print(msg, flush=True))

    if cluster == "none":
        log("[INFO] CPU affinity: disabled (cpu_affinity=none)")
        return []
    if cluster not in ("performance", "efficiency"):
        log(f"[WARN] CPU affinity: invalid cluster {cluster!r}; skipping")
        return []

    if max_freqs is None:
        max_freqs = read_core_max_freqs()
    cpus = select_cpus(cluster, max_freqs)
    if not cpus:
        log(f"[WARN] CPU affinity: no '{cluster}' cores detected; skipping")
        return []

    if set_affinity is None:
        set_affinity = getattr(os, "sched_setaffinity", None)
    if set_affinity is None:
        log("[WARN] CPU affinity: sched_setaffinity unavailable on this platform")
        return []

    try:
        set_affinity(0, set(cpus))
    except Exception as exc:  # pragma: no cover - platform-dependent
        log(f"[WARN] CPU affinity: failed to pin to {cluster} cores {cpus}: {exc}")
        return []

    log(f"[INFO] CPU affinity: pinned to {cluster} cores {cpus}")
    return cpus
