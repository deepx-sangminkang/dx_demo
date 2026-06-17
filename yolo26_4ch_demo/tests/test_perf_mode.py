"""Unit tests for perf_mode (DVFS clock-pinning planning + apply/restore).

Real sysfs is board-only, so reads/writes/globs are injected.
"""

from __future__ import annotations

from demo import perf_mode as pm


class _FakeFs:
    def __init__(self, files):
        # files: {path: value}
        self.files = dict(files)
        self.writes = []
        self.fail_writes = set()

    def read(self, path):
        return self.files.get(path)

    def write(self, path, value):
        if path in self.fail_writes:
            return False
        self.files[path] = value
        self.writes.append((path, value))
        return True

    def glob(self, pattern):
        import fnmatch

        return [p for p in self.files if fnmatch.fnmatch(p, pattern)]


def _board_fs(cpu_gov="ondemand", gpu_min="300000000", gpu_max="1000000000"):
    return _FakeFs(
        {
            "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor": cpu_gov,
            "/sys/devices/system/cpu/cpufreq/policy4/scaling_governor": cpu_gov,
            "/sys/devices/system/cpu/cpufreq/policy6/scaling_governor": cpu_gov,
            # The devfreq directory itself, so the ``*.gpu`` glob can match it.
            "/sys/class/devfreq/fb000000.gpu": "",
            "/sys/class/devfreq/fb000000.gpu/min_freq": gpu_min,
            "/sys/class/devfreq/fb000000.gpu/max_freq": gpu_max,
        }
    )


def test_plan_sets_cpu_performance_and_pins_gpu_min_to_max():
    fs = _board_fs()
    plan = pm.plan_performance(fs.read, fs.glob)
    assert plan["/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"] == "performance"
    assert plan["/sys/devices/system/cpu/cpufreq/policy4/scaling_governor"] == "performance"
    assert plan["/sys/class/devfreq/fb000000.gpu/min_freq"] == "1000000000"


def test_plan_skips_nodes_already_at_target():
    fs = _board_fs(cpu_gov="performance", gpu_min="1000000000", gpu_max="1000000000")
    plan = pm.plan_performance(fs.read, fs.glob)
    assert plan == {}


def test_enable_returns_original_values_for_restore():
    fs = _board_fs()
    saved = pm.enable_performance(fs.read, fs.write, fs.glob, log=lambda m: None)
    # CPU governors now performance
    assert fs.files["/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"] == "performance"
    # GPU min pinned to max
    assert fs.files["/sys/class/devfreq/fb000000.gpu/min_freq"] == "1000000000"
    # Saved holds the originals
    assert saved["/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"] == "ondemand"
    assert saved["/sys/class/devfreq/fb000000.gpu/min_freq"] == "300000000"


def test_enable_does_not_save_when_write_fails():
    fs = _board_fs()
    fs.fail_writes.add("/sys/devices/system/cpu/cpufreq/policy0/scaling_governor")
    saved = pm.enable_performance(fs.read, fs.write, fs.glob, log=lambda m: None)
    # The failed node must not be in saved (nothing to restore).
    assert "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor" not in saved
    # Other nodes still applied.
    assert "/sys/class/devfreq/fb000000.gpu/min_freq" in saved


def test_restore_writes_back_saved_values():
    fs = _board_fs()
    saved = pm.enable_performance(fs.read, fs.write, fs.glob, log=lambda m: None)
    pm.restore(saved, fs.write)
    assert fs.files["/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"] == "ondemand"
    assert fs.files["/sys/class/devfreq/fb000000.gpu/min_freq"] == "300000000"


def test_enable_noop_when_nothing_to_change_logs_and_returns_empty():
    fs = _board_fs(cpu_gov="performance", gpu_min="1000000000", gpu_max="1000000000")
    logs = []
    saved = pm.enable_performance(fs.read, fs.write, fs.glob, log=logs.append)
    assert saved == {}
    assert any("performance settings" in m for m in logs)
