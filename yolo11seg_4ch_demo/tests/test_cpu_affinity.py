"""Unit tests for CPU cluster detection and affinity selection."""

from __future__ import annotations

import pytest

from demo import cpu_affinity as ca


# RK3588-like layout: cpu0-3 A55 @1.8GHz, cpu4-5 A76 @2.256GHz, cpu6-7 @2.304GHz.
RK3588 = {0: 1800000, 1: 1800000, 2: 1800000, 3: 1800000,
          4: 2256000, 5: 2256000, 6: 2304000, 7: 2304000}


def test_classify_clusters_rk3588():
    clusters = ca.classify_clusters(RK3588)
    assert clusters["efficiency"] == [0, 1, 2, 3]
    # 2.256 and 2.304 GHz are within tolerance -> one performance cluster.
    assert clusters["performance"] == [4, 5, 6, 7]


def test_classify_clusters_uniform():
    uniform = {0: 2000000, 1: 2000000, 2: 2000000, 3: 2000000}
    clusters = ca.classify_clusters(uniform)
    assert clusters["efficiency"] == [0, 1, 2, 3]
    assert clusters["performance"] == [0, 1, 2, 3]


def test_classify_clusters_empty():
    clusters = ca.classify_clusters({})
    assert clusters == {"efficiency": [], "performance": []}


def test_select_cpus_performance_and_efficiency():
    assert ca.select_cpus("performance", RK3588) == [4, 5, 6, 7]
    assert ca.select_cpus("efficiency", RK3588) == [0, 1, 2, 3]


def test_select_cpus_none():
    assert ca.select_cpus("none", RK3588) == []


def test_select_cpus_invalid():
    with pytest.raises(ValueError):
        ca.select_cpus("bogus", RK3588)


def test_apply_affinity_pins_performance_cluster():
    pinned = {}

    def fake_set(pid, cpus):
        pinned["pid"] = pid
        pinned["cpus"] = set(cpus)

    logs = []
    result = ca.apply_affinity(
        "performance", max_freqs=RK3588, set_affinity=fake_set, logger=logs.append
    )
    assert result == [4, 5, 6, 7]
    assert pinned == {"pid": 0, "cpus": {4, 5, 6, 7}}
    assert any("performance" in m for m in logs)


def test_apply_affinity_none_is_noop():
    called = []
    result = ca.apply_affinity(
        "none", max_freqs=RK3588, set_affinity=lambda *a: called.append(a)
    )
    assert result == []
    assert called == []


def test_apply_affinity_no_cores_skips():
    called = []
    result = ca.apply_affinity(
        "performance", max_freqs={}, set_affinity=lambda *a: called.append(a)
    )
    assert result == []
    assert called == []


def test_apply_affinity_failure_is_swallowed():
    def boom(pid, cpus):
        raise OSError("nope")

    result = ca.apply_affinity(
        "performance", max_freqs=RK3588, set_affinity=boom, logger=lambda m: None
    )
    assert result == []
