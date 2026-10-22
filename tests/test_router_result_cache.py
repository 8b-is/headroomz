"""The content router's compression cache stays within its memory budget."""

from __future__ import annotations

import subprocess
import sys

from headroom.cache.result_store import ResultStore, open_result_store
from headroom.transforms.content_router import CompressionCache, content_cache_key


def test_results_stay_within_the_memory_budget_least_recently_used_first():
    cache = CompressionCache(max_bytes=10_000)
    for i in range(100):
        cache.put(f"k{i}", "x" * 1000, 0.5, "kompress")
        cache.get("k0")  # k0 stays recently used
    assert cache.stats["cache_bytes"] <= 10_000
    assert cache.get("k0") is not None
    assert cache.get("k1") is None  # evicted, not kept forever


def test_an_evicted_result_comes_back_from_disk_unchanged(tmp_path):
    cache = CompressionCache(max_bytes=3_000, store=ResultStore(tmp_path / "c.db", 1800))
    cache.put("first", "compressed once", 0.4, "kompress")
    for i in range(10):
        cache.put(f"k{i}", "y" * 1000, 0.5, "kompress")
    # Out of memory, but read back: no second compression, same bytes.
    assert cache.get("first") == ("compressed once", 0.4, "kompress")
    assert cache.stats["cache_disk_hits"] == 1


def test_the_skip_set_is_bounded_too():
    cache = CompressionCache(max_bytes=1024 * 1024)
    for i in range(50_000):
        cache.mark_skip(f"s{i}")
    assert cache.skip_size <= 2000
    assert cache.is_skipped("s49999")


def test_keys_are_the_same_in_every_process():
    code = (
        "from headroom.transforms.content_router import content_cache_key;"
        "print(content_cache_key('same text', 0.5, gated=True))"
    )
    other = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert other == content_cache_key("same text", 0.5, gated=True)
    assert content_cache_key("same text", 0.5) != other  # gated tool output kept apart


def test_stateless_mode_writes_no_cache_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("HEADROOM_STATELESS", "1")
    assert open_result_store(1800) is None
    assert not (tmp_path / "compression_cache.db").exists()
