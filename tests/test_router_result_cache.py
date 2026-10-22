"""The content router's compression cache stays within its memory budget."""

from __future__ import annotations

import subprocess
import sys

from headroom.cache.result_store import ResultStore, open_result_store
from headroom.transforms.content_router import (
    CompressionCache,
    ContentRouter,
    ContentRouterConfig,
    content_cache_key,
)


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


def test_a_result_in_use_is_kept_and_an_idle_one_is_dropped(tmp_path, monkeypatch):
    # Nothing signals a session's end: a result expires once unused for the
    # ttl, from memory and from disk, while a session still reading it keeps it.
    import time as time_module

    clock = [1000.0]
    monkeypatch.setattr(time_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time_module, "time", lambda: clock[0])
    store = ResultStore(tmp_path / "c.db", ttl_seconds=100)
    cache = CompressionCache(ttl_seconds=100, store=store)
    cache.put("live", "kept", 0.5, "kompress")
    cache.put("idle", "dropped", 0.5, "kompress")
    for _ in range(5):  # 'live' is read every 60 s, for 300 s
        clock[0] += 60
        assert cache.get("live") is not None
    assert cache.get("idle") is None
    assert store.get("idle") is None
    assert store.get("live") == ("kept", 0.5, "kompress")


def test_routers_configured_differently_never_share_output(tmp_path, monkeypatch):
    # The proxy can run two routers (per-provider Kompress overrides), and a
    # restart can change settings; both read the same file.
    monkeypatch.setenv("HEADROOM_COMPRESSION_CACHE_PATH", str(tmp_path / "c.db"))
    lossy = ContentRouter(ContentRouterConfig(result_cache_disk=True))
    lossless = ContentRouter(ContentRouterConfig(result_cache_disk=True, lossless=True))
    lossy._cache.put("k", "lossy output", 0.3, "kompress")
    assert lossless._cache.get("k") is None
    same = ContentRouter(ContentRouterConfig(result_cache_disk=True))
    assert same._cache.get("k") == ("lossy output", 0.3, "kompress")


def test_the_namespace_is_the_same_in_every_process():
    code = (
        "from headroom.transforms.content_router import ContentRouterConfig, _config_namespace;"
        "print(_config_namespace(ContentRouterConfig()))"
    )
    seen = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**__import__("os").environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("1", "2")
    }
    assert len(seen) == 1


def test_a_broken_database_is_a_miss_not_a_failed_request(tmp_path):
    store = ResultStore(tmp_path / "c.db", 1800)
    cache = CompressionCache(store=store)
    store._db.close()
    cache.put("k", "text", 0.5, "kompress")  # no raise
    cache.clear()
    assert cache.get("k") is None


def test_a_result_pointing_at_a_ccr_original_is_not_kept_on_disk(tmp_path):
    store = ResultStore(tmp_path / "c.db", 1800)
    cache = CompressionCache(store=store)
    cache.put("k", "first lines <<ccr:0123456789abcdef>>", 0.2, "x")
    assert store.get("k") is None
