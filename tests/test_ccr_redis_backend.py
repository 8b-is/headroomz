"""One CCR store for every replica: HEADROOM_CCR_BACKEND=redis.

Behind a gateway that spreads requests across replicas, the retrieval of a
<<ccr:hash>> marker can land on a replica other than the one that compressed.
"""

from __future__ import annotations

import pytest

fakeredis = pytest.importorskip("fakeredis")

from headroom.cache.backends.redis import RedisBackend  # noqa: E402
from headroom.cache.compression_store import (  # noqa: E402
    CompressionStore,
    _create_default_ccr_backend,
)


@pytest.fixture
def server():
    return fakeredis.FakeServer()


def _replica(server) -> CompressionStore:
    backend = RedisBackend("redis://localhost", client=fakeredis.FakeRedis(server=server))
    return CompressionStore(backend=backend)


def test_a_marker_made_on_one_replica_is_retrieved_on_another(server):
    hash_key = _replica(server).store("the full original output " * 50, "short")
    entry = _replica(server).retrieve(hash_key)
    assert entry is not None and entry.original_content.startswith("the full original output")


def test_an_entry_expires_in_redis_when_it_expires(server):
    store = _replica(server)
    hash_key = store.store("original " * 40, "short", ttl=60)
    ttl_ms = fakeredis.FakeRedis(server=server).pttl(f"headroom:ccr:{hash_key}")
    assert 0 < ttl_ms <= 60_000


def test_deployments_with_different_prefixes_do_not_share(server, monkeypatch):
    hash_key = _replica(server).store("original " * 40, "short")
    monkeypatch.setenv("HEADROOM_REDIS_KEY_PREFIX", "staging")
    assert _replica(server).retrieve(hash_key) is None


def test_redis_down_is_a_miss_and_never_fails_compression(monkeypatch):
    import redis

    class Down(fakeredis.FakeRedis):
        def execute_command(self, *args, **kwargs):
            raise redis.ConnectionError("connection refused")

    store = CompressionStore(backend=RedisBackend("redis://localhost", client=Down()))
    hash_key = store.store("original " * 40, "short")  # no raise
    assert store.retrieve(hash_key) is None


def test_a_corrupt_entry_is_a_miss(server):
    store = _replica(server)
    fakeredis.FakeRedis(server=server).set("headroom:ccr:abc123def456", "not json")
    assert store.retrieve("abc123def456") is None


def test_the_env_selects_the_redis_backend(monkeypatch):
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "redis")
    monkeypatch.setenv("HEADROOM_REDIS_URL", "redis://127.0.0.1:6399/0")
    assert isinstance(_create_default_ccr_backend(), RedisBackend)


def test_a_url_that_is_not_redis_falls_back_to_the_local_store(monkeypatch):
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "redis")
    monkeypatch.setenv("HEADROOM_REDIS_URL", "memcached://cache:11211")
    assert _create_default_ccr_backend() is None


def test_redis_is_skipped_for_a_while_after_a_connection_error():
    import redis

    calls = []

    class Down(fakeredis.FakeRedis):
        def execute_command(self, *args, **kwargs):
            calls.append(args[0])
            raise redis.ConnectionError("connection refused")

    backend = RedisBackend("redis://localhost", client=Down())
    for _ in range(5):
        backend.get("abc123def456")
    assert len(calls) == 1  # one failure, then back-off


def test_storing_never_scans_the_shared_keyspace(server, monkeypatch):
    # A per-replica cap would count (SCAN) the whole store on every write and
    # evict other replicas' live entries; Redis expires entries by TTL.
    store = _replica(server)
    store._max_entries = 2
    monkeypatch.setattr(RedisBackend, "_scan", lambda self: pytest.fail("scanned"))
    hashes = [store.store(f"original {i} " * 40, f"short {i}") for i in range(5)]
    # Re-stores (a marker met again every turn) and retrievals neither.
    for _ in range(300):
        store.store("original 0 " * 40, "short 0")
    assert all(store.retrieve(h) is not None for h in hashes)
    assert not store._eviction_heap and not store._expiration_heap  # nothing grows


def test_a_redis_that_reads_but_refuses_writes_logs_once(caplog):
    import redis

    class Full(fakeredis.FakeRedis):
        def set(self, *args, **kwargs):
            raise redis.ResponseError("OOM command not allowed")

    store = CompressionStore(backend=RedisBackend("redis://localhost", client=Full()))
    with caplog.at_level("INFO", logger="headroom.cache.redis_client"):
        for i in range(5):
            store.store(f"original {i} " * 40, "short")
            store.retrieve("abc123def456")
    messages = [r.getMessage() for r in caplog.records]
    assert sum("failed" in m for m in messages) == 1
    assert not any("works again" in m for m in messages)


def test_a_prefix_with_glob_characters_matches_only_itself(server, monkeypatch):
    _replica(server).store("ours " * 40, "short")
    monkeypatch.setenv("HEADROOM_REDIS_KEY_PREFIX", "head*")
    assert RedisBackend("redis://localhost", client=fakeredis.FakeRedis(server=server)).count() == 0


def test_stats_and_retrieval_never_read_or_rewrite_whole_entries(server, monkeypatch):
    store = _replica(server)
    hashes = [
        store.store(f"original {i} " * 40, f"short {i}", original_tokens=80, compressed_tokens=2)
        for i in range(5)
    ]
    reads, writes = [], []
    monkeypatch.setattr(fakeredis.FakeRedis, "get", lambda self, k: reads.append(k))
    stats = store.get_stats()
    assert stats["entry_count"] == 5 and not reads
    # Numeric totals (/stats contract): what this replica stored and retrieved.
    assert (stats["total_original_tokens"], stats["total_retrievals"]) == (400, 0)
    monkeypatch.undo()
    monkeypatch.setattr(fakeredis.FakeRedis, "set", lambda self, *a, **kw: writes.append(a))
    assert store.retrieve(hashes[0]) is not None
    assert not writes
    assert store.get_stats()["total_retrievals"] == 1


def test_credentials_in_a_redis_error_are_not_logged(caplog):
    import redis

    class Refusing(fakeredis.FakeRedis):
        def get(self, *args, **kwargs):
            raise redis.ConnectionError("cannot reach redis://app:s3cret@cache:6379/0")

    backend = RedisBackend("redis://localhost", client=Refusing())
    with caplog.at_level("WARNING", logger="headroom.cache.redis_client"):
        assert backend.get("abc") is None
    assert "s3cret" not in caplog.text and "//***@cache" in caplog.text


def test_a_write_redis_refuses_adds_no_tokens(caplog):
    import redis

    class Full(fakeredis.FakeRedis):
        def set(self, *args, **kwargs):
            raise redis.ResponseError("OOM command not allowed")

    store = CompressionStore(backend=RedisBackend("redis://localhost", client=Full()))
    store.store("original " * 40, "short", original_tokens=80, compressed_tokens=2)
    stats = store.get_stats()
    assert (stats["total_original_tokens"], stats["totals_scope"]) == (0, "this_replica")
