"""Tests for request coalescer."""

import asyncio

import pytest

from headroom.proxy.coalescer import RequestCoalescer


@pytest.mark.asyncio
async def test_single_request_not_batched():
    c = RequestCoalescer(window_sec=0.1, max_batch=3)
    result = await c.submit({"model": "test", "messages": [{"role": "user", "content": "hi"}]})
    assert result["coalesced"] is False
    await c.close()


@pytest.mark.asyncio
async def test_rapid_requests_batched():
    c = RequestCoalescer(window_sec=0.3, max_batch=3)
    tasks = [
        c.submit({"model": "test", "messages": [{"role": "user", "content": f"msg {i}"}]})
        for i in range(3)
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        assert r["coalesced"] is True
        assert r["batch_size"] == 3
    await c.close()


@pytest.mark.asyncio
async def test_max_batch_triggers_flush():
    c = RequestCoalescer(window_sec=10.0, max_batch=2)
    tasks = [
        c.submit({"model": "test", "messages": [{"role": "user", "content": f"msg {i}"}]})
        for i in range(2)
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        assert r["coalesced"] is True
        assert r["batch_size"] == 2
    await c.close()


@pytest.mark.asyncio
async def test_deduplication():
    c = RequestCoalescer(window_sec=0.1, max_batch=3)
    tasks = [
        c.submit(
            {
                "model": "test",
                "messages": [
                    {"role": "user", "content": "same message"},
                    {"role": "user", "content": f"unique {i}"},
                ],
            }
        )
        for i in range(3)
    ]
    results = await asyncio.gather(*tasks)
    merged = results[0]["payload"]
    messages = merged.get("messages", [])
    same_count = sum(1 for m in messages if m.get("content") == "same message")
    assert same_count == 1
    await c.close()


@pytest.mark.asyncio
async def test_solo_after_window():
    c = RequestCoalescer(window_sec=0.1, max_batch=5)
    r1 = await c.submit({"model": "test", "messages": [{"role": "user", "content": "first"}]})
    assert r1["coalesced"] is False
    await asyncio.sleep(0.2)
    r2 = await c.submit({"model": "test", "messages": [{"role": "user", "content": "second"}]})
    assert r2["coalesced"] is False
    await c.close()


@pytest.mark.asyncio
async def test_async_context_manager_closes_cleanly():
    async with RequestCoalescer(window_sec=0.01, max_batch=2) as coalescer:
        result = await coalescer.submit(
            {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        )

    assert result["coalesced"] is False


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_strand_batch_members():
    """Cancelling a pending request must not raise InvalidStateError or strand later callers."""
    c = RequestCoalescer(window_sec=0.2, max_batch=2)
    alice_task = asyncio.create_task(
        c.submit({"model": "test", "messages": [{"role": "user", "content": "alice"}]})
    )
    await asyncio.sleep(0.02)
    alice_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await alice_task

    bob_res, carol_res = await asyncio.gather(
        c.submit({"model": "test", "messages": [{"role": "user", "content": "bob"}]}),
        c.submit({"model": "test", "messages": [{"role": "user", "content": "carol"}]}),
    )
    assert bob_res["coalesced"] is True
    assert carol_res["coalesced"] is True
    assert bob_res["batch_size"] == 2
    assert c.get_stats()["cancelled_requests"] >= 1
    await c.close()


@pytest.mark.asyncio
async def test_repeated_conversation_turns_preserved_within_request():
    """Repeated messages within one request must not be deduplicated away."""
    c = RequestCoalescer(window_sec=0.1, max_batch=2)
    req1 = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "shared intro"},
            {"role": "user", "content": "retry"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "retry"},
        ],
    }
    req2 = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "shared intro"},
            {"role": "user", "content": "follow up"},
        ],
    }
    r1, r2 = await asyncio.gather(c.submit(req1), c.submit(req2))
    merged_msgs = r1["payload"]["messages"]
    retry_count = sum(1 for m in merged_msgs if m.get("content") == "retry")
    intro_count = sum(1 for m in merged_msgs if m.get("content") == "shared intro")
    assert intro_count == 1
    assert retry_count == 2
    assert r2["payload"]["messages"] == merged_msgs
    await c.close()


@pytest.mark.asyncio
async def test_incompatible_settings_partitioned_into_separate_batches():
    """Requests with different model, system, tools, or scope must not inherit each other's settings."""
    c = RequestCoalescer(window_sec=0.05, max_batch=3)
    r_model_a, r_model_b, r_sys_b = await asyncio.gather(
        c.submit(
            {
                "model": "claude-sonnet-4-5",
                "system": "sys-a",
                "messages": [{"role": "user", "content": "a"}],
            }
        ),
        c.submit(
            {
                "model": "claude-haiku-4-5",
                "system": "sys-a",
                "messages": [{"role": "user", "content": "b"}],
            }
        ),
        c.submit(
            {
                "model": "claude-sonnet-4-5",
                "system": "sys-b",
                "messages": [{"role": "user", "content": "c"}],
            }
        ),
    )
    assert r_model_a["coalesced"] is False
    assert r_model_a["payload"]["model"] == "claude-sonnet-4-5"
    assert r_model_a["payload"]["system"] == "sys-a"

    assert r_model_b["coalesced"] is False
    assert r_model_b["payload"]["model"] == "claude-haiku-4-5"

    assert r_sys_b["coalesced"] is False
    assert r_sys_b["payload"]["system"] == "sys-b"
    await c.close()


@pytest.mark.asyncio
async def test_dispatch_fn_executes_and_propagates_results_and_errors():
    """Coalescer must execute dispatch_fn once per compatible batch and deliver response/errors."""
    dispatched_payloads = []

    async def fake_dispatch(payload):
        dispatched_payloads.append(payload)
        return {"id": "msg_batch_1", "usage": {"input_tokens": 42, "output_tokens": 10}}

    c = RequestCoalescer(window_sec=0.2, max_batch=2, dispatch_fn=fake_dispatch)
    r1, r2 = await asyncio.gather(
        c.submit({"model": "test", "messages": [{"role": "user", "content": "one"}]}),
        c.submit({"model": "test", "messages": [{"role": "user", "content": "two"}]}),
    )
    assert len(dispatched_payloads) == 1
    assert r1["coalesced"] is True
    assert r1["response"]["id"] == "msg_batch_1"
    assert r2["response"]["id"] == "msg_batch_1"
    assert c.get_stats()["api_calls_saved"] == 1
    await c.close()

    async def failing_dispatch(_payload):
        raise RuntimeError("upstream unavailable")

    c_err = RequestCoalescer(window_sec=0.2, max_batch=2, dispatch_fn=failing_dispatch)
    with pytest.raises(RuntimeError, match="upstream unavailable"):
        await asyncio.gather(
            c_err.submit({"model": "test", "messages": [{"role": "user", "content": "one"}]}),
            c_err.submit({"model": "test", "messages": [{"role": "user", "content": "two"}]}),
        )
    assert c_err.get_stats()["dispatch_errors"] == 1
    await c_err.close()
