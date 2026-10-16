"""Per-item original/sent measurement feeding the savings record."""

from __future__ import annotations

import json
import threading
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from headroom.proxy import savings
from headroom.proxy.loopback_guard import require_loopback
from headroom.proxy.savings import ItemLedger, ItemMeasure, prefix_key, responses_items
from headroom.proxy.server import ProxyConfig, create_app


def _words(text: str) -> int:
    return len(text.split(","))


def _items(*texts: str) -> list[tuple[str, str]]:
    return [(t, t) for t in texts]


def _turn(ledger, original, sent, count=_words, *, succeeded=True):
    """Measure one request and settle it, as the outcome funnel does."""
    m = ledger.measure("conv", original, sent, count)
    ledger.settle(m, succeeded=succeeded)
    return m


def test_a_conversation_is_measured_against_its_previous_request():
    ledger = ItemLedger()
    calls: list[str] = []

    def count(text: str) -> int:
        calls.append(text)
        return _words(text)

    first = ['["t1","t2"]', '"sys"', '{"a":1,"b":2,"c":3}']
    sent1 = ['["t1"]', '"sys"', '{"a":1,"b":2,"c":3}']
    second = [*first, '{"d":4,"e":5}']
    sent2 = [*sent1, '{"d":4}']

    assert _turn(ledger, _items(*first), _items(*sent1), count) == ItemMeasure(
        sent_tokens=5, removed_old=0, removed_new=1, sent_shared_prefix=0, bust=0
    )
    assert _turn(ledger, _items(*second), _items(*sent2), count) == ItemMeasure(
        sent_tokens=6, removed_old=1, removed_new=1, sent_shared_prefix=5, bust=0
    )
    # Re-sent items are tokenized once, not once per turn.
    assert len(calls) == len(set(calls))


def test_items_follow_the_providers_prompt_order():
    body = {"tools": [{"name": "t"}], "instructions": "be brief", "input": [{"x": 1}, {"y": 2}]}
    assert responses_items(body) == _items('[{"name":"t"}]', '"be brief"', '{"x":1}', '{"y":2}')
    assert responses_items({"input": "hi"}) == _items("[]", '""', '"hi"')


def test_reasoning_ciphertext_is_identity_not_billed_text():
    """Codex replays encrypted reasoning every turn; counting the ciphertext
    made Headroom's count drift far above the provider's bill."""
    item = {"type": "reasoning", "summary": [], "encrypted_content": "gAAAA" * 50}
    [(identity, billable)] = responses_items({"input": [item]})[2:]
    assert "gAAAA" in identity
    assert billable == '{"type":"reasoning","summary":[]}'


def test_a_folded_transcript_keeps_the_old_items_old():
    """A hook folding two items into one shifts every later position; pairing
    by position then priced untouched new items as removed."""
    ledger = ItemLedger()
    t, i = "[]", '"sys"'
    a, b, c, d = ("," * 9 + x for x in "abcd")  # 10 tokens each under _words
    _turn(ledger, _items(t, i, a, b, c), _items(t, i, a, b, c), _words)
    m = _turn(ledger, _items(t, i, a, b, c, d), _items(t, i, "f", c, d), _words)
    assert (m.removed_old, m.removed_new) == (19, 0)
    # The fold moved c: its sent copy is outside the shared prefix, a bust.
    assert m.bust == 11 and m.aligned is False


def test_a_fold_after_an_item_compressed_in_place_keeps_the_prefix():
    """The common Codex case: an early tool output compressed every turn, then
    a hook folds the new items. The compressed item must still pair at the head,
    or every later item is priced as removed and as a bust."""
    ledger = ItemLedger()
    t, i = "[]", '"sys"'
    big, small = "," * 99 + "x", "x"  # 100 and 1 tokens under _words
    n1, n2 = "," * 9 + "n", "," * 9 + "m"
    _turn(ledger, _items(t, i, big), _items(t, i, small), _words)
    m = _turn(ledger, _items(t, i, big, n1, n2), _items(t, i, small, "f"), _words)
    assert (m.sent_shared_prefix, m.bust) == (3, 0)
    assert (m.removed_old, m.removed_new) == (99, 19)
    assert m.aligned is False


def test_an_approximate_pairing_marks_the_record_estimated():
    m = ItemMeasure(
        sent_tokens=10, removed_old=0, removed_new=5, sent_shared_prefix=0, bust=0, aligned=False
    )
    assert savings.compute_savings_record(m, None, model="unpriced-model").estimated


def test_a_failed_request_does_not_become_the_baseline():
    ledger = ItemLedger()
    first = _items("[]", '"sys"', '"a"')
    _turn(ledger, first, first)
    _turn(ledger, _items("[]", '"other"', '"b"'), _items("[]", '"other"', '"b"'), succeeded=False)
    assert _turn(ledger, first, first).sent_shared_prefix == 3


def test_a_late_settling_older_turn_keeps_the_newer_baseline():
    ledger = ItemLedger()
    old, new = _items("[]", '"a"'), _items("[]", '"a"', '"b"')
    slow = ledger.measure("conv", old, old, _words)
    fast = ledger.measure("conv", new, new, _words)
    ledger.settle(fast, succeeded=True)
    ledger.settle(slow, succeeded=True)
    assert _turn(ledger, new, new).sent_shared_prefix == 3


def test_requests_awaiting_their_outcome_are_bounded_by_items(monkeypatch):
    monkeypatch.setattr(savings, "MAX_STORED_ITEMS", 4)
    ledger = ItemLedger()
    abandoned = ledger.measure("a", _items("[]", '"x"', '"y"'), _items("[]", '"x"', '"y"'), _words)
    kept = ledger.measure("b", _items("[]", '"z"'), _items("[]", '"z"'), _words)
    # The oldest pending request was forgotten; settling it is a no-op.
    ledger.settle(abandoned, succeeded=True)
    ledger.settle(kept, succeeded=True)
    # Each probe is itself a pending request under the budget: kept first.
    assert (
        ledger.measure("b", _items("[]", '"z"'), _items("[]", '"z"'), _words).sent_shared_prefix
        == 2
    )
    again = _items("[]", '"x"', '"y"')
    assert ledger.measure("a", again, again, _words).sent_shared_prefix == 0


def test_a_slow_older_measure_never_evicts_a_newer_one(monkeypatch):
    """JerrettDavis's #4240 interleaving: ticket 1 stalls inside token
    counting while ticket 2 measures four items. Eviction must drop the older
    ticket, not the newer one that finished first."""
    monkeypatch.setattr(savings, "MAX_STORED_ITEMS", 4)
    ledger = ItemLedger()
    counting, release = threading.Event(), threading.Event()

    def stalled(text: str) -> int:
        counting.set()
        assert release.wait(timeout=10)
        return _words(text)

    older: list[ItemMeasure] = []
    slow = _items("[]", '"x"')
    thread = threading.Thread(target=lambda: older.append(ledger.measure("a", slow, slow, stalled)))
    thread.start()
    assert counting.wait(timeout=10)
    new = _items("[]", '"b1"', '"b2"', '"b3"')
    newer = ledger.measure("b", new, new, _words)
    release.set()
    thread.join(timeout=10)

    ledger.settle(newer, succeeded=True)
    ledger.settle(older[0], succeeded=True)
    assert ledger.measure("b", new, new, _words).sent_shared_prefix == 4


def test_settled_and_pending_items_share_one_budget(monkeypatch):
    monkeypatch.setattr(savings, "MAX_STORED_ITEMS", 4)
    ledger = ItemLedger()
    idle = _items("[]", '"a1"', '"a2"', '"a3"')
    _turn(ledger, idle, idle)
    active = _items("[]", '"b1"', '"b2"', '"b3"')
    in_flight = ledger.measure("b", active, active, _words)
    # Admitting the in-flight request evicted the idle baseline at once.
    assert ledger._stored + ledger._pending_items == 4
    ledger.settle(in_flight, succeeded=True)
    assert ledger.measure("b", active, active, _words).sent_shared_prefix == 4
    assert ledger.measure("conv", idle, idle, _words).sent_shared_prefix == 0


def test_a_request_larger_than_the_budget_is_measured_but_never_kept(monkeypatch):
    """It cannot become the next baseline, so the conversation keeps the last
    one that fit (a shorter shared prefix, which errs toward the miss rate) and
    the records it feeds are marked estimated. Other conversations keep theirs."""
    monkeypatch.setattr(savings, "MAX_STORED_ITEMS", 4)
    ledger = ItemLedger()
    other = _items("[]")
    ledger.settle(ledger.measure("other", other, other, _words), succeeded=True)
    base = _items("[]", '"s"', '"u1"')
    _turn(ledger, base, base)
    grown = [*base, *_items('"u2"', '"u3"')]
    m = _turn(ledger, grown, grown)
    assert (m.sent_shared_prefix, m.aligned) == (3, False)
    longer = [*grown, *_items('"u4"')]
    m = _turn(ledger, longer, longer)
    assert (m.sent_shared_prefix, m.aligned) == (3, False)
    assert ledger.measure("other", other, other, _words).sent_shared_prefix == 1


def test_messages_stand_in_for_input():
    assert responses_items({"messages": [{"role": "user", "content": "hi"}]})[2:] == _items(
        '{"role":"user","content":"hi"}'
    )


def test_side_requests_do_not_replace_the_conversations_previous_request():
    main = _items("[]", '"agent"', '"hi"')
    side = _items("[]", '"summarize"', '"hi"')
    assert prefix_key("conv-1", "m", main) != prefix_key("conv-1", "m", side)
    assert prefix_key("conv-1", "m", main) == prefix_key(
        "conv-1", "m", [*main, ('"more"', '"more"')]
    )
    assert prefix_key(None, "m", main) == prefix_key(None, "m", [*main, ('"more"', '"more"')])
    assert prefix_key(None, "m", main) != prefix_key(None, "other", main)


def _sse(usage: dict) -> str:
    done = {"type": "response.completed", "response": {"id": "r", "output": [], "usage": usage}}
    return f"event: response.completed\ndata: {json.dumps(done)}\n\n"


@pytest.mark.parametrize("stream", [False, True])
@respx.mock
def test_responses_outcomes_carry_the_measure(monkeypatch, stream):
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
        )
    )
    app.dependency_overrides[require_loopback] = lambda: None
    outcomes: list[Any] = []

    # Capture at the funnel's first effect, so the real funnel (which settles
    # the measure) still runs.
    monkeypatch.setattr("headroom.telemetry.session.record_outcome", outcomes.append)
    monkeypatch.setattr(savings, "_ledger", ItemLedger())
    usage = {"input_tokens": 10, "output_tokens": 1}
    respx.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(
            200,
            text=_sse(usage),
            headers={"content-type": "text/event-stream"},
        )
        if stream
        else httpx.Response(200, json={"id": "r", "output": [], "usage": usage})
    )
    turn1 = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "a"}]}]
    turn2 = [
        *turn1,
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "b"}]},
    ]
    with TestClient(app) as client:
        for items in (turn1, turn2):
            r = client.post(
                "/v1/responses",
                json={
                    "model": "gpt-4o",
                    "instructions": "be brief",
                    "input": items,
                    "stream": stream,
                    "client_metadata": {"session_id": "conv-test"},
                },
                headers={"authorization": "Bearer sk-test"},
            )
            assert r.status_code == 200
            r.read()

    first, second = (o.savings_measure for o in outcomes)
    assert first.removed == 0 and first.sent_shared_prefix == 0
    # Nothing was compressed, so the whole first request is the shared prefix.
    assert second.sent_shared_prefix == first.sent_tokens
    assert second.removed == 0 and second.bust == 0
