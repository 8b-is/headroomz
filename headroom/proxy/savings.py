"""One per-request savings record: counterfactual cost minus actual cost.

Every savings surface (/stats, the durable ledger, persistent savings, the
dashboard, the PERF line) is meant to derive from the :class:`SavingsRecord`
this module computes, so no surface prices savings its own way.

Two rulers meet here and are only ever combined through RATIOS:

* Headroom's tokenizer gives per-item counts of the request as the client sent
  it (ORIGINAL) and as Headroom forwarded it (SENT). Both are counted with the
  same tokenizer, so a tokenizer bias cancels in their difference.
* The provider's usage block gives exact billed totals (uncached, cache read,
  cache write) but no per-item detail.

``k = billed / sent`` converts Headroom units into provider units, calibrated
on this request's own content. The counterfactual cache behaviour mirrors the
actual request's: removed tokens in items the previous request already carried
would have been cache reads to the extent the actual request hit its cache
(``warm``); removed tokens in new items would have been misses, written to
cache in the proportion the actual request wrote (``w``). Old items whose sent
copy Headroom changed since the previous request (``bust``) are misses that
Headroom caused, charged back against the saving.

Output tokens are identical in the counterfactual by construction; output
shaping is measured separately against its holdout and stays out of this
number.

``headroom.pricing`` is imported inside functions, as elsewhere on the proxy
path: its package ``__init__`` pulls in litellm (~4s), which must not land on
proxy startup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from headroom.log_safety import describe_exception, safe_id

if TYPE_CHECKING:
    from headroom.pricing.counterfactual import CacheMix, CacheRates

logger = logging.getLogger(__name__)

#: OpenAI caches prompts of at least 1024 tokens, in 128-token blocks.
OPENAI_MIN_CACHEABLE_TOKENS = 1024
OPENAI_CACHE_BLOCK_TOKENS = 128
#: Used when the catalog has no ``prompt_cache_min_tokens`` for a model.
DEFAULT_MIN_CACHEABLE_TOKENS = 1024


@dataclass(frozen=True, slots=True)
class ItemCounts:
    """One request item (tools block, system, or one input/message item).

    Hashes identify the item's content; counts are in Headroom tokens. Lists of
    items are aligned by position between ORIGINAL and SENT, so an item
    Headroom dropped entirely keeps its slot with ``sent_hash=""`` and
    ``sent_tokens=0``. Tool-schema deferral is removal inside the tools item.
    """

    original_hash: str
    original_tokens: int
    sent_hash: str
    sent_tokens: int


@dataclass(frozen=True)
class ItemMeasure:
    """The Headroom-unit quantities the savings formula needs."""

    #: ``S``: tokens of the whole sent body.
    sent_tokens: int
    #: ``R_old``: removed tokens (signed: an item Headroom grew counts against
    #: it) in items within the longest prefix of the ORIGINAL body identical to
    #: the previous ORIGINAL body.
    removed_old: int
    #: ``R_new``: removed tokens in every other item, signed likewise.
    removed_new: int
    #: ``P_S``: tokens of the longest prefix of the SENT body identical to the
    #: previous SENT body.
    sent_shared_prefix: int
    #: Sent tokens of old items that fell outside the sent shared prefix: cache
    #: misses caused only by Headroom changing what it had sent before.
    bust: int
    #: False when Headroom changed the item count and the original/sent pairing
    #: had to be approximated: the totals are exact, the old/new split is not.
    aligned: bool = True

    @property
    def removed(self) -> int:
        return self.removed_old + self.removed_new


@dataclass(frozen=True)
class SavingsRecord:
    """What one request saved, in dollars and in both token rulers."""

    saved_usd: float
    actual_input_cost_usd: float
    counterfactual_input_cost_usd: float
    #: Removed tokens in Headroom units (``R``).
    removed_tokens: int
    #: The same removal in provider units (``k * R``).
    saved_tokens: float
    k: float
    warm: float
    write_share: float
    #: True when usage was missing (aborted stream) or summed over server-side
    #: iterations, so the cache split could not be read from this request.
    estimated: bool = False
    #: False when the catalog has no price for the model.
    priced: bool = True


def _common_prefix_len(a: Sequence[str], b: Sequence[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def measure_items(
    items: Sequence[ItemCounts],
    previous: Sequence[ItemCounts] | None,
) -> ItemMeasure:
    """Split one request's removal into old/new items against the previous request.

    ``previous`` is the same conversation's previous request, or ``None`` on its
    first request, where everything is new and nothing is shared.
    """
    n_old = _common_prefix_len(
        [i.original_hash for i in items], [i.original_hash for i in previous or ()]
    )
    n_sent_shared = _common_prefix_len(
        [i.sent_hash for i in items], [i.sent_hash for i in previous or ()]
    )
    removed = [i.original_tokens - i.sent_tokens for i in items]
    sent = [max(0, i.sent_tokens) for i in items]
    return ItemMeasure(
        sent_tokens=sum(sent),
        removed_old=sum(removed[:n_old]),
        removed_new=sum(removed[n_old:]),
        sent_shared_prefix=sum(sent[:n_sent_shared]),
        bust=sum(sent[n_sent_shared:n_old]),
    )


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


#: One request item as (identity text, billable text): the identity is hashed
#: to tell items apart, the billable text is what gets tokenized.
Item = tuple[str, str]


def _item(value: Any) -> Item:
    identity = _dump(value)
    if (
        isinstance(value, dict)
        and value.get("type") == "reasoning"
        and "encrypted_content" in value
    ):
        # The provider bills the reasoning it decrypts, not the ciphertext,
        # which would otherwise be most of a long Codex transcript's count.
        billable = _dump({k: v for k, v in value.items() if k != "encrypted_content"})
        return identity, billable
    return identity, identity


def responses_items(body: dict[str, Any]) -> list[Item]:
    """A ``/v1/responses`` body as items in prompt order.

    Tools, then instructions, then each ``input`` item: the order the provider
    caches the prompt in, which is what the shared-prefix comparison needs.
    Codex's ``additional_tools`` input items stay where the client put them.
    """
    items = [_item(body.get("tools") or []), _item(body.get("instructions") or "")]
    # OpenAI-compatible upstreams accept ``messages`` in place of ``input``.
    raw = body.get("input") if body.get("input") is not None else body.get("messages")
    if isinstance(raw, list):
        items.extend(_item(item) for item in raw)
    elif raw:
        items.append(_item(raw))
    return items


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=8).hexdigest()


#: Items kept across all conversations' previous requests. A long Codex
#: session carries 1000+ items; bounding items, not conversations, bounds memory.
MAX_STORED_ITEMS = 100_000


def _align(
    original: list[tuple[str, int]],
    sent: list[tuple[str, int]],
    previous: Sequence[ItemCounts] | None,
) -> tuple[list[ItemCounts], bool]:
    """Pair original and sent items, and say whether the pairing is exact.

    Same item count: by position (compression rewrites items in place). A
    different count (a hook folded or dropped items, or Headroom inserted
    some): identical items at either end pair up; the unmatched originals in
    between are removed, and the unmatched sent items in between are folded
    into the first of their slots. The totals stay exact either way; only
    which items count as old is approximate, hence ``aligned=False``. An item
    compressed in place still pairs at the head when the previous request
    paired the same original with the same sent copy at that position.

    Equal counts pair by position even if a fold and an insert cancel out.
    """
    if len(original) == len(sent):
        return [ItemCounts(oh, ot, sh, st) for (oh, ot), (sh, st) in zip(original, sent)], True
    limit = min(len(original), len(sent))
    prev = previous or ()

    def same_as_before(i: int) -> bool:
        return (
            i < len(prev)
            and original[i][0] == prev[i].original_hash
            and sent[i][0] == prev[i].sent_hash
        )

    head = 0
    while head < limit and (original[head][0] == sent[head][0] or same_as_before(head)):
        head += 1
    tail = 0
    while (
        tail < limit - head
        and original[len(original) - 1 - tail][0] == sent[len(sent) - 1 - tail][0]
    ):
        tail += 1
    mid_original = original[head : len(original) - tail]
    mid_sent = sent[head : len(sent) - tail]
    if len(mid_sent) == 1:
        folded_hash = mid_sent[0][0]
    else:
        folded_hash = _digest("\x00".join(h for h, _ in mid_sent)) if mid_sent else ""
    folded_tokens = sum(t for _, t in mid_sent)
    middle = [ItemCounts(oh, ot, "", 0) for oh, ot in mid_original] or [ItemCounts("", 0, "", 0)]
    middle[0] = ItemCounts(
        middle[0].original_hash, middle[0].original_tokens, folded_hash, folded_tokens
    )
    pairs = [ItemCounts(oh, ot, sh, st) for (oh, ot), (sh, st) in zip(original[:head], sent[:head])]
    pairs += middle
    pairs += [ItemCounts(h, t, h, t) for h, t in original[len(original) - tail :]]
    return pairs, False


#: Measured requests awaiting their outcome. A request whose outcome never
#: arrives (a dropped connection) is forgotten once this many are newer.
MAX_PENDING_REQUESTS = 4096


class ItemLedger:
    """The previous request of each conversation, for the old/new item split.

    One instance per process. A request is measured against the conversation's
    last SUCCESSFUL request and becomes the baseline only once its own outcome
    succeeds (:meth:`settle`): a failed call never reached the provider's cache.
    Requests are sequenced at measure time, so a slow earlier turn settling
    late cannot replace a newer turn's baseline. Items re-sent from the
    baseline reuse its token counts, so a long transcript is tokenized once
    rather than every turn.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._previous: OrderedDict[str, tuple[int, list[ItemCounts]]] = OrderedDict()
        self._pending: OrderedDict[str, tuple[str, int, list[ItemCounts]]] = OrderedDict()
        self._stored = 0
        self._seq = 0

    def measure(
        self,
        key: str,
        original: Sequence[Item],
        sent: Sequence[Item],
        count: Callable[[str], int],
        *,
        request_id: str,
    ) -> ItemMeasure:
        """Measure one request against ``key``'s baseline; settle it later."""
        with self._lock:
            self._seq += 1
            seq = self._seq
            entry = self._previous.get(key)
        previous = entry[1] if entry else None
        known: dict[str, int] = {}
        for item in previous or ():
            known[item.original_hash] = item.original_tokens
            known[item.sent_hash] = item.sent_tokens

        def counted(items: Sequence[Item]) -> list[tuple[str, int]]:
            out = []
            for identity, billable in items:
                digest = _digest(identity)
                tokens = known.get(digest)
                if tokens is None:
                    tokens = known[digest] = int(count(billable))
                out.append((digest, tokens))
            return out

        items, aligned = _align(counted(original), counted(sent), previous)
        with self._lock:
            self._pending[request_id] = (key, seq, items)
            while len(self._pending) > MAX_PENDING_REQUESTS:
                self._pending.popitem(last=False)
        measure = measure_items(items, previous)
        return measure if aligned else replace(measure, aligned=False)

    def settle(self, request_id: str, *, succeeded: bool) -> None:
        """Make a measured request its conversation's baseline if it succeeded."""
        with self._lock:
            pending = self._pending.pop(request_id, None)
            if pending is None or not succeeded:
                return
            key, seq, items = pending
            current = self._previous.get(key)
            if current is not None and current[0] > seq:
                return
            self._stored -= len(current[1]) if current else 0
            self._previous[key] = (seq, items)
            self._previous.move_to_end(key)
            self._stored += len(items)
            while self._stored > MAX_STORED_ITEMS and len(self._previous) > 1:
                _, (_, evicted) = self._previous.popitem(last=False)
                self._stored -= len(evicted)


_ledger = ItemLedger()


def get_item_ledger() -> ItemLedger:
    """Process-wide ledger: one conversation spans many requests and handlers."""
    return _ledger


def prefix_key(conversation: str | None, model: str, original: Sequence[Item]) -> str:
    """Which previous request a request is compared against.

    The conversation's explicit identity (``savings_conversation_key``) plus
    the tools and instructions: a side request in the same session (compaction,
    review) carries a different prefix and must not replace the conversation's
    previous request. Without an identity, the model plus the first input item
    stand in. Requests that collide on the key overwrite each other's previous
    request, shortening the shared prefix: more of the removal is then priced
    as new, at the miss rate, which overstates the saving.
    """
    head = "\x00".join(identity for identity, _ in original[:2])
    if conversation:
        # The model too: the provider's cache is per model, and so are counts.
        return _digest("\x00".join([conversation, model, head]))
    return _digest("\x00".join([model, *(identity for identity, _ in original[:3])]))


def _input_cost(
    rates: CacheRates, *, uncached: float, read: float, write_5m: float, write_1h: float
) -> float:
    return (
        uncached * rates.uncached
        + read * rates.read
        + write_5m * rates.write_5m
        + write_1h * rates.write_1h
    )


def _min_cacheable_tokens(model: str, provider: str | None) -> int:
    if provider == "openai":
        return OPENAI_MIN_CACHEABLE_TOKENS
    from headroom.pricing.counterfactual import catalog_min_cacheable_tokens

    return catalog_min_cacheable_tokens(model) or DEFAULT_MIN_CACHEABLE_TOKENS


def _unpriced(measure: ItemMeasure, *, k: float, estimated: bool) -> SavingsRecord:
    return SavingsRecord(
        saved_usd=0.0,
        actual_input_cost_usd=0.0,
        counterfactual_input_cost_usd=0.0,
        removed_tokens=measure.removed,
        saved_tokens=k * measure.removed,
        k=k,
        warm=0.0,
        write_share=0.0,
        estimated=estimated,
        priced=False,
    )


def compute_savings_record(
    measure: ItemMeasure,
    usage: CacheMix | None,
    *,
    model: str,
    provider: str | None = None,
    estimated: bool = False,
) -> SavingsRecord:
    """Price one request's removal as ``counterfactual_cost - actual_cost``.

    ``usage`` is the provider's billed input as a :class:`CacheMix` (normalized
    here, so an inferred OpenAI write counts as the uncached slice it is).
    ``None`` or an empty mix means the provider reported nothing (an aborted
    stream); the record then prices at list with ``k = 1`` and is marked
    ``estimated``.

    For a request the provider ran as several server-side iterations, pass the
    FIRST iteration's usage when it is available (``estimated=False``); the
    later iterations are a cost the caller books separately. Pass the summed
    usage with ``estimated=True`` only when per-iteration usage is not
    available: the sum describes more prompts than the one sent body, so
    ``k`` is not calibrated from it (``k = 1``) and only its cache ratios are
    used. ``provider`` must be given for OpenAI's 128-token cache blocks to
    apply.

    Never raises: a savings figure must not be able to fail a request.
    """
    try:
        record = _compute(measure, usage, model=model, provider=provider, estimated=estimated)
        # An approximate item pairing leaves the totals exact but not the
        # old/new split the price depends on.
        return record if measure.aligned else replace(record, estimated=True)
    except Exception as exc:  # noqa: BLE001 - pricing must never fail a request
        logger.debug(
            "savings: record failed for model=%s: %s", safe_id(model), describe_exception(exc)
        )
        return _unpriced(measure, k=1.0, estimated=True)


def _compute(
    measure: ItemMeasure,
    usage: CacheMix | None,
    *,
    model: str,
    provider: str | None,
    estimated: bool,
) -> SavingsRecord:
    from headroom.pricing.counterfactual import LONG_CONTEXT_THRESHOLD_TOKENS, resolve_rates

    def rates_for(billed: float, *, for_billing: bool = False) -> CacheRates | None:
        return resolve_rates(
            model,
            long_context=billed > LONG_CONTEXT_THRESHOLD_TOKENS,
            provider=provider,
            for_billing=for_billing,
        )

    removed = measure.removed
    mix = usage.normalized() if usage is not None else None
    if mix is None or mix.billed <= 0:
        actual_rates = rates_for(measure.sent_tokens)
        cf_rates = rates_for(measure.sent_tokens + removed)
        if actual_rates is None or cf_rates is None:
            return _unpriced(measure, k=1.0, estimated=True)
        actual = measure.sent_tokens * actual_rates.uncached
        counterfactual = (measure.sent_tokens + removed) * cf_rates.uncached
        return SavingsRecord(
            saved_usd=counterfactual - actual,
            actual_input_cost_usd=actual,
            counterfactual_input_cost_usd=counterfactual,
            removed_tokens=removed,
            saved_tokens=float(removed),
            k=1.0,
            warm=0.0,
            write_share=0.0,
            estimated=True,
        )

    billed = mix.billed
    writes = mix.write_5m + mix.write_1h
    k = billed / measure.sent_tokens if measure.sent_tokens > 0 and not estimated else 1.0
    hit = mix.read / billed
    sigma = measure.sent_shared_prefix / measure.sent_tokens if measure.sent_tokens > 0 else 0.0
    warm = min(1.0, hit / sigma) if sigma > 0 else 0.0
    misses = mix.uncached + writes
    w = writes / misses if misses > 0 else 0.0

    # The counterfactual prefix must itself have been cacheable.
    if k * (measure.sent_shared_prefix + measure.removed_old) < _min_cacheable_tokens(
        model, provider
    ):
        warm = 0.0

    # Tokens that would have been read instead of missed. Capped at everything
    # the counterfactual missed at all: a bust cannot turn more tokens into
    # reads than the request actually missed plus what Headroom removed, and
    # without the cap the counterfactual buckets go negative.
    # It can be negative when an old item grew; it never removes more reads
    # than the request had.
    extra_read = max(
        -float(mix.read),
        min(k * (measure.removed_old + measure.bust) * warm, misses + k * removed),
    )
    if provider == "openai" and extra_read > 0:
        # Hits come in whole 128-token blocks; the remainder stays a miss.
        total_read = mix.read + extra_read
        blocks = (total_read // OPENAI_CACHE_BLOCK_TOKENS) * OPENAI_CACHE_BLOCK_TOKENS
        extra_read = max(blocks, mix.read) - mix.read
    extra_miss = k * removed - extra_read
    extra_write = extra_miss * w
    w5_share = mix.write_5m / writes if writes > 0 else 1.0

    # The billed buckets price at billing rates on both sides (the
    # counterfactual's at its own context tier); only the tokens Headroom
    # changed price at the savings-estimate rates.
    cf_billed = billed + k * removed
    actual_rates = rates_for(billed, for_billing=True)
    cf_billing_rates = rates_for(cf_billed, for_billing=True)
    cf_rates = rates_for(cf_billed)
    if actual_rates is None or cf_billing_rates is None or cf_rates is None:
        return _unpriced(measure, k=k, estimated=estimated)

    def billed_cost(rates: CacheRates) -> float:
        return _input_cost(
            rates,
            uncached=mix.uncached,
            read=mix.read,
            write_5m=mix.write_5m,
            write_1h=mix.write_1h,
        )

    actual = billed_cost(actual_rates)
    counterfactual = billed_cost(cf_billing_rates) + _input_cost(
        cf_rates,
        uncached=extra_miss - extra_write,
        read=extra_read,
        write_5m=extra_write * w5_share,
        write_1h=extra_write * (1.0 - w5_share),
    )
    return SavingsRecord(
        saved_usd=counterfactual - actual,
        actual_input_cost_usd=actual,
        counterfactual_input_cost_usd=counterfactual,
        removed_tokens=removed,
        saved_tokens=k * removed,
        k=k,
        warm=warm,
        write_share=w,
        estimated=estimated,
    )


__all__ = [
    "Item",
    "ItemCounts",
    "ItemLedger",
    "ItemMeasure",
    "SavingsRecord",
    "compute_savings_record",
    "get_item_ledger",
    "measure_items",
    "prefix_key",
    "responses_items",
]
