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

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from headroom.log_safety import describe_exception, safe_id

if TYPE_CHECKING:
    from headroom.pricing.counterfactual import CacheMix, CacheRates

logger = logging.getLogger(__name__)

#: OpenAI caches prompts of at least 1024 tokens, in 128-token blocks.
OPENAI_MIN_CACHEABLE_TOKENS = 1024
OPENAI_CACHE_BLOCK_TOKENS = 128
#: Used when the catalog has no ``prompt_cache_min_tokens`` for a model.
DEFAULT_MIN_CACHEABLE_TOKENS = 1024


@dataclass(frozen=True)
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
    ``estimated``. ``estimated=True`` also marks a request whose usage was
    summed across server-side iterations. ``provider`` must be given for
    OpenAI's 128-token cache blocks to apply.

    Never raises: a savings figure must not be able to fail a request.
    """
    try:
        return _compute(measure, usage, model=model, provider=provider, estimated=estimated)
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

    def rates_for(billed: float) -> CacheRates | None:
        return resolve_rates(
            model, long_context=billed > LONG_CONTEXT_THRESHOLD_TOKENS, provider=provider
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
    k = billed / measure.sent_tokens if measure.sent_tokens > 0 else 1.0
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
    extra_read = min(k * (measure.removed_old + measure.bust) * warm, misses + k * removed)
    if provider == "openai":
        # Hits come in whole 128-token blocks; the remainder stays a miss.
        total_read = mix.read + extra_read
        blocks = (total_read // OPENAI_CACHE_BLOCK_TOKENS) * OPENAI_CACHE_BLOCK_TOKENS
        extra_read = max(blocks, mix.read) - mix.read
    extra_miss = k * removed - extra_read
    extra_write = extra_miss * w
    w5_share = mix.write_5m / writes if writes > 0 else 1.0

    actual_rates = rates_for(billed)
    cf_rates = rates_for(billed + k * removed)
    if actual_rates is None or cf_rates is None:
        return _unpriced(measure, k=k, estimated=estimated)

    actual = _input_cost(
        actual_rates,
        uncached=mix.uncached,
        read=mix.read,
        write_5m=mix.write_5m,
        write_1h=mix.write_1h,
    )
    counterfactual = _input_cost(
        cf_rates,
        uncached=mix.uncached + extra_miss - extra_write,
        read=mix.read + extra_read,
        write_5m=mix.write_5m + extra_write * w5_share,
        write_1h=mix.write_1h + extra_write * (1.0 - w5_share),
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
    "ItemCounts",
    "ItemMeasure",
    "SavingsRecord",
    "compute_savings_record",
    "measure_items",
]
