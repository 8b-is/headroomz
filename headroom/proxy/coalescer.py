"""Request coalescer — merge rapid compatible prompts into batched requests.

When an agent sends multiple compatible non-streaming requests within a
configurable window (default 5s), buffer them by compatibility key
(model, system, tools, generation parameters, and auth scope), share only
their verified common conversation prefix, and dispatch through the
configured upstream sender.

Benefits:
- Reduces API call overhead (N compatible calls -> 1 call)
- Shared conversation/system prefix is sent once instead of N times
- Preserves intra-request repeated turns, per-request settings, and
  caller cancellation isolation
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DispatchFn = Callable[[dict[str, Any]], Awaitable[Any]]

_INTERNAL_KEYS = frozenset({"_coalesce_scope"})


def _canonical_json(value: Any) -> str:
    """Deterministic serialization for request compatibility & prefix comparison."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return repr(value)


def _compatibility_key(payload: dict[str, Any], scope: str | None = None) -> str:
    """Compute a compatibility key from all non-message request fields + scope.

    Requests only batch together when their model, system prompt, tools,
    generation settings, and auth/endpoint scope match exactly.
    """
    non_message_fields = {
        k: v for k, v in payload.items() if k != "messages" and k not in _INTERNAL_KEYS
    }
    effective_scope = scope if scope is not None else str(payload.get("_coalesce_scope", ""))
    return _canonical_json({"scope": effective_scope, "fields": non_message_fields})


def _strip_internal_keys(payload: dict[str, Any]) -> dict[str, Any]:
    if not any(k in payload for k in _INTERNAL_KEYS):
        return dict(payload)
    return {k: v for k, v in payload.items() if k not in _INTERNAL_KEYS}


@dataclass
class CoalescedRequest:
    """A request waiting to be batched."""

    payload: dict[str, Any]
    scope: str | None = None
    dispatch_fn: DispatchFn | None = None
    future: asyncio.Future[dict[str, Any]] = field(init=False)
    arrived_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.future = asyncio.get_running_loop().create_future()


class RequestCoalescer:
    """Buffer rapid compatible requests and merge them into batches.

    When compatible requests arrive within ``window_sec`` of each other,
    they are grouped by non-message settings + scope and merged into a
    single batch preserving all intra-request turns.
    """

    def __init__(
        self,
        window_sec: float = 5.0,
        max_batch: int = 5,
        dispatch_fn: DispatchFn | None = None,
    ):
        self.window_sec = window_sec
        self.max_batch = max_batch
        self.dispatch_fn = dispatch_fn
        self._pending: list[CoalescedRequest] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None
        self._stats: dict[str, int] = {
            "requests_submitted": 0,
            "requests_coalesced": 0,
            "batches_flushed": 0,
            "api_calls_saved": 0,
            "cancelled_requests": 0,
            "dispatch_errors": 0,
        }

    async def __aenter__(self) -> RequestCoalescer:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    def get_stats(self) -> dict[str, int]:
        """Return a snapshot of coalescer operational counters."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset coalescer operational counters."""
        for key in self._stats:
            self._stats[key] = 0

    async def submit(
        self,
        payload: dict[str, Any],
        *,
        scope: str | None = None,
        dispatch_fn: DispatchFn | None = None,
    ) -> dict[str, Any]:
        """Submit a request. May be batched with other compatible rapid requests."""
        req = CoalescedRequest(payload=payload, scope=scope, dispatch_fn=dispatch_fn)

        async with self._lock:
            self._stats["requests_submitted"] += 1
            self._pending.append(req)
            active_count = sum(1 for r in self._pending if not r.future.done())

            if active_count >= self.max_batch:
                # Batch is full — flush immediately
                await self._flush()
            elif self._flush_task is None or self._flush_task.done():
                # Start the flush timer
                self._flush_task = asyncio.create_task(self._delayed_flush())

        try:
            return await req.future
        except asyncio.CancelledError:
            async with self._lock:
                if req in self._pending:
                    self._pending.remove(req)
                    self._stats["cancelled_requests"] += 1
                if not any(not r.future.done() for r in self._pending):
                    self._pending.clear()
                    if self._flush_task and not self._flush_task.done():
                        self._flush_task.cancel()
                        self._flush_task = None
            raise

    async def _delayed_flush(self) -> None:
        """Wait for window_sec, then flush pending requests."""
        try:
            await asyncio.sleep(self.window_sec)
        except asyncio.CancelledError:
            return
        async with self._lock:
            await self._flush()

    async def _flush(self) -> None:
        """Group pending requests by compatibility, dispatch, and resolve futures."""
        if self._flush_task and self._flush_task is not asyncio.current_task():
            self._flush_task.cancel()
        self._flush_task = None

        if not self._pending:
            return

        raw_pending = self._pending
        self._pending = []

        # Drop requests whose callers cancelled before flush
        active_pending: list[CoalescedRequest] = []
        for req in raw_pending:
            if req.future.done():
                self._stats["cancelled_requests"] += 1
            else:
                active_pending.append(req)

        if not active_pending:
            return

        # Partition by compatibility key so requests with different model,
        # system, tools, generation settings, or auth scope never mix.
        groups: OrderedDict[str, list[CoalescedRequest]] = OrderedDict()
        for req in active_pending:
            key = _compatibility_key(req.payload, req.scope)
            groups.setdefault(key, []).append(req)

        for group in groups.values():
            await self._flush_compatible_group(group)

    async def _flush_compatible_group(self, group: list[CoalescedRequest]) -> None:
        """Dispatch a single compatible group (solo or merged) and resolve waiters."""
        live_group = [r for r in group if not r.future.done()]
        if not live_group:
            return

        self._stats["batches_flushed"] += 1
        is_coalesced = len(live_group) > 1
        if is_coalesced:
            outbound_payload = self._merge_requests([r.payload for r in live_group])
            self._stats["requests_coalesced"] += len(live_group)
            self._stats["api_calls_saved"] += len(live_group) - 1
            logger.info(
                "coalesced %d compatible requests into 1 batch (saved %d API calls)",
                len(live_group),
                len(live_group) - 1,
            )
        else:
            outbound_payload = _strip_internal_keys(live_group[0].payload)

        effective_dispatch = live_group[0].dispatch_fn or self.dispatch_fn
        upstream_response: Any = None
        if effective_dispatch is not None:
            try:
                upstream_response = await effective_dispatch(outbound_payload)
            except Exception as exc:
                self._stats["dispatch_errors"] += 1
                for req in live_group:
                    if not req.future.done():
                        req.future.set_exception(exc)
                return

        batch_size = len(live_group)
        for idx, req in enumerate(live_group):
            if req.future.done():
                self._stats["cancelled_requests"] += 1
                continue
            result_envelope: dict[str, Any] = {
                "coalesced": is_coalesced,
                "batch_size": batch_size,
                "request_index": idx,
                "payload": outbound_payload,
            }
            if effective_dispatch is not None:
                result_envelope["response"] = upstream_response
            req.future.set_result(result_envelope)

    def _merge_requests(self, payloads: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge compatible payloads while preserving every intra-request turn.

        Identifies only the verified longest common message prefix across all
        payloads in the batch, emits that shared prefix once, and appends each
        request's full suffix in order so repeated turns within a single
        conversation (e.g., ``user: retry`` after ``assistant: done``) and
        tool-call/tool-result pairs are never dropped.
        """
        if not payloads:
            return {}

        merged = _strip_internal_keys(payloads[0])
        message_lists: list[list[Any]] = [list(p.get("messages", [])) for p in payloads]

        if not message_lists:
            merged["messages"] = []
            merged["_coalesced_from"] = len(payloads)
            merged["_coalesced_slices"] = []
            return merged

        # Find the longest common message prefix across all requests in the batch
        min_len = min(len(msgs) for msgs in message_lists)
        prefix_len = 0
        for idx in range(min_len):
            first_key = _canonical_json(message_lists[0][idx])
            if all(_canonical_json(msgs[idx]) == first_key for msgs in message_lists[1:]):
                prefix_len = idx + 1
            else:
                break

        all_messages: list[Any] = list(message_lists[0][:prefix_len])
        slices: list[dict[str, int]] = []
        for msgs in message_lists:
            suffix = msgs[prefix_len:]
            start = len(all_messages)
            all_messages.extend(suffix)
            end = len(all_messages)
            slices.append({"prefix_len": prefix_len, "start": start, "end": end})

        merged["messages"] = all_messages
        merged["_coalesced_from"] = len(payloads)
        merged["_coalesced_slices"] = slices

        return merged

    async def close(self) -> None:
        """Flush any remaining requests and shut down."""
        async with self._lock:
            if self._flush_task:
                self._flush_task.cancel()
                self._flush_task = None
            await self._flush()


# ── Singleton & Proxy Integration Helpers ────────────────────────────
_default_coalescer: RequestCoalescer | None = None


def coalescing_enabled(headers: dict[str, str] | Any | None = None) -> bool:
    """Check whether request coalescing is enabled via env var or request header."""
    if headers is not None:
        header_val = ""
        if hasattr(headers, "get"):
            header_val = str(
                headers.get("x-headroom-coalesce") or headers.get("X-Headroom-Coalesce") or ""
            ).strip()
        if header_val.lower() in ("1", "true", "yes", "on"):
            return True
        if header_val.lower() in ("0", "false", "no", "off"):
            return False
    env_val = os.environ.get("HEADROOM_COALESCE_REQUESTS", "").strip().lower()
    return env_val in ("1", "true", "yes", "on")


def get_coalescer(
    window_sec: float = 5.0,
    max_batch: int = 5,
    dispatch_fn: DispatchFn | None = None,
) -> RequestCoalescer:
    """Get or create the default coalescer."""
    global _default_coalescer
    if _default_coalescer is None:
        _default_coalescer = RequestCoalescer(
            window_sec=window_sec,
            max_batch=max_batch,
            dispatch_fn=dispatch_fn,
        )
    elif dispatch_fn is not None and _default_coalescer.dispatch_fn is None:
        _default_coalescer.dispatch_fn = dispatch_fn
    return _default_coalescer


def get_coalescer_stats() -> dict[str, int]:
    """Return stats for the default coalescer without instantiating if unused."""
    if _default_coalescer is None:
        return {
            "requests_submitted": 0,
            "requests_coalesced": 0,
            "batches_flushed": 0,
            "api_calls_saved": 0,
            "cancelled_requests": 0,
            "dispatch_errors": 0,
        }
    return _default_coalescer.get_stats()


def reset_coalescer() -> None:
    """Reset the default coalescer singleton and its stats."""
    global _default_coalescer
    if _default_coalescer is not None:
        _default_coalescer.reset_stats()
    _default_coalescer = None
