"""Redis connections for state that several Headroom replicas share.

The CCR store (``HEADROOM_CCR_BACKEND=redis``) and shared ``/v1/compress``
sessions keep their keys under ``HEADROOM_REDIS_KEY_PREFIX`` (default
``headroom``), so two deployments can share one Redis without sharing state.

Redis holds tool output and transcripts verbatim: it must be private to the
deployment, with authentication and TLS (``rediss://``). Plain ``redis://`` to a
host other than this machine is accepted with a warning.

Shared state is an optimisation of a request, never a condition for it: a Redis
error is a miss, logged when it starts and when Redis is back, and Redis is not
asked again for :data:`BACKOFF_SECONDS` after a connection failure, so an outage
costs requests nothing but the sharing.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar
from urllib.parse import urlsplit

from headroom.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

PREFIX_ENV = "HEADROOM_REDIS_KEY_PREFIX"
#: Seconds Redis is skipped after a connection error.
BACKOFF_SECONDS = 5.0
#: Seconds a command may take. Commands carry small values (an entry, a few
#: messages), so a slow one is an outage, not a large transfer.
TIMEOUT_SECONDS = 1.0

T = TypeVar("T")


def key_prefix() -> str:
    return os.environ.get(PREFIX_ENV, "").strip() or "headroom"


def connect(url: str, *, setting: str) -> Any:
    """A client for ``url``. Raises :class:`ConfigurationError`, naming
    ``setting``, when the URL is not Redis or the client is not installed.
    Does not connect: an unreachable Redis does not stop startup."""
    scheme = urlsplit(url).scheme
    if scheme not in ("redis", "rediss", "unix"):
        raise ConfigurationError(f"{setting} must be a redis://, rediss:// or unix:// URL")
    try:
        import redis
    except ImportError as exc:
        raise ConfigurationError(
            f"{setting} names Redis but the client is not installed: "
            "pip install 'headroom-ai[redis]'"
        ) from exc
    from headroom.proxy.loopback_guard import is_loopback_host

    host = urlsplit(url).hostname
    if scheme == "redis" and not is_loopback_host(host):
        logger.warning(
            "%s: plain redis:// to %s; Redis will hold tool output and transcripts, "
            "so use rediss:// with authentication on a private network",
            setting,
            host,
        )
    return redis.Redis.from_url(
        url, socket_timeout=TIMEOUT_SECONDS, socket_connect_timeout=TIMEOUT_SECONDS
    )


class Guard:
    """Runs Redis calls for one feature: an error is a miss, logged when it
    starts and when Redis is back; after a connection error Redis is skipped
    for :data:`BACKOFF_SECONDS`."""

    def __init__(self, feature: str) -> None:
        import redis

        self._feature = feature
        self._errors: tuple[type[BaseException], ...] = (redis.RedisError,)
        self._connection_errors = (redis.ConnectionError, redis.TimeoutError)
        self._lock = threading.Lock()
        self._skip_until = 0.0
        self._failing = False

    def call(self, what: str, fn: Callable[[], T], default: T) -> T:
        if time.monotonic() < self._skip_until:
            return default
        try:
            result = fn()
        except self._errors as exc:
            with self._lock:
                if isinstance(exc, self._connection_errors):
                    self._skip_until = time.monotonic() + BACKOFF_SECONDS
                if not self._failing:
                    self._failing = True
                    logger.warning(
                        "%s: Redis %s failed (%s: %s); carrying on without shared state",
                        self._feature,
                        what,
                        type(exc).__name__,
                        exc,
                    )
            return default
        if self._failing:
            with self._lock:
                if self._failing:
                    self._failing = False
                    logger.info("%s: Redis is reachable again", self._feature)
        return result
