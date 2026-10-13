"""Both Anthropic translation paths forward the caller's service tier unchanged."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from tests._dotenv import importorskip_no_env_leak

litellm = importorskip_no_env_leak("litellm")

from headroom.backends import litellm as backend_module  # noqa: E402


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("tier", [None, "default", "flex", "priority"])
async def test_anthropic_service_tier_is_forwarded_in_both_paths(
    monkeypatch, streaming, tier
) -> None:
    backend = backend_module.LiteLLMBackend(provider="openrouter")
    body = {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hello"}]}
    if tier is not None:
        body["service_tier"] = tier
    original = deepcopy(body)

    async def chunks():
        for chunk in []:
            yield chunk

    response = (
        chunks()
        if streaming
        else litellm.ModelResponse(
            model="claude-sonnet-4-5",
            choices=[{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
    )
    completion = AsyncMock(return_value=response)
    monkeypatch.setattr(backend_module, "acompletion", completion)
    if streaming:
        events = [event async for event in backend.stream_message(body, {})]
        assert events[-1].event_type == "message_stop"
    else:
        assert (await backend.send_message(body, {})).status_code == 200
    completion.assert_awaited_once()
    kwargs = completion.call_args.kwargs
    if tier is None:
        assert "service_tier" not in kwargs
    else:
        assert kwargs["service_tier"] == tier
    assert body == original
