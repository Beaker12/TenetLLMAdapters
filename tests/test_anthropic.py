from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from tenet_core.llm.client import LLMParams

from tenet_llm_adapters import AnthropicAdapter


async def test_list_models_maps_beta_fields() -> None:
    m1 = SimpleNamespace(
        id="claude-3-7-sonnet-20250219",
        display_name="Claude 3.7 Sonnet",
        max_input_tokens=200000,
        max_tokens=8192,
        capabilities=SimpleNamespace(image_input=True, thinking=True),
    )
    m2 = SimpleNamespace(
        id="claude-3-5-haiku-20241022",
        display_name="Claude 3.5 Haiku",
        max_input_tokens=200000,
        max_tokens=4096,
        capabilities=SimpleNamespace(image_input=False, thinking=False),
    )

    async def _pager():
        for item in [m1, m2]:
            yield item

    mock_client = MagicMock()
    mock_client.models = MagicMock()
    mock_client.models.list = AsyncMock(return_value=_pager())

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        adapter = AnthropicAdapter(api_key="test-key")

    models = await adapter.list_models()

    assert [m.model_id for m in models] == [
        "claude-3-7-sonnet-20250219",
        "claude-3-5-haiku-20241022",
    ]
    assert models[0].context_window == 200000
    assert models[0].max_output_tokens == 8192
    assert models[0].supports_reasoning is True
    assert models[1].supports_reasoning is False
    assert models[0].supports_vision is True
    assert models[0].provider_metadata.get("provider") == "anthropic"
    assert "raw_model" in models[0].provider_metadata


async def test_generate_falls_back_to_streaming_for_long_requests() -> None:
    usage = SimpleNamespace(
        input_tokens=321,
        output_tokens=123,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    final_message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="streamed final response")],
        usage=usage,
        stop_reason="end_turn",
        id="req-stream-1",
    )

    stream_cm = AsyncMock()
    stream_cm.__aenter__.return_value = stream_cm
    stream_cm.__aexit__.return_value = False
    stream_cm.get_final_message = AsyncMock(return_value=final_message)

    mock_client = MagicMock()
    mock_client.messages = MagicMock()
    mock_client.messages.create = AsyncMock(
        side_effect=ValueError(
            "Streaming is required for operations that may take longer than 10 minutes."
        )
    )
    mock_client.messages.stream = MagicMock(return_value=stream_cm)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        adapter = AnthropicAdapter(api_key="test-key")

    response = await adapter.generate(
        [SimpleNamespace(role="user", content="hello")],
        "claude-sonnet-4-6",
        max_tokens=64000,
    )

    assert response.content == "streamed final response"
    assert response.input_tokens == 321
    assert response.output_tokens == 123
    mock_client.messages.create.assert_awaited_once()
    mock_client.messages.stream.assert_called_once()
    stream_cm.get_final_message.assert_awaited_once()


async def test_generate_accepts_params() -> None:
    usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=4,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    response_obj = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        usage=usage,
        stop_reason="end_turn",
        id="req-params-1",
    )

    mock_client = MagicMock()
    mock_client.messages = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=response_obj)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        adapter = AnthropicAdapter(api_key="test-key")

    result = await adapter.generate(
        [SimpleNamespace(role="user", content="hello")],
        "claude-sonnet-4-6",
        params=LLMParams(max_tokens=256, temperature=0.2, stop_sequences=["END"]),
    )

    assert result.content == "ok"
    kwargs = mock_client.messages.create.await_args.kwargs
    assert kwargs["max_tokens"] == 256
    assert kwargs["temperature"] == 0.2
    assert kwargs["stop_sequences"] == ["END"]


async def test_stream_accepts_event_type_content_block_delta_variant() -> None:
    usage = SimpleNamespace(
        input_tokens=12,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    final_message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello from final")],
        usage=usage,
        stop_reason="end_turn",
        id="req-stream-variant-1",
    )

    class RawDeltaEvent:
        def __init__(self, *, delta: SimpleNamespace) -> None:
            self.type = "content_block_delta"
            self.delta = delta

    stream_cm = AsyncMock()
    stream_cm.__aenter__.return_value = stream_cm
    stream_cm.__aexit__.return_value = False
    stream_cm.get_final_message = AsyncMock(return_value=final_message)

    stream_cm.__aiter__.return_value = [
        RawDeltaEvent(delta=SimpleNamespace(type="text_delta", text="hello ")),
        RawDeltaEvent(delta=SimpleNamespace(type="text_delta", text="world")),
    ]

    mock_client = MagicMock()
    mock_client.messages = MagicMock()
    mock_client.messages.stream = MagicMock(return_value=stream_cm)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        adapter = AnthropicAdapter(api_key="test-key")

    chunks = []
    async for chunk in adapter.stream(
        [SimpleNamespace(role="user", content="hello")],
        "claude-sonnet-4-6",
    ):
        chunks.append(chunk)

    assert chunks[0].delta == "hello "
    assert chunks[1].delta == "world"
    assert chunks[-1].stop_reason == "end_turn"

