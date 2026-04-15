from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenet_llm_adapters import OpenAIAdapter


async def test_list_models_returns_thin_discovered_models() -> None:
    data = [
        SimpleNamespace(id="gpt-4o"),
        SimpleNamespace(id="gpt-4.1-mini"),
        SimpleNamespace(id=None),
    ]
    models_result = SimpleNamespace(data=data)

    mock_client = MagicMock()
    mock_client.models = MagicMock()
    mock_client.models.list = AsyncMock(return_value=models_result)

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        adapter = OpenAIAdapter(api_key="test-key")

    models = await adapter.list_models()

    assert [m.model_id for m in models] == ["gpt-4o", "gpt-4.1-mini"]
    assert all(m.provider == "openai-compatible" for m in models)
    assert all(m.context_window is None for m in models)
    assert models[0].provider_metadata.get("provider") == "openai-compatible"
    assert isinstance(models[0].provider_metadata.get("raw_model"), dict)


async def test_generate_raises_runtime_error_when_choices_missing() -> None:
    malformed_response = SimpleNamespace(choices=None, usage=None, id="req-123")

    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=malformed_response)

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        adapter = OpenAIAdapter(api_key="test-key")

    with pytest.raises(RuntimeError, match="did not include any choices"):
        await adapter.generate(
            [SimpleNamespace(role="user", content="hello")],
            "qwen/test-model",
        )


async def test_stream_terminal_chunk_includes_reasoning_tokens() -> None:
    async def _fake_stream():
        yield SimpleNamespace(
            id="chunk-1",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="hello", reasoning_content=""),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        yield SimpleNamespace(
            id="chunk-2",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="", reasoning_content=""),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=5267,
                completion_tokens=99,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=98),
            ),
        )

    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_fake_stream())

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        adapter = OpenAIAdapter(api_key="test-key")

    chunks = []
    async for chunk in adapter.stream(
        [SimpleNamespace(role="user", content="hello")],
        "qwen/test-model",
    ):
        chunks.append(chunk)

    assert chunks[-1].input_tokens == 5267
    assert chunks[-1].output_tokens == 99
    assert chunks[-1].thinking_tokens == 98
