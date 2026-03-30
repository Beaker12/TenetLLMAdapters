from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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
