from __future__ import annotations

import respx
from httpx import Response
from tenetcore.llm.client import Message

from tenet_llm_adapters import CohereAdapter


@respx.mock
async def test_cohere_list_models_uses_name_field_for_model_id() -> None:
    route = respx.get("https://api.cohere.com/v2/models").mock(
        return_value=Response(
            200,
            json={
                "models": [
                    {
                        "name": "command-r-plus",
                        "id": "ignored-id",
                        "context_length": 128000,
                        "endpoints": ["chat"],
                    }
                ]
            },
        )
    )

    adapter = CohereAdapter(api_key="c-key")
    models = await adapter.list_models()

    assert route.called
    assert len(models) == 1
    assert models[0].model_id == "command-r-plus"
    assert models[0].provider == "cohere"
    assert models[0].context_window == 128000


@respx.mock
async def test_cohere_generate_parses_text_and_usage() -> None:
    route = respx.post("https://api.cohere.com/v2/chat").mock(
        return_value=Response(
            200,
            json={
                "message": {
                    "content": [
                        {"type": "text", "text": "hello from cohere"},
                    ]
                },
                "usage": {
                    "billed_units": {
                        "input_tokens": 9,
                        "output_tokens": 5,
                    }
                },
                "finish_reason": "COMPLETE",
            },
        )
    )

    adapter = CohereAdapter(api_key="c-key")
    result = await adapter.generate(
        messages=[Message(role="user", content="hi")],
        model="command-r-plus",
    )

    assert route.called
    assert result.content == "hello from cohere"
    assert result.input_tokens == 9
    assert result.output_tokens == 5
    assert result.stop_reason == "complete"
