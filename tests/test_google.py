from __future__ import annotations

import respx
from httpx import Response
from tenetcore.llm.client import Message

from tenet_llm_adapters import GoogleAdapter


@respx.mock
async def test_google_list_models_maps_input_token_limit() -> None:
    route = respx.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": "g-key"},
    ).mock(
        return_value=Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/gemini-1.5-pro",
                        "displayName": "Gemini 1.5 Pro",
                        "inputTokenLimit": 2097152,
                        "outputTokenLimit": 8192,
                        "supportedGenerationMethods": [
                            "generateContent",
                            "streamGenerateContent",
                        ],
                    }
                ]
            },
        )
    )

    adapter = GoogleAdapter(api_key="g-key")
    models = await adapter.list_models()

    assert route.called
    assert len(models) == 1
    assert models[0].model_id == "gemini-1.5-pro"
    assert models[0].provider == "google"
    assert models[0].context_window == 2097152
    assert models[0].supports_streaming is True
    assert models[0].provider_metadata.get("provider") == "google"
    assert models[0].provider_metadata.get("raw_model", {}).get("name") == "models/gemini-1.5-pro"


@respx.mock
async def test_google_generate_parses_text_and_usage() -> None:
    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
        params={"key": "g-key"},
    ).mock(
        return_value=Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "hello from gemini"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 7,
                },
            },
        )
    )

    adapter = GoogleAdapter(api_key="g-key")
    result = await adapter.generate(
        messages=[Message(role="user", content="hi")],
        model="gemini-1.5-pro",
    )

    assert route.called
    assert result.content == "hello from gemini"
    assert result.input_tokens == 11
    assert result.output_tokens == 7
    assert result.stop_reason == "stop"
