from __future__ import annotations

import respx
from httpx import Response
from tenet_core.llm.client import LLMParams, Message

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
    assert models[0].provider_metadata.get("provider") == "cohere"
    assert models[0].provider_metadata.get("raw_model", {}).get("name") == "command-r-plus"


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


@respx.mock
async def test_cohere_generate_accepts_params() -> None:
    route = respx.post("https://api.cohere.com/v2/chat").mock(
        return_value=Response(
            200,
            json={
                "message": {
                    "content": [{"type": "text", "text": "hello params"}],
                },
                "usage": {"billed_units": {"input_tokens": 1, "output_tokens": 1}},
                "finish_reason": "COMPLETE",
            },
        )
    )

    adapter = CohereAdapter(api_key="c-key")
    result = await adapter.generate(
        messages=[Message(role="user", content="hi")],
        model="command-r-plus",
        params=LLMParams(max_tokens=128, temperature=0.3, stop_sequences=["END"]),
    )

    assert route.called
    body = route.calls[0].request.content.decode("utf-8")
    assert '"max_tokens":128' in body
    assert '"temperature":0.3' in body
    assert '"stop_sequences":["END"]' in body
    assert result.content == "hello params"


@respx.mock
async def test_cohere_stream_emits_terminal_chunk() -> None:
    stream_payload = "\n".join([
        '{"type":"content-delta","id":"evt-1","delta":{"message":{"content":{"text":"hello "}}}}',
        '{"type":"content-delta","id":"evt-2","delta":{"message":{"content":{"text":"world"}}}}',
        '{"type":"message-end","id":"evt-3","finish_reason":"COMPLETE","usage":{"billed_units":{"input_tokens":7,"output_tokens":4}}}',
    ])
    route = respx.post("https://api.cohere.com/v2/chat").mock(
        return_value=Response(200, text=stream_payload)
    )

    adapter = CohereAdapter(api_key="c-key")
    chunks = []
    async for chunk in adapter.stream(
        messages=[Message(role="user", content="hi")],
        model="command-r-plus",
    ):
        chunks.append(chunk)

    assert route.called
    assert chunks[0].delta == "hello "
    assert chunks[1].delta == "world"
    assert chunks[-1].stop_reason == "complete"
    assert chunks[-1].input_tokens == 7
    assert chunks[-1].output_tokens == 4
    assert chunks[-1].request_id == "evt-3"
