# Tenet Platform
# Copyright (C) 2026 Stuart W. Parkhurst
#
# This file is part of the Tenet Platform.
# Licensed under the GNU Affero General Public License v3.0
# See LICENSE file or https://www.gnu.org/licenses/agpl-3.0.html

from __future__ import annotations

import os
from unittest.mock import patch

import respx
from httpx import Response
from tenet_core.llm.client import LLMParams, Message

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


@respx.mock
async def test_google_generate_accepts_params() -> None:
    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
        params={"key": "g-key"},
    ).mock(
        return_value=Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "hello from params"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 4,
                    "candidatesTokenCount": 3,
                },
            },
        )
    )

    adapter = GoogleAdapter(api_key="g-key")
    result = await adapter.generate(
        messages=[Message(role="user", content="hi")],
        model="gemini-1.5-pro",
        params=LLMParams(temperature=0.2, stop_sequences=["END"]),
    )

    assert route.called
    body = route.calls[0].request.content.decode("utf-8")
    assert '"temperature":0.2' in body
    assert '"stopSequences":["END"]' in body
    assert result.content == "hello from params"


@respx.mock
async def test_google_stream_emits_terminal_chunk() -> None:
    stream_payload = "\n".join([
        'data: {"responseId":"resp-1","candidates":[{"content":{"parts":[{"text":"hello "}]}}]}',
        'data: {"responseId":"resp-1","candidates":[{"content":{"parts":[{"text":"world"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":9}}',
    ])
    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:streamGenerateContent",
        params={"key": "g-key", "alt": "sse"},
    ).mock(return_value=Response(200, text=stream_payload))

    adapter = GoogleAdapter(api_key="g-key")
    chunks = []
    async for chunk in adapter.stream(
        messages=[Message(role="user", content="hi")],
        model="gemini-1.5-pro",
    ):
        chunks.append(chunk)

    assert route.called
    assert chunks[0].delta == "hello "
    assert chunks[1].delta == "world"
    assert chunks[-1].stop_reason == "stop"
    assert chunks[-1].input_tokens == 12
    assert chunks[-1].output_tokens == 9
    assert chunks[-1].request_id == "resp-1"


def test_gateway_url_sets_base_for_google() -> None:
    """When TENET_LLM_GATEWAY_URL is set, GoogleAdapter._base must use the gateway URL."""
    gateway = "http://test-gateway:8430"
    with patch.dict(os.environ, {"TENET_LLM_GATEWAY_URL": gateway}, clear=False):
        adapter = GoogleAdapter(api_key="g-key")

    assert adapter._base == gateway


def test_gateway_url_overrides_explicit_base_url_for_google() -> None:
    """Gateway URL takes precedence over an explicit base_url kwarg."""
    gateway = "http://test-gateway:8430"
    with patch.dict(os.environ, {"TENET_LLM_GATEWAY_URL": gateway}, clear=False):
        adapter = GoogleAdapter(api_key="g-key", base_url="https://custom.example.com/v1beta")

    assert adapter._base == gateway
