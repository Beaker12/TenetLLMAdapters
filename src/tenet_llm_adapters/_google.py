"""Google Gemini adapter for the Tenet LLM plugin system."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
from tenet_core.llm.client import LLMChunk, LLMResponse, Message, ToolCall, ToolDef

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from tenet_core.llm import DiscoveredModel

_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GoogleAdapter:
    """Google Gemini adapter using the REST API via httpx."""

    def __init__(self, api_key: str, *, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base = (base_url or _BASE).rstrip("/")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GoogleAdapter:
        return cls(
            api_key=config.get("api_key", ""),
            base_url=config.get("base_url"),
        )

    async def list_models(self) -> list[DiscoveredModel]:
        """List available models from the Google Generative Language API."""
        from tenet_core.llm import DiscoveredModel

        url = f"{self._base}/models?key={self._api_key}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10.0)
            resp.raise_for_status()

        raw = resp.json().get("models", [])
        result: list[DiscoveredModel] = []
        for m in raw:
            name = m.get("name", "")
            if not name:
                continue
            model_id = name.split("/")[-1]
            methods = m.get("supportedGenerationMethods", [])
            result.append(
                DiscoveredModel(
                    model_id=model_id,
                    provider="google",
                    display_name=m.get("displayName"),
                    context_window=m.get("inputTokenLimit"),
                    max_output_tokens=m.get("outputTokenLimit"),
                    supports_tools="generateContent" in methods,
                    supports_streaming="streamGenerateContent" in methods,
                    supports_vision=None,
                    supports_reasoning=None,
                    supports_batch=False,
                    provider_metadata={
                        "provider": "google",
                        "raw_model": m,
                        "supported_generation_methods": methods,
                    },
                )
            )
        return result

    async def generate(
        self,
        messages: list[Message],
        model: str,
        *,
        tools: list[ToolDef] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate a response via Gemini generateContent REST endpoint."""
        payload = self._build_payload(
            messages, tools, max_tokens, temperature, stop_sequences
        )
        url = f"{self._base}/models/{model}:generateContent?key={self._api_key}"
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=120.0)
            resp.raise_for_status()
        return self._parse_response(resp.json(), model)

    async def stream(
        self,
        messages: list[Message],
        model: str,
        *,
        tools: list[ToolDef] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """Stream response chunks via Gemini streamGenerateContent endpoint."""
        payload = self._build_payload(
            messages, tools, max_tokens, temperature, stop_sequences
        )
        url = (
            f"{self._base}/models/{model}:streamGenerateContent"
            f"?key={self._api_key}&alt=sse"
        )
        async with httpx.AsyncClient() as client, client.stream(
            "POST", url, json=payload, timeout=120.0
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                chunk_data = json.loads(line[5:].strip())
                text = ""
                for part in (
                    chunk_data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [])
                ):
                    text += part.get("text", "")
                if text:
                    yield LLMChunk(delta=text)

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None,
        max_tokens: int,
        temperature: float,
        stop_sequences: list[str] | None,
    ) -> dict[str, Any]:
        """Convert TenetCore messages to Gemini API payload."""
        system_parts: list[dict[str, str]] = []
        contents: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append({"text": msg.content})
            elif msg.role == "user":
                contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == "assistant":
                parts: list[dict[str, Any]] = []
                if msg.content:
                    parts.append({"text": msg.content})
                for tc in msg.tool_calls:
                    parts.append(
                        {"functionCall": {"name": tc.name, "args": tc.arguments}}
                    )
                contents.append({"role": "model", "parts": parts})
            elif msg.role == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": msg.name or "tool",
                                    "response": {"output": msg.content},
                                }
                            }
                        ],
                    }
                )

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}
        if stop_sequences:
            payload["generationConfig"]["stopSequences"] = stop_sequences
        if tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                        for t in tools
                    ]
                }
            ]
        return payload

    def _parse_response(self, data: dict[str, Any], model: str) -> LLMResponse:
        candidates = data.get("candidates", [{}])
        candidate = candidates[0] if candidates else {}
        content = candidate.get("content", {})
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for part in content.get("parts", []):
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    ToolCall(
                        id=f"call_{fc['name']}",
                        name=fc["name"],
                        arguments=fc.get("args", {}),
                    )
                )

        usage = data.get("usageMetadata", {})
        return LLMResponse(
            content="".join(text_parts) or "",
            model=model,
            tool_calls=tool_calls,
            input_tokens=usage.get("promptTokenCount") or 0,
            output_tokens=usage.get("candidatesTokenCount") or 0,
            stop_reason=(candidate.get("finishReason") or "STOP").lower(),
        )
