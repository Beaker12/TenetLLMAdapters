"""Cohere adapter for the Tenet LLM plugin system."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
from tenetcore.llm.client import LLMChunk, LLMResponse, Message, ToolCall, ToolDef

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from tenetcore.llm import DiscoveredModel

_BASE = "https://api.cohere.com"


class CohereAdapter:
    """Cohere LLM adapter using Cohere v2 API via httpx."""

    def __init__(self, api_key: str, *, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base = (base_url or _BASE).rstrip("/")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CohereAdapter:
        return cls(
            api_key=config.get("api_key", ""),
            base_url=config.get("base_url"),
        )

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def list_models(self) -> list[DiscoveredModel]:
        """List available models from the Cohere v2 models endpoint."""
        from tenetcore.llm import DiscoveredModel

        url = f"{self._base}/v2/models"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=self._auth_headers(), timeout=10.0)
            resp.raise_for_status()

        raw = resp.json().get("models", [])
        result: list[DiscoveredModel] = []
        for m in raw:
            name = m.get("name")
            if not name:
                continue
            endpoints = m.get("endpoints", [])
            result.append(
                DiscoveredModel(
                    model_id=name,
                    provider="cohere",
                    context_window=m.get("context_length"),
                    supports_tools="chat" in endpoints,
                    supports_streaming="chat" in endpoints,
                    supports_batch=False,
                    provider_metadata={
                        "provider": "cohere",
                        "raw_model": m,
                        "endpoints": endpoints,
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
        """Generate a response via Cohere v2 /chat endpoint."""
        payload = self._build_payload(
            messages,
            model,
            tools,
            max_tokens,
            temperature,
            stop_sequences,
            stream=False,
        )
        url = f"{self._base}/v2/chat"
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                json=payload,
                headers=self._auth_headers(),
                timeout=120.0,
            )
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
        """Stream response chunks via Cohere v2 /chat with stream=True."""
        payload = self._build_payload(
            messages,
            model,
            tools,
            max_tokens,
            temperature,
            stop_sequences,
            stream=True,
        )
        url = f"{self._base}/v2/chat"
        async with httpx.AsyncClient() as client, client.stream(
            "POST",
            url,
            json=payload,
            headers=self._auth_headers(),
            timeout=120.0,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "content-delta":
                    text = (
                        event.get("delta", {})
                        .get("message", {})
                        .get("content", {})
                        .get("text", "")
                    )
                    if text:
                        yield LLMChunk(delta=text)

    def _build_payload(
        self,
        messages: list[Message],
        model: str,
        tools: list[ToolDef] | None,
        max_tokens: int,
        temperature: float,
        stop_sequences: list[str] | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        cohere_msgs: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                cohere_msgs.append({"role": "system", "content": msg.content})
            elif msg.role == "user":
                cohere_msgs.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                parts: list[dict[str, Any]] = []
                if msg.content:
                    parts.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    parts.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                cohere_msgs.append({"role": "assistant", "content": parts})
            elif msg.role == "tool":
                cohere_msgs.append(
                    {
                        "role": "tool",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id or "",
                                "content": [
                                    {"type": "text", "text": msg.content or ""}
                                ],
                            }
                        ],
                    }
                )

        payload: dict[str, Any] = {
            "model": model,
            "messages": cohere_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if stop_sequences:
            payload["stop_sequences"] = stop_sequences
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        return payload

    def _parse_response(self, data: dict[str, Any], model: str) -> LLMResponse:
        message = data.get("message", {})
        content_parts = message.get("content", [])
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for part in content_parts:
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=part.get("id", ""),
                        name=part.get("name", ""),
                        arguments=part.get("input", {}),
                    )
                )

        usage = data.get("usage", {}).get("billed_units", {})
        return LLMResponse(
            content="".join(text_parts) or "",
            model=model,
            tool_calls=tool_calls,
            input_tokens=usage.get("input_tokens") or 0,
            output_tokens=usage.get("output_tokens") or 0,
            stop_reason=(data.get("finish_reason") or "complete").lower(),
        )
