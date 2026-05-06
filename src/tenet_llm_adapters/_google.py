"""Google Gemini adapter for the Tenet LLM plugin system."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx
from tenet_core.llm.client import (
    LLMChunk,
    LLMParams,
    LLMResponse,
    Message,
    ToolCall,
    ToolDef,
    resolve_params,
)
from tenet_core.prompt.formatter import format_system_prompt

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from tenet_core.llm import DiscoveredModel

_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GoogleAdapter:
    """Google Gemini adapter using the REST API via httpx."""

    @staticmethod
    def _normalize_base_url(base_url: str | None) -> str:
        if not base_url:
            return _BASE
        candidate = base_url.strip().rstrip("/")
        if not candidate:
            return _BASE
        try:
            parsed = urlparse(candidate)
            path = (parsed.path or "").rstrip("/")
            if path.endswith("/models"):
                path = path[: -len("/models")]
            normalized = parsed._replace(path=path).geturl().rstrip("/")
            return normalized or _BASE
        except Exception:
            return candidate

    def __init__(self, api_key: str, *, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base = self._normalize_base_url(base_url)

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
        params: LLMParams | None = None,
    ) -> LLMResponse:
        """Generate a response via Gemini generateContent REST endpoint."""
        p = resolve_params(
            params,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )
        payload = self._build_payload(
            messages,
            tools,
            p.max_tokens,
            p.temperature if p.temperature is not None else 0.0,
            p.stop_sequences,
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
        params: LLMParams | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """Stream response chunks via Gemini streamGenerateContent endpoint."""
        p = resolve_params(
            params,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )
        payload = self._build_payload(
            messages,
            tools,
            p.max_tokens,
            p.temperature if p.temperature is not None else 0.0,
            p.stop_sequences,
        )
        url = (
            f"{self._base}/models/{model}:streamGenerateContent"
            f"?key={self._api_key}&alt=sse"
        )
        final_reason = "stop"
        final_usage: dict[str, Any] = {}
        request_id = ""
        async with httpx.AsyncClient() as client, client.stream(
            "POST", url, json=payload, timeout=120.0
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                chunk_data = json.loads(line[5:].strip())
                request_id = (
                    chunk_data.get("responseId")
                    or chunk_data.get("id")
                    or request_id
                )
                text = ""
                candidates = chunk_data.get("candidates") or []
                first_candidate = candidates[0] if candidates else {}
                for part in (first_candidate.get("content", {}).get("parts", [])):
                    text += part.get("text", "")
                if text:
                    yield LLMChunk(delta=text, request_id=request_id)
                finish_reason = first_candidate.get("finishReason")
                if isinstance(finish_reason, str) and finish_reason:
                    final_reason = finish_reason.lower()
                usage = chunk_data.get("usageMetadata")
                if isinstance(usage, dict) and usage:
                    final_usage = usage
        yield LLMChunk(
            delta="",
            stop_reason=final_reason,
            input_tokens=int(final_usage.get("promptTokenCount") or 0),
            output_tokens=int(final_usage.get("candidatesTokenCount") or 0),
            request_id=request_id,
        )

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
                system_parts.append({"text": format_system_prompt(msg.content, "markdown")})
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
        tool_call_index = 0

        for part in content.get("parts", []):
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_call_index += 1
                tool_calls.append(
                    ToolCall(
                        id=f"call_{fc['name']}_{tool_call_index}",
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
