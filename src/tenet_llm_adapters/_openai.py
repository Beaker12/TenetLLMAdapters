"""OpenAI-compatible adapter implementation for the Tenet LLM plugin system."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any
_OPENAI_BATCH_MODELS: frozenset[str] = frozenset({
    "gpt-4o", "gpt-4o-2024-11-20", "gpt-4o-2024-08-06",
    "gpt-4o-mini", "gpt-4o-mini-2024-07-18",
    "gpt-3.5-turbo", "gpt-3.5-turbo-0125",
    "gpt-4-turbo", "gpt-4-turbo-preview",
    "o3-mini",
})

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from tenetcore.llm import DiscoveredModel

import openai
from tenetcore.llm.client import LLMChunk, LLMResponse, Message, ToolCall, ToolDef

logger = logging.getLogger(__name__)


class OpenAIAdapter:
    """OpenAI-compatible LLM adapter."""

    def __init__(self, api_key: str, *, base_url: str | None = None) -> None:
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**kwargs)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> OpenAIAdapter:
        """Instantiate from a ``BackendConfig.model_dump()`` dict."""
        return cls(
            api_key=config.get("api_key", ""),
            base_url=config.get("base_url"),
        )

    @staticmethod
    def _to_plain_dict(value: Any) -> dict[str, Any]:
        """Best-effort conversion of SDK objects to plain dicts."""
        if isinstance(value, dict):
            return dict(value)
        for attr in ("model_dump", "to_dict"):
            fn = getattr(value, attr, None)
            if callable(fn):
                try:
                    data = fn()
                    if isinstance(data, dict):
                        return dict(data)
                except Exception:  # noqa: BLE001
                    pass
        if hasattr(value, "__dict__"):
            data = getattr(value, "__dict__", None)
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if not k.startswith("_")}
        return {}

    async def list_models(self) -> list[DiscoveredModel]:
        """List available models from the OpenAI-compatible /v1/models endpoint."""
        from tenetcore.llm import DiscoveredModel

        result: list[DiscoveredModel] = []
        models = await self._client.models.list()
        for m in models.data:
            if not getattr(m, "id", None):
                continue
            raw_model = self._to_plain_dict(m)
            result.append(
                DiscoveredModel(
                    model_id=m.id,
                    provider="openai-compatible",
                    supports_batch=m.id in _OPENAI_BATCH_MODELS,
                    provider_metadata={
                        "provider": "openai-compatible",
                        "raw_model": raw_model,
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
        """Generate response via OpenAI-compatible Chat Completions API."""
        api_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "tool":
                api_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id or "",
                })
            elif msg.role == "assistant" and msg.tool_calls:
                m: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
                api_messages.append(m)
            else:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = [
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
        if stop_sequences:
            kwargs["stop"] = stop_sequences

        logger.debug("OpenAI API call: model=%s", model)
        response = await self._client.chat.completions.create(**kwargs)
        return self._parse_response(response, model)

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
        """Stream response chunks via the OpenAI Chat Completions streaming API."""
        return self._stream_impl(
            messages,
            model,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )

    async def _stream_impl(
        self,
        messages: list[Message],
        model: str,
        *,
        tools: list[ToolDef] | None = None,
            max_tokens: int = 16384,
        temperature: float = 0.0,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        api_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "tool":
                api_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id or "",
                })
            elif msg.role == "assistant" and msg.tool_calls:
                m: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
                api_messages.append(m)
            else:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = [
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
        if stop_sequences:
            kwargs["stop"] = stop_sequences

        logger.debug("OpenAI streaming call: model=%s", model)
        last_chunk: Any = None
        # Track reasoning_content separately for thinking models (e.g., Qwen3).
        # Only fall back to it if the model emitted zero content tokens (e.g. when
        # max_tokens is exhausted mid-thinking-chain).
        reasoning_text = ""
        got_content = False
        async for chunk in await self._client.chat.completions.create(**kwargs):
            last_chunk = chunk
            if not chunk.choices:
                continue
            delta_obj = chunk.choices[0].delta
            delta_text = delta_obj.content or ""
            reasoning_delta = getattr(delta_obj, "reasoning_content", None) or ""
            if delta_text:
                got_content = True
                yield LLMChunk(delta=delta_text)
            elif reasoning_delta:
                reasoning_text += reasoning_delta
        if not got_content and reasoning_text:
            # Fallback: model exhausted tokens during reasoning with no content yield.
            yield LLMChunk(delta=reasoning_text)
        usage = getattr(last_chunk, "usage", None) if last_chunk is not None else None
        finish_reason = (
            last_chunk.choices[0].finish_reason
            if last_chunk is not None and last_chunk.choices
            else "stop"
        )
        yield LLMChunk(
            delta="",
            stop_reason=finish_reason or "stop",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            request_id=getattr(last_chunk, "id", "") or "" if last_chunk else "",
        )

    def _parse_response(self, response: Any, model: str) -> LLMResponse:
        choices = getattr(response, "choices", None)
        if not choices:
            payload = self._to_plain_dict(response)
            summary = json.dumps(payload, default=str)[:500] if payload else repr(response)
            raise RuntimeError(
                "OpenAI-compatible response did not include any choices "
                f"for model '{model}': {summary}"
            )

        choice = choices[0]
        msg = choice.message
        tool_calls: list[ToolCall] = []

        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args if isinstance(args, dict) else {},
                    )
                )

        usage = response.usage
        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            model=model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            stop_reason=choice.finish_reason or "stop",
            request_id=getattr(response, "id", "") or "",
        )
