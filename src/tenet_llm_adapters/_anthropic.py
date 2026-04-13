"""Anthropic adapter implementation for the Tenet LLM plugin system."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import anthropic
from tenet_core.llm.client import LLMChunk, LLMResponse, Message, ToolCall, ToolDef

_ANTHROPIC_BATCH_MODELS: frozenset[str] = frozenset({
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
})

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from tenet_core.llm import DiscoveredModel

logger = logging.getLogger(__name__)


class AnthropicAdapter:
    """Anthropic (Claude) LLM adapter."""

    def __init__(self, api_key: str, *, base_url: str | None = None) -> None:
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AnthropicAdapter:
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

    @staticmethod
    def _capability_supported(capabilities: dict[str, Any], key: str) -> bool | None:
        """Return capability support state from Anthropic capability objects."""
        raw = capabilities.get(key)
        if isinstance(raw, dict):
            supported = raw.get("supported")
            return bool(supported) if isinstance(supported, bool) else None
        if isinstance(raw, bool):
            return raw
        return None

    async def list_models(self) -> list[DiscoveredModel]:
        """List available models from Anthropic beta models endpoint."""
        from tenet_core.llm import DiscoveredModel

        result: list[DiscoveredModel] = []
        async for m in await self._client.models.list():
            raw_model = self._to_plain_dict(m)
            raw_caps = raw_model.get("capabilities") if isinstance(raw_model, dict) else None
            capabilities = raw_caps if isinstance(raw_caps, dict) else {}

            supports_batch = self._capability_supported(capabilities, "batch")
            supports_vision = self._capability_supported(capabilities, "image_input")
            supports_reasoning = self._capability_supported(capabilities, "thinking")

            if supports_batch is None:
                supports_batch = m.id in _ANTHROPIC_BATCH_MODELS

            if supports_vision is None:
                caps_obj = getattr(m, "capabilities", None)
                val = getattr(caps_obj, "image_input", None) if caps_obj else None
                supports_vision = val if isinstance(val, bool) else None

            if supports_reasoning is None:
                caps_obj = getattr(m, "capabilities", None)
                val = getattr(caps_obj, "thinking", None) if caps_obj else None
                supports_reasoning = val if isinstance(val, bool) else None

            result.append(
                DiscoveredModel(
                    model_id=m.id,
                    provider="anthropic",
                    display_name=getattr(m, "display_name", None),
                    context_window=getattr(m, "max_input_tokens", None),
                    max_output_tokens=getattr(m, "max_tokens", None),
                    supports_tools=True,
                    supports_streaming=True,
                    supports_vision=supports_vision,
                    supports_reasoning=supports_reasoning,
                    supports_batch=supports_batch,
                    provider_metadata={
                        "provider": "anthropic",
                        "raw_model": raw_model,
                        "capabilities": capabilities,
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
        """Generate response via Anthropic Messages API."""
        system_prompt, api_messages, api_tools = self._build_api_payload(
            messages, tools
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if api_tools:
            kwargs["tools"] = api_tools
        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences

        logger.debug("Anthropic API call: model=%s", model)
        response = await self._client.messages.create(**kwargs)
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
        """Stream response chunks via the Anthropic Messages streaming API."""
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
        max_tokens: int = 4096,
        temperature: float = 0.0,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        system_prompt, api_messages, api_tools = self._build_api_payload(
            messages, tools
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if api_tools:
            kwargs["tools"] = api_tools
        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences

        logger.debug("Anthropic streaming call: model=%s", model)
        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield LLMChunk(delta=text)
            final = await stream.get_final_message()
            yield LLMChunk(
                delta="",
                stop_reason=final.stop_reason or "end_turn",
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
                request_id=getattr(final, "id", "") or "",
            )

    async def count_tokens(
        self,
        messages: list[Message],
        model: str,
        *,
        tools: list[ToolDef] | None = None,
    ) -> int:
        """Count tokens server-side via Anthropic Token Counting API."""
        system_prompt, api_messages, api_tools = self._build_api_payload(
            messages, tools
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if api_tools:
            kwargs["tools"] = api_tools

        logger.debug("Anthropic token count: model=%s", model)
        result = await self._client.messages.count_tokens(**kwargs)
        return result.input_tokens

    def _build_api_payload(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        system_prompt = ""
        api_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "tool":
                api_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id or "",
                                "content": msg.content,
                            }
                        ],
                    }
                )
            elif msg.role == "assistant" and msg.tool_calls:
                content: list[dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                api_messages.append({"role": "assistant", "content": content})
            else:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        api_tools: list[dict[str, Any]] = []
        if tools:
            api_tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]

        return system_prompt, api_messages, api_tools

    def _parse_response(self, response: Any, model: str) -> LLMResponse:
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        usage = response.usage
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

        return LLMResponse(
            content="\n".join(content_parts),
            tool_calls=tool_calls,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            stop_reason=response.stop_reason or "end_turn",
            request_id=getattr(response, "id", "") or "",
            thinking_content="\n".join(thinking_parts),
            thinking_tokens=0,  # Anthropic doesn't report thinking token count separately
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
        )
