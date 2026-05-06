"""Anthropic adapter implementation for the Tenet LLM plugin system."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import anthropic
from tenet_core.llm.client import (
    LLMChunk,
    LLMParams,
    LLMResponse,
    Message,
    ToolCall,
    ToolDef,
    resolve_params,
)

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


_LONG_REQUEST_STREAMING_ERROR = (
    "Streaming is required for operations that may take longer than 10 minutes."
)


class AnthropicAdapter:
    """Anthropic (Claude) LLM adapter."""

    @staticmethod
    def _normalize_base_url(base_url: str | None) -> str | None:
        if not base_url:
            return None
        candidate = base_url.strip().rstrip("/")
        if not candidate:
            return None
        try:
            parsed = urlparse(candidate)
            path = (parsed.path or "").rstrip("/")
            for suffix in ("/v1/messages", "/v1/models", "/v1"):
                if path.endswith(suffix):
                    path = path[: -len(suffix)]
                    break
            normalized = parsed._replace(path=path).geturl().rstrip("/")
            return normalized or None
        except Exception:
            return candidate

    def __init__(self, api_key: str, *, base_url: str | None = None) -> None:
        kwargs: dict[str, Any] = {"api_key": api_key}
        normalized = self._normalize_base_url(base_url)
        if normalized:
            kwargs["base_url"] = normalized
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
        params: LLMParams | None = None,
    ) -> LLMResponse:
        """Generate response via Anthropic Messages API."""
        p = resolve_params(
            params,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )
        system_prompt, api_messages, api_tools = self._build_api_payload(
            messages, tools
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": p.max_tokens,
            "temperature": p.temperature if p.temperature is not None else 0.0,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if api_tools:
            kwargs["tools"] = api_tools
        if p.stop_sequences:
            kwargs["stop_sequences"] = p.stop_sequences

        logger.debug("Anthropic API call: model=%s", model)
        try:
            response = await self._client.messages.create(**kwargs)
        except ValueError as exc:
            if _LONG_REQUEST_STREAMING_ERROR not in str(exc):
                raise
            logger.info(
                "Anthropic non-streaming request requires streaming; retrying via streaming API | model=%s",
                model,
            )
            response = await self._generate_via_streaming_api(**kwargs)
        return self._parse_response(response, model)

    async def _generate_via_streaming_api(self, **kwargs: Any) -> Any:
        """Execute a request via Anthropic streaming and return the final message."""
        async with self._client.messages.stream(**kwargs) as stream:
            return await stream.get_final_message()

    def stream(
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
        """Stream response chunks via the Anthropic Messages streaming API."""
        return self._stream_impl(
            messages,
            model,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
            params=params,
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
        params: LLMParams | None = None,
    ) -> AsyncIterator[LLMChunk]:
        p = resolve_params(
            params,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )
        system_prompt, api_messages, api_tools = self._build_api_payload(
            messages, tools
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": p.max_tokens,
            "temperature": p.temperature if p.temperature is not None else 0.0,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if api_tools:
            kwargs["tools"] = api_tools
        if p.stop_sequences:
            kwargs["stop_sequences"] = p.stop_sequences

        logger.debug("Anthropic streaming call: model=%s", model)
        streamed_text_parts: list[str] = []
        streamed_thinking_parts: list[str] = []
        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                event_type_name = type(event).__name__
                event_type = str(getattr(event, "type", "") or "")
                is_content_block_delta = (
                    event_type == "content_block_delta"
                    or event_type_name.endswith("ContentBlockDeltaEvent")
                )

                if is_content_block_delta:
                    delta = getattr(event, "delta", None)
                    if delta is None:
                        continue
                    delta_type = str(getattr(delta, "type", "") or "")
                    if delta_type == "text_delta":
                        text = str(getattr(delta, "text", "") or "")
                        if text:
                            streamed_text_parts.append(text)
                            yield LLMChunk(delta=text)
                    elif delta_type == "thinking_delta":
                        thinking = str(getattr(delta, "thinking", "") or "")
                        if thinking:
                            streamed_thinking_parts.append(thinking)
                            yield LLMChunk(thinking_delta=thinking)
            final = await stream.get_final_message()

            # Some Anthropic SDK versions may not emit delta events in the
            # same class shape; backfill from final blocks so content is never
            # silently dropped.
            if not streamed_text_parts and getattr(final, "content", None):
                fallback_text_parts: list[str] = []
                for block in final.content:
                    if getattr(block, "type", None) == "text":
                        text = str(getattr(block, "text", "") or "")
                        if text:
                            fallback_text_parts.append(text)
                fallback_text = "\n".join(fallback_text_parts)
                if fallback_text:
                    yield LLMChunk(delta=fallback_text)

            if not streamed_thinking_parts and getattr(final, "content", None):
                fallback_thinking_parts: list[str] = []
                for block in final.content:
                    if getattr(block, "type", None) == "thinking":
                        thinking = str(getattr(block, "thinking", "") or "")
                        if thinking:
                            fallback_thinking_parts.append(thinking)
                fallback_thinking = "\n".join(fallback_thinking_parts)
                if fallback_thinking:
                    yield LLMChunk(thinking_delta=fallback_thinking)

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
