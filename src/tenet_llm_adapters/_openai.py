"""OpenAI-compatible adapter implementation for the Tenet LLM plugin system."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
_OPENAI_BATCH_MODELS: frozenset[str] = frozenset({
    "gpt-4o", "gpt-4o-2024-11-20", "gpt-4o-2024-08-06",
    "gpt-4o-mini", "gpt-4o-mini-2024-07-18",
    "gpt-3.5-turbo", "gpt-3.5-turbo-0125",
    "gpt-4-turbo", "gpt-4-turbo-preview",
    "o3-mini",
})

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from tenet_core.llm import DiscoveredModel

import openai
from tenet_core.llm.client import LLMChunk, LLMParams, LLMResponse, Message, ToolCall, ToolDef, resolve_params

logger = logging.getLogger(__name__)


def _longest_suffix_prefix(text: str, tag: str) -> int:
    """Return the length of the longest suffix of *text* that is a prefix of *tag*.

    Used to detect partial thinking-tag matches at chunk boundaries so we can
    buffer those characters rather than flushing them as content.
    """
    max_len = min(len(text), len(tag) - 1)
    for length in range(max_len, 0, -1):
        if text.endswith(tag[:length]):
            return length
    return 0


class OpenAIAdapter:
    """OpenAI-compatible LLM adapter."""

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
            if path.endswith("/v1/models"):
                path = path[: -len("/models")]
            elif path.endswith("/models"):
                path = path[: -len("/models")] or "/v1"
            elif not path:
                path = "/v1"
            normalized = parsed._replace(path=path).geturl().rstrip("/")
            return normalized or None
        except Exception:
            return candidate

    def __init__(self, api_key: str, *, base_url: str | None = None) -> None:
        kwargs: dict[str, Any] = {"api_key": api_key}
        normalized = self._normalize_base_url(base_url)
        if normalized:
            kwargs["base_url"] = normalized
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

    @staticmethod
    def _normalize_messages(
        api_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Reorder and merge system messages to satisfy llama.cpp chat-template rules.

        llama.cpp (and many local-model servers) require the system message to be
        the very first message in the conversation.  When the agent loop injects
        context as a mid-conversation system message, the payload arrives here
        with system messages scattered through the list.  This method collects all
        system messages, merges their content (newline-separated), and hoists the
        merged message to index 0.  All other messages remain in their original
        relative order.  OpenAI-hosted endpoints are not affected because they
        accept system messages at any position.
        """
        system_parts: list[str] = []
        non_system: list[dict[str, Any]] = []
        for m in api_messages:
            if m.get("role") == "system":
                content = m.get("content") or ""
                if content:
                    system_parts.append(content)
            else:
                non_system.append(m)
        if not system_parts:
            return non_system
        merged_system: dict[str, Any] = {
            "role": "system",
            "content": "\n\n".join(system_parts),
        }
        return [merged_system, *non_system]

    async def list_models(self) -> list[DiscoveredModel]:
        """List available models from the OpenAI-compatible /v1/models endpoint."""
        from tenet_core.llm import DiscoveredModel

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
        params: LLMParams | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Generate response via OpenAI-compatible Chat Completions API."""
        p = resolve_params(params, max_tokens=max_tokens, temperature=temperature, stop_sequences=stop_sequences)
        effective_temp = p.temperature if p.temperature is not None else 0.0
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
        api_messages = self._normalize_messages(api_messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": p.max_tokens,
            "temperature": effective_temp,
        }
        if p.top_p is not None:
            kwargs["top_p"] = p.top_p
        # OpenAI reasoning_effort for o1/o3 models
        if p.reasoning is not None and p.reasoning != "off":
            kwargs["reasoning_effort"] = p.reasoning
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
        if p.stop_sequences:
            kwargs["stop"] = p.stop_sequences
        if extra_headers:
            kwargs["extra_headers"] = dict(extra_headers)

        logger.debug("OpenAI API call: model=%s", model)
        request_client = (
            self._client.with_options(default_headers=dict(extra_headers))
            if extra_headers
            else self._client
        )
        response = await request_client.chat.completions.create(**kwargs)
        return self._parse_response(response, model)

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
        thinking_tags: tuple[str, str] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """Stream response chunks via the OpenAI Chat Completions streaming API.

        Args:
            thinking_tags: Optional ``(start_tag, end_tag)`` pair for models that
                embed reasoning inside the response text (e.g. Qwen3 uses
                ``("<think>", "</think>")``, DeepSeek-R1 uses the same).  When
                provided, text between the tags is emitted as ``thinking_delta``
                chunks and stripped from the regular ``delta`` stream.
        """
        return self._stream_impl(
            messages,
            model,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
            params=params,
            thinking_tags=thinking_tags,
            extra_headers=extra_headers,
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
        params: LLMParams | None = None,
        thinking_tags: tuple[str, str] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        p = resolve_params(params, max_tokens=max_tokens, temperature=temperature, stop_sequences=stop_sequences)
        effective_temp = p.temperature if p.temperature is not None else 0.0
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

        api_messages = self._normalize_messages(api_messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": p.max_tokens,
            "temperature": effective_temp,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if p.top_p is not None:
            kwargs["top_p"] = p.top_p
        # OpenAI reasoning_effort for o1/o3 models
        if p.reasoning is not None and p.reasoning != "off":
            kwargs["reasoning_effort"] = p.reasoning
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
        if p.stop_sequences:
            kwargs["stop"] = p.stop_sequences
        if extra_headers:
            kwargs["extra_headers"] = dict(extra_headers)

        logger.debug("OpenAI streaming call: model=%s", model)
        last_chunk: Any = None
        tool_call_parts: dict[int, dict[str, Any]] = {}
        # Track reasoning_content separately for thinking models (e.g., Qwen3).
        # Only fall back to it if the model emitted zero content tokens (e.g. when
        # max_tokens is exhausted mid-thinking-chain).
        reasoning_text = ""
        got_content = False
        # State for inline thinking-tag parsing (e.g. Qwen3 <think>...</think>).
        in_thinking_tag = False
        tag_buffer = ""  # accumulates chars when we may be mid-tag
        think_start = thinking_tags[0] if thinking_tags else None
        think_end = thinking_tags[1] if thinking_tags else None

        request_client = (
            self._client.with_options(default_headers=dict(extra_headers))
            if extra_headers
            else self._client
        )

        try:
            async for chunk in await request_client.chat.completions.create(**kwargs):
                last_chunk = chunk
                if not chunk.choices:
                    continue
                delta_obj = chunk.choices[0].delta
                delta_tool_calls = getattr(delta_obj, "tool_calls", None) or []
                if delta_tool_calls:
                    for tc in delta_tool_calls:
                        idx_raw = getattr(tc, "index", None)
                        try:
                            idx = int(idx_raw) if idx_raw is not None else 0
                        except (TypeError, ValueError):
                            idx = 0

                        part = tool_call_parts.setdefault(
                            idx,
                            {
                                "id": "",
                                "name": "",
                                "arguments": "",
                                "type": "function",
                            },
                        )
                        tc_id = getattr(tc, "id", None)
                        if isinstance(tc_id, str) and tc_id:
                            part["id"] = tc_id
                        tc_type = getattr(tc, "type", None)
                        if isinstance(tc_type, str) and tc_type:
                            part["type"] = tc_type

                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            fn_name = getattr(fn, "name", None)
                            if isinstance(fn_name, str) and fn_name:
                                part["name"] = fn_name
                            fn_args = getattr(fn, "arguments", None)
                            if isinstance(fn_args, str) and fn_args:
                                part["arguments"] += fn_args

                delta_text = delta_obj.content or ""
                # Native reasoning_content field (o1/o3/DeepSeek API).
                reasoning_delta = getattr(delta_obj, "reasoning_content", None) or ""
                if reasoning_delta:
                    yield LLMChunk(thinking_delta=reasoning_delta)
                if delta_text:
                    if think_start and think_end:
                        # Parse inline thinking tags out of the text stream.
                        remaining = tag_buffer + delta_text
                        tag_buffer = ""
                        while remaining:
                            if in_thinking_tag:
                                end_pos = remaining.find(think_end)
                                if end_pos == -1:
                                    yield LLMChunk(thinking_delta=remaining)
                                    remaining = ""
                                else:
                                    if end_pos > 0:
                                        yield LLMChunk(thinking_delta=remaining[:end_pos])
                                    in_thinking_tag = False
                                    remaining = remaining[end_pos + len(think_end):]
                            else:
                                start_pos = remaining.find(think_start)
                                if start_pos == -1:
                                    # Check for partial tag at the end of the chunk.
                                    partial = _longest_suffix_prefix(remaining, think_start)
                                    if partial:
                                        tag_buffer = remaining[len(remaining) - partial:]
                                        text_part = remaining[: len(remaining) - partial]
                                    else:
                                        text_part = remaining
                                    if text_part:
                                        got_content = True
                                        yield LLMChunk(delta=text_part)
                                    remaining = ""
                                else:
                                    if start_pos > 0:
                                        got_content = True
                                        yield LLMChunk(delta=remaining[:start_pos])
                                    in_thinking_tag = True
                                    remaining = remaining[start_pos + len(think_start):]
                    else:
                        got_content = True
                        reasoning_text = reasoning_text  # keep for fallback check
                        yield LLMChunk(delta=delta_text)
                elif not reasoning_delta:
                    # No content and no native reasoning — accumulate for fallback.
                    pass
        except json.JSONDecodeError as exc:
            # Some OpenAI-compatible proxies may emit trailing empty/invalid SSE
            # frames after valid chunks. If we already captured content or tool
            # calls, gracefully finish instead of hard-failing the whole turn.
            if got_content or tool_call_parts:
                logger.warning(
                    "OpenAI stream ended with JSON decode error after partial output; "
                    "continuing with collected chunks: %s",
                    exc,
                )
            else:
                raise
        # Flush any tag_buffer remainder as plain text (tag never closed).
        if tag_buffer:
            got_content = True
            yield LLMChunk(delta=tag_buffer)
        if not got_content and reasoning_text:
            # Fallback: model exhausted tokens during reasoning with no content yield.
            yield LLMChunk(delta=reasoning_text)
        usage = getattr(last_chunk, "usage", None) if last_chunk is not None else None
        finish_reason = (
            last_chunk.choices[0].finish_reason
            if last_chunk is not None and last_chunk.choices
            else "stop"
        )
        reasoning_tokens = 0
        if usage:
            completion_details = getattr(usage, "completion_tokens_details", None)
            if completion_details:
                reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0

        parsed_tool_calls: list[ToolCall] = []
        if tool_call_parts:
            for idx in sorted(tool_call_parts.keys()):
                part = tool_call_parts[idx]
                args_raw = part.get("arguments", "") or ""
                args_obj: dict[str, Any] = {}
                if isinstance(args_raw, str) and args_raw.strip():
                    try:
                        parsed = json.loads(args_raw)
                        if isinstance(parsed, dict):
                            args_obj = parsed
                    except json.JSONDecodeError:
                        args_obj = {"_raw": args_raw}

                parsed_tool_calls.append(
                    ToolCall(
                        id=str(part.get("id") or f"tool_call_{idx}"),
                        name=str(part.get("name") or "unknown_tool"),
                        arguments=args_obj,
                    )
                )

        yield LLMChunk(
            delta="",
            stop_reason=finish_reason or "stop",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            thinking_tokens=reasoning_tokens,
            request_id=getattr(last_chunk, "id", "") or "" if last_chunk else "",
            tool_calls=parsed_tool_calls,
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
        # Extract reasoning/thinking tokens if available (e.g. o1, Qwen3).
        reasoning_tokens = 0
        cached_tokens = 0
        if usage:
            completion_details = getattr(usage, "completion_tokens_details", None)
            if completion_details:
                reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details:
                cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0

        # Fallback: reasoning models (o1, Qwen3) may exhaust the token
        # budget on chain-of-thought, leaving msg.content empty while the
        # actual output sits in msg.reasoning_content.  Mirror the streaming
        # path's fallback so callers always receive usable content.
        content = msg.content or ""
        if not content:
            content = getattr(msg, "reasoning_content", "") or ""

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            stop_reason=choice.finish_reason or "stop",
            request_id=getattr(response, "id", "") or "",
            thinking_tokens=reasoning_tokens,
            cache_read_tokens=cached_tokens,
        )
