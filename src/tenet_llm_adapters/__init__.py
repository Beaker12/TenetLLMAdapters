"""Cloud LLM adapters for the Tenet platform.

Provides Anthropic, OpenAI-compatible, Google Gemini, and Cohere adapters.
Each adapter registers via the ``tenet.llm_adapters`` entry-point group and
implements both ``LLMAdapter`` (generation) and ``LLMDiscovery`` (model listing).

Install with the extra that matches the provider you need, e.g.::

    pip install tenet-llm-adapters[anthropic]
    pip install tenet-llm-adapters[openai]
    pip install tenet-llm-adapters[google,cohere]
    pip install tenet-llm-adapters[all]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tenet_llm_adapters._anthropic import AnthropicAdapter
    from tenet_llm_adapters._cohere import CohereAdapter
    from tenet_llm_adapters._google import GoogleAdapter
    from tenet_llm_adapters._openai import OpenAIAdapter


def __getattr__(name: str) -> Any:
    if name == "AnthropicAdapter":
        from tenet_llm_adapters._anthropic import AnthropicAdapter

        return AnthropicAdapter
    if name == "OpenAIAdapter":
        from tenet_llm_adapters._openai import OpenAIAdapter

        return OpenAIAdapter
    if name == "GoogleAdapter":
        from tenet_llm_adapters._google import GoogleAdapter

        return GoogleAdapter
    if name == "CohereAdapter":
        from tenet_llm_adapters._cohere import CohereAdapter

        return CohereAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AnthropicAdapter", "OpenAIAdapter", "GoogleAdapter", "CohereAdapter"]
