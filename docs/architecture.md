# Architecture — TenetLLMAdapters

```{graphviz}
digraph TenetLLMAdaptersArch {
  graph [fontname="Arial" fontsize=10 rankdir=TB nodesep=0.3 ranksep=0.5 bgcolor=white]
  node  [shape=box style="rounded,filled" fontname="Arial" fontsize=10 penwidth=1 color="#4a5568"]
  edge  [fontname="Arial" fontsize=9 color="#4a5568" arrowsize=0.7]

  TenetCore [label="TenetCore\nAdapter Registry" fillcolor="#e8f0fe" shape=ellipse]
  EntryPoints [label="entry_points\n(tenet.llm_adapters)" fillcolor="#feebc8"]
  LazyInit [label="__getattr__\nLazy Import" fillcolor="#f7f9fc"]

  subgraph cluster_adapters {
    label="Provider Adapters" style=solid color="#63b3ed" fontname="Arial" fontsize=11 bgcolor=white
    OpenAI [label="OpenAIAdapter\n(openai SDK)" fillcolor="#fff7e6"]
    Anthropic [label="AnthropicAdapter\n(anthropic SDK)" fillcolor="#fff7e6"]
    Google [label="GoogleAdapter\n(httpx REST)" fillcolor="#fff7e6"]
    Cohere [label="CohereAdapter\n(httpx REST)" fillcolor="#fff7e6"]
  }

  subgraph cluster_contract {
    label="Adapter Contract" style=solid color="#63b3ed" fontname="Arial" fontsize=11 bgcolor=white
    Generate [label="generate()" fillcolor="#edf2f7"]
    Stream [label="stream()" fillcolor="#edf2f7"]
    ListModels [label="list_models()" fillcolor="#edf2f7"]
    FromConfig [label="from_config()" fillcolor="#edf2f7"]
  }

  subgraph cluster_types {
    label="Canonical Types" style=solid color="#63b3ed" fontname="Arial" fontsize=11 bgcolor=white
    LLMResponse [label="LLMResponse" fillcolor="#c6f6d5"]
    LLMChunk [label="LLMChunk" fillcolor="#c6f6d5"]
    ToolCall [label="ToolCall" fillcolor="#c6f6d5"]
  }

  TenetCore -> EntryPoints [label="discover"]
  EntryPoints -> LazyInit [label="load class"]
  LazyInit -> OpenAI
  LazyInit -> Anthropic
  LazyInit -> Google
  LazyInit -> Cohere

  OpenAI -> Generate
  OpenAI -> Stream
  OpenAI -> ListModels
  Anthropic -> Generate
  Anthropic -> Stream

  Generate -> LLMResponse [label="return"]
  Stream -> LLMChunk [label="yield"]
}
```

## Adapter Pattern

Each provider adapter is a standalone class that:

1. Accepts `api_key` and optional `base_url` in its constructor.
2. Exposes a `from_config(config: dict)` class method for factory instantiation from `BackendConfig.model_dump()` dicts.
3. Implements `generate()`, `stream()`, and `list_models()` with a uniform signature.
4. Returns TenetCore canonical types: `LLMResponse`, `LLMChunk`, `ToolCall`.

There is no abstract base class. The contract is structural (duck-typed), enforced by TenetCore's adapter loader.

## Provider Abstraction

All adapters translate between TenetCore's canonical message format (`Message`, `ToolDef`, `ToolCall`) and provider-specific API formats:

```
TenetCore Message[]  ──►  _build_api_payload()  ──►  Provider API format
Provider response    ──►  _parse_response()     ──►  LLMResponse / LLMChunk
```

Translation responsibilities:

| Concern | OpenAI | Anthropic | Google | Cohere |
|---|---|---|---|---|
| System message | `role: "system"` | Extracted to `system` kwarg | `systemInstruction.parts` | `role: "system"` |
| Tool results | `role: "tool"` + `tool_call_id` | `role: "user"` + `tool_result` content block | `functionResponse` in user parts | `role: "tool"` + `tool_result` |
| Tool calls in assistant | `tool_calls[]` with JSON-string args | `tool_use` content blocks | `functionCall` in model parts | `tool_use` content blocks |
| Streaming | SDK `stream=True` iterator | `messages.stream()` context manager | SSE `alt=sse` | NDJSON lines |

## MCP Integration

Adapters are not MCP servers themselves. They are registered via Python entry points:

```toml
[project.entry-points."tenet.llm_adapters"]
anthropic        = "tenet_llm_adapters._anthropic:AnthropicAdapter"
openai-compatible = "tenet_llm_adapters._openai:OpenAIAdapter"
google           = "tenet_llm_adapters._google:GoogleAdapter"
cohere           = "tenet_llm_adapters._cohere:CohereAdapter"
```

TenetCore's adapter registry (`adapter_registry.py`) discovers these at startup via `importlib.metadata.entry_points(group="tenet.llm_adapters")` and instantiates them with backend configuration.

## Module Declaration

The `declaration.py` module registers via the `tenet.module_declarations` entry point:

```python
ModuleDeclaration(
    module_id="tenet_llm_adapters",
    module_version="1.0.0",
    module_category="core",
    required=True,
    tunables=[],
)
```

No tunables are declared — all configuration flows through `BackendConfig` in the instance manifest.

## Lazy Import Strategy

`__init__.py` uses `__getattr__` for lazy imports. Adapter classes are only imported when accessed, avoiding import-time SDK load failures when a provider SDK is not installed:

```python
def __getattr__(name: str) -> Any:
    if name == "AnthropicAdapter":
        from tenet_llm_adapters._anthropic import AnthropicAdapter
        return AnthropicAdapter
    ...
```

This means `import tenet_llm_adapters` always succeeds regardless of installed extras.

## SDK vs REST Adapters

| Adapter | Transport | Rationale |
|---|---|---|
| `OpenAIAdapter` | `openai` Python SDK | Mature SDK with streaming, tool calling, usage tracking |
| `AnthropicAdapter` | `anthropic` Python SDK | Required for Messages API, streaming context manager, token counting |
| `GoogleAdapter` | `httpx` direct REST | No official async Python SDK; REST API is straightforward |
| `CohereAdapter` | `httpx` direct REST | v2 API is simple enough; avoids `cohere` SDK dependency |

## Streaming Architecture

All adapters implement streaming as async generators yielding `LLMChunk`:

1. Content chunks: `LLMChunk(delta="text")`
2. Final sentinel: `LLMChunk(delta="", stop_reason=..., input_tokens=..., output_tokens=..., request_id=...)`

The final chunk always carries usage metadata. OpenAI uses `stream_options={"include_usage": True}` to get usage in the last streamed chunk. Anthropic gets it from `stream.get_final_message()`. Google and Cohere extract it from the last SSE/NDJSON payload.

All adapters accept provider-agnostic `LLMParams` in both `generate()` and `stream()`. Adapters that do not support a parameter ignore it explicitly at the adapter boundary rather than requiring TenetCore call-site branching.

### OpenAI Reasoning Content Fallback

`OpenAIAdapter._stream_impl` tracks `reasoning_content` deltas separately. If the model emits zero content tokens (e.g., Qwen3 exhausting `max_tokens` during a thinking chain), the collected reasoning text is yielded as a fallback.

## Error Handling

- SDK adapters (OpenAI, Anthropic) propagate SDK exceptions directly.
- REST adapters (Google, Cohere) call `resp.raise_for_status()`, raising `httpx.HTTPStatusError` on 4xx/5xx.
- `_parse_response()` in `OpenAIAdapter` raises `RuntimeError` if the response contains no choices.
- `_to_plain_dict()` (shared helper in OpenAI and Anthropic) handles SDK object serialization for error diagnostics.

## Batch Model Detection

Both OpenAI and Anthropic maintain hardcoded frozensets of known batch-capable models:

- `_OPENAI_BATCH_MODELS`: gpt-4o variants, gpt-3.5-turbo, gpt-4-turbo, o3-mini
- `_ANTHROPIC_BATCH_MODELS`: Claude 3.7/3.5/3 variants

Anthropic also reads batch capability from the model's `capabilities` object when the API provides it, falling back to the frozenset.
