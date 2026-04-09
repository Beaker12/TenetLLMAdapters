# Source Tree — TenetLLMAdapters

## Package Layout

```
src/tenet_llm_adapters/
├── __init__.py          # Lazy-import module with __getattr__
├── declaration.py       # TenetCore module declaration
├── _openai.py           # OpenAI-compatible adapter
├── _anthropic.py        # Anthropic (Claude) adapter
├── _google.py           # Google Gemini adapter
├── _cohere.py           # Cohere v2 adapter
tests/
├── conftest.py          # Shared fixtures
├── test_openai.py       # OpenAI adapter tests
├── test_anthropic.py    # Anthropic adapter tests
├── test_google.py       # Google adapter tests
├── test_cohere.py       # Cohere adapter tests
docs/
├── README.md            # Overview
├── architecture.md      # Architecture deep-dive
├── source-tree.md       # This file
├── providers.md         # Per-provider reference
├── configuration.md     # Configuration reference
├── operations.md        # Operations runbook
├── integration.md       # Integration contracts
└── gaps-and-drift.md    # Gap analysis
pyproject.toml           # Package metadata, entry points, extras
```

## File Details

### `__init__.py`

| Symbol | Kind | Description |
|---|---|---|
| `__getattr__(name)` | function | Lazy-loads adapter classes on first access |
| `__all__` | list | `["AnthropicAdapter", "OpenAIAdapter", "GoogleAdapter", "CohereAdapter"]` |

### `declaration.py`

| Symbol | Kind | Description |
|---|---|---|
| `_MODULE_ID` | str | `"tenet_llm_adapters"` |
| `_MODULE_VERSION` | str | `"1.0.0"` |
| `_TUNABLES` | list | Empty list |
| `get_declaration()` | function | Returns `ModuleDeclaration` for TenetCore registration |

### `_openai.py`

| Symbol | Kind | Description |
|---|---|---|
| `_OPENAI_BATCH_MODELS` | frozenset | Known batch-capable model IDs |
| `OpenAIAdapter` | class | OpenAI-compatible LLM adapter |
| `OpenAIAdapter.__init__(api_key, *, base_url)` | method | Creates `openai.AsyncOpenAI` client |
| `OpenAIAdapter.from_config(config)` | classmethod | Factory from config dict |
| `OpenAIAdapter._to_plain_dict(value)` | staticmethod | SDK object → plain dict |
| `OpenAIAdapter.list_models()` | async method | Lists models via `/v1/models` |
| `OpenAIAdapter.generate(messages, model, ...)` | async method | Chat Completions (non-streaming) |
| `OpenAIAdapter.stream(messages, model, ...)` | async method | Chat Completions (streaming) |
| `OpenAIAdapter._stream_impl(...)` | async method | Internal streaming with reasoning fallback |
| `OpenAIAdapter._parse_response(response, model)` | method | SDK response → `LLMResponse` |

### `_anthropic.py`

| Symbol | Kind | Description |
|---|---|---|
| `_ANTHROPIC_BATCH_MODELS` | frozenset | Known batch-capable model IDs |
| `AnthropicAdapter` | class | Anthropic (Claude) LLM adapter |
| `AnthropicAdapter.__init__(api_key, *, base_url)` | method | Creates `anthropic.AsyncAnthropic` client |
| `AnthropicAdapter.from_config(config)` | classmethod | Factory from config dict |
| `AnthropicAdapter._to_plain_dict(value)` | staticmethod | SDK object → plain dict |
| `AnthropicAdapter._capability_supported(capabilities, key)` | staticmethod | Extracts capability bool from nested dict |
| `AnthropicAdapter.list_models()` | async method | Lists models via beta models endpoint |
| `AnthropicAdapter.generate(messages, model, ...)` | async method | Messages API (non-streaming) |
| `AnthropicAdapter.stream(messages, model, ...)` | async method | Messages API (streaming) |
| `AnthropicAdapter._stream_impl(...)` | async method | Internal streaming via `messages.stream()` |
| `AnthropicAdapter.count_tokens(messages, model, *, tools)` | async method | Server-side token counting |
| `AnthropicAdapter._build_api_payload(messages, tools)` | method | Converts to Anthropic message format |
| `AnthropicAdapter._parse_response(response, model)` | method | SDK response → `LLMResponse` |

### `_google.py`

| Symbol | Kind | Description |
|---|---|---|
| `_BASE` | str | `"https://generativelanguage.googleapis.com/v1beta"` |
| `GoogleAdapter` | class | Google Gemini REST adapter |
| `GoogleAdapter.__init__(api_key, *, base_url)` | method | Stores API key and base URL |
| `GoogleAdapter.from_config(config)` | classmethod | Factory from config dict |
| `GoogleAdapter.list_models()` | async method | Lists models via `/v1beta/models` |
| `GoogleAdapter.generate(messages, model, ...)` | async method | `generateContent` endpoint |
| `GoogleAdapter.stream(messages, model, ...)` | async method | `streamGenerateContent` SSE endpoint |
| `GoogleAdapter._build_payload(messages, tools, ...)` | method | Converts to Gemini payload format |
| `GoogleAdapter._parse_response(data, model)` | method | JSON response → `LLMResponse` |

### `_cohere.py`

| Symbol | Kind | Description |
|---|---|---|
| `_BASE` | str | `"https://api.cohere.com"` |
| `CohereAdapter` | class | Cohere v2 API adapter |
| `CohereAdapter.__init__(api_key, *, base_url)` | method | Stores API key and base URL |
| `CohereAdapter.from_config(config)` | classmethod | Factory from config dict |
| `CohereAdapter._auth_headers()` | method | Returns `Authorization: Bearer` header |
| `CohereAdapter.list_models()` | async method | Lists models via `/v2/models` |
| `CohereAdapter.generate(messages, model, ...)` | async method | `/v2/chat` (non-streaming) |
| `CohereAdapter.stream(messages, model, ...)` | async method | `/v2/chat` (streaming NDJSON) |
| `CohereAdapter._build_payload(messages, model, ...)` | method | Converts to Cohere v2 message format |
| `CohereAdapter._parse_response(data, model)` | method | JSON response → `LLMResponse` |

## Entry Points (pyproject.toml)

### `tenet.llm_adapters`

| Key | Target |
|---|---|
| `anthropic` | `tenet_llm_adapters._anthropic:AnthropicAdapter` |
| `openai-compatible` | `tenet_llm_adapters._openai:OpenAIAdapter` |
| `google` | `tenet_llm_adapters._google:GoogleAdapter` |
| `cohere` | `tenet_llm_adapters._cohere:CohereAdapter` |

### `tenet.module_declarations`

| Key | Target |
|---|---|
| `tenet_llm_adapters` | `tenet_llm_adapters.declaration:get_declaration` |
