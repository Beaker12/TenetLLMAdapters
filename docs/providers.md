# Provider Reference — TenetLLMAdapters

```{graphviz}
digraph TenetProviders {
  graph [fontname="Arial" fontsize=10 rankdir=LR nodesep=0.3 ranksep=0.5 bgcolor=white]
  node  [shape=box style="rounded,filled" fontname="Arial" fontsize=10 penwidth=1 color="#4a5568"]
  edge  [fontname="Arial" fontsize=9 color="#4a5568" arrowsize=0.7]

  Factory [label="Adapter\nFactory" fillcolor="#e6fffa"]

  subgraph cluster_providers {
    label="Provider Adapters" style=solid color="#63b3ed" fontname="Arial" fontsize=11 bgcolor=white
    OpenAI [label="OpenAIAdapter\nopenai-compatible\n(openai SDK)" fillcolor="#fff7e6"]
    Anthropic [label="AnthropicAdapter\nanthropic\n(anthropic SDK)" fillcolor="#fff7e6"]
    Google [label="GoogleAdapter\ngoogle\n(httpx REST)" fillcolor="#fff7e6"]
    Cohere [label="CohereAdapter\ncohere\n(httpx REST)" fillcolor="#fff7e6"]
  }

  subgraph cluster_capabilities {
    label="Capabilities" style=solid color="#63b3ed" fontname="Arial" fontsize=11 bgcolor=white
    Generate [label="generate()" fillcolor="#c6f6d5"]
    Stream [label="stream()" fillcolor="#c6f6d5"]
    ListModels [label="list_models()" fillcolor="#c6f6d5"]
    CountTokens [label="count_tokens()\n(Anthropic only)" fillcolor="#edf2f7"]
    Batch [label="Batch Support\n(OpenAI models)" fillcolor="#edf2f7"]
  }

  subgraph cluster_apis {
    label="Provider APIs" style=solid color="#63b3ed" fontname="Arial" fontsize=11 bgcolor=white
    OAIAPI [label="OpenAI\n/v1/chat/completions" fillcolor="#f3e8ff"]
    AAPI [label="Anthropic\n/v1/messages" fillcolor="#f3e8ff"]
    GAPI [label="Google Gemini\ngenerateContent" fillcolor="#f3e8ff"]
    CAPI [label="Cohere\n/v2/chat" fillcolor="#f3e8ff"]
  }

  Factory -> OpenAI
  Factory -> Anthropic
  Factory -> Google
  Factory -> Cohere

  OpenAI -> Generate
  OpenAI -> Stream
  OpenAI -> ListModels
  OpenAI -> Batch
  Anthropic -> Generate
  Anthropic -> Stream
  Anthropic -> CountTokens
  Google -> Generate
  Google -> Stream
  Cohere -> Generate
  Cohere -> Stream

  OpenAI -> OAIAPI [style=dashed]
  Anthropic -> AAPI [style=dashed]
  Google -> GAPI [style=dashed]
  Cohere -> CAPI [style=dashed]
}
```

## OpenAI (`OpenAIAdapter`)

**Entry-point key:** `openai-compatible`
**Module:** `tenet_llm_adapters._openai`
**SDK:** `openai` (async client, `openai.AsyncOpenAI`)

### Authentication

| Parameter | Source | Notes |
|---|---|---|
| `api_key` | `BackendConfig.api_key` | Required for OpenAI; optional for self-hosted OAI-compatible servers |
| `base_url` | `BackendConfig.base_url` | Override for Azure OpenAI, self-hosted endpoints, etc. |

### API Surface

| Method | Endpoint | Description |
|---|---|---|
| `list_models()` | `GET /v1/models` | Lists all available models |
| `generate()` | `POST /v1/chat/completions` | Non-streaming chat completion |
| `stream()` | `POST /v1/chat/completions` (stream) | Streaming with `stream_options: {include_usage: true}` |

### Tool Calling

Tools are sent as `tools[]` with `type: "function"` format. Tool call arguments are JSON-serialized strings in responses, parsed back to dicts.

### Batch Models

Hardcoded frozenset `_OPENAI_BATCH_MODELS`:
- `gpt-4o`, `gpt-4o-2024-11-20`, `gpt-4o-2024-08-06`
- `gpt-4o-mini`, `gpt-4o-mini-2024-07-18`
- `gpt-3.5-turbo`, `gpt-3.5-turbo-0125`
- `gpt-4-turbo`, `gpt-4-turbo-preview`
- `o3-mini`

### Streaming Notes

- Default `max_tokens` in `_stream_impl` is `16384` (higher than `generate`'s `4096`)
- Tracks `reasoning_content` deltas for thinking models (e.g., Qwen3); yields reasoning as fallback if no content tokens emitted
- Final chunk carries `stop_reason`, `input_tokens`, `output_tokens`, `request_id`

### DiscoveredModel Fields

| Field | Value |
|---|---|
| `provider` | `"openai-compatible"` |
| `supports_batch` | `True` if model ID in `_OPENAI_BATCH_MODELS` |
| `provider_metadata.raw_model` | Raw model dict from API |

---

## Anthropic (`AnthropicAdapter`)

**Entry-point key:** `anthropic`
**Module:** `tenet_llm_adapters._anthropic`
**SDK:** `anthropic` (async client, `anthropic.AsyncAnthropic`)

### Authentication

| Parameter | Source | Notes |
|---|---|---|
| `api_key` | `BackendConfig.api_key` | Anthropic API key |
| `base_url` | `BackendConfig.base_url` | Override for proxy/custom endpoints |

### API Surface

| Method | Endpoint | Description |
|---|---|---|
| `list_models()` | `GET /v1/models` (beta) | Lists all available models with capabilities |
| `generate()` | `POST /v1/messages` | Non-streaming Messages API |
| `stream()` | `POST /v1/messages` (stream) | Streaming via `messages.stream()` context manager |
| `count_tokens()` | `POST /v1/messages/count_tokens` | Server-side token counting |

`generate()` and `stream()` now also accept provider-agnostic `LLMParams` values propagated from TenetCore (`max_tokens`, `temperature`, `stop_sequences`).

### System Message Handling

System messages are extracted from the message list and passed as the `system` keyword argument to the Messages API — not included in the `messages[]` array.

### Tool Calling

- Outbound: `input_schema` format (Anthropic-native)
- Tool results: wrapped as `role: "user"` with `tool_result` content block
- Tool calls in assistant: `tool_use` content blocks

### Batch Models

Hardcoded frozenset `_ANTHROPIC_BATCH_MODELS`:
- `claude-3-7-sonnet-20250219`
- `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`
- `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`

Also reads `capabilities.batch.supported` from the API response when available.

### Capability Extraction

`_capability_supported(capabilities, key)` handles two API shapes:
1. `{"batch": {"supported": true}}` — nested dict
2. `{"batch": true}` — flat bool

Extracts `image_input` (vision) and `thinking` (reasoning) capabilities similarly.

### DiscoveredModel Fields

| Field | Value |
|---|---|
| `provider` | `"anthropic"` |
| `display_name` | From API |
| `context_window` | `max_input_tokens` |
| `max_output_tokens` | `max_tokens` |
| `supports_tools` | Always `True` |
| `supports_streaming` | Always `True` |
| `supports_vision` | From `capabilities.image_input` |
| `supports_reasoning` | From `capabilities.thinking` |
| `supports_batch` | From API capabilities or frozenset fallback |
| `provider_metadata.capabilities` | Raw capabilities dict |

---

## Google Gemini (`GoogleAdapter`)

**Entry-point key:** `google`
**Module:** `tenet_llm_adapters._google`
**Transport:** `httpx` direct REST (no vendor SDK)

### Authentication

| Parameter | Source | Notes |
|---|---|---|
| `api_key` | `BackendConfig.api_key` | Google AI API key |
| `base_url` | `BackendConfig.base_url` | Override; defaults to `https://generativelanguage.googleapis.com/v1beta` |

API key is passed as `?key=` query parameter on every request.

### API Surface

| Method | Endpoint | Description |
|---|---|---|
| `list_models()` | `GET /v1beta/models` | Lists models with generation methods |
| `generate()` | `POST /v1beta/models/{model}:generateContent` | Non-streaming |
| `stream()` | `POST /v1beta/models/{model}:streamGenerateContent?alt=sse` | SSE streaming |

### Message Mapping

| TenetCore Role | Gemini Mapping |
|---|---|
| `system` | `systemInstruction.parts[].text` |
| `user` | `role: "user"`, `parts[].text` |
| `assistant` | `role: "model"`, `parts[].text` + `parts[].functionCall` |
| `tool` | `role: "user"`, `parts[].functionResponse` |

### Tool Calling

Tools are sent as `tools[].functionDeclarations[]`. Tool calls in responses are `functionCall` parts. Tool call IDs are synthesized as `call_{function_name}_{index}` to avoid collisions when the same function is called multiple times in one response.

### Streaming Notes

- `stream()` emits incremental `LLMChunk(delta=...)` chunks for content.
- A terminal chunk is always emitted with `stop_reason`, `input_tokens`, `output_tokens`, and `request_id` (if provided by Google).
- `generate()` and `stream()` accept `LLMParams` and map to Gemini generation config.

### DiscoveredModel Fields

| Field | Value |
|---|---|
| `provider` | `"google"` |
| `display_name` | `displayName` from API |
| `context_window` | `inputTokenLimit` |
| `max_output_tokens` | `outputTokenLimit` |
| `supports_tools` | `"generateContent" in supportedGenerationMethods` |
| `supports_streaming` | `"streamGenerateContent" in supportedGenerationMethods` |
| `supports_batch` | Always `False` |

### Timeouts

- `list_models()`: 10s
- `generate()` / `stream()`: 120s

---

## Cohere (`CohereAdapter`)

**Entry-point key:** `cohere`
**Module:** `tenet_llm_adapters._cohere`
**Transport:** `httpx` direct REST (no vendor SDK)

### Authentication

| Parameter | Source | Notes |
|---|---|---|
| `api_key` | `BackendConfig.api_key` | Cohere API key |
| `base_url` | `BackendConfig.base_url` | Override; defaults to `https://api.cohere.com` |

API key is sent as `Authorization: Bearer {api_key}` header.

### API Surface

| Method | Endpoint | Description |
|---|---|---|
| `list_models()` | `GET /v2/models` | Lists models with endpoint capabilities |
| `generate()` | `POST /v2/chat` | Non-streaming v2 chat |
| `stream()` | `POST /v2/chat` (stream=true) | Streaming NDJSON |

### Message Mapping

| TenetCore Role | Cohere v2 Mapping |
|---|---|
| `system` | `role: "system"`, `content` as string |
| `user` | `role: "user"`, `content` as string |
| `assistant` | `role: "assistant"`, `content[]` with `text` and `tool_use` parts |

### Streaming Notes

- `stream()` emits incremental `LLMChunk(delta=...)` chunks from NDJSON `content-delta` events.
- A terminal chunk is always emitted with `stop_reason`, `input_tokens`, `output_tokens`, and `request_id` when present.
- `generate()` and `stream()` accept `LLMParams` and map to `max_tokens`, `temperature`, and `stop_sequences`.
| `tool` | `role: "tool"`, `content[].tool_result` |

### Streaming

Cohere v2 streaming emits NDJSON lines. The adapter filters for `type: "content-delta"` events and extracts `delta.message.content.text`.

### DiscoveredModel Fields

| Field | Value |
|---|---|
| `provider` | `"cohere"` |
| `context_window` | `context_length` |
| `supports_tools` | `"chat" in endpoints` |
| `supports_streaming` | `"chat" in endpoints` |
| `supports_batch` | Always `False` |

### Usage Tracking

Cohere reports usage under `usage.billed_units.{input_tokens, output_tokens}`.

### Timeouts

- `list_models()`: 10s
- `generate()` / `stream()`: 120s
