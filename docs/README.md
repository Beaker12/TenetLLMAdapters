# TenetLLMAdapters

Cloud LLM provider adapters for the Tenet platform. Provides a uniform adapter interface over four cloud API providers, registered as `tenet.llm_adapters` entry points and consumed by TenetCore's LLM backend layer.

## Providers

| Provider | Entry-Point Key | Adapter Class | SDK / Transport | Default Endpoint |
|---|---|---|---|---|
| OpenAI (+ compatible) | `openai-compatible` | `OpenAIAdapter` | `openai` SDK (async) | OpenAI default |
| Anthropic | `anthropic` | `AnthropicAdapter` | `anthropic` SDK (async) | Anthropic default |
| Google Gemini | `google` | `GoogleAdapter` | `httpx` REST | `https://generativelanguage.googleapis.com/v1beta` |
| Cohere | `cohere` | `CohereAdapter` | `httpx` REST | `https://api.cohere.com` |

## Capabilities per Provider

| Capability | OpenAI | Anthropic | Google | Cohere |
|---|---|---|---|---|
| `generate()` | Yes | Yes | Yes | Yes |
| `stream()` | Yes | Yes | Yes | Yes |
| `list_models()` | Yes | Yes | Yes | Yes |
| `count_tokens()` | — | Yes | — | — |
| Tool calling | Yes | Yes | Yes | Yes |
| Batch model detection | Yes | Yes | — | — |
| Vision metadata | — | Yes | — | — |
| Reasoning metadata | — | Yes | — | — |

## Architecture

```
TenetCore
  │
  │  tenet.llm_adapters entry points
  ▼
┌──────────────────────────────────────────────────┐
│              TenetLLMAdapters                     │
│                                                   │
│  ┌──────────────┐  ┌────────────────┐            │
│  │ OpenAIAdapter│  │AnthropicAdapter│            │
│  │ (openai SDK) │  │(anthropic SDK) │            │
│  └──────────────┘  └────────────────┘            │
│  ┌──────────────┐  ┌────────────────┐            │
│  │GoogleAdapter │  │ CohereAdapter  │            │
│  │ (httpx REST) │  │ (httpx REST)   │            │
│  └──────────────┘  └────────────────┘            │
│                                                   │
│  Common types: Message, LLMResponse, LLMChunk,   │
│  ToolCall, ToolDef (from tenet_core.llm.client)  │
└──────────────────────────────────────────────────┘
```

All adapters share a common interface contract:
- `__init__(api_key, *, base_url=None)` — constructor
- `from_config(config: dict) -> Self` — factory from `BackendConfig.model_dump()`
- `generate(messages, model, *, tools, max_tokens, temperature, stop_sequences) -> LLMResponse`
- `stream(messages, model, ...) -> AsyncIterator[LLMChunk]`
- `list_models() -> list[DiscoveredModel]`

## Installation

```bash
# Base (no provider SDKs)
pip install tenet-llm-adapters

# Single provider
pip install tenet-llm-adapters[openai]
pip install tenet-llm-adapters[anthropic]
pip install tenet-llm-adapters[google]
pip install tenet-llm-adapters[cohere]

# All provider SDKs
pip install tenet-llm-adapters[all]
```

## Dependencies

| Dependency | Version | Scope |
|---|---|---|
| `tenet-core` | `>=1.0.0,<2.0` | Required |
| `httpx` | `>=0.27,<1.0` | Required (Google, Cohere transport + base) |
| `anthropic` | `>=0.40,<1.0` | Extra: `anthropic` |
| `openai` | `>=1.50,<2.0` | Extra: `openai` |

Google and Cohere adapters use `httpx` directly (no vendor SDK).

## Module Declaration

Registered via `tenet.module_declarations` entry point:
- **module_id:** `tenet_llm_adapters`
- **module_version:** `1.1.0`
- **module_category:** `core`
- **required:** `true`
- **tunables:** `llm.gateway_url` (str, default `""`) — base URL of a TenetLLMGateway instance; when set, the gateway adapter is used for all providers

## EGRF References

| Layer | Document |
|---|---|
| Requirements | `../TenetEGRF/docs/requirements/components/SRS_TENETLLMADAPTERS.md` |
| Architecture | `../TenetEGRF/docs/architecture/ARCH_TENETLLMADAPTERS.md` |
| Realization | `../TenetEGRF/docs/realization/PLAN_TENETLLMADAPTERS.md` |
| Verification | `../TenetEGRF/docs/verification/VER_TENET_LLM_ADAPTERS.md` |

## License

AGPL-3.0-only
