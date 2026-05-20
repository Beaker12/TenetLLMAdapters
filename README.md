# TenetLLMAdapters

| Testing | [![CI](https://github.com/Beaker12/TenetLLMAdapters/actions/workflows/ci.yml/badge.svg)](https://github.com/Beaker12/TenetLLMAdapters/actions/workflows/ci.yml) |
|---|---|
| Meta | [![License](https://img.shields.io/badge/license-AGPL--3.0--only-blue)](LICENSE) ![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white) |

Cloud LLM adapter integrations for the Tenet platform. Provides Anthropic, OpenAI-compatible, Google Gemini, and Cohere adapters, all registered via the `tenet.llm_adapters` entry-point group and implementing the TenetCore `LLMAdapter` and `LLMDiscovery` protocols.

> Part of the [Tenet Platform](https://tenet.tools)


## Platform Context

- **Requirements:** [SRS_TENETLLMADAPTERS](../TenetEGRF/docs/requirements/components/SRS_TENETLLMADAPTERS.md)
- **Architecture:** [ARCH_TENETLLMADAPTERS](../TenetEGRF/docs/architecture/ARCH_TENETLLMADAPTERS.md)
- **Realization:** [PLAN_TENETLLMADAPTERS](../TenetEGRF/docs/realization/PLAN_TENETLLMADAPTERS.md)
- **Verification:** [VER_TENETLLMADAPTERS](../TenetEGRF/docs/verification/VER_TENETLLMADAPTERS.md)
- **Governance:** [TenetOS](../TenetOS/README.md)

---

## Architecture

Four provider adapters under `src/tenet_llm_adapters/`. All are loaded lazily at import time via `__getattr__` so that a missing optional SDK never breaks the import.

| Adapter | Entry-point key | Transport | Provider API |
|---------|-----------------|-----------|-------------|
| `AnthropicAdapter` | `anthropic` | `anthropic` SDK | Anthropic Messages API (`/v1/messages`) |
| `OpenAIAdapter` | `openai-compatible` | `openai` SDK | OpenAI Chat Completions (`/v1/chat/completions`) |
| `GoogleAdapter` | `google` | `httpx` REST | Google Generative Language (`/v1beta/models/{model}:generateContent`) |
| `CohereAdapter` | `cohere` | `httpx` REST | Cohere v2 Chat (`/v2/chat`) |

Each adapter implements:
- `generate(messages, model, *, tools, temperature, stop_sequences, params)` → `LLMResponse`
- `stream(messages, model, *, tools, temperature, stop_sequences, params)` → `AsyncIterator[LLMChunk]`
- `list_models()` → `list[DiscoveredModel]`
- `from_config(config: dict)` — class method for factory instantiation

`router.py` contains `MLRouterClient`, a helper that classifies a query against an optional classifier service to select a cost tier (`low`, `medium`, or `high`). It is not an adapter itself; it is a routing aid for callers that support tiered model selection.

### Anthropic-specific capabilities

- Resolves `max_tokens` from the TenetCore model registry; falls back to 4096.
- Propagates `anthropic-beta` headers for extended-thinking models (`claude-opus-4`, `claude-sonnet-4`, and variants) using the `interleaved-thinking-2025-05-14` beta.
- SOCKS proxy guard: when a SOCKS proxy env var is set but `socksio` is not installed, the adapter disables environment proxy loading to avoid httpx destructor warnings.
- Automatic streaming fallback: if the Anthropic API returns the long-request error requiring streaming, `generate()` retries via `_generate_via_streaming_api()`.
- `count_tokens()` calls the Anthropic Token Counting API server-side.

### OpenAI-specific capabilities

- Supports `reasoning_effort` for o1/o3 models via `LLMParams.reasoning`.
- `thinking_tags` parameter on `stream()` parses inline reasoning tags (e.g. `<think>...</think>` for Qwen3/DeepSeek-R1) out of the text stream and emits them as `LLMChunk.thinking_delta`.
- System message normalization: merges and hoists all system messages to index 0 for llama.cpp-compatible endpoints.
- Graceful JSON decode error recovery: if the stream ends with an invalid SSE frame after partial output, the error is logged and already-collected chunks are returned.

---

## Installation

```bash
pip install tenet-llm-adapters                    # base package only (no provider SDKs)
pip install tenet-llm-adapters[anthropic]         # + anthropic SDK
pip install tenet-llm-adapters[openai]            # + openai SDK
pip install tenet-llm-adapters[google,cohere]     # Google and Cohere use httpx (already included)
pip install tenet-llm-adapters[all]               # anthropic + openai
```

The `google` and `cohere` extras install no additional packages because both adapters use `httpx`, which is a base dependency.

---

## Quick Start

```python
from tenet_llm_adapters import AnthropicAdapter

adapter = AnthropicAdapter(api_key="sk-ant-...")
response = await adapter.generate(
    messages=[{"role": "user", "content": "Hello"}],
    model="claude-sonnet-4-6",
)
print(response.content)

# Streaming
async for chunk in adapter.stream(messages=[...], model="claude-sonnet-4-6"):
    if chunk.delta:
        print(chunk.delta, end="", flush=True)
```

```python
from tenet_llm_adapters import OpenAIAdapter

adapter = OpenAIAdapter(api_key="sk-...", base_url="http://localhost:11434/v1")
models = await adapter.list_models()
```

---

## Configuration

### Adapter Constructor / Config Dict

All adapters accept `api_key` and optional `base_url` in their constructor, or via `from_config(config)` where `config` is a `BackendConfig.model_dump()` dict.

| Field | Type | Description |
|-------|------|-------------|
| `api_key` | `str` | Provider API key. Required for cloud providers. |
| `base_url` | `str \| None` | Override default API endpoint. `None` uses the provider default. |

### Gateway Routing — Module Tunable

Workspace-level tunable `llm.gateway_url` controls routing through TenetLLMGateway:

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `llm.gateway_url` | `str` | `""` | When set, all LLM adapter requests route through this TenetLLMGateway URL. Overrides `TENET_LLM_GATEWAY_URL` env var. Empty string disables gateway routing. |

When `llm.gateway_url` is configured, all adapters call `_resolve_base_url()` which checks (in order): `TENET_LLM_GATEWAY_URL` env var (highest priority), manifest `base_url`, provider default. If the env var is set, it overrides the tunable, allowing runtime gateway override without redeployment.

Per-provider defaults:

| Provider | Default endpoint |
|----------|-----------------|
| OpenAI | OpenAI SDK default |
| Anthropic | Anthropic SDK default |
| Google | `https://generativelanguage.googleapis.com/v1beta` |
| Cohere | `https://api.cohere.com` |

---

## Public API

| Export | Description |
|--------|-------------|
| `AnthropicAdapter` | Anthropic (Claude) adapter |
| `OpenAIAdapter` | OpenAI-compatible adapter |
| `GoogleAdapter` | Google Gemini adapter |
| `CohereAdapter` | Cohere adapter |

All four are importable from `tenet_llm_adapters` directly. Imports are lazy — the provider SDK is only loaded when the class is first accessed.

---

## Project Layout

```
src/tenet_llm_adapters/
├── __init__.py          # Lazy imports, __all__
├── _anthropic.py        # AnthropicAdapter — Anthropic Messages API
├── _openai.py           # OpenAIAdapter — OpenAI-compatible Chat Completions
├── _google.py           # GoogleAdapter — Google Generative Language REST
├── _cohere.py           # CohereAdapter — Cohere v2 Chat REST
├── router.py            # MLRouterClient — classifier-based cost-tier selection
└── declaration.py       # Module declaration (module_category="core", required=True)
tests/
docs/
  architecture.md
  configuration.md
  providers.md
  integration.md
  gaps-and-drift.md
  source-tree.md
```

---

## Testing

```bash
pytest tests/
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Uses [Conventional Commits](https://www.conventionalcommits.org/), [SemVer](https://semver.org/), [Keep a Changelog](https://keepachangelog.com/).

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

---

## License

Tenet Platform is licensed under a **dual-license model**:

| Tier | License | Cost |
|------|---------|------|
| **Open Source** | [AGPL-3.0](LICENSE) | Free — must share modifications if deployed as a network service |
| **Commercial** | Commercial License | Paid subscription — no AGPL obligations |

See [LICENSING.md](LICENSING.md) for full terms.

Copyright (C) 2026 Stuart W. Parkhurst
