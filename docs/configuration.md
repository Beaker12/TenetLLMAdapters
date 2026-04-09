# Configuration — TenetLLMAdapters

## Overview

TenetLLMAdapters is configured through TenetCore's `BackendConfig` in instance manifests. There are no standalone environment variables or config files specific to this package. All configuration flows through the adapter factory method `from_config(config: dict)`.

```{graphviz}
digraph TenetLLMAdaptersConfig {
  graph [fontname="Arial" fontsize=10 rankdir=LR nodesep=0.3 ranksep=0.5 bgcolor=white]
  node  [shape=box style="rounded,filled" fontname="Arial" fontsize=10 penwidth=1 color="#4a5568"]
  edge  [fontname="Arial" fontsize=9 color="#4a5568" arrowsize=0.7]

  Manifest [label="Instance\nManifest" fillcolor="#feebc8"]

  subgraph cluster_backend {
    label="BackendConfig" style=solid color="#63b3ed" fontname="Arial" fontsize=11 bgcolor=white
    ApiKey [label="api_key" fillcolor="#fed7d7"]
    BaseUrl [label="base_url\n(optional)" fillcolor="#edf2f7"]
  }

  subgraph cluster_gen_params {
    label="Generation Parameters" style=solid color="#63b3ed" fontname="Arial" fontsize=11 bgcolor=white
    Model [label="model\n(e.g. gpt-4o)" fillcolor="#fff7e6"]
    MaxTokens [label="max_tokens\n(4096)" fillcolor="#fff7e6"]
    Temp [label="temperature\n(0.0)" fillcolor="#fff7e6"]
    Tools [label="tools\n(ToolDef[])" fillcolor="#fff7e6"]
  }

  subgraph cluster_extras {
    label="Install Extras" style=solid color="#63b3ed" fontname="Arial" fontsize=11 bgcolor=white
    ExtOpenAI [label="openai\n(openai SDK)" fillcolor="#f7f9fc"]
    ExtAnthropic [label="anthropic\n(anthropic SDK)" fillcolor="#f7f9fc"]
    ExtGoogle [label="google\n(httpx)" fillcolor="#f7f9fc"]
    ExtCohere [label="cohere\n(httpx)" fillcolor="#f7f9fc"]
  }

  Adapter [label="Adapter\nInstance" fillcolor="#c6f6d5"]

  Manifest -> ApiKey
  Manifest -> BaseUrl
  ApiKey -> Adapter [label="auth"]
  BaseUrl -> Adapter [label="endpoint"]
  Model -> Adapter
  MaxTokens -> Adapter
  Temp -> Adapter
  Tools -> Adapter
}
```

## Backend Configuration Fields

Each adapter reads from the `BackendConfig.model_dump()` dictionary:

| Field | Type | Used By | Description |
|---|---|---|---|
| `api_key` | `str` | All adapters | Provider API key. Required for cloud providers. |
| `base_url` | `str \| None` | All adapters | Override default API endpoint. `None` = provider default. |

## Per-Provider Defaults

| Provider | Default Endpoint | Source |
|---|---|---|
| OpenAI | OpenAI SDK default | `openai` library configuration |
| Anthropic | Anthropic SDK default | `anthropic` library configuration |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta` | `_google._BASE` |
| Cohere | `https://api.cohere.com` | `_cohere._BASE` |

## Generation Parameters

All adapters accept these parameters in `generate()` and `stream()`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | — | Model identifier (e.g., `gpt-4o`, `claude-3-5-sonnet-20241022`) |
| `max_tokens` | `int` | `4096` | Maximum output tokens. OpenAI streaming defaults to `16384`. |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `stop_sequences` | `list[str] \| None` | `None` | Stop sequences |
| `tools` | `list[ToolDef] \| None` | `None` | Tool definitions for function calling |

## Module Declaration Tunables

The module declaration (`declaration.py`) declares zero tunables. All runtime behavior is controlled via `BackendConfig` in the instance manifest.

## Install Extras

Install extras control which provider SDKs are available at runtime:

| Extra | Packages Installed |
|---|---|
| `anthropic` | `anthropic>=0.40,<1.0` |
| `openai` | `openai>=1.50,<2.0` |
| `google` | (none — uses `httpx`, already a base dependency) |
| `cohere` | (none — uses `httpx`, already a base dependency) |
| `all` | `anthropic` + `openai` |

The `google` and `cohere` extras resolve to no additional packages since `httpx` is a base dependency.

## Environment Variables

The underlying provider SDKs respect their standard environment variables:

| Variable | SDK | Effect |
|---|---|---|
| `OPENAI_API_KEY` | `openai` | Fallback API key if not provided in config |
| `OPENAI_BASE_URL` | `openai` | Fallback base URL |
| `ANTHROPIC_API_KEY` | `anthropic` | Fallback API key if not provided in config |

These are SDK-level fallbacks. The adapter always passes `api_key` and `base_url` from config explicitly, so config values take precedence.

## HTTP Timeouts

| Operation | Timeout | Providers |
|---|---|---|
| `list_models()` | 10s | Google, Cohere |
| `generate()` | 120s | Google, Cohere |
| `stream()` | 120s | Google, Cohere |

OpenAI and Anthropic timeouts are managed by their respective SDKs with their own defaults.
