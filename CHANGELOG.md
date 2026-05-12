# Changelog

All notable changes to TenetLLMAdapters will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **LLM gateway routing support** (`_gateway.py` new file, `_anthropic.py`, `_openai.py`, `_google.py`, `_cohere.py`):
  - New `_resolve_base_url(provider_default, config_override) -> str` helper checks `TENET_LLM_GATEWAY_URL` env var first (highest priority), then manifest `base_url`, then provider default
  - All four cloud adapters (Anthropic, OpenAI, Google, Cohere) now call `_resolve_base_url()` before constructing SDK clients
  - When `TENET_LLM_GATEWAY_URL` is set, all cloud traffic routes through the gateway transparently
  - Enables workspace-level and platform-level LLM request routing through proxy/gateway infrastructure

- **Workspace-level gateway URL configuration** (`declaration.py`):
  - New `llm.gateway_url` tunable for workspace-level gateway URL override without env var injection
  - Allows per-workspace gateway routing without affecting other workspaces sharing the same TenetGUI instance

- **Anthropic model beta headers** (`_anthropic.py`):
  - New `_MODEL_BETAS` dict mapping model prefixes to anthropic-beta header values (interleaved thinking beta)
  - New `get_model_betas(model_id: str) -> list[str]` helper that returns beta header values for a given model
  - Both `messages.create()` and `messages.stream()` now pass resolved betas via `betas=` kwarg when non-empty
  - Enables early access to extended thinking, agent framework, and other beta features per model

- **Anthropic adapter max_tokens resolution** (`_anthropic.py`):
  - New `_resolve_max_tokens(model: str) -> int` helper function that queries TenetCore's model registry for `max_output_tokens` capability with safe fallback to 4096 on registry miss or exception
  - Both `messages.create()` (async) and `messages.stream()` (async streaming) now explicitly pass `max_tokens` resolved from model capabilities
  - Improves reliability for models with strict output token limits and prevents `BadRequestError` on oversized responses
  - Default 4096-token limit applies when registry unavailable or model lacks declared maximum

### Removed

- **Legacy OpenAI batch model support** (`_openai.py`) — `gpt-3.5-turbo`, `gpt-3.5-turbo-0125`, `gpt-4-turbo`, and `gpt-4-turbo-preview` removed from `_OPENAI_BATCH_MODELS`; batch routing now targets current model generations only.

### Changed

- **LLM gateway URL resolution consolidation** (`_gateway.py`, `_anthropic.py`, `_cohere.py`, `_google.py`, `_openai.py`):
  - Consolidated `_resolve_base_url()` logic into standalone `_gateway.py` module for DRY principle
  - All four cloud adapters now import and use shared gateway resolver
  - Simplifies future gateway routing enhancements across all provider backends

- **Anthropic API logging** (`_anthropic.py`): Debug logs now include resolved `max_tokens` value for both `messages.create()` and `messages.stream()` calls

## [1.1.0] — 2026-05-12

### Changed
- Version aligned to implementation audit: all R-CLOUD-001–010 and R-DECL-1 requirements met, plus mature extensions beyond spec (gateway routing, thinking-delta streaming, SOCKS proxy guard, inline thinking-tag parsing, LLMParams passthrough, reasoning_effort support).
## [1.0.0] - 2026-03-30

### Added

- Initial package with Anthropic, OpenAI-compatible, Google, and Cohere cloud adapters.
- Discovery support via `list_models()` on all adapters.
- `tenet.llm_adapters` and `tenet.module_declarations` entry-point registration.
