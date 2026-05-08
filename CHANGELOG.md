# Changelog

All notable changes to TenetLLMAdapters will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

### Changed

- **Anthropic API logging** (`_anthropic.py`): Debug logs now include resolved `max_tokens` value for both `messages.create()` and `messages.stream()` calls, improving transparency into output token budget enforcement per model
- Planned RC target: `1.0.1-rc.1` for the coordinated module metadata rollout.
- Declaration metadata now marks TenetLLMAdapters as a required `core` module, matching the plugin inventory and RC gating behavior expected by TenetGUI and TenetCore.

## [1.0.0] - 2026-03-30

### Added

- Initial package with Anthropic, OpenAI-compatible, Google, and Cohere cloud adapters.
- Discovery support via `list_models()` on all adapters.
- `tenet.llm_adapters` and `tenet.module_declarations` entry-point registration.
