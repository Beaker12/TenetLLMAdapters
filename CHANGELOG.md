# Changelog

All notable changes to TenetLLMAdapters will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Planned RC target: `1.0.1-rc.1` for the coordinated module metadata rollout.
- Declaration metadata now marks TenetLLMAdapters as a required `core` module,
  matching the plugin inventory and RC gating behavior expected by TenetGUI and
  TenetCore.

## [1.0.0] - 2026-03-30

### Added

- Initial package with Anthropic, OpenAI-compatible, Google, and Cohere cloud adapters.
- Discovery support via `list_models()` on all adapters.
- `tenet.llm_adapters` and `tenet.module_declarations` entry-point registration.
