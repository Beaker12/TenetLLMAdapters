# TenetLLMAdapters Documentation

## Purpose
TenetLLMAdapters provides cloud-provider LLM adapters for Tenet, including Anthropic, OpenAI-compatible, Google, and Cohere providers.

## Architecture and boundaries
- Product code: `src/tenet_llm_adapters/`
- Provider adapters registered via `tenet.llm_adapters` entry points.
- Boundary: adapter layer only; orchestration logic remains in TenetCore.

## APIs/Interfaces (or MCP/CLI surface)
- Package interfaces for adapter invocation and model discovery.
- Entry-point registration consumed by TenetCore provider loading.

## Configuration
- Provider keys/endpoints configured through TenetCore backend config and environment variables.
- Optional extras install provider-specific SDK dependencies.

## Operations and run commands
- Install base package: `pip install tenet-llm-adapters`
- Install extras: `pip install tenet-llm-adapters[anthropic]` (or other provider extras)
- Run tests/lint via repository tooling.

## Testing and verification
- Validate generation and model-list behavior per provider.
- Validate error mapping and timeout behavior consistency.

## Security/governance notes
- Governed by TenetOS repository policies.
- Keep provider credentials externalized via environment/config, not source.

## EGRF reference set (SRS/ARCH/PLAN/VER path strings)
- `../TenetEGRF/docs/requirements/components/SRS_TENETLLMADAPTERS.md`
- `../TenetEGRF/docs/architecture/ARCH_TENETLLMADAPTERS.md`
- `../TenetEGRF/docs/realization/PLAN_TENETLLMADAPTERS.md`
- `../TenetEGRF/docs/verification/VER_TENET_LLM_ADAPTERS.md`
