# TenetLLMAdapters Integration

## Integration contracts
- TenetCore loads adapters from `tenet.llm_adapters` entry points.
- Adapters expose generation and model discovery interfaces.

## Dependencies
- Optional provider SDK dependencies by adapter extra.
- Consumed by TenetCore LLM backend configuration.

## Integration boundaries
- Adapter package does not orchestrate workflows.
- Governance remains external to adapter package runtime.

## Integration verification
- Validate adapter discovery through entry-point loading.
- Validate per-provider generation and listing contract tests.
