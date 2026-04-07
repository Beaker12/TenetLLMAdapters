# TenetLLMAdapters Operations Runbook

## Core operations
- Install required provider extras for target deployment.
- Validate provider connectivity and model discovery behavior.

## Routine checks
- Validate provider API compatibility after SDK updates.
- Validate retry/error handling under rate limits and transient faults.

## Incident response
- Auth failures: verify provider API key and endpoint configuration.
- Unsupported model errors: validate provider model inventory and adapter mapping.

## Verification commands
- `pytest tests/`
- `make lint` (if available)
