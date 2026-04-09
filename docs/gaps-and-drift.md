# Gaps and Drift — TenetLLMAdapters

## Observed Inconsistencies

### 1. Streaming `max_tokens` Mismatch (OpenAI)

`OpenAIAdapter.generate()` defaults to `max_tokens=4096`, but `_stream_impl()` defaults to `max_tokens=16384`. The public `stream()` method passes the caller's `max_tokens` (default `4096`) through to `_stream_impl()`, which then shadows it with its own `16384` default. This is a latent bug — the caller's explicit `max_tokens` **does** get forwarded, but if `stream()` is called without `max_tokens`, it receives `4096` from `stream()`'s signature, not `16384` from `_stream_impl()`. The `_stream_impl` default of `16384` is dead code.

**Impact:** Low. The default `4096` from `stream()` always wins.
**Fix:** Align `_stream_impl` default to `4096` or remove the redundant default.

### 2. No Abstract Base Class

The adapter contract is structural only. No ABC or Protocol class is defined. This makes it impossible to statically verify adapter compliance.

**Impact:** Medium. New adapters could silently omit methods.
**Fix:** Define a `Protocol` class in `tenet_core.llm` or in this package.

### 3. Google Tool Call ID Synthesis

`GoogleAdapter._parse_response()` generates tool call IDs as `f"call_{fc['name']}"`. If a response contains multiple calls to the same function, they get identical IDs, which breaks TenetCore's tool result routing.

**Impact:** Medium for multi-tool-call scenarios with duplicate function names.
**Fix:** Use a counter or UUID suffix.

### 4. `all` Extra Excludes Google and Cohere

`pyproject.toml` defines `all = ["tenet-llm-adapters[anthropic]", "tenet-llm-adapters[openai]"]`. Since Google and Cohere only need `httpx` (already a base dependency), this is technically correct, but semantically misleading.

**Impact:** Low. Users may expect `[all]` to install something for these providers.
**Fix:** Document explicitly or add them to `all` as no-ops.

### 5. No Retry or Circuit Breaker

Adapters do not implement retry logic. This is by design (retries belong in TenetCore), but it means transient failures always propagate. If TenetCore does not implement retries yet, there is no resilience layer.

**Impact:** Depends on TenetCore implementation.

### 6. Cohere Streaming: No Final Usage Chunk

`CohereAdapter.stream()` yields `content-delta` events but never emits a final `LLMChunk` with `stop_reason` and token counts. The stream ends without a sentinel chunk.

**Impact:** Medium. TenetCore may expect a sentinel chunk with usage data.
**Fix:** Parse `message-end` event type from the Cohere stream for usage + stop reason, then yield a final sentinel.

### 7. Google Streaming: No Final Usage Chunk

`GoogleAdapter.stream()` yields text deltas but never emits a final sentinel `LLMChunk` with `stop_reason` and usage metadata.

**Impact:** Medium. Same as Cohere — missing sentinel.
**Fix:** Track the last chunk's `usageMetadata` and `finishReason`, then yield a final sentinel.

### 8. Batch Model Staleness

Both `_OPENAI_BATCH_MODELS` and `_ANTHROPIC_BATCH_MODELS` are hardcoded frozensets. They require manual updates when providers add new batch-capable models.

**Impact:** Low. Anthropic has API-driven detection as primary, frozenset as fallback.

## Documentation vs Code Drift

| Document | Status | Notes |
|---|---|---|
| Previous `docs/README.md` | Replaced | Was a placeholder with no provider details |
| Previous `docs/operations.md` | Replaced | Was a stub with no actionable commands |
| Previous `docs/integration.md` | Replaced | Was a stub with no contract details |
| `SRS_TENETLLMADAPTERS.md` | Not verified | EGRF requirements doc not read for this audit |
| `ARCH_TENETLLMADAPTERS.md` | Not verified | EGRF architecture doc not read for this audit |
