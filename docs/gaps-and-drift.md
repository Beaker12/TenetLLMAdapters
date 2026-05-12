# Gaps and Drift — TenetLLMAdapters

Last audited: 2026-05-12

## Resolved gaps (fixed in current code)

The following issues identified in earlier audits are resolved in the current source.

### Cohere streaming sentinel chunk
`CohereAdapter.stream()` now yields a terminal `LLMChunk` with `stop_reason`,
`input_tokens`, `output_tokens`, and `request_id` after consuming all NDJSON
events. Token counts are sourced from `usage.billed_units` in the final event.

### Google streaming sentinel chunk
`GoogleAdapter.stream()` now yields a terminal `LLMChunk` after the SSE stream
completes, carrying `stop_reason`, `input_tokens` (`promptTokenCount`), and
`output_tokens` (`candidatesTokenCount`).

### Google tool call ID uniqueness
`GoogleAdapter._parse_response()` now uses a `tool_call_index` counter, producing
IDs in the form `call_{function_name}_{index}`. Multiple calls to the same function
in a single response receive unique IDs.

### OpenAI `_stream_impl` default `max_tokens` dead code
`OpenAIAdapter` now resolves `max_tokens` through `LLMParams` / `resolve_params`
rather than a per-method default. Both `generate()` and `stream()` delegate to
`resolve_params(params, ...)`, and `_resolve_max_tokens(model)` is used for the
Anthropic adapter. The OpenAI streaming path no longer has a `16384` default.

---

## Open gaps

### 1. No abstract base class or Protocol enforcement

The adapter contract is structural only. No ABC or `Protocol` class is defined in
this package or in TenetCore's public interface. New adapters can silently omit
methods; the failure only appears at call time.

**Impact:** Medium. New adapters could partially implement the interface.
**Fix:** Define a `typing.Protocol` class in `tenet_core.llm` or in this package
and annotate all adapter classes as implementing it.

### 2. `all` extra excludes Google and Cohere semantically

`pyproject.toml` defines `all = ["tenet-llm-adapters[anthropic]", "tenet-llm-adapters[openai]"]`.
Google and Cohere require only `httpx` (already a base dependency), so the extra
is technically correct, but users may expect `[all]` to include something for
Google and Cohere providers.

**Impact:** Low. All four adapters are functional regardless.
**Fix:** Add a comment in `pyproject.toml` explaining why `[all]` omits these, or
add them as explicit (no-op) extras for documentation clarity.

### 3. No retry or circuit breaker

Adapters propagate upstream errors directly to the caller. Retry logic belongs in
TenetCore, but if TenetCore does not implement retries, there is no resilience layer.

**Impact:** Depends on TenetCore implementation.
**Status:** By design (R-LLM-022: no retry in adapter). Track at TenetCore level.

### 4. Batch model frozensets require manual updates

`_OPENAI_BATCH_MODELS` and `_ANTHROPIC_BATCH_MODELS` are hardcoded. Anthropic
has API-driven detection as primary with the frozenset as fallback, but OpenAI
batch support is frozenset-only.

**Impact:** Low. Incorrect `supports_batch` on newly released models until the
frozenset is updated.

### 5. `MLRouterClient` has no tests or documentation

`router.py` provides `MLRouterClient`, a cost-tier classifier client. It is not
registered as an entry point and has no dedicated tests or doc references.

**Impact:** Low. The module is a utility for TenetCore callers; it defaults safely
to `"medium"` when the classifier service is unavailable.
**Fix:** Add unit tests for `predict_tier()` covering the endpoint-unavailable
fallback and the label-to-tier mapping.

---

## Documentation vs code drift

| Document | Status | Notes |
|---|---|---|
| `docs/gaps-and-drift.md` | Updated 2026-05-11 | This file |
| `docs/architecture.md` | Current | Reflects lazy import, SDK vs REST split, streaming sentinel, thinking support |
| `docs/configuration.md` | Current | Reflects `LLMParams` generation parameters and HTTP timeouts |
| `docs/providers.md` | Current | Per-provider API surface, batch models, capability extraction |
| `SRS_TENETLLMADAPTERS.md` | Current | Requirements verified against implementation |
| `ARCH_TENETLLMADAPTERS.md` | Current | Architecture decisions aligned |
