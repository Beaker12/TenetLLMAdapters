# MODULE_OPERATIONS_PLAYBOOK — TenetLLMAdapters

```{graphviz}
digraph TenetLLMAdaptersOps {
  graph [fontname="Arial" fontsize=10 rankdir=LR nodesep=0.3 ranksep=0.5 bgcolor=white]
  node  [shape=box style="rounded,filled" fontname="Arial" fontsize=10 penwidth=1 color="#4a5568"]
  edge  [fontname="Arial" fontsize=9 color="#4a5568" arrowsize=0.7]

  Operator [label="Operator" fillcolor="#e8f0fe" shape=ellipse]

  subgraph cluster_checks {
    label="Routine Checks" style=solid color="#63b3ed" fontname="Arial" fontsize=11 bgcolor=white
    Tests [label="pytest tests/" fillcolor="#c6f6d5"]
    EntryPt [label="Entry-point\nregistration" fillcolor="#edf2f7"]
    ModelList [label="list_models()\ncorrectness" fillcolor="#edf2f7"]
    Lint [label="ruff check\n+ format" fillcolor="#f7f9fc"]
  }

  subgraph cluster_incidents {
    label="Incident Response" style=solid color="#63b3ed" fontname="Arial" fontsize=11 bgcolor=white
    Auth [label="Auth Failure\n401/403" fillcolor="#fed7d7"]
    RateLimit [label="Rate Limit\n429" fillcolor="#fed7d7"]
    NotFound [label="Model Not\nFound 404" fillcolor="#fed7d7"]
    ImportErr [label="SDK Import\nError" fillcolor="#fed7d7"]
  }

  ProviderAPI [label="Provider\nAPIs" fillcolor="#f3e8ff"]

  Operator -> Tests [label="verify"]
  Operator -> EntryPt [label="validate"]
  Operator -> ModelList [label="weekly"]
  Operator -> Lint [label="quality"]
  Tests -> ProviderAPI [label="mock calls"]
  ProviderAPI -> Auth [label="failure"]
  ProviderAPI -> RateLimit [label="throttled"]
  ProviderAPI -> NotFound [label="missing"]
}
```

## Installation

```bash
# Editable dev install with all extras
pip install -e ".[all,dev]"

# Production: install only needed providers
pip install tenet-llm-adapters[anthropic]
pip install tenet-llm-adapters[openai]
```

## Verification

```bash
# Run all tests
pytest tests/

# Run tests for a single provider
pytest tests/test_openai.py
pytest tests/test_anthropic.py
pytest tests/test_google.py
pytest tests/test_cohere.py

# Lint
ruff check src/ tests/
ruff format --check src/ tests/
```

## Routine Checks

| Check | Command / Action | Frequency |
|---|---|---|
| Provider SDK compatibility | Run test suite after SDK version bumps | On dependency update |
| Entry-point registration | `python -c "from importlib.metadata import entry_points; print([e.name for e in entry_points(group='tenet.llm_adapters')])"` | On release |
| Model list correctness | Call `list_models()` against each provider API | Weekly / on new model announcement |
| Batch model frozenset currency | Compare `_OPENAI_BATCH_MODELS` and `_ANTHROPIC_BATCH_MODELS` against provider docs | Quarterly |

## Incident Response

### Auth Failure (401/403)

1. Verify API key in `BackendConfig` or environment variable.
2. Check key permissions / expiry with provider dashboard.
3. If using `base_url` override, verify endpoint accepts the key.

### Rate Limit (429)

1. TenetLLMAdapters does not implement retry logic; retries are TenetCore's responsibility.
2. Check provider rate limit headers in the exception.
3. Scale down concurrent agent sessions or upgrade provider plan.

### Model Not Found (404)

1. Verify model ID string matches provider's current model inventory.
2. Run `list_models()` to see currently available models.
3. For OpenAI-compatible endpoints, verify the server has the model loaded.

### Empty Response / No Choices

- `OpenAIAdapter._parse_response()` raises `RuntimeError` with response details if no choices returned.
- Check model availability and request parameters.

### SDK Import Error

- If adapter class access raises `ImportError`, the corresponding extra is not installed.
- Lazy imports in `__init__.py` delay the error until the adapter class is actually used.

## Build & Release

```bash
# Build wheel
python -m build

# Publish (requires PyPI credentials)
twine upload dist/*
```

## Dependency Matrix

| Dependency | Pinned Range | Notes |
|---|---|---|
| `tenet-core` | `>=1.0.0,<2.0` | Core types and declarations |
| `httpx` | `>=0.27,<1.0` | Google + Cohere transport |
| `anthropic` | `>=0.40,<1.0` | Extra: `anthropic` |
| `openai` | `>=1.50,<2.0` | Extra: `openai` |
| `pytest` | `>=8.0,<9.0` | Dev only |
| `pytest-asyncio` | `>=0.23,<1.0` | Dev only |
| `respx` | `>=0.21,<1.0` | Dev only (httpx mock) |
| `ruff` | `>=0.3,<1.0` | Dev only |
