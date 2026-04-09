# Integration — TenetLLMAdapters

```{graphviz}
digraph TenetLLMAdaptersIntegration {
  graph [fontname="Arial" fontsize=10 rankdir=LR nodesep=0.3 ranksep=0.5 bgcolor=white]
  node  [shape=box style="rounded,filled" fontname="Arial" fontsize=10 penwidth=1 color="#4a5568"]
  edge  [fontname="Arial" fontsize=9 color="#4a5568" arrowsize=0.7]

  Manifest [label="Instance\nManifest" fillcolor="#feebc8"]
  BackendCfg [label="BackendConfig\n(provider, api_key,\nbase_url, model)" fillcolor="#fff7e6"]
  Registry [label="adapter_registry\nentry_points" fillcolor="#edf2f7"]
  Factory [label="from_config()" fillcolor="#fff7e6"]

  subgraph cluster_adapters {
    label="Adapter Instances" style=solid color="#63b3ed" fontname="Arial" fontsize=11 bgcolor=white
    OpenAI [label="OpenAI\nAdapter" fillcolor="#fff7e6"]
    Anthropic [label="Anthropic\nAdapter" fillcolor="#fff7e6"]
    Google [label="Google\nAdapter" fillcolor="#fff7e6"]
    Cohere [label="Cohere\nAdapter" fillcolor="#fff7e6"]
  }

  subgraph cluster_apis {
    label="LLM Provider APIs" style=solid color="#63b3ed" fontname="Arial" fontsize=11 bgcolor=white
    OpenAIAPI [label="OpenAI API" fillcolor="#f3e8ff"]
    AnthropicAPI [label="Anthropic API" fillcolor="#f3e8ff"]
    GoogleAPI [label="Google Gemini" fillcolor="#f3e8ff"]
    CohereAPI [label="Cohere API" fillcolor="#f3e8ff"]
  }

  Response [label="LLMResponse\nLLMChunk" fillcolor="#c6f6d5"]

  Manifest -> BackendCfg [label="extract"]
  BackendCfg -> Registry [label="provider key"]
  Registry -> Factory [label="class lookup"]
  Factory -> OpenAI
  Factory -> Anthropic
  Factory -> Google
  Factory -> Cohere

  OpenAI -> OpenAIAPI [label="generate/stream"]
  Anthropic -> AnthropicAPI
  Google -> GoogleAPI
  Cohere -> CohereAPI

  OpenAIAPI -> Response
  AnthropicAPI -> Response
  GoogleAPI -> Response
  CohereAPI -> Response
}
```

## How TenetCore Discovers Adapters

TenetCore loads adapters at startup via Python entry points:

```
importlib.metadata.entry_points(group="tenet.llm_adapters")
```

Each entry point maps a provider key to an adapter class:

| Key | Class | Package |
|---|---|---|
| `openai-compatible` | `OpenAIAdapter` | `tenet-llm-adapters` |
| `anthropic` | `AnthropicAdapter` | `tenet-llm-adapters` |
| `google` | `GoogleAdapter` | `tenet-llm-adapters` |
| `cohere` | `CohereAdapter` | `tenet-llm-adapters` |

The same `tenet.llm_adapters` group is shared with `tenet-llm-local`, which registers `ollama`, `lmstudio`, and `local` keys.

## Instantiation Flow

```
Instance Manifest
    └── BackendConfig (provider, api_key, base_url, model)
         └── adapter_registry.py
              └── entry_points["tenet.llm_adapters"][provider]
                   └── AdapterClass.from_config(config.model_dump())
                        └── adapter instance ready for generate/stream
```

TenetCore calls `from_config(config)` with a dict containing at minimum `api_key` and optionally `base_url`. The adapter is then used for all `generate()` and `stream()` calls on that backend.

## Adapter Interface Contract

Every adapter must implement:

```python
class SomeAdapter:
    def __init__(self, api_key: str, *, base_url: str | None = None) -> None: ...

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SomeAdapter": ...

    async def list_models(self) -> list[DiscoveredModel]: ...

    async def generate(
        self,
        messages: list[Message],
        model: str,
        *,
        tools: list[ToolDef] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse: ...

    async def stream(
        self,
        messages: list[Message],
        model: str,
        *,
        tools: list[ToolDef] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[LLMChunk]: ...
```

Optional additions (not required by TenetCore):
- `count_tokens(messages, model, *, tools) -> int` — Anthropic only

## Shared Types (from `tenet_core.llm.client`)

| Type | Role |
|---|---|
| `Message` | Input message with `role`, `content`, `tool_calls`, `tool_call_id`, `name` |
| `ToolDef` | Tool definition with `name`, `description`, `parameters` (JSON Schema) |
| `ToolCall` | Tool invocation with `id`, `name`, `arguments` (dict) |
| `LLMResponse` | Complete response with `content`, `tool_calls`, `model`, token counts, `stop_reason`, `request_id` |
| `LLMChunk` | Streaming delta with `delta`, `stop_reason`, token counts, `request_id` |
| `DiscoveredModel` | Model metadata with `model_id`, `provider`, capabilities, architecture info |

## Module Declaration Integration

`tenet.module_declarations` entry point registers the module with TenetCore:

```python
ModuleDeclaration(
    module_id="tenet_llm_adapters",
    module_version="1.0.0",
    module_category="core",
    required=True,
    tunables=[],
)
```

This allows TenetCore to verify the module is present and report its version.

## Boundary Rules

- Adapters perform **only** API translation — no orchestration, caching, retry, or routing.
- Retry logic belongs in TenetCore.
- Model selection and fallback belong in TenetCore's backend configuration.
- Governance checks are external to adapters.

## Integration with TenetLLMLocal

Both packages register under the same `tenet.llm_adapters` entry-point group. TenetCore treats all adapters uniformly regardless of which package provides them. Key distinction:
- **TenetLLMAdapters** — cloud providers with no URL restrictions
- **TenetLLMLocal** — local providers with SSRF guard enforcement

## Integration Verification

| Test | Validates |
|---|---|
| Entry-point load test | `tenet.llm_adapters` keys resolve to correct classes |
| `from_config()` test | Factory creates usable adapter from config dict |
| `generate()` contract test | Returns `LLMResponse` with expected fields |
| `stream()` contract test | Yields `LLMChunk` sequence ending with sentinel |
| `list_models()` contract test | Returns `list[DiscoveredModel]` with valid fields |
| Tool calling round-trip | Tools sent → tool calls parsed back correctly |
