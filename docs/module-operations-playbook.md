# TenetLLMAdapters Software Operations and Integration Guide

## Executive Summary

Provider adapter layer for LLM integrations.

This document is an implementation-focused reference for engineering, SRE, QA, and security teams.
It captures runtime interfaces, operational controls, API and CLI commands, and validation workflows.

## Architecture and Runtime Topology

```{graphviz}
digraph TenetLLMAdaptersRuntime {
  graph [fontname="Arial" fontsize=10 rankdir=LR nodesep=0.3 ranksep=0.5 bgcolor=white]
  node  [shape=box style="rounded,filled" fontname="Arial" fontsize=10 penwidth=1 color="#4a5568"]
  edge  [fontname="Arial" fontsize=9 color="#4a5568" arrowsize=0.7]

  TenetCore [label="TenetCore\n(caller)" fillcolor="#e8f0fe" shape=ellipse]

  subgraph cluster_interface {
    label="Interface Layer" style=solid color="#63b3ed" fontname="Arial" fontsize=11 bgcolor=white
    EntryPoints [label="Entry Points\n(tenet.llm_adapters)" fillcolor="#e6fffa"]
    Factory [label="from_config()\nFactory" fillcolor="#fff7e6"]
  }

  subgraph cluster_adapters {
    label="Provider Adapters" style=solid color="#63b3ed" fontname="Arial" fontsize=11 bgcolor=white
    OpenAI [label="OpenAI" fillcolor="#fff7e6"]
    Anthropic [label="Anthropic" fillcolor="#fff7e6"]
    Google [label="Google" fillcolor="#fff7e6"]
    Cohere [label="Cohere" fillcolor="#fff7e6"]
  }

  subgraph cluster_apis {
    label="External APIs" style=solid color="#63b3ed" fontname="Arial" fontsize=11 bgcolor=white
    OpenAIAPI [label="OpenAI API" fillcolor="#f3e8ff"]
    AnthropicAPI [label="Anthropic API" fillcolor="#f3e8ff"]
    GoogleAPI [label="Google Gemini" fillcolor="#f3e8ff"]
    CohereAPI [label="Cohere API" fillcolor="#f3e8ff"]
  }

  TenetCore -> EntryPoints [label="discover"]
  EntryPoints -> Factory [label="instantiate"]
  Factory -> OpenAI
  Factory -> Anthropic
  Factory -> Google
  Factory -> Cohere
  OpenAI -> OpenAIAPI [label="generate/stream"]
  Anthropic -> AnthropicAPI
  Google -> GoogleAPI
  Cohere -> CohereAPI
}
```

## Component Boundaries

| Boundary | Responsibility |
|---|---|
| Interface layer | Accepts CLI/API/tool invocations and validates request shape. |
| Core logic | Executes module-specific orchestration, validation, and transformation. |
| Persistence/state | Stores module artifacts, indexes, checkpoints, or generated outputs. |
| Integration adapters | Communicates with peer Tenet modules and external systems. |
| Governance hooks | Applies profile/policy constraints where TenetOS integration is required. |

## CLI Command Reference

No project.scripts CLI entrypoints were discovered in pyproject.toml for this module.

## API Surface Reference

No decorator-defined HTTP routes were discovered during repository scan.

## Environment Variable Contract

| Variable | Operational Use |
|---|---|
| `ANTHROPIC_API_KEY` | Referenced in source/config for runtime behavior or governance integration. |
| `OPENAI_API_KEY` | Referenced in source/config for runtime behavior or governance integration. |
| `TENETOS_HOME` | Referenced in source/config for runtime behavior or governance integration. |
| `TENETOS_PROFILE` | Referenced in source/config for runtime behavior or governance integration. |

## Configuration and Request Structures

```json
{
  "module": "TenetLLMAdapters",
  "operation": "validate",
  "request_id": "req-20260407-001",
  "inputs": {
    "strict": true,
    "trace": true
  }
}
```

```json
{
  "status": "ok",
  "module": "TenetLLMAdapters",
  "result": {
    "validated": true,
    "warnings": []
  },
  "timestamp": "2026-04-07T00:00:00Z"
}
```

## Runtime Procedures

1. Preflight: verify profile/environment values and local dependencies.
2. Start process: launch module service or command in foreground for first-run validation.
3. Validate API/CLI: execute health/status commands and representative module workflows.
4. Observe: collect logs, error envelopes, and timing characteristics for baseline metrics.
5. Recover: apply known rollback and restart sequences when validation fails.

## Security and Governance Controls

1. Use TenetOS resolve/verify before finalizing changes in governed repositories.
2. Never embed plaintext secrets, tokens, or private keys in docs examples.
3. Validate trust boundaries for request headers, proxy chains, and external adapters.
4. Keep examples aligned with documented architectural invariants for this module.

## Testing and Verification Matrix

| Layer | Verification Activity | Expected Result |
|---|---|---|
| Unit | Run module unit tests | All tests pass with no regressions in critical paths. |
| Contract | Validate CLI/API payload contracts | Inputs and outputs match documented schema. |
| Integration | Exercise dependent services/modules | Hand-offs succeed and error handling is deterministic. |
| Governance | Run TenetOS policy checks (where required) | Resolve/verify reports pass or policy exceptions are documented. |

## Incident Response and Troubleshooting

| Symptom | Probable Cause | Mitigation |
|---|---|---|
| Startup failure | Missing config, profile mismatch, or absent dependency | Validate env vars and config paths, then restart cleanly. |
| Request validation errors | Contract drift between caller and module | Compare payload against documented JSON examples and route expectations. |
| Timeouts or degraded throughput | Upstream dependency saturation/unavailability | Check dependency health and apply retry/backoff policy. |
| Inconsistent outputs | Version/config drift | Re-run with pinned config and capture reproducible traces. |

## Source Anchors

Use these canonical repo docs as supplementary references:
- docs/README.md
- docs/integration.md
- docs/operations.md
