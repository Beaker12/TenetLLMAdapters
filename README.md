# TenetLLMAdapters

Cloud LLM adapter package for the Tenet platform.

This repository provides four adapters that register through the
`tenet.llm_adapters` entry-point group and implement both generation and
live model discovery:

- `anthropic` -> `AnthropicAdapter`
- `openai-compatible` -> `OpenAIAdapter`
- `google` -> `GoogleAdapter`
- `cohere` -> `CohereAdapter`

## Platform Context

- **Requirements:** [SRS_TENETLLMADAPTERS](../TenetEGRF/docs/requirements/components/SRS_TENETLLMADAPTERS.md)
- **Architecture:** [ARCH_TENETLLMADAPTERS](../TenetEGRF/docs/architecture/ARCH_TENETLLMADAPTERS.md)
- **Realization:** [PLAN_TENETLLMADAPTERS](../TenetEGRF/docs/realization/PLAN_TENETLLMADAPTERS.md)
- **Verification:** [VER_TENET_LLM_ADAPTERS](../TenetEGRF/docs/verification/VER_TENET_LLM_ADAPTERS.md)
- **Governance:** [TenetOS](../TenetOS/README.md)

## Install

Base install includes no cloud SDK dependencies:

```bash
pip install tenet-llm-adapters
```

Install provider-specific extras as needed:

```bash
pip install tenet-llm-adapters[anthropic]
pip install tenet-llm-adapters[openai]
pip install tenet-llm-adapters[google,cohere]
pip install tenet-llm-adapters[all]
```

## TenetCore usage

Set the provider in a TenetCore backend config and point API keys to environment
variables:

```yaml
llm:
  backends:
    claude:
      provider: anthropic
      api_key: "${ANTHROPIC_API_KEY}"
    shared-openai:
      provider: openai-compatible
      api_key: "${OPENAI_API_KEY}"
    gemini:
      provider: google
      api_key: "${GOOGLE_API_KEY}"
    cohere:
      provider: cohere
      api_key: "${COHERE_API_KEY}"
```

## Entry points

```toml
[project.entry-points."tenet.llm_adapters"]
anthropic = "tenet_llm_adapters._anthropic:AnthropicAdapter"
openai-compatible = "tenet_llm_adapters._openai:OpenAIAdapter"
google = "tenet_llm_adapters._google:GoogleAdapter"
cohere = "tenet_llm_adapters._cohere:CohereAdapter"
```
