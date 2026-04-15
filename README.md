# TenetLLMAdapters

TenetLLMAdapters provides cloud LLM adapter integrations for the Tenet platform.

## Badges/Status

[![CI](https://github.com/Beaker12/TenetLLMAdapters/actions/workflows/ci.yml/badge.svg)](https://github.com/Beaker12/TenetLLMAdapters/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
[![License: AGPL-3.0-only](https://img.shields.io/badge/license-AGPL--3.0--only-blue.svg)](LICENSE)

## Platform Context + EGRF Traceability

- Requirements: [SRS_TENETLLMADAPTERS](../TenetEGRF/docs/requirements/components/SRS_TENETLLMADAPTERS.md)
- Architecture: [ARCH_TENETLLMADAPTERS](../TenetEGRF/docs/architecture/ARCH_TENETLLMADAPTERS.md)
- Realization: [PLAN_TENETLLMADAPTERS](../TenetEGRF/docs/realization/PLAN_TENETLLMADAPTERS.md)
- Verification: [VER_TENET_LLM_ADAPTERS](../TenetEGRF/docs/verification/VER_TENET_LLM_ADAPTERS.md)
- Governance: [TenetOS](../TenetOS/README.md)

## Overview

Adapters register through `tenet.llm_adapters` entry points for Anthropic, OpenAI-compatible, Google, and Cohere providers.

## Installation

```bash
pip install tenet-llm-adapters
# optional extras
# pip install tenet-llm-adapters[anthropic]
# pip install tenet-llm-adapters[openai]
# pip install tenet-llm-adapters[google,cohere]
# pip install tenet-llm-adapters[all]
```

## Quick Start / Usage

Use the package by setting provider names in TenetCore backend configuration.

## Configuration

Provider credentials and endpoint configuration are supplied by environment variables and host runtime backend configuration.

## API/CLI Surface

- Python entry-point group: `tenet.llm_adapters`
- Standalone CLI: not applicable

## Architecture

- Product code: `src/tenet_llm_adapters/`
- Adapter implementations: `_anthropic.py`, `_openai.py`, `_google.py`, `_cohere.py`

## Testing

```bash
pytest tests/
```

## Development & Contributing

- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

AGPL-3.0-only. See [LICENSE](LICENSE).
