# Copilot Instructions

This repository is governed by TenetOS via MCP (Model Context Protocol).
TenetOS is discovered via the `TENETOS_HOME` env var or the sibling `../TenetOS`
directory; no local `tenetos.kernel.json` is needed in this repo.

## Governance Model

- **Profile:** `python-minimal` (set via `TENETOS_PROFILE` env var; resolved by TenetOS server at sibling `../TenetOS`)
- **Enforcement:** TenetOS MCP server provides `resolve_constraints`, `verify`,
  `propose_plan`, `explain_violation`, and `capabilities` tools
- **No local kernel or policies** — governance lives entirely in the TenetOS repo

## Agent Rules

1. Before modifying code, call TenetOS `resolve_constraints` with changed files.
2. Follow all MUST/MUST NOT rules from resolved constraints.
3. Before completing any task, call TenetOS `verify`.
4. Use Conventional Commits.
5. Do not duplicate policy rules into this file.

## Architecture (TenetLLMAdapters)

- Product code lives in `src/tenet_llm_adapters/` (standard Python src layout)
- Tests in `tests/`


## EGRF References

- Requirements: `../TenetEGRF/docs/requirements/components/SRS_TENETLLMADAPTERS.md`
- Architecture: `../TenetEGRF/docs/architecture/ARCH_TENETLLMADAPTERS.md`
- Realization: `../TenetEGRF/docs/realization/PLAN_TENETLLMADAPTERS.md`
- Verification: `../TenetEGRF/docs/verification/VER_TENET_LLM_ADAPTERS.md`

## What NOT to Do

- Do not invent new architecture patterns without updating docs
- Do not add dependencies without explicit approval
- Do not modify files outside the task scope
- Do not copy TenetOS files into this repo — use MCP
