"""Module declaration for TenetLLMAdapters."""

from __future__ import annotations

from tenet_core.config.declarations import ModuleDeclaration, TunableDeclaration

_MODULE_ID = "tenet_llm_adapters"
_MODULE_VERSION = "1.0.0"

_TUNABLES: list[TunableDeclaration] = []


def get_declaration() -> ModuleDeclaration:
    """Return the TenetLLMAdapters module declaration."""
    return ModuleDeclaration(
        module_id=_MODULE_ID,
        module_version=_MODULE_VERSION,
        module_category="core",
        required=True,
        tunables=_TUNABLES,
    )
