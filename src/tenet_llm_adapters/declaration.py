# Tenet Platform
# Copyright (C) 2025 Stuart W. Parkhurst
#
# This file is part of the Tenet Platform.
# Licensed under the GNU Affero General Public License v3.0
# See LICENSE file or https://www.gnu.org/licenses/agpl-3.0.html

"""Module declaration for TenetLLMAdapters."""

from __future__ import annotations

from tenet_core.config.declarations import ModuleDeclaration, TunableDeclaration

_MODULE_ID = "tenet_llm_adapters"
_MODULE_VERSION = "1.1.0.dev0"

_TUNABLES: list[TunableDeclaration] = [
    TunableDeclaration(
        key="llm.gateway_url",
        value_type="str",
        default="",
        description=(
            "When set, all LLM adapter requests route through this TenetLLMGateway URL. "
            "Overrides TENET_LLM_GATEWAY_URL env var. Empty string disables gateway routing."
        ),
        mutability_scope="workspace",
        owner=_MODULE_ID,
    ),
]


def get_declaration() -> ModuleDeclaration:
    """Return the TenetLLMAdapters module declaration."""
    return ModuleDeclaration(
        module_id=_MODULE_ID,
        module_version=_MODULE_VERSION,
        module_category="core",
        required=True,
        tunables=_TUNABLES,
    )
