# Tenet Platform
# Copyright (C) 2025 Stuart W. Parkhurst
#
# This file is part of the Tenet Platform.
# Licensed under the GNU Affero General Public License v3.0
# See LICENSE file or https://www.gnu.org/licenses/agpl-3.0.html

"""TenetLLMGateway routing helper.

When ``TENET_LLM_GATEWAY_URL`` is set, all cloud LLM adapters route their
requests through the gateway URL instead of calling provider APIs directly.
The gateway speaks the same wire protocol as the upstream provider, so it can
be used as a drop-in ``base_url`` replacement.
"""

from __future__ import annotations

import os

GATEWAY_URL_ENV = "TENET_LLM_GATEWAY_URL"


def _resolve_base_url(
    provider_default_url: str | None,
    config_base_url: str | None,
) -> str | None:
    """Return the effective base URL for an LLM SDK client.

    Priority (highest first):
    1. ``TENET_LLM_GATEWAY_URL`` environment variable — routes all traffic
       through TenetLLMGateway.
    2. ``config_base_url`` — explicit override from the backend config.
    3. ``provider_default_url`` — SDK / adapter default (may be ``None``
       for SDKs that have a baked-in default, e.g. ``anthropic``).
    """
    gateway = os.getenv(GATEWAY_URL_ENV, "").strip()
    if gateway:
        return gateway.rstrip("/")
    return config_base_url or provider_default_url
