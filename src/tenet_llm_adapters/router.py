# Tenet Platform
# Copyright (C) 2025 Stuart W. Parkhurst
#
# This file is part of the Tenet Platform.
# Licensed under the GNU Affero General Public License v3.0
# See LICENSE file or https://www.gnu.org/licenses/agpl-3.0.html

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Label → cost tier mapping
_LABEL_TO_TIER: dict[str, str] = {
    "needs_reasoning": "high",
    "no_reasoning": "low",
    "fallback": "medium",
}


class MLRouterClient:
    """Calls a classifier service to pick a cost tier for a query.

    If the classifier service is unavailable, defaults to 'medium'.
    """

    def __init__(self, endpoint_url: str = "") -> None:
        self._endpoint_url = endpoint_url

    async def predict_tier(
        self,
        text: str,
        turn_number: int = 0,
        prompt_char_count: int = 0,
    ) -> str:
        """Return a cost tier: 'low', 'medium', or 'high'."""
        if not self._endpoint_url:
            return "medium"
        try:
            import urllib.request, json as _json, asyncio
            payload = _json.dumps(
                {"text": text[:2000], "turn_number": turn_number, "prompt_char_count": prompt_char_count}
            ).encode()
            loop = asyncio.get_event_loop()
            def _call() -> str:
                req = urllib.request.Request(
                    self._endpoint_url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=2) as resp:
                    data = _json.loads(resp.read())
                return data.get("predicted_label", "fallback")
            label = await loop.run_in_executor(None, _call)
            return _LABEL_TO_TIER.get(label, "medium")
        except Exception as exc:
            logger.debug("ML router unavailable (%s); defaulting to medium", exc)
            return "medium"
