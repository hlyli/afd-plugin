# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Deterministic development-only fault injection for recovery tests."""

from __future__ import annotations

from afd_plugin.config import AFDConfig


class InjectedFFNForwardFailure(RuntimeError):
    """Raised intentionally before FFN compute to exercise recovery."""


class FFNForwardFaultInjector:
    """Raise once before forward on the configured physical FFN role rank."""

    def __init__(self, config: AFDConfig, physical_rank: int) -> None:
        self._target_rank = config.fault_injection_ffn_rank
        self._physical_rank = physical_rank
        self._injected = False

    def before_forward(self) -> None:
        if self._injected or self._target_rank != self._physical_rank:
            return
        # Mark first so callers that catch the exception do not inject forever.
        self._injected = True
        raise InjectedFFNForwardFailure(
            "injected AFD FFN failure before forward on physical role rank "
            f"{self._physical_rank}",
        )


__all__ = ["FFNForwardFaultInjector", "InjectedFFNForwardFailure"]
