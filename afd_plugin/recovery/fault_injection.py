# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Deterministic development-only fault injection for recovery tests."""

from __future__ import annotations

from afd_plugin.config import AFDConfig


class InjectedFFNForwardFailure(RuntimeError):
    """Raised intentionally before FFN compute to exercise recovery."""

    def __init__(self, phase: str, physical_rank: int) -> None:
        self.phase = phase
        self.physical_rank = physical_rank
        super().__init__(
            f"injected AFD FFN failure {phase.replace('_', ' ')} on physical "
            f"role rank {physical_rank}",
        )


class FFNForwardFaultInjector:
    """Raise once before forward on the configured physical FFN role rank."""

    def __init__(self, config: AFDConfig, physical_rank: int) -> None:
        self._target_rank = config.fault_injection_ffn_rank
        self._physical_rank = physical_rank
        self._phase = config.fault_injection_phase
        self._injected = False

    def before_forward(self) -> None:
        self._inject("before_forward")

    def before_step(self) -> None:
        self._inject("before_step")

    def _inject(self, phase: str) -> None:
        if (
            self._injected
            or self._target_rank != self._physical_rank
            or self._phase != phase
        ):
            return
        # Mark first so callers that catch the exception do not inject forever.
        self._injected = True
        raise InjectedFFNForwardFailure(phase, self._physical_rank)


__all__ = ["FFNForwardFaultInjector", "InjectedFFNForwardFailure"]
