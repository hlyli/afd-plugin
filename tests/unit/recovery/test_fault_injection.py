from __future__ import annotations

import pytest

from afd_plugin.config import AFDConfig
from afd_plugin.recovery import (
    FFNForwardFaultInjector,
    InjectedFFNForwardFailure,
)


def test_target_ffn_rank_raises_once_before_forward():
    injector = FFNForwardFaultInjector(
        AFDConfig(
            fault_injection_ffn_rank=1,
            num_attention_ranks=2,
            num_ffn_ranks=2,
        ),
        physical_rank=1,
    )

    with pytest.raises(InjectedFFNForwardFailure, match="physical role rank 1"):
        injector.before_forward()

    injector.before_forward()


def test_non_target_ffn_rank_does_not_raise():
    injector = FFNForwardFaultInjector(
        AFDConfig(
            fault_injection_ffn_rank=1,
            num_attention_ranks=2,
            num_ffn_ranks=2,
        ),
        physical_rank=0,
    )

    injector.before_forward()


def test_disabled_fault_injector_does_not_raise():
    injector = FFNForwardFaultInjector(AFDConfig(), physical_rank=0)

    injector.before_forward()


def test_safe_boundary_injector_raises_before_step_only():
    injector = FFNForwardFaultInjector(
        AFDConfig(
            fault_injection_ffn_rank=0,
            fault_injection_phase="before_step",
        ),
        physical_rank=0,
    )

    injector.before_forward()
    with pytest.raises(InjectedFFNForwardFailure, match="before step"):
        injector.before_step()
