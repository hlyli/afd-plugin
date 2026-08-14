from __future__ import annotations

import pytest

from afd_plugin.config import AFDConfig
from afd_plugin.recovery import (
    AFDRecoveryCoordinator,
    AFDRuntimeTopology,
    FailedAFDRank,
    RecoveryPhase,
)


def _topology() -> AFDRuntimeTopology:
    return AFDRuntimeTopology.from_config(
        AFDConfig(num_attention_ranks=4, num_ffn_ranks=2),
    )


def test_ffn_failure_compacts_logical_ranks_in_next_epoch():
    topology = _topology().without_rank(FailedAFDRank("ffn", 0))

    assert topology.epoch == 1
    assert topology.ffn_physical_ranks == (1,)
    assert topology.logical_rank("ffn", 1) == 0
    assert topology.attention_physical_ranks == (0, 1, 2, 3)


def test_topology_rejects_last_ffn_failure():
    topology = AFDRuntimeTopology.from_config(AFDConfig())

    with pytest.raises(ValueError, match="requires an FFN rank"):
        topology.without_rank(FailedAFDRank("ffn", 0))


def test_failed_rank_rejects_unknown_role():
    with pytest.raises(ValueError, match="unknown AFD runtime role"):
        FailedAFDRank("invalid", 0)  # type: ignore[arg-type]


def test_coordinator_runs_ordered_recovery_lifecycle():
    coordinator = AFDRecoveryCoordinator(_topology())
    failed_rank = FailedAFDRank("ffn", 0)

    assert coordinator.begin_recovery(failed_rank, reason="injected").phase is (
        RecoveryPhase.QUIESCING
    )
    reconfiguring = coordinator.mark_quiesced()
    assert reconfiguring.phase is RecoveryPhase.RECONFIGURING
    assert reconfiguring.topology.epoch == 1
    assert coordinator.mark_reconfigured(1).phase is RecoveryPhase.RESUMING
    running = coordinator.mark_resumed(1)

    assert running.phase is RecoveryPhase.RUNNING
    assert running.topology.ffn_physical_ranks == (1,)
    assert running.failed_rank is None


def test_coordinator_rejects_stale_epoch():
    coordinator = AFDRecoveryCoordinator(_topology())
    coordinator.begin_recovery(FailedAFDRank("ffn", 1), reason="injected")
    coordinator.mark_quiesced()

    with pytest.raises(ValueError, match="stale AFD recovery epoch"):
        coordinator.mark_reconfigured(0)


def test_coordinator_rejects_overlapping_recovery():
    coordinator = AFDRecoveryCoordinator(_topology())
    coordinator.begin_recovery(FailedAFDRank("ffn", 0), reason="first")

    with pytest.raises(RuntimeError, match="expected 'running'"):
        coordinator.begin_recovery(FailedAFDRank("ffn", 1), reason="second")


def test_coordinator_records_terminal_recovery_failure():
    coordinator = AFDRecoveryCoordinator(_topology())
    coordinator.begin_recovery(FailedAFDRank("ffn", 0), reason="timeout")

    snapshot = coordinator.mark_failed("survivors did not quiesce")

    assert snapshot.phase is RecoveryPhase.FAILED
    assert snapshot.failure_reason == "survivors did not quiesce"
