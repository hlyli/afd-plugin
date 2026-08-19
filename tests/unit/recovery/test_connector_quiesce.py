from types import SimpleNamespace

import pytest

from afd_plugin.config import AFDConfig

pytest.importorskip("vllm")

from afd_plugin.connectors.base import AFDConnectorBase
from afd_plugin.recovery import (
    AFDFailureNotice,
    AFDRecoveryCoordinator,
    AFDRecoveryQuiescing,
    AFDRuntimeTopology,
    FailedAFDRank,
    RecoveryPhase,
)


def _connector() -> SimpleNamespace:
    connector = SimpleNamespace()
    connector.recovery_coordinator = AFDRecoveryCoordinator(
        AFDRuntimeTopology.from_config(
            AFDConfig(num_attention_ranks=2, num_ffn_ranks=2),
        ),
    )
    return connector


def test_failure_notice_moves_connector_to_quiescing():
    connector = _connector()

    AFDConnectorBase._on_failure_notice(
        connector,
        AFDFailureNotice(1, FailedAFDRank("ffn", 0)),
    )

    snapshot = connector.recovery_coordinator.snapshot()
    assert snapshot.phase is RecoveryPhase.QUIESCING
    assert snapshot.failed_rank == FailedAFDRank("ffn", 0)


def test_connector_rejects_new_work_while_quiescing():
    connector = _connector()
    AFDConnectorBase._on_failure_notice(
        connector,
        AFDFailureNotice(1, FailedAFDRank("ffn", 0)),
    )

    with pytest.raises(AFDRecoveryQuiescing, match="is quiescing"):
        AFDConnectorBase.ensure_recovery_running(connector)
