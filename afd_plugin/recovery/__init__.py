# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Fault-recovery primitives for AFD disaggregated runtimes."""

from afd_plugin.recovery.channel import AFDFailureNotice, AFDRecoveryChannel
from afd_plugin.recovery.coordinator import (
    AFDRecoveryCoordinator,
    RecoveryPhase,
    RecoverySnapshot,
)
from afd_plugin.recovery.fault_injection import (
    FFNForwardFaultInjector,
    InjectedFFNForwardFailure,
)
from afd_plugin.recovery.topology import AFDRuntimeTopology, FailedAFDRank


class AFDRecoveryQuiescing(RuntimeError):
    """Raised before new AFD work when recovery has requested quiescence."""


__all__ = [
    "AFDRecoveryCoordinator",
    "AFDRecoveryQuiescing",
    "AFDFailureNotice",
    "AFDRecoveryChannel",
    "AFDRuntimeTopology",
    "FFNForwardFaultInjector",
    "FailedAFDRank",
    "InjectedFFNForwardFailure",
    "RecoveryPhase",
    "RecoverySnapshot",
]
