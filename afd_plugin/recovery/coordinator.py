# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Thread-safe state machine coordinating one AFD rank recovery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Condition, RLock

from afd_plugin.recovery.topology import AFDRuntimeTopology, FailedAFDRank


class RecoveryPhase(str, Enum):
    RUNNING = "running"
    QUIESCING = "quiescing"
    RECONFIGURING = "reconfiguring"
    RESUMING = "resuming"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RecoverySnapshot:
    """Observable coordinator state safe to share with worker threads."""

    phase: RecoveryPhase
    topology: AFDRuntimeTopology
    failed_rank: FailedAFDRank | None = None
    failure_reason: str | None = None


class AFDRecoveryCoordinator:
    """Serialize recovery and publish epoch-scoped topology changes.

    The coordinator deliberately does not perform backend operations. Engine
    and worker code acknowledges quiescence, connector code recreates groups,
    and this class enforces the ordering shared by those components.
    """

    def __init__(self, topology: AFDRuntimeTopology) -> None:
        self._condition = Condition(RLock())
        self._snapshot = RecoverySnapshot(RecoveryPhase.RUNNING, topology)

    def snapshot(self) -> RecoverySnapshot:
        with self._condition:
            return self._snapshot

    def begin_recovery(
        self,
        failed_rank: FailedAFDRank,
        *,
        reason: str,
        epoch: int | None = None,
    ) -> RecoverySnapshot:
        """Begin recovery and compute, but do not publish, new membership."""

        if not reason.strip():
            raise ValueError("AFD recovery reason must not be empty")
        with self._condition:
            current = self._snapshot
            if (
                current.phase is RecoveryPhase.QUIESCING
                and current.failed_rank == failed_rank
                and (epoch is None or epoch == current.topology.epoch + 1)
            ):
                return current
            self._require_phase(RecoveryPhase.RUNNING)
            expected_epoch = current.topology.epoch + 1
            if epoch is not None and epoch != expected_epoch:
                raise ValueError(
                    f"stale AFD recovery epoch {epoch}; expected {expected_epoch}",
                )
            # Validate the rank before changing observable state.
            current.topology.without_rank(failed_rank)
            self._snapshot = RecoverySnapshot(
                RecoveryPhase.QUIESCING,
                current.topology,
                failed_rank,
                reason,
            )
            self._condition.notify_all()
            return self._snapshot

    def mark_quiesced(
        self,
        *,
        preserve_membership: bool = False,
        next_topology: AFDRuntimeTopology | None = None,
    ) -> RecoverySnapshot:
        """Publish the next topology after all participating ranks stop work.

        ``preserve_membership`` supports transient-failure recovery where the
        failed process remains alive and rejoins a recreated communicator with
        its original rank.
        """

        with self._condition:
            self._require_phase(RecoveryPhase.QUIESCING)
            failed_rank = self._snapshot.failed_rank
            assert failed_rank is not None
            if preserve_membership and next_topology is not None:
                raise ValueError(
                    "preserve_membership and next_topology are mutually exclusive",
                )
            if next_topology is not None:
                expected_epoch = self._snapshot.topology.epoch + 1
                if next_topology.epoch != expected_epoch:
                    raise ValueError(
                        f"next topology epoch is {next_topology.epoch}; expected "
                        f"{expected_epoch}",
                    )
                published_topology = next_topology
            elif preserve_membership:
                published_topology = self._snapshot.topology.next_epoch()
            else:
                published_topology = self._snapshot.topology.without_rank(failed_rank)
            self._snapshot = RecoverySnapshot(
                RecoveryPhase.RECONFIGURING,
                published_topology,
                failed_rank,
                self._snapshot.failure_reason,
            )
            self._condition.notify_all()
            return self._snapshot

    def mark_reconfigured(self, epoch: int) -> RecoverySnapshot:
        """Record that communication resources for ``epoch`` are ready."""

        with self._condition:
            self._require_phase(RecoveryPhase.RECONFIGURING)
            self._require_epoch(epoch)
            self._snapshot = RecoverySnapshot(
                RecoveryPhase.RESUMING,
                self._snapshot.topology,
                self._snapshot.failed_rank,
                self._snapshot.failure_reason,
            )
            self._condition.notify_all()
            return self._snapshot

    def mark_resumed(self, epoch: int) -> RecoverySnapshot:
        """Commit recovery and allow inference under the new epoch."""

        with self._condition:
            self._require_phase(RecoveryPhase.RESUMING)
            self._require_epoch(epoch)
            self._snapshot = RecoverySnapshot(
                RecoveryPhase.RUNNING,
                self._snapshot.topology,
            )
            self._condition.notify_all()
            return self._snapshot

    def mark_failed(self, reason: str) -> RecoverySnapshot:
        """Move an in-progress recovery into a terminal failed state."""

        if not reason.strip():
            raise ValueError("AFD recovery failure reason must not be empty")
        with self._condition:
            if self._snapshot.phase in {RecoveryPhase.RUNNING, RecoveryPhase.FAILED}:
                raise RuntimeError(
                    "AFD recovery can fail only while recovery is in progress",
                )
            self._snapshot = RecoverySnapshot(
                RecoveryPhase.FAILED,
                self._snapshot.topology,
                self._snapshot.failed_rank,
                reason,
            )
            self._condition.notify_all()
            return self._snapshot

    def _require_phase(self, expected: RecoveryPhase) -> None:
        if self._snapshot.phase is not expected:
            raise RuntimeError(
                f"AFD recovery phase is {self._snapshot.phase.value!r}, "
                f"expected {expected.value!r}",
            )

    def _require_epoch(self, epoch: int) -> None:
        if self._snapshot.topology.epoch != epoch:
            raise ValueError(
                f"stale AFD recovery epoch {epoch}; current epoch is "
                f"{self._snapshot.topology.epoch}",
            )


__all__ = ["AFDRecoveryCoordinator", "RecoveryPhase", "RecoverySnapshot"]
