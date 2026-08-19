# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Out-of-band CPU notification channel for AFD recovery events."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import torch
import torch.distributed as dist

from afd_plugin.recovery.topology import AFDRuntimeRole, FailedAFDRank

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

logger = logging.getLogger(__name__)

RECOVERY_PROTOCOL_VERSION: Final[int] = 1
FAILURE_EVENT: Final[int] = 1
SHUTDOWN_EVENT: Final[int] = 2
RECOVERY_MESSAGE_FIELDS: Final[int] = 5
RECOVERY_LISTENER_POLL_SECONDS: Final[float] = 1.0
RECOVERY_LISTENER_SHUTDOWN_SECONDS: Final[float] = 2.0
RECOVERY_RECONFIGURATION_TIMEOUT_SECONDS: Final[float] = 1800.0

_ROLE_TO_CODE: Final[dict[str, int]] = {"attention": 1, "ffn": 2}
_CODE_TO_ROLE: Final[dict[int, AFDRuntimeRole]] = {1: "attention", 2: "ffn"}


@dataclass(frozen=True, slots=True)
class AFDFailureNotice:
    """Failure notification broadcast independently of accelerator groups."""

    epoch: int
    failed_rank: FailedAFDRank


class AFDRecoveryChannel:
    """Listen for fixed-size failure notices on a dedicated Gloo group.

    Only the background listener participates in the periodic status
    collective. Normal inference code observes ``failure_event`` and never
    waits for a failure message that may not come.
    """

    def __init__(
        self,
        process_group: ProcessGroup,
        *,
        world_rank: int,
        world_size: int,
        notice_callback: Callable[[AFDFailureNotice], None] | None = None,
        poll_interval_seconds: float = RECOVERY_LISTENER_POLL_SECONDS,
    ) -> None:
        if not 0 <= world_rank < world_size:
            raise ValueError(
                f"recovery world rank {world_rank} is outside size {world_size}",
            )
        self.process_group = process_group
        self.world_rank = world_rank
        self.world_size = world_size
        if poll_interval_seconds <= 0:
            raise ValueError("recovery poll interval must be positive")
        self.poll_interval_seconds = poll_interval_seconds
        self.notice_callback = notice_callback
        self.failure_event = threading.Event()
        self.recovery_ready_event = threading.Event()
        self.recovery_ready_event.set()
        self._shutdown_event = threading.Event()
        self._notice_lock = threading.Lock()
        self._latest_notice: AFDFailureNotice | None = None
        self._recovery_error: str | None = None
        self._outbound_lock = threading.Lock()
        self._outbound_message = _empty_recovery_message()
        self._listener_thread: threading.Thread | None = None

    @property
    def latest_notice(self) -> AFDFailureNotice | None:
        with self._notice_lock:
            return self._latest_notice

    def start(self) -> None:
        if self._listener_thread is not None and self._listener_thread.is_alive():
            return
        self._shutdown_event.clear()
        self._listener_thread = threading.Thread(
            target=self._listen,
            name=f"afd-recovery-listener-{self.world_rank}",
            daemon=True,
        )
        self._listener_thread.start()

    def report_failure(self, notice: AFDFailureNotice) -> None:
        """Publish one failure notice to every other AFD rank."""

        # Block local data-plane work immediately, but let the listener process
        # the notice so every rank invokes callbacks in the same collective
        # order.
        self.recovery_ready_event.clear()
        with self._outbound_lock:
            self._outbound_message = encode_failure_notice(notice)

    def close(self) -> None:
        with self._outbound_lock:
            self._outbound_message = _shutdown_message()
        listener = self._listener_thread
        if listener is not None:
            listener.join(timeout=RECOVERY_LISTENER_SHUTDOWN_SECONDS)
        self._shutdown_event.set()
        self._listener_thread = None

    def wait_until_recovery_ready(
        self,
        timeout_seconds: float = RECOVERY_RECONFIGURATION_TIMEOUT_SECONDS,
    ) -> None:
        """Wait until every original rank finishes handling the latest notice."""

        if timeout_seconds <= 0:
            raise ValueError("recovery readiness timeout must be positive")
        if self.recovery_ready_event.is_set():
            recovery_is_ready = True
        else:
            recovery_is_ready = self.recovery_ready_event.wait(timeout_seconds)
        if not recovery_is_ready:
            raise TimeoutError(
                "AFD recovery ranks did not become ready within "
                f"{timeout_seconds} seconds",
            )
        with self._notice_lock:
            recovery_error = self._recovery_error
        if recovery_error is not None:
            raise RuntimeError(recovery_error)

    def _listen(self) -> None:
        try:
            while not self._shutdown_event.is_set():
                with self._outbound_lock:
                    message = self._outbound_message.clone()
                dist.all_reduce(
                    message,
                    op=dist.ReduceOp.MAX,
                    group=self.process_group,
                )
                event = int(message[1].item())
                if event == FAILURE_EVENT:
                    is_new_notice = self._record_notice(
                        decode_failure_notice(message),
                    )
                    if is_new_notice:
                        # Every original rank retains this CPU group. Callback
                        # completion means local teardown/rebuild is done (or
                        # the failed rank has elected not to rejoin).
                        with self._notice_lock:
                            local_recovery_succeeded = self._recovery_error is None
                        recovery_status = torch.tensor(
                            [int(local_recovery_succeeded)],
                            dtype=torch.int64,
                            device="cpu",
                        )
                        dist.all_reduce(
                            recovery_status,
                            op=dist.ReduceOp.MIN,
                            group=self.process_group,
                        )
                        if int(recovery_status.item()) == 0:
                            with self._notice_lock:
                                if self._recovery_error is None:
                                    self._recovery_error = (
                                        "AFD recovery failed on another rank"
                                    )
                        self.recovery_ready_event.set()
                elif event == SHUTDOWN_EVENT:
                    self._shutdown_event.set()
                    break
                elif event != 0:
                    raise ValueError(f"unknown AFD recovery event {event}")
                self._shutdown_event.wait(self.poll_interval_seconds)
        except Exception:
            if not self._shutdown_event.is_set():
                logger.exception("AFD recovery listener failed")

    def _record_notice(self, notice: AFDFailureNotice) -> bool:
        notify = False
        with self._notice_lock:
            current = self._latest_notice
            if current is None or notice.epoch > current.epoch:
                self._latest_notice = notice
                self.failure_event.set()
                self.recovery_ready_event.clear()
                self._recovery_error = None
                notify = True
            elif notice == current:
                self.failure_event.set()
        if notify and self.notice_callback is not None:
            try:
                self.notice_callback(notice)
            except Exception as exc:
                logger.exception("AFD failure-notice callback failed")
                with self._notice_lock:
                    self._recovery_error = f"AFD local recovery failed: {exc}"
        return notify


def encode_failure_notice(notice: AFDFailureNotice) -> torch.Tensor:
    role_code = _ROLE_TO_CODE[notice.failed_rank.role]
    return torch.tensor(
        [
            RECOVERY_PROTOCOL_VERSION,
            FAILURE_EVENT,
            notice.epoch,
            role_code,
            notice.failed_rank.physical_rank,
        ],
        dtype=torch.int64,
        device="cpu",
    )


def decode_failure_notice(message: torch.Tensor) -> AFDFailureNotice:
    values = [int(value) for value in message.tolist()]
    if len(values) != RECOVERY_MESSAGE_FIELDS:
        raise ValueError(f"invalid AFD recovery message size {len(values)}")
    version, event, epoch, role_code, physical_rank = values
    if version != RECOVERY_PROTOCOL_VERSION:
        raise ValueError(f"unsupported AFD recovery protocol version {version}")
    if event != FAILURE_EVENT:
        raise ValueError(f"unknown AFD recovery event {event}")
    if role_code not in _CODE_TO_ROLE:
        raise ValueError(f"unknown AFD recovery role code {role_code}")
    return AFDFailureNotice(
        epoch=epoch,
        failed_rank=FailedAFDRank(_CODE_TO_ROLE[role_code], physical_rank),
    )


def _empty_recovery_message() -> torch.Tensor:
    return torch.zeros(RECOVERY_MESSAGE_FIELDS, dtype=torch.int64, device="cpu")


def _shutdown_message() -> torch.Tensor:
    message = _empty_recovery_message()
    message[0] = RECOVERY_PROTOCOL_VERSION
    message[1] = SHUTDOWN_EVENT
    return message


__all__ = [
    "AFDFailureNotice",
    "AFDRecoveryChannel",
    "decode_failure_notice",
    "encode_failure_notice",
]
