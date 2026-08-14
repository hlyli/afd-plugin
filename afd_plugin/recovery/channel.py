# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Out-of-band CPU notification channel for AFD recovery events."""

from __future__ import annotations

import logging
import threading
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
    ) -> None:
        if not 0 <= world_rank < world_size:
            raise ValueError(
                f"recovery world rank {world_rank} is outside size {world_size}",
            )
        self.process_group = process_group
        self.world_rank = world_rank
        self.world_size = world_size
        self.failure_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._notice_lock = threading.Lock()
        self._latest_notice: AFDFailureNotice | None = None
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

        self._record_notice(notice)
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
                    self._record_notice(decode_failure_notice(message))
                elif event == SHUTDOWN_EVENT:
                    self._shutdown_event.set()
                    break
                elif event != 0:
                    raise ValueError(f"unknown AFD recovery event {event}")
                self._shutdown_event.wait(RECOVERY_LISTENER_POLL_SECONDS)
        except Exception:
            if not self._shutdown_event.is_set():
                logger.exception("AFD recovery listener failed")

    def _record_notice(self, notice: AFDFailureNotice) -> None:
        with self._notice_lock:
            current = self._latest_notice
            if current is None or notice.epoch >= current.epoch:
                self._latest_notice = notice
                self.failure_event.set()


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
