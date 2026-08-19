from __future__ import annotations

import multiprocessing
import os
import socket
import time
from datetime import timedelta

import pytest
import torch.distributed as dist

from afd_plugin.recovery import (
    AFDFailureNotice,
    AFDRecoveryChannel,
    FailedAFDRank,
)

RECOVERY_TEST_WORLD_SIZE = 2


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _channel_process(
    rank: int,
    port: int,
    inject_failure: bool,
    result_queue: multiprocessing.Queue,
) -> None:
    channel = None
    try:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=RECOVERY_TEST_WORLD_SIZE,
            timeout=timedelta(seconds=10),
        )
        channel = AFDRecoveryChannel(
            dist.group.WORLD,
            world_rank=rank,
            world_size=RECOVERY_TEST_WORLD_SIZE,
            poll_interval_seconds=0.02,
        )
        channel.start()
        if inject_failure and rank == 1:
            time.sleep(0.1)
            channel.report_failure(
                AFDFailureNotice(1, FailedAFDRank("ffn", 0)),
            )

        if inject_failure:
            observed = channel.failure_event.wait(timeout=5)
            notice = channel.latest_notice
            result_queue.put(
                (
                    rank,
                    "notice",
                    observed,
                    None if notice is None else notice.epoch,
                    None if notice is None else notice.failed_rank.role,
                    None if notice is None else notice.failed_rank.physical_rank,
                ),
            )
        else:
            time.sleep(0.2)
            result_queue.put((rank, "idle", channel.failure_event.is_set()))

        # Exercise staggered close: the first rank's shutdown status must let
        # every listener leave its periodic collective without a timeout.
        if rank == 1:
            time.sleep(0.1)
        channel.close()
        channel = None
    except Exception as exc:
        result_queue.put((rank, "error", type(exc).__name__, str(exc)))
    finally:
        if channel is not None:
            channel.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_channel_processes(*, inject_failure: bool) -> list[tuple]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    port = _available_port()
    processes = [
        context.Process(
            target=_channel_process,
            args=(rank, port, inject_failure, result_queue),
        )
        for rank in range(RECOVERY_TEST_WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    results = [result_queue.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
        assert not process.is_alive(), "Gloo recovery test process did not exit"
        assert process.exitcode == 0
    return sorted(results)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_gloo_channel_remains_idle_without_failure():
    results = _run_channel_processes(inject_failure=False)

    assert results == [(0, "idle", False), (1, "idle", False)]


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_gloo_channel_propagates_failure_to_every_rank():
    results = _run_channel_processes(inject_failure=True)

    assert results == [
        (0, "notice", True, 1, "ffn", 0),
        (1, "notice", True, 1, "ffn", 0),
    ]
