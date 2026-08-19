from __future__ import annotations

import pytest
import torch

from afd_plugin.recovery import (
    AFDFailureNotice,
    AFDRecoveryChannel,
    FailedAFDRank,
)
from afd_plugin.recovery.channel import (
    decode_failure_notice,
    encode_failure_notice,
)


def test_failure_notice_round_trip():
    notice = AFDFailureNotice(
        epoch=3,
        failed_rank=FailedAFDRank("ffn", 7),
    )

    assert decode_failure_notice(encode_failure_notice(notice)) == notice


def test_failure_notice_uses_cpu_tensor():
    message = encode_failure_notice(
        AFDFailureNotice(1, FailedAFDRank("attention", 2)),
    )

    assert message.device.type == "cpu"
    assert message.dtype is torch.int64


def test_failure_notice_rejects_unknown_protocol():
    message = encode_failure_notice(AFDFailureNotice(1, FailedAFDRank("ffn", 0)))
    message[0] = 999

    with pytest.raises(ValueError, match="protocol version"):
        decode_failure_notice(message)


def test_channel_records_notice_once_and_invokes_callback_once():
    notices = []
    channel = AFDRecoveryChannel(
        object(),  # type: ignore[arg-type]
        world_rank=0,
        world_size=2,
        notice_callback=notices.append,
    )
    notice = AFDFailureNotice(1, FailedAFDRank("ffn", 0))

    channel._record_notice(notice)
    channel._record_notice(notice)

    assert channel.failure_event.is_set()
    assert channel.latest_notice == notice
    assert notices == [notice]


def test_channel_rejects_non_positive_poll_interval():
    with pytest.raises(ValueError, match="poll interval must be positive"):
        AFDRecoveryChannel(
            object(),  # type: ignore[arg-type]
            world_rank=0,
            world_size=1,
            poll_interval_seconds=0,
        )
