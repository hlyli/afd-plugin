from __future__ import annotations

import pytest
import torch

from afd_plugin.recovery import AFDFailureNotice, FailedAFDRank
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
