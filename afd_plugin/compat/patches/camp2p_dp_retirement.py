# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Retire tail Attention engines after CAMP2P startup recovery.

This patch reuses vLLM's public ``/scale_elastic_ep`` control path without
running native Elastic EP expert migration. CAMP2P owns its data plane while
the patch rebuilds vLLM's Attention DP groups and updates frontend routing.
"""

from __future__ import annotations

import asyncio

import msgspec.msgpack
from vllm.v1.engine.core_client import DPLBAsyncMPClient

from afd_plugin.config import parse_optional_afd_config

_original_scale_elastic_ep = DPLBAsyncMPClient.scale_elastic_ep


def _is_camp2p_tail_recovery(client: DPLBAsyncMPClient) -> bool:
    config = parse_optional_afd_config(client.vllm_config, validate=False)
    return (
        config is not None
        and config.role == "attention"
        and config.connector == "CAMP2pAFDConnector"
        and config.fault_injection_phase == "before_step"
        and config.fault_injection_ffn_rank == config.num_ffn_ranks - 1
    )


# Patch reason: native Elastic EP scale-down reconfigures vLLM's DP/EP
# communicators, which conflicts with CAMP2P's independently rebuilt HCCL
# groups on Ascend.
# Patch functionality: for the CAMP2P startup PoC only, rebuild Attention DP
# groups, retire tail Attention identities from routing, and notify
# DPCoordinator of the new count. Other setups retain the native method.
# Signature: matches vLLM 0.19.1 exactly; no added parameters.
async def scale_elastic_ep(
    self: DPLBAsyncMPClient,
    new_data_parallel_size: int,
) -> None:
    if not _is_camp2p_tail_recovery(self):
        await _original_scale_elastic_ep(self, new_data_parallel_size)
        return

    current_size = len(self.core_engines)
    config = parse_optional_afd_config(self.vllm_config, validate=False)
    assert config is not None
    attention_group_size = config.num_attention_ranks // config.num_ffn_ranks
    expected_size = current_size - attention_group_size
    if new_data_parallel_size != expected_size:
        raise ValueError(
            "CAMP2P tail recovery must retire exactly one Attention group: "
            f"expected DP size {expected_size}, got {new_data_parallel_size}",
        )
    if self.reqs_in_flight:
        raise RuntimeError(
            "CAMP2P startup recovery cannot retire Attention engines while "
            "requests are in flight",
        )

    # ### PATCH START: CAMP2P tail DP reconstruction and retirement
    master_ip, coord_store_port = self._setup_elastic_ep_reconfig_bootstrap()
    await asyncio.gather(
        *(
            self._call_utility_async(
                "reconfigure_afd_attention_dp",
                new_data_parallel_size,
                master_ip,
                coord_store_port,
                engine=engine,
            )
            for engine in self.core_engines
        ),
    )
    self.core_engines = self.core_engines[:new_data_parallel_size]
    self.lb_engines = self.lb_engines[:new_data_parallel_size]
    self.engine_ranks_managed = self.engine_ranks_managed[:new_data_parallel_size]
    self.eng_start_index = (
        new_data_parallel_size * self.client_index
    ) // self.client_count
    self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
    self._ensure_stats_update_task()
    scale_down_marker = msgspec.msgpack.encode(
        ("SCALE_ELASTIC_EP", new_data_parallel_size),
    )
    await self.first_req_send_socket.send(scale_down_marker)
    # ### PATCH END: CAMP2P tail DP reconstruction and retirement


DPLBAsyncMPClient.scale_elastic_ep = scale_elastic_ep


__all__ = ["scale_elastic_ep"]
