# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU Attention-side worker for AFD execution."""

from __future__ import annotations

from typing import Any

from vllm.distributed.elastic_ep.standby_state import (
    create_standby_groups,
    pop_standby_groups,
)
from vllm.distributed.parallel_state import _replace_active_groups
from vllm.v1.worker.workspace import init_workspace_manager
from vllm_ascend.worker.worker import NPUWorker

from afd_plugin.compat.npu import (
    apply_afd_ascend_patches_if_needed,
    fail_if_unsupported_npu_afd_features,
    fix_all2all_backend_for_afd,
    npu_afd_num_ubatches,
)
from afd_plugin.model_executor.models.model_utils import get_afd_model_config
from afd_plugin.v1.worker.npu.attention_model_runner import (
    AFDNPUAttentionModelRunner,
)
from afd_plugin.validation import (
    NPU_ATTENTION_WORKER_FQCN,
    assert_compatible_afd_stack,
)


class AFDNPUAttentionWorker(NPUWorker):
    """Attention worker that creates an AFD-aware vLLM-Ascend runner."""

    afd_expected_role = "attention"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        apply_afd_ascend_patches_if_needed()
        super().__init__(*args, **kwargs)

    def init_device(self) -> None:
        assert_compatible_afd_stack(
            self.vllm_config,
            caller="AFDNPUAttentionWorker.init_device",
            expected_role="attention",
            expected_worker_qualname_override=NPU_ATTENTION_WORKER_FQCN,
        )
        fail_if_unsupported_npu_afd_features(self.vllm_config)
        fix_all2all_backend_for_afd(self.vllm_config)
        if self.use_v2_model_runner:
            raise RuntimeError(
                "AFD NPU Attention supports only vllm-ascend model runner v1",
            )

        self.device = self._init_device()
        init_workspace_manager(
            self.device,
            npu_afd_num_ubatches(self.vllm_config),
        )
        self.vllm_config.model_config = get_afd_model_config(
            self.vllm_config.model_config,
        )
        self.model_runner = AFDNPUAttentionModelRunner(
            self.vllm_config,
            self.device,
        )

    def reconfigure_afd_attention_dp(
        self,
        new_data_parallel_size: int,
        master_ip: str,
        coord_store_port: int,
        retire_current_rank: bool,
    ) -> None:
        """Replace Attention DP groups after a CAMP2P group retirement."""

        parallel_config = self.vllm_config.parallel_config
        if not retire_current_rank:
            create_standby_groups(
                new_dp_size=new_data_parallel_size,
                new_world_size_across_dp=(
                    parallel_config.world_size * new_data_parallel_size
                ),
                master_ip=master_ip,
                coord_store_port=coord_store_port,
                enable_eplb=False,
            )

        # Every original rank participates in destruction of the old groups.
        # Retired workers install no replacement and remain parked in their
        # EngineCore; surviving workers atomically promote the standby groups.
        replacement_groups = (
            {
                "world": None,
                "dp": None,
                "ep": None,
                "eplb": None,
                "node_count": None,
            }
            if retire_current_rank
            else pop_standby_groups()
        )
        _replace_active_groups(**replacement_groups)

        if retire_current_rank:
            return

        parallel_config.data_parallel_size = new_data_parallel_size
        parallel_config.data_parallel_master_ip = master_ip
        parallel_config._coord_store_port = coord_store_port
        self.model_runner.dp_size = new_data_parallel_size
        self.model_runner.dp_rank = parallel_config.data_parallel_rank


__all__ = ["AFDNPUAttentionWorker"]
