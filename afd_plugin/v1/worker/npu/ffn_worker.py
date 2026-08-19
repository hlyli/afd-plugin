# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU FFN-side worker for AFD execution."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.worker.workspace import init_workspace_manager
from vllm_ascend.worker.worker import NPUWorker

from afd_plugin.compat.npu import (
    apply_afd_ascend_patches_if_needed,
    fail_if_unsupported_npu_afd_features,
    fix_all2all_backend_for_afd,
    npu_afd_num_ubatches,
)
from afd_plugin.model_executor.models.model_utils import get_afd_model_config
from afd_plugin.recovery import (
    AFDFailureNotice,
    AFDRecoveryQuiescing,
    FailedAFDRank,
    InjectedFFNForwardFailure,
)
from afd_plugin.v1.worker.npu.ffn_model_runner import AFDNPUFFNModelRunner
from afd_plugin.validation import NPU_FFN_WORKER_FQCN, assert_compatible_afd_stack

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

logger = logging.getLogger(__name__)


class AFDNPUFFNWorker(NPUWorker):
    """FFN worker that owns a connector-driven NPU daemon loop."""

    afd_expected_role = "ffn"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        apply_afd_ascend_patches_if_needed()
        super().__init__(*args, **kwargs)
        self._ffn_thread: threading.Thread | None = None
        self._ffn_shutdown_event: threading.Event | None = None
        self._ffn_loop_error: BaseException | None = None

    def init_device(self) -> None:
        assert_compatible_afd_stack(
            self.vllm_config,
            caller="AFDNPUFFNWorker.init_device",
            expected_role="ffn",
            expected_worker_qualname_override=NPU_FFN_WORKER_FQCN,
        )
        fail_if_unsupported_npu_afd_features(self.vllm_config)
        fix_all2all_backend_for_afd(self.vllm_config)
        if self.use_v2_model_runner:
            raise RuntimeError("AFD NPU FFN supports only vllm-ascend MRv1")

        self.device = self._init_device()
        init_workspace_manager(
            self.device,
            npu_afd_num_ubatches(self.vllm_config),
        )
        self.vllm_config.model_config = get_afd_model_config(
            self.vllm_config.model_config,
        )
        self.model_runner = AFDNPUFFNModelRunner(self.vllm_config, self.device)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return {}

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.model_runner.initialize_kv_cache(kv_cache_config)
        self.model_runner.initialize_afd_connector()
        self.start_ffn_server_loop()

    def compile_or_warm_up_model(self) -> float:
        return 0.0

    def execute_model(self, scheduler_output: SchedulerOutput) -> None:
        raise RuntimeError(
            "AFD NPU FFN workers are connector-driven; scheduler-driven "
            "execute_model() is not supported.",
        )

    def start_ffn_server_loop(self) -> None:
        if self._ffn_thread is not None and self._ffn_thread.is_alive():
            self.raise_ffn_loop_error_if_any()
            return

        self.raise_ffn_loop_error_if_any()
        connector = self.model_runner.connector
        if not connector.is_initialized:
            self.model_runner.initialize_afd_connector()

        self._ffn_shutdown_event = threading.Event()
        self._ffn_loop_error = None

        def ffn_worker_loop() -> None:
            try:
                self._run_ffn_server_loop()
            except InjectedFFNForwardFailure as exc:
                self._report_injected_failure()
                if exc.phase == "before_step":
                    self._wait_for_injected_recovery()
                    logger.warning(
                        "AFD NPU FFN worker completed safe-boundary recovery",
                    )
                else:
                    self._ffn_loop_error = exc
                    logger.exception("Injected AFD NPU FFN worker failure")
            except AFDRecoveryQuiescing:
                logger.info("AFD NPU FFN worker quiesced for recovery")
            except Exception as exc:
                self._ffn_loop_error = exc
                logger.exception("AFD NPU FFN worker loop failed")

        self._ffn_thread = threading.Thread(
            target=ffn_worker_loop,
            name="afd-npu-ffn-worker-loop",
            daemon=True,
        )
        self._ffn_thread.start()

    def _report_injected_failure(self) -> None:
        connector = self.model_runner.connector
        channel = connector.recovery_channel
        if channel is None:
            logger.error("Injected NPU FFN failure has no recovery channel")
            return
        latest_notice = channel.latest_notice
        next_epoch = 1 if latest_notice is None else latest_notice.epoch + 1
        channel.report_failure(
            AFDFailureNotice(
                epoch=next_epoch,
                failed_rank=FailedAFDRank("ffn", connector.role_rank),
            ),
        )

    def _wait_for_injected_recovery(self) -> None:
        """Wait for all original ranks to finish the injected recovery epoch."""

        channel = self.model_runner.connector.recovery_channel
        if channel is None:
            raise RuntimeError("Injected NPU FFN recovery has no recovery channel")
        channel.wait_until_recovery_ready()

    def _run_ffn_server_loop(self) -> None:
        event = self._ffn_shutdown_event
        if event is None:
            return

        torch.npu.set_device(self.device)
        self._wait_for_peer_safe_boundary_injection()
        while not event.is_set():
            self.model_runner.fault_injector.before_step()
            self.model_runner.connector.ensure_recovery_running()
            if self.model_runner.connector.control_plane is None:
                self.model_runner.execute_connector_driven_step()
                torch.npu.synchronize()
                continue

            payload = self.model_runner.connector.control_plane.recv_dp_metadata_list()
            dp_metadata_list = payload.dp_metadata_list
            is_attn_graph_capturing = payload.is_graph_capturing
            is_warmup = payload.is_warmup

            self.model_runner.execute_ffn_step(
                dp_metadata_list=dp_metadata_list,
                is_graph_capturing=is_attn_graph_capturing,
                is_warmup=is_warmup,
            )
            torch.npu.synchronize()

    def _wait_for_peer_safe_boundary_injection(self) -> None:
        """Keep peer FFNs off old groups during startup fault injection."""

        connector = self.model_runner.connector
        config = connector.afd_config
        target_rank = config.fault_injection_ffn_rank
        if (
            config.fault_injection_phase != "before_step"
            or target_rank is None
            or connector.role_rank == target_rank
        ):
            return
        channel = connector.recovery_channel
        if channel is None:
            raise RuntimeError("Peer NPU FFN recovery has no recovery channel")
        channel.failure_event.wait()
        channel.wait_until_recovery_ready()

    def raise_ffn_loop_error_if_any(self) -> None:
        error = self._ffn_loop_error
        if error is not None:
            self._ffn_loop_error = None
            raise RuntimeError("AFD NPU FFN worker loop failed") from error

    def stop_ffn_server_loop(self) -> None:
        event = self._ffn_shutdown_event
        if event is not None:
            event.set()
        try:
            self.model_runner.shutdown()
        finally:
            thread = self._ffn_thread
            if thread is not None:
                thread.join(timeout=5)
            self._ffn_thread = None
            self._ffn_shutdown_event = None
        self.raise_ffn_loop_error_if_any()

    def shutdown(self) -> None:
        self.stop_ffn_server_loop()
        super().shutdown()


__all__ = ["AFDNPUFFNWorker"]
