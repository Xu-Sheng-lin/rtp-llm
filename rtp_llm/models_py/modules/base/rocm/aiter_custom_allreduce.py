"""Aiter CustomAllreduce wrapper for ROCm AllReduce.

Uses ``aiter.dist.device_communicators.custom_all_reduce.CustomAllreduce``
(the upstream vLLM-style wrapper) under the hood. Key benefit over a
low-level-ops wrapper:

- ``CustomAllreduce`` records input/output addresses to an internal
  unregistered queue during HIPGraph capture and batch-registers their
  IPC handles via ``register_graph_buffers`` after capture exits. The
  kernel then accesses peer pointers through a C++-maintained slot table
  (indirection), which is robust against PyTorch caching allocator reuse
  — the same scenario that triggered the ``trt-allreduce-stale-ipc-cache``
  BUGFIX (BF16 → Inf → NaN at TP4 / 200+ requests with fast path).

API kept compatible with ``rocm_rccl.py`` callers (``aiter_ar_manager``
singleton with ``ensure_initialized`` / ``should_use`` / ``allreduce``
/ ``close``). Capture-flow helpers (``enter_capture`` / ``exit_capture``
/ ``has_pending_capture`` / ``consume_capture``) are added so the dispatch
in ``hipgraph_capture_all_reduce`` can include aiter as a graph-safe tier.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SIZE = 128 * 1024 * 1024  # 128 MB
_SUPPORTED_WORLD_SIZES = {2, 4, 8}

# Aiter CustomAllreduce supports bf16 and fp16 (no fp8 in graph path).
_SUPPORTED_DTYPES = (
    torch.bfloat16,
    torch.float16,
)


class _AiterARManager:
    """Singleton managing ``aiter.CustomAllreduce`` lifecycle.

    The underlying ``CustomAllreduce`` instance is created lazily on the
    first ``ensure_initialized`` call, after the TP process group is known.
    A gloo (non-NCCL) backend group is created internally for the IPC
    handshake (broadcast_object_list on CPU tensors).
    """

    def __init__(self) -> None:
        self.group: Optional[ProcessGroup] = None
        self.gloo_group: Optional[ProcessGroup] = None
        self.device_id: Optional[int] = None
        self.rank: int = 0
        self.world_size: int = 1
        self.ar = None  # CustomAllreduce instance
        self.max_size: int = _DEFAULT_MAX_SIZE
        self.initialized = False
        self.disabled = False
        self._is_capture_active = False

    def initialize(self, group: ProcessGroup, device_id: int) -> None:
        if self.initialized and self.group is group and self.device_id == device_id:
            return

        world_size = dist.get_world_size(group=group)
        if world_size not in _SUPPORTED_WORLD_SIZES:
            logger.warning(
                "aiter CustomAllreduce: unsupported world_size=%d (supported: %s)",
                world_size,
                sorted(_SUPPORTED_WORLD_SIZES),
            )
            self.disabled = True
            self.initialized = True
            return

        try:
            from aiter.dist.device_communicators.custom_all_reduce import (
                CustomAllreduce,
            )
        except Exception as exc:
            logger.warning("aiter CustomAllreduce: import failed: %s", exc)
            self.disabled = True
            self.initialized = True
            return

        # CustomAllreduce asserts the bound group is non-NCCL backend
        # (uses TCP store + CPU broadcast for IPC handshake).
        try:
            gloo_group = dist.new_group(backend="gloo")
        except Exception as exc:
            logger.warning(
                "aiter CustomAllreduce: gloo group creation failed: %s",
                exc,
            )
            self.disabled = True
            self.initialized = True
            return

        try:
            ar = CustomAllreduce(
                group=gloo_group,
                device=device_id,
                max_size=self.max_size,
                enable_register_for_capturing=True,
            )
        except Exception as exc:
            logger.warning(
                "aiter CustomAllreduce: instantiate failed: %s",
                exc,
                exc_info=True,
            )
            self.disabled = True
            self.initialized = True
            return

        if ar.disabled:
            logger.warning(
                "aiter CustomAllreduce: instance is disabled (cross-node, "
                "world_size, or hardware check failed)"
            )
            self.disabled = True
            self.initialized = True
            return

        self.group = group
        self.gloo_group = gloo_group
        self.device_id = device_id
        self.rank = dist.get_rank(group=group)
        self.world_size = world_size
        self.ar = ar
        self.initialized = True
        self.disabled = False
        logger.info(
            "aiter CustomAllreduce initialized: rank=%d, ws=%d, device=%d",
            self.rank,
            world_size,
            device_id,
        )

    def ensure_initialized(self, group: ProcessGroup, device_id: int) -> bool:
        """Lazy-init on first call. Returns True if usable."""
        if not self.initialized:
            self.initialize(group, device_id)
        return self.initialized and not self.disabled

    def should_use(self, tensor: Tensor, group: ProcessGroup, device_id: int) -> bool:
        """Check whether *tensor* is eligible. State-only — never triggers
        (re-)initialization. Call ``ensure_initialized`` outside of stream
        capture beforehand."""
        if not self.initialized or self.disabled or self.ar is None:
            return False
        if self.group is not group or self.device_id != device_id:
            return False
        if tensor.dtype not in _SUPPORTED_DTYPES:
            return False
        return self.ar.should_custom_ar(tensor)

    def allreduce(self, tensor: Tensor) -> Tensor:
        """AllReduce *tensor*. Returns a NEW tensor (out-of-place semantics).

        In HIPGraph capture mode the wrapper records input/output addresses
        for later IPC registration (zero staging copy). Outside capture, a
        staging-buffer fast path is used.
        """
        if not self.initialized or self.disabled or self.ar is None:
            raise RuntimeError("aiter CustomAllreduce not ready")
        out = self.ar.custom_all_reduce(tensor)
        if out is None:
            raise RuntimeError(
                f"aiter CustomAllreduce refused tensor "
                f"(shape={tuple(tensor.shape)}, dtype={tensor.dtype})"
            )
        return out

    def close(self) -> None:
        """Release the underlying CustomAllreduce + IPC handles. Idempotent."""
        if self.ar is not None:
            try:
                close_fn = getattr(self.ar, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception as exc:
                logger.warning("aiter CustomAllreduce close failed: %s", exc)
        self.ar = None
        self.group = None
        self.gloo_group = None
        self.device_id = None
        self.initialized = False
        self.disabled = True

    # -- HIPGraph capture flow integration -------------------------------------
    # rtp-llm's CudaGraphRunner runs:
    #   enter_hipgraph_capture_mode  ->  enter_capture()  (set _IS_CAPTURING)
    #   ... HIPGraph capture runs, AR calls record IPC slots ...
    #   exit_hipgraph_capture_mode   ->  exit_capture()   (clear flag + flush)
    #
    # We flush eagerly in exit_capture (not deferred to finish_session)
    # because CudaGraphRunner runs replayAndSyncCheck() right after each
    # per-batch capture — before any finish-session hook fires. Deferred
    # flush would leave the slot table uninitialised and the replay would
    # GPU-fault.
    def enter_capture(self) -> None:
        if not self.initialized or self.disabled or self.ar is None:
            return
        self.ar._IS_CAPTURING = True
        self._is_capture_active = True

    def exit_capture(self) -> None:
        """Clear capture flag AND immediately register any pending IPC handles.
        Must NOT be called inside a HIPGraph capture stream."""
        if not self.initialized or self.disabled or self.ar is None:
            return
        self.ar._IS_CAPTURING = False
        self._is_capture_active = False
        if torch.cuda.is_current_stream_capturing():
            logger.warning(
                "aiter exit_capture: stream still capturing; deferring flush"
            )
            return
        try:
            from aiter.ops.custom_all_reduce import get_graph_buffer_count

            count = get_graph_buffer_count(self.ar._ptr)
        except Exception:
            count = None
        if count == 0:
            return
        try:
            self.ar.register_graph_buffers()
            # Ensure register_graph_buffers' device-side per_call_ptrs writes
            # are visible to the subsequent graph replay (different stream).
            torch.cuda.synchronize()
        except Exception as exc:
            logger.warning(
                "aiter register_graph_buffers failed: %s",
                exc,
                exc_info=True,
            )

    def has_pending_capture(self) -> bool:
        if not self.initialized or self.disabled or self.ar is None:
            return False
        try:
            from aiter.ops.custom_all_reduce import get_graph_buffer_count

            return get_graph_buffer_count(self.ar._ptr) > 0
        except Exception:
            return False

    def consume_capture(self) -> None:
        """Drain pending graph-IPC unregistered list (collective). Safe to
        call only outside HIPGraph capture stream."""
        if not self.initialized or self.disabled or self.ar is None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "aiter consume_capture must not run during stream capture."
            )
        if self.has_pending_capture():
            try:
                self.ar.register_graph_buffers()
            except Exception as exc:
                logger.warning(
                    "aiter consume_capture register_graph_buffers failed: %s",
                    exc,
                    exc_info=True,
                )


aiter_ar_manager = _AiterARManager()
