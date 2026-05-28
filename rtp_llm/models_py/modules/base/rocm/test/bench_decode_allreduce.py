"""Decode AllReduce microbenchmark: trt_allreduce vs aiter Custom AR vs FlyDSL Custom AR.

Measures latency across various tensor sizes that appear in decode (M=1..128).
Requires TP=2 (2 GPUs). Run with:

    torchrun --nproc_per_node=2 bench_decode_allreduce.py

Environment variables:
    WARMUP_ITERS: warmup iterations (default 50)
    BENCH_ITERS: benchmark iterations (default 200)
"""

import ctypes
import logging
import os
import time
from typing import Callable, List, Optional, Tuple

# Preload libpython3.10.so from conda before rtp_llm tries to load it from /usr/local
_LIBPYTHON_PATHS = [
    "/opt/conda310/lib/libpython3.10.so",
    "/usr/local/lib/libpython3.10.so",
]
for _p in _LIBPYTHON_PATHS:
    if os.path.exists(_p):
        ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
        break

import torch
import torch.distributed as dist

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

WARMUP_ITERS = int(os.environ.get("WARMUP_ITERS", "50"))
BENCH_ITERS = int(os.environ.get("BENCH_ITERS", "200"))
HIDDEN_SIZE = 4096  # Qwen3.5-9B hidden_size (config.json: hidden_size=4096)
INTERMEDIATE_SIZE = 12288  # Qwen3.5-9B intermediate_size (gate/up/down proj)


def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def teardown():
    dist.barrier()
    dist.destroy_process_group()


def bench_fn(
    fn: Callable[[torch.Tensor], torch.Tensor],
    tensor: torch.Tensor,
    warmup: int = WARMUP_ITERS,
    iters: int = BENCH_ITERS,
) -> float:
    """Benchmark a single allreduce function. Returns median latency in us.

    Uses CUDA events for true GPU timing (avoids ~8us host-side
    perf_counter+sync measurement overhead on MI308X).
    """
    for _ in range(warmup):
        fn(tensor)
    torch.cuda.synchronize()

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        ev_start.record()
        fn(tensor)
        ev_end.record()
        ev_end.synchronize()
        times.append(ev_start.elapsed_time(ev_end) * 1000.0)  # ms -> us

    times.sort()
    median = times[len(times) // 2]
    p90 = times[int(len(times) * 0.9)]
    return median, p90


# ---------------------------------------------------------------------------
# Backend: NCCL (baseline)
# ---------------------------------------------------------------------------
def make_nccl_ar() -> Optional[Callable]:
    def nccl_allreduce(tensor: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(tensor, group=dist.group.WORLD)
        return tensor

    return nccl_allreduce


# ---------------------------------------------------------------------------
# Backend: trt_allreduce
# ---------------------------------------------------------------------------
def make_trt_ar(rank: int) -> Optional[Callable]:
    try:
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
            allreduce as trtllm_allreduce,
        )
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
            ensure_trtllm_comm_initialized,
        )

        if not ensure_trtllm_comm_initialized(dist.group.WORLD, rank):
            log.warning("trt_allreduce: init failed (disabled)")
            return None

        def trt_ar(tensor: torch.Tensor) -> torch.Tensor:
            return trtllm_allreduce(tensor, dist.group.WORLD, rank)

        trt_ar._eager_fn = trt_ar
        return trt_ar
    except Exception as e:
        log.warning(f"trt_allreduce: unavailable ({e})")
        return None


# ---------------------------------------------------------------------------
# Backend: aiter Custom AR (low-level ops, bypass broken wrapper)
# ---------------------------------------------------------------------------
def _exchange_ipc_handles(ptr: int, rank: int, world_size: int, device: torch.device):
    """Exchange IPC handle for a meta_buffer pointer across all ranks.

    Returns (handles, offsets, _keeper) where _keeper must be kept alive
    as long as the handles pointers are in use.
    """
    import aiter as aiter_mod

    handle_size = 64  # sizeof(hipIpcMemHandle_t)
    handle_cpu = torch.empty(handle_size, dtype=torch.uint8, device="cpu")
    aiter_mod.get_meta_buffer_ipc_handle(ptr, handle_cpu.data_ptr())

    local_gpu = handle_cpu.to(device)
    gathered_gpu = torch.empty(
        handle_size * world_size, dtype=torch.uint8, device=device
    )
    dist.all_gather_into_tensor(gathered_gpu, local_gpu, group=dist.group.WORLD)
    torch.cuda.synchronize()

    gathered_cpu = gathered_gpu.cpu()
    # Each rank's handle must be a separate contiguous tensor so data_ptr is stable
    handle_tensors = []
    handles = []
    offsets = []
    for i in range(world_size):
        start = i * handle_size
        end = start + handle_size
        h = gathered_cpu[start:end].clone()  # contiguous copy
        handle_tensors.append(h)
        handles.append(int(h.data_ptr()))
        offsets.append(0)
    return handles, offsets, handle_tensors  # keep handle_tensors alive!


def make_aiter_ar(
    rank: int, world_size: int, max_size: int = 128 * 1024 * 1024
) -> Optional[Callable]:
    try:
        import aiter as aiter_mod
        from aiter.ops.custom_all_reduce import all_reduce as aiter_low_level_ar
        from aiter.utility.dtypes import torch_to_aiter_pybind

        device = torch.device(f"cuda:{rank}")
        stream = torch.cuda.current_stream().cuda_stream

        # Allocate meta buffer (returns ptr as int)
        meta_ptr = aiter_mod.allocate_meta_buffer(
            aiter_mod.meta_size() + max_size * 2, stream
        )

        # Allocate rank_data
        rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)

        # Allocate staging buffer (for unregistered allreduce)
        buffer_ptr = aiter_mod.allocate_meta_buffer(max_size, stream)

        # Exchange meta IPC handles
        meta_handles, meta_offsets, _meta_keep = _exchange_ipc_handles(
            meta_ptr, rank, world_size, device
        )

        # init_custom_ar — returns (Tensor_dummy, fa_handle_int) or just int
        fa_result = aiter_mod.init_custom_ar(
            meta_ptr,
            rank_data.data_ptr(),
            rank_data.numel(),
            meta_handles,
            meta_offsets,
            rank,
            True,  # fully_connected
        )
        fa = fa_result[1] if isinstance(fa_result, (tuple, list)) else fa_result

        # Exchange buffer IPC handles and register as input+output staging
        buf_handles, buf_offsets, _buf_keep = _exchange_ipc_handles(
            buffer_ptr, rank, world_size, device
        )
        # Register the staging buffer for both input and output
        aiter_mod.register_input_buffer(fa, buffer_ptr, buf_handles, buf_offsets)
        aiter_mod.register_output_buffer(fa, buffer_ptr, buf_handles, buf_offsets)

        dist.barrier()
        log.info(f"[rank {rank}] aiter Custom AR initialized: fa={fa}")

        def aiter_ar(tensor: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(tensor)
            cur_stream = torch.cuda.current_stream().cuda_stream
            aiter_low_level_ar(
                fa,
                torch_to_aiter_pybind(tensor),
                torch_to_aiter_pybind(out),
                True,  # use_new
                False,  # open_fp8_quant
                buffer_ptr,  # reg_inp_ptr (staging buffer)
                max_size,  # reg_inp_bytes
                buffer_ptr,  # reg_out_ptr (staging buffer)
                max_size,  # reg_out_bytes
                cur_stream,
            )
            return out

        return aiter_ar
    except Exception as e:
        log.warning(f"aiter Custom AR: init failed ({e})")
        import traceback

        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Backend: FlyDSL Custom AR
# ---------------------------------------------------------------------------
def make_flydsl_ar(
    rank: int, world_size: int, max_size: int = 128 * 1024 * 1024
) -> Optional[Callable]:
    try:
        import sys

        # Try pip-installed flydsl first, then fallback to gpu-wiki path
        flydsl_path = "/home/quqing/gpu-wiki/reference-kernels/amd/cdna/flydsl/FlyDSL"
        if flydsl_path not in sys.path:
            sys.path.insert(0, flydsl_path)

        from custom_all_reduce import FlyDSLAllreduce
        from custom_all_reduce_kernel import make_allreduce_kernels

        device = torch.device(f"cuda:{rank}")

        # Pre-compile kernels sequentially per rank to avoid JIT cache race
        for r in range(world_size):
            if rank == r:
                make_allreduce_kernels(
                    N=HIDDEN_SIZE, dtype_str="bf16", world_size=world_size, threads=512
                )
            dist.barrier()

        # FlyDSL uses broadcast_object_list(device="cpu") which requires gloo backend
        gloo_group = dist.new_group(backend="gloo")
        ar_obj = FlyDSLAllreduce(
            group=gloo_group,
            device=device,
            max_size=max_size,
            world_size=world_size,
            rank=rank,
            full_nvlink=True,
        )

        dist.barrier()
        # Warm-up call to ensure kernels are ready
        warmup_tensor = torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        _r = ar_obj.custom_all_reduce(warmup_tensor)
        torch.cuda.synchronize()
        dist.barrier()
        log.info(f"[rank {rank}] FlyDSL Custom AR initialized")

        def flydsl_ar(tensor: torch.Tensor) -> torch.Tensor:
            orig_shape = tensor.shape
            flat = tensor.view(-1)
            result = ar_obj.custom_all_reduce(flat)
            if result is None:
                dist.all_reduce(tensor, group=dist.group.WORLD)
                return tensor
            return result.view(orig_shape)

        flydsl_ar._ar_obj = ar_obj  # expose for graph variant
        return flydsl_ar
    except Exception as e:
        log.warning(f"FlyDSL Custom AR: init failed ({e})")
        import traceback

        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Backend: FlyDSL Custom AR + HIPGraph capture
# ---------------------------------------------------------------------------
def make_flydsl_graph_ar(
    flydsl_eager_fn: Callable,
    pre_capture_shapes: List[Tuple[int, torch.dtype]],
    device: torch.device,
) -> Optional[Callable]:
    """Wrap an eager FlyDSL allreduce with per-shape HIPGraph capture+replay.

    Pre-captures one graph per shape inside a SINGLE FlyDSL capture() context
    so the IPC handle exchange happens once collectively.  Subsequent calls
    just replay — no Python dispatch overhead.
    """
    if flydsl_eager_fn is None:
        return None
    ar_obj = getattr(flydsl_eager_fn, "_ar_obj", None)
    if ar_obj is None:
        return None

    graph_cache: dict = {}

    # Pre-allocate buffers and warm-up kernels (eager path) before capture.
    bufs = {}
    for numel, dtype in pre_capture_shapes:
        inp = torch.randn(numel, dtype=dtype, device=device)
        out = torch.empty_like(inp)
        bufs[(numel, str(dtype))] = (inp, out)
        for _ in range(3):
            ar_obj.custom_all_reduce(inp, out=out)
    torch.cuda.synchronize()
    dist.barrier()

    # Capture all shapes inside ONE FlyDSL capture context so the IPC
    # exchange (collective on exit) happens once for all of them.
    with ar_obj.capture():
        for numel, dtype in pre_capture_shapes:
            inp, out = bufs[(numel, str(dtype))]
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                ar_obj.custom_all_reduce(inp, out=out)
            graph_cache[(numel, str(dtype))] = (g, inp, out)
    torch.cuda.synchronize()
    dist.barrier()
    log.info(f"FlyDSL_Graph pre-captured {len(graph_cache)} shapes")

    # If BENCH_FLYDSL_NOCOPY=1, the bench function replays without copying the
    # caller's input into the captured buffer (measures graph replay overhead
    # only, simulating the "model writes directly to AR input buffer" case).
    no_copy = os.environ.get("BENCH_FLYDSL_NOCOPY", "0") == "1"

    def flydsl_graph_ar(tensor: torch.Tensor) -> torch.Tensor:
        orig_shape = tensor.shape
        flat = tensor.view(-1)
        key = (flat.numel(), str(flat.dtype))
        entry = graph_cache.get(key)
        if entry is None:
            raise RuntimeError(
                f"FlyDSL_Graph: shape {key} not pre-captured. "
                f"Available: {list(graph_cache.keys())}"
            )
        g, inp_buf, out_buf = entry
        if not no_copy:
            inp_buf.copy_(flat)
        g.replay()
        return out_buf.view(orig_shape)

    return flydsl_graph_ar


# ---------------------------------------------------------------------------
# Generic graph-capture wrapper for backends that don't need a custom
# capture() context manager (e.g. trt_allreduce). Output buffer is whatever
# the eager fn returns into; we pre-capture per shape.
# ---------------------------------------------------------------------------
def make_graph_wrapper(
    eager_fn: Callable,
    pre_capture_2d_shapes: List[Tuple[Tuple[int, int], torch.dtype]],
    device: torch.device,
    name: str = "graph",
) -> Optional[Callable]:
    """Pre-capture per 2D shape (M, hidden). Tensor passed at call time must
    match a captured 2D shape exactly (trt_allreduce checks shape[-1])."""
    if eager_fn is None:
        return None
    graph_cache: dict = {}
    bufs = {}
    for shape, dtype in pre_capture_2d_shapes:
        inp = torch.randn(*shape, dtype=dtype, device=device)
        bufs[(shape, str(dtype))] = inp
        for _ in range(3):
            eager_fn(inp)
    torch.cuda.synchronize()
    dist.barrier()

    for shape, dtype in pre_capture_2d_shapes:
        inp = bufs[(shape, str(dtype))]
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = eager_fn(inp)
            graph_cache[(shape, str(dtype))] = (g, inp, out)
        except Exception as e:
            log.warning(f"{name}: capture failed for shape={shape}: {e}")
            return None
    torch.cuda.synchronize()
    dist.barrier()
    log.info(f"{name} pre-captured {len(graph_cache)} shapes")

    def graph_fn(tensor: torch.Tensor) -> torch.Tensor:
        key = (tuple(tensor.shape), str(tensor.dtype))
        entry = graph_cache.get(key)
        if entry is None:
            raise RuntimeError(f"{name}: shape {key} not pre-captured")
        g, inp_buf, out_buf = entry
        g.replay()
        return out_buf

    return graph_fn


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def main():
    rank, world_size = setup()
    assert world_size == 2, f"This benchmark requires TP=2, got world_size={world_size}"

    device = torch.device(f"cuda:{rank}")

    # Tensor sizes to benchmark (simulating decode scenarios)
    # M = batch size in decode, AR tensor shape = (M, hidden_size), dtype = bf16
    # Qwen3.5-9B: allreduce happens after attn output proj and MLP down proj,
    # both have shape (M, hidden_size=4096)
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    hidden_sizes = [HIDDEN_SIZE]

    # Initialize backends
    backends = {}
    backends["NCCL"] = make_nccl_ar()

    pre_capture_2d = [
        ((M, h), torch.bfloat16) for h in hidden_sizes for M in batch_sizes
    ]
    pre_capture_1d = [
        (M * h, torch.bfloat16) for h in hidden_sizes for M in batch_sizes
    ]

    trt = make_trt_ar(rank)
    if trt:
        backends["trt_allreduce"] = trt
        if os.environ.get("BENCH_TRT_GRAPH", "0") == "1":
            trt_g = make_graph_wrapper(trt, pre_capture_2d, device, name="trt_Graph")
            if trt_g:
                backends["trt_Graph"] = trt_g

    aiter = make_aiter_ar(rank, world_size)
    if aiter:
        backends["aiter_CustomAR"] = aiter

    if os.environ.get("BENCH_FLYDSL", "0") == "1":
        flydsl = make_flydsl_ar(rank, world_size)
        if flydsl:
            if os.environ.get("BENCH_FLYDSL_EAGER", "1") == "1":
                backends["FlyDSL_CustomAR"] = flydsl
            if os.environ.get("BENCH_FLYDSL_GRAPH", "1") == "1":
                flydsl_g = make_flydsl_graph_ar(flydsl, pre_capture_1d, device)
                if flydsl_g:
                    backends["FlyDSL_Graph"] = flydsl_g

    if rank == 0:
        log.info(f"Backends available: {list(backends.keys())}")
        log.info(f"Hidden sizes: {hidden_sizes}, dtype: bf16")
        log.info(f"Warmup: {WARMUP_ITERS}, Bench iters: {BENCH_ITERS}")

    TRT_SUPPORTED = {1024, 2048, 2560, 4096, 5120}

    for hidden in hidden_sizes:
        if rank == 0:
            log.info(f"\n{'='*60}")
            log.info(f"Hidden = {hidden}")
            log.info(f"{'='*60}")
            header = f"{'M':>5} {'bytes':>10}"
            for name in backends:
                header += f" | {name:>14s}(p50) {'(p90)':>7s}"
            log.info(header)
            log.info("-" * len(header))

        for M in batch_sizes:
            numel = M * hidden
            byte_size = numel * 2  # bf16 = 2 bytes
            tensor = torch.randn(M, hidden, dtype=torch.bfloat16, device=device)

            results = {}
            for name, fn in backends.items():
                try:
                    if name == "trt_allreduce" and hidden not in TRT_SUPPORTED:
                        results[name] = (None, None)
                        continue
                    test_tensor = tensor.clone()
                    median, p90 = bench_fn(fn, test_tensor)
                    results[name] = (median, p90)
                except Exception as e:
                    if rank == 0:
                        log.warning(f"  {name} failed at M={M},H={hidden}: {e}")
                    results[name] = (None, None)

            if rank == 0:
                line = f"{M:>5} {byte_size:>10}"
                for name in backends:
                    med, p90 = results.get(name, (None, None))
                    if med is not None:
                        line += f" | {med:>10.1f} us {p90:>7.1f}"
                    else:
                        line += f" | {'N/A':>10s} {'N/A':>7s}"
                log.info(line)

    if rank == 0:
        log.info("")
        log.info("Done.")

    teardown()


if __name__ == "__main__":
    main()
