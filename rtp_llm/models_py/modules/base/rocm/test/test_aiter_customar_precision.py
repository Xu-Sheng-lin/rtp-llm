"""aiter CustomAllreduce precision test vs NCCL reference.

Two test modes:
  1. Sanity check: rank-r feeds (r+1), expect sum = 1+2+...+ws
  2. Random bf16/fp16 inputs over many shapes/seeds, compare vs NCCL allreduce

Decision criterion: TP=2 sum of 2 random ~N(0,1) bf16 values gives a single
bf16 add — should be bit-exact (max_abs_diff = 0). Any non-zero diff is a
red flag that the AR is doing intermediate rounding.

Run:
    torchrun --nproc_per_node=2 test_aiter_customar_precision.py
"""

import ctypes
import logging
import os
from typing import Tuple

# Preload libpython3.10.so (workaround for rtp_llm/ops hardcoded path)
for _p in ["/opt/conda310/lib/libpython3.10.so", "/usr/local/lib/libpython3.10.so"]:
    if os.path.exists(_p):
        ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
        break

import torch
import torch.distributed as dist

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

HIDDEN = 4096  # Qwen3.5-9B
SEEDS_PER_SHAPE = 30


def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def make_aiter_ar(rank: int, world_size: int):
    """Returns an aiter.CustomAllreduce instance (works in eager mode without graph)."""
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    # aiter requires non-NCCL backend group (uses TCP store + gloo for IPC handshake)
    gloo_group = dist.new_group(backend="gloo")
    ar = CustomAllreduce(
        group=gloo_group,
        device=rank,
        max_size=128 * 1024 * 1024,
        enable_register_for_capturing=True,
    )
    if ar.disabled:
        raise RuntimeError("aiter CustomAllreduce got disabled at init")
    return ar


def aiter_allreduce(ar, tensor: torch.Tensor) -> torch.Tensor:
    """Run aiter custom_all_reduce in eager mode (uses internal staging buffer
    for unregistered input, kernel itself is the same as graph-mode path)."""
    out = ar.custom_all_reduce(tensor)
    if out is None:
        raise RuntimeError(
            f"aiter refused: shape={tuple(tensor.shape)} dtype={tensor.dtype}"
        )
    return out


def nccl_allreduce(tensor: torch.Tensor) -> torch.Tensor:
    out = tensor.clone()
    dist.all_reduce(out, group=dist.group.WORLD)
    return out


def compare(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float]:
    """Return (max_abs_diff, rel_err, mean_abs_diff). Comparison in fp32."""
    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    ref_scale = bf.abs().max().item()
    rel = max_abs / max(ref_scale, 1e-9)
    return max_abs, rel, mean_abs


def sanity_check(ar, rank: int, world_size: int, device: torch.device):
    """rank-r feeds (r+1); expect sum = ws*(ws+1)/2."""
    expected = sum(r + 1 for r in range(world_size))
    inp = torch.full((1, 4096), float(rank + 1), dtype=torch.bfloat16, device=device)
    out = aiter_allreduce(ar, inp.clone())
    torch.cuda.synchronize()
    expected_t = torch.full_like(inp, float(expected))
    diff = (out.float() - expected_t.float()).abs().max().item()
    if rank == 0:
        log.info(
            f"Sanity: rank-r feeds (r+1), expect {expected}, got "
            f"{out[0, 0].item()}, max_diff={diff:.6f}"
        )
    assert (
        diff < 1e-3
    ), f"AR not computing sum! got {out[0,0].item()} expected {expected}"


def main():
    rank, world_size = setup()
    assert world_size == 2, f"This test requires TP=2, got {world_size}"
    device = torch.device(f"cuda:{rank}")

    log.info(f"[rank {rank}] Initializing aiter CustomAllreduce...")
    ar = make_aiter_ar(rank, world_size)
    log.info(f"[rank {rank}] ready")

    sanity_check(ar, rank, world_size, device)
    dist.barrier()

    # Test matrix
    test_shapes = [
        (1, 4096),
        (2, 4096),
        (4, 4096),
        (8, 4096),
        (16, 4096),
        (32, 4096),
        (64, 4096),
        (128, 4096),
        (1, 1024),
        (1, 2048),
        (1, 5120),
        (1, 8192),
    ]
    dtypes = [torch.bfloat16, torch.float16]

    if rank == 0:
        log.info(
            f"\n{'Shape':>16} {'dtype':>8} {'worst':>6} | "
            f"{'max_abs':>10} {'rel_err':>10} {'mean_abs':>10}  result"
        )
        log.info("-" * 90)

    failures = []
    summary = {}

    for dtype in dtypes:
        for shape in test_shapes:
            max_abs_max = 0.0
            rel_max = 0.0
            mean_abs_max = 0.0

            for seed in range(SEEDS_PER_SHAPE):
                # Each rank gets different random data (seed * 100 + rank), so
                # the AR has something non-trivial to sum.
                torch.manual_seed(seed * 100 + rank)
                inp = torch.randn(*shape, dtype=dtype, device=device)

                ref = nccl_allreduce(inp.clone())
                got = aiter_allreduce(ar, inp.clone())
                torch.cuda.synchronize()
                max_abs, rel, mean_abs = compare(got, ref)
                if max_abs > max_abs_max:
                    max_abs_max = max_abs
                if rel > rel_max:
                    rel_max = rel
                if mean_abs > mean_abs_max:
                    mean_abs_max = mean_abs

                tol_abs = 1.5e-2 if dtype == torch.bfloat16 else 5e-3
                if max_abs > tol_abs or rel > 1e-2:
                    failures.append((shape, dtype, seed, max_abs, rel))

            if rank == 0:
                tol_abs = 1.5e-2 if dtype == torch.bfloat16 else 5e-3
                ok = (max_abs_max <= tol_abs) and (rel_max <= 1e-2)
                tag = "OK" if ok else "FAIL"
                log.info(
                    f"{str(shape):>16} {str(dtype).replace('torch.',''):>8}  worst | "
                    f"{max_abs_max:>10.6f} {rel_max:>10.6f} {mean_abs_max:>10.6f}  {tag}"
                )
                summary[(str(shape), str(dtype))] = (ok, max_abs_max, rel_max)

    if rank == 0:
        log.info("\n" + "=" * 90)
        n_total = len(summary)
        n_pass = sum(1 for v in summary.values() if v[0])
        log.info(f"Summary: {n_pass}/{n_total} shape×dtype passed")
        if failures:
            log.info(f"Failures ({len(failures)} cases):")
            for shape, dtype, seed, max_abs, rel in failures[:15]:
                log.info(
                    f"  shape={shape} dtype={dtype} seed={seed}: "
                    f"max_abs={max_abs:.6f} rel={rel:.6f}"
                )
            if len(failures) > 15:
                log.info(f"  ... and {len(failures) - 15} more")
        else:
            log.info("All within bf16 tolerance (max_abs<=1.5e-2, rel<=1e-2).")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
