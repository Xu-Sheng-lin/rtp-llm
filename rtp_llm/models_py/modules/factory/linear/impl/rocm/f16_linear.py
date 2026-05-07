"""ROCm F16 (non-quantized) Linear implementation"""

import os
from functools import lru_cache
from typing import Optional

import torch
from aiter import hipb_create_extension, hipb_findallsols, hipb_mm

from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.ops import HWKernelConfig


class RocmF16LinearBase(LinearBase):
    """ROCm F16 (non-quantized) Linear"""

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        raise NotImplementedError("Subclasses must implement `can_handle`.")

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_config: object = None,
        weight_scale_2: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            weight, weight_scales, input_scales, bias, quant_config, weight_scale_2
        )
        self.weight = weight
        self.bias = bias
        # findall mitigation state (instance-level)
        self._chosen_sol: Optional[int] = None
        self._compatible_sols: Optional[list] = None  # intersection across seen M's
        self._sols_by_m: dict = {}  # cache: M -> list[sol_idx] from findallsols
        self._instance_id = id(self) % 100000

    @staticmethod
    @lru_cache(maxsize=1)
    def init_hipblas():
        hipb_create_extension()

    def _pick_findall_sol(self, input: torch.Tensor, bpreshuffle: bool = False) -> int:
        if self._chosen_sol is not None:
            return self._chosen_sol
        if torch.cuda.is_current_stream_capturing():
            return -1
        sols = sorted(
            hipb_findallsols(
                input,
                self.weight,
                bias=self.bias,
                out_dtype=input.dtype,
                scaleA=None,
                scaleB=None,
                scaleC=None,
                bpreshuffle=bpreshuffle,
            )
        )
        self._chosen_sol = min(sols) if sols else -1
        if _FINDALL_LOG:
            m = input.shape[0]
            print(
                f"[FINDALL] inst={self._instance_id} LOCK M={m} "
                f"N={self.weight.shape[1]} K={self.weight.shape[0]} "
                f"sols={len(sols)} chosen={self._chosen_sol}",
                flush=True,
            )
        return self._chosen_sol

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement `forward`.")


class RocmF16LinearWithSwizzle(RocmF16LinearBase):

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"],
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        return (
            weight_scales is None
            and hw_kernel_config is not None
            and hw_kernel_config.use_swizzleA
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.init_hipblas()
        sol_idx = -1
        if _USE_FINDALL:
            sol_idx = self._pick_findall_sol(input, bpreshuffle=True)
        return hipb_mm(
            input,
            self.weight,
            solution_index=sol_idx,
            bias=self.bias,
            out_dtype=input.dtype,
            scaleA=None,
            scaleB=None,
            scaleOut=None,
            bpreshuffle=True,
        )


_USE_TORCH_F16_LINEAR = os.environ.get("RTP_F16_LINEAR_TORCH", "0") == "1"
_USE_FP32_ACC = os.environ.get("RTP_F16_LINEAR_FP32_ACC", "0") == "1"
_FIXED_SOL = os.environ.get("RTP_F16_LINEAR_FIXED_SOL", "")
_FIXED_SOL_INT = int(_FIXED_SOL) if _FIXED_SOL.lstrip("-").isdigit() else None
_USE_FINDALL = os.environ.get("RTP_F16_LINEAR_FINDALL", "0") == "1"
_FINDALL_LOG = os.environ.get("RTP_F16_LINEAR_FINDALL_LOG", "0") == "1"


class RocmF16LinearNoSwizzle(RocmF16LinearBase):

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"],
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        if weight_scales is not None:
            return False
        if hw_kernel_config is None:
            return True
        elif not hw_kernel_config.use_swizzleA:
            return True
        else:
            return False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if _USE_FP32_ACC:
            out = torch.matmul(input.float(), self.weight.float())
            if self.bias is not None:
                out = out + self.bias.float()
            return out.to(input.dtype)
        if _USE_TORCH_F16_LINEAR:
            out = torch.matmul(input, self.weight)
            if self.bias is not None:
                out = out + self.bias
            return out
        self.init_hipblas()
        sol_idx = _FIXED_SOL_INT if _FIXED_SOL_INT is not None else -1
        if _USE_FINDALL:
            sol_idx = self._pick_findall_sol(input)
        return hipb_mm(
            input,
            self.weight,
            solution_index=sol_idx,
            bias=self.bias,
            out_dtype=input.dtype,
            scaleA=None,
            scaleB=None,
            scaleOut=None,
            bpreshuffle=False,
        )
