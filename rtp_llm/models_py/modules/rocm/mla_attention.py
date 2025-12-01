import logging
from typing import Any, Dict, List, Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.mla.flashinfer_mla import check_attention_inputs
from rtp_llm.models_py.modules.fmha import FMHADecodeImplBase, FMHAPrefillImplBase
from rtp_llm.ops.compute_ops import (
    KVCache,
    ParamsBase,
    PyAttentionInputs,
    rtp_llm_ops
)

from librtp_compute_ops.rtp_llm_ops import (
    apply_rope_pos_ids_cos_sin_cache,
    append_paged_mla_kv_cache
)

from aiter.mla import (
    mla_decode_fwd,
    mla_prefill_fwd
)

class MlaParams(ParamsBase):
    def __init__(
        self,
        input_lengths: torch.Tensor
    ):
        super().__init__()

        self.max_seq_len = input_lengths.max().item()
        self.batch_size = input_lengths.size(0)

class AiterMlaPrefillOp:
    def __init__(self, config: GptInitModelParameters):
        self.head_num = config.head_num
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        # Create prefill parameters using pure Python implementation
        self.fmha_params = MlaParams(input_lengths=attn_inputs.input_lengths)
        return self.fmha_params

    def forward(self, q, kv_buffer, fmha_params):
        max_seqlen_q = fmha_params.max_seqlen_q

        qo_indptr = torch.zeros(fmha_params.batch_size + 1, dtype=torch.int)
        kv_indptr = torch.zeros(fmha_params.batch_size + 1, dtype=torch.int)
        kv_last_page_lens = torch.ones(fmha_params.batch_size, dtype=torch.int)
        total_q = qo_indptr[-1].item()
        total_kv = kv_indptr[-1].item()
        num_page = kv_buffer.size(0)
        kv_indices = torch.randint(0, num_page, (total_kv,), dtype=torch.int)

        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        v_head_dim = self.qk_nope_head_dim
        sm_scale = 1.0 / (qk_head_dim ** 0.5)

        out_dtype = torch.bfloat16
        out_asm = torch.empty((total_q, self.head_num, v_head_dim), dtype=out_dtype).fill_(-1)

        attn_logits, attn_lse = mla_prefill_fwd(
            q=q, # shape: [num_seqs, num_heads, head_size]
            kv_buffer=kv_buffer, # shape: [num_page, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
            o=out_asm, # shape: [num_seqs, num_heads, v_head_dim]
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_lens=kv_last_page_lens,
            max_seqlen_q=max_seqlen_q,
            sm_scale=sm_scale
        )

        return attn_logits

class AiterMlaDecodeOp:
    def __init__(self, config: GptInitModelParameters):
        self.head_num = config.head_num
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim

        self.kv_cache_data_type = config.kv_cache_data_type
        self.use_asm_pa = config.hw_kernel_config.use_asm_pa
        self.enable_cuda_graph = (
            config.gpt_init_params.hw_kernel_config.enable_cuda_graph
        )

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        # Create decode parameters using pure Python implementation
        self.fmha_params = MlaParams(input_lengths=attn_inputs.input_lengths)
        return self.fmha_params

    def forward(self, q, kv_buffer, fmha_params):
        max_seqlen_q = fmha_params.max_seqlen_q

        qo_indptr = torch.zeros(fmha_params.batch_size + 1, dtype=torch.int)
        kv_indptr = torch.zeros(fmha_params.batch_size + 1, dtype=torch.int)
        kv_last_page_lens = torch.ones(fmha_params.batch_size, dtype=torch.int)
        total_q = qo_indptr[-1].item()
        total_kv = kv_indptr[-1].item()
        num_page = kv_buffer.size(0)
        kv_indices = torch.randint(0, num_page, (total_kv,), dtype=torch.int)

        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        v_head_dim = self.qk_nope_head_dim
        sm_scale = 1.0 / (qk_head_dim ** 0.5)

        out_dtype = torch.bfloat16
        out_asm = torch.empty((total_q, self.head_num, v_head_dim), dtype=out_dtype).fill_(-1)

        attn_logits, attn_lse = mla_decode_fwd(
            q=q,  # shape: [num_seqs, num_heads, head_size]
            kv_buffer=kv_buffer,  # shape: [num_page, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
            o=out_asm,  # shape: [num_seqs, num_heads, v_head_dim]
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_lens=kv_last_page_lens,
            max_seqlen_q=max_seqlen_q,
            sm_scale=sm_scale
        )

        return attn_logits

class AiterMlaRotaryEmbeddingOp:
    """Rocm rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        cos_sin_cache: torch.Tensor | None,
        kv_lora_rank: int,
        rope_head_dim: int,
        token_per_block: int,
        is_neox_style: bool,
    ) -> None:
        if cos_sin_cache is None:
            raise Exception(f"RotaryEmbedding need cos_sin_cache but got none")
        super().__init__()
        self.head_size = head_size
        self.is_neox_style = is_neox_style
        self.cos_sin_cache = cos_sin_cache
        self.kv_lora_rank = kv_lora_rank
        self.rope_head_dim = rope_head_dim
        self.token_per_block = token_per_block

    def prepare(self, attention_inputs: PyAttentionInputs):
        check_attention_inputs(attention_inputs)
        return rtp_llm_ops.fill_mla_params(
            attention_inputs.prefix_lengths,
            attention_inputs.sequence_lengths,
            attention_inputs.input_lengths,
            attention_inputs.kv_cache_block_id_host,
            self.token_per_block,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        append_ckv_t: torch.Tensor,
        rope_params: Any,
        kv_cache: Optional[KVCache] = None,
    ):
        apply_rope_pos_ids_cos_sin_cache(
            q=query,
            k=key.unsqueeze(1),
            q_rope=query,
            k_rope=key.unsqueeze(1),
            cos_sin_cache=self.cos_sin_cache,
            pos_ids=rope_params.positions,
            interleave=self.is_neox_style,
        )

        if kv_cache is not None:
            k_cache, v_cache = torch.split(
                kv_cache.k_cache_base, [self.kv_lora_rank, self.rope_head_dim], dim=-1
            )

            append_paged_mla_kv_cache(
                append_ckv_t,
                key,
                rope_params.batch_indice,
                rope_params.positions,
                k_cache,
                v_cache,
                rope_params.page_indice,
                rope_params.decode_page_indptr,
                rope_params.paged_kv_last_page_len,
            )

PREFILL_MLA_IMPS: List[type[FMHAPrefillImplBase]] = []
DECODE_MLA_IMPS: List[type[FMHADecodeImplBase]] = []

try:

    class AiterMlaPrefillImpl(FMHAPrefillImplBase):
        def __init__(
            self,
            config: GptInitModelParameters,
            attn_inputs: PyAttentionInputs,
            weights: List[Dict[str, torch.Tensor]],
            cos_sin_cache: torch.Tensor,
            absorb_opt_len: int = 1024,
            use_trt_fmha: bool = False,
        ) -> None:
            # trt prefill not support reuse cache yet
            super().__init__(
                AiterMlaPrefillOp(
                    config=config
                ),
                AiterMlaRotaryEmbeddingOp(
                    head_size=config.nope_head_dim,
                    cos_sin_cache=cos_sin_cache,
                    kv_lora_rank=config.kv_lora_rank,
                    rope_head_dim=config.rope_head_dim,
                    token_per_block=config.seq_size_per_block,
                    is_neox_style=False,
                ),
                attn_inputs,
            )
            self.max_seq_len = attn_inputs.input_lengths.max().item()

            self.has_reuse_cache = False
            if attn_inputs.prefix_lengths is not None:
                self.has_reuse_cache = attn_inputs.prefix_lengths.max().item() > 0

            self.absorb_opt_len = absorb_opt_len
            self.aborb_fmha = AiterMlaDecodeOp(
                config=config,
            )
            self.aborb_fmha.prepare(attn_inputs)

        def compute_prefill_context(
            self,
            q: torch.Tensor,
            compressed_kv: torch.Tensor
        ):
            """Compute prefill context with optimized cache reuse logic."""
            if q.size(0) < self.absorb_opt_len and self.has_reuse_cache:
                return self._handle_short_sequence(q, compressed_kv)
            else:
                return self._handle_long_sequence(
                    q, compressed_kv
                )

        def _handle_long_sequence(
            self,
            q: torch.Tensor,
            compressed_kv: torch.Tensor
        ):
            """Handle long sequences using cache reuse operation."""
            # Handle cache reuse for longer sequences
            return self.fmha_impl.forward(
                q, compressed_kv, self.fmha_params
            )

        def _handle_short_sequence(
            self, q: torch.Tensor, compressed_kv: torch.Tensor
        ) -> torch.Tensor:
            """Handle short sequences using absorb operation."""
            # Split query into nope and pe components
            q_nope, q_pe = torch.split(
                q,
                [self.aborb_fmha.qk_nope_head_dim, self.aborb_fmha.qk_rope_head_dim],
                dim=-1,
            )

            return self.aborb_fmha.forward(
                q_nope, compressed_kv, self.fmha_params
            )

        def forward(
            self,
            q: torch.Tensor,
            compressed_kv: torch.Tensor,
            k_pe: torch.Tensor,
            kv_cache: Optional[KVCache],
            layer_id: int,
        ):

            assert self.rope_kvcache_impl is not None and self.rope_params is not None
            q_pe = q[:, :, self.fmha_impl.qk_nope_head_dim :]
            self.rope_kvcache_impl.forward(
                q_pe, k_pe, compressed_kv, self.rope_params, kv_cache
            )

            if (
                self.attn_inputs.is_prefill
                and self.attn_inputs.cache_store_inputs
                and self.write_cache_store_impl is not None
            ):
                self.write_cache_store_impl(kv_cache)
            assert self.fmha_impl is not None
            return self.compute_prefill_context(
                q, compressed_kv
            )

    PREFILL_MLA_IMPS.append(AiterMlaPrefillImpl)

except ImportError:
    logging.info("AiterMlaPrefillImpl not available, skipped.")

try:

    class AiterMlaDecodeImpl(FMHADecodeImplBase):

        def __init__(
            self,
            config: GptInitModelParameters,
            attn_inputs: PyAttentionInputs,
            weights: List[Dict[str, torch.Tensor]],
            cos_sin_cache: torch.Tensor,
        ) -> None:
            super().__init__(
                AiterMlaDecodeOp(
                    config=config,
                ),
                AiterMlaRotaryEmbeddingOp(
                    head_size=config.nope_head_dim,
                    cos_sin_cache=cos_sin_cache,
                    kv_lora_rank=config.kv_lora_rank,
                    rope_head_dim=config.rope_head_dim,
                    token_per_block=config.seq_size_per_block,
                    is_neox_style=False,
                ),
                attn_inputs,
            )

        def forward(
            self,
            q: torch.Tensor,
            compressed_kv: torch.Tensor,
            k_pe: torch.Tensor,
            kv_cache: Optional[KVCache],
            layer_id: int,
        ):
            assert self.rope_kvcache_impl is not None and self.rope_params is not None
            q_pe = q[:, :, self.fmha_impl.qk_nope_head_dim :]
            self.rope_kvcache_impl.forward(
                q_pe, k_pe, compressed_kv, self.rope_params, kv_cache
            )

            if (
                self.attn_inputs.is_prefill
                and self.attn_inputs.cache_store_inputs
                and self.write_cache_store_impl is not None
            ):
                self.write_cache_store_impl(kv_cache)
            assert self.fmha_impl is not None
            res = self.fmha_impl.forward(
                q, compressed_kv, self.fmha_params
            )
            return res

    DECODE_MLA_IMPS.append(AiterMlaDecodeImpl)

except ImportError:
    logging.info("AiterMlaDecodeImpl not available, skipped.")
