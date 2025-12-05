import logging
from typing import Any, Dict, List, Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.linear_factory import LinearFactory
from rtp_llm.models_py.modules.common.mla.utils import check_attention_inputs
from rtp_llm.models_py.modules.common.mha import FMHADecodeImplBase, FMHAPrefillImplBase
from rtp_llm.ops.compute_ops import (
    KVCache,
    ParamsBase,
    PyAttentionInputs,
    rtp_llm_ops
)
from rtp_llm.utils.model_weight import W

import aiter
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
        self.max_seqlen_q = self.max_seq_len
        self.max_seqlen_k = self.max_seq_len
        self.batch_size = input_lengths.size(0)

def rocm_fill_mla_params(
        t_prefix_lengths: torch.Tensor,
        t_sequence_lengths: torch.Tensor,
        t_input_lengths: torch.Tensor,
        t_kv_cache_block_id_host: torch.Tensor,
        seq_size_per_block: int
) -> MlaParams:
    params = MlaParams(t_input_lengths)

    def to_cpu(tensor):
        return tensor.cpu().numpy() if tensor is not None else None

    sequence_lengths = to_cpu(t_sequence_lengths)
    input_lengths = to_cpu(t_input_lengths)
    prefix_lengths = to_cpu(t_prefix_lengths) if t_prefix_lengths.numel() > 0 else None
    kv_cache_block_id = to_cpu(t_kv_cache_block_id_host) if t_kv_cache_block_id_host.numel() > 0 else None

    batch_size = input_lengths.shape[0]
    max_batch_blocks = kv_cache_block_id.shape[1] if kv_cache_block_id is not None else -1

    batch_indice = []
    positions = []
    paged_kv_last_page_len = []
    kvlen = []
    page_indice = []
    reuse_cache_page_indice = []
    decode_page_indptr = [0]
    prefill_page_indptr = [0]
    qo_indptr = [0]
    batch_reuse_info_vec = []

    total_page_idx = 0
    accu_q_len = 0
    accu_kv_len = 0
    max_kv_len = 0
    batch_start_idx = 0

    for i in range(batch_size):
        if prefix_lengths is not None: # prefill
            input_length = input_lengths[i]
            prefix_length = prefix_lengths[i]

            batch_indice.extend([i] * input_length)
            positions.extend([j + prefix_length for j in range(input_length)])

            seq_len = input_length + prefix_length
            accu_q_len += input_length
            accu_kv_len += seq_len
            if seq_len > max_kv_len:
                max_kv_len = seq_len

            page_num = (prefix_length + seq_size_per_block - 1) // seq_size_per_block
            if kv_cache_block_id is not None:
                reuse_cache_page_indice.extend(
                    [kv_cache_block_id[i * max_batch_blocks + j] for j in range(page_num)]
                )

            if prefix_length > 0:
                batch_reuse_info_vec.append([i, prefix_length, batch_start_idx, page_num])
                batch_start_idx += page_num
            else:
                batch_reuse_info_vec.append([i, 0, 0, 0])
        else: # decode
            batch_indice.append(i)
            positions.append(sequence_lengths[i])
            seq_len = sequence_lengths[i] + 1
            accu_q_len += 1
            accu_kv_len += 1

        last_page_len = (seq_len - 1) % seq_size_per_block + 1
        paged_kv_last_page_len.append(last_page_len)
        kvlen.append(seq_len)

        page_num = (seq_len + seq_size_per_block - 1) // seq_size_per_block
        if kv_cache_block_id is not None:
            page_indice.extend([kv_cache_block_id[i * max_batch_blocks + j] for j in range(page_num)])
            total_page_idx += page_num

        decode_page_indptr.append(total_page_idx)
        prefill_page_indptr.append(accu_kv_len)
        qo_indptr.append(accu_q_len)

    def to_hip_tensor(data, dtype=torch.int32):
        return torch.tensor(data, dtype=dtype, device=torch.device("cuda"))

    params.batch_indice = to_hip_tensor(batch_indice)
    params.page_indice = to_hip_tensor(page_indice)
    params.reuse_cache_page_indice = to_hip_tensor(reuse_cache_page_indice)
    params.decode_page_indptr = to_hip_tensor(decode_page_indptr)
    params.prefill_page_indptr = to_hip_tensor(prefill_page_indptr)
    params.paged_kv_last_page_len = to_hip_tensor(paged_kv_last_page_len)
    params.qo_indptr = to_hip_tensor(qo_indptr)
    params.kvlen = to_hip_tensor(kvlen)
    params.positions = to_hip_tensor(positions)
    params.max_seqlen_k = max_kv_len

    if len(reuse_cache_page_indice) > 0:
        flat_info = [elem for sublist in batch_reuse_info_vec for elem in sublist]
        params.batch_reuse_info_vec = to_hip_tensor(flat_info).view(
            len(batch_reuse_info_vec),
            len(batch_reuse_info_vec[0])
        )

    return params

class AiterMlaPrefillOp:
    def __init__(self, config: GptInitModelParameters,
                 weights: List[Dict[str, torch.Tensor]]):
        self.config = config
        self.head_num = config.head_num // config.tp_size
        self.head_num_kv = config.head_num_kv // config.tp_size
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.nope_head_dim
        self.qk_rope_head_dim = config.rope_head_dim
        self.page_size = config.seq_size_per_block
        self.v_head_dim = config.v_head_dim
        self.weights = weights

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        # Create prefill parameters using pure Python implementation
        return rocm_fill_mla_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            self.page_size,
        )

    def forward(self,
                q: torch.Tensor,
                compressed_kv: torch.Tensor,
                k_pe: torch.Tensor,
                fmha_params: Any,
                layer_id: int):
        k_weight = self.weights[layer_id].get(W.mla_kc, None)
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        q_nope = torch.bmm(q_nope.transpose(0, 1), k_weight)
        q_nope = q_nope.transpose(0, 1)
        q = torch.cat([q_nope, q_pe], dim=-1)

        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
        self.k_nope_proj = LinearFactory.create_linear_from_weights(
            self.weights[layer_id], W.mla_k_nope_w, W.mla_k_nope_s, None, self.config
        )

        self.v_proj = LinearFactory.create_linear_from_weights(
            self.weights[layer_id], W.mla_v_w, W.mla_v_s, None, self.config
        )

        k_nope = self.k_nope_proj(compressed_kv)
        value_states = self.v_proj(compressed_kv)

        k_nope = k_nope.view(-1, self.head_num_kv, self.qk_nope_head_dim)
        value_states = value_states.view(-1, self.head_num_kv, self.v_head_dim)

        k = k_pe.new_empty(
            k_pe.size(0), self.head_num_kv, self.qk_rope_head_dim + self.qk_nope_head_dim
        )
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_pe

        res = aiter.flash_attn_varlen_func(
            q,  # Query张量: (total_q, nheads, headdim_q) - 批次中所有query token的总数
            k,  # Key张量: (total_k, nheads_k, headdim_q) - 批次中所有key token的总数
            value_states,  # Value张量: (total_k, nheads_k, headdim_v) - 批次中所有value token的总数
            fmha_params.qo_indptr,  # Query累积序列长度: (batch_size + 1,) dtype=int32 - 用于索引q张量
            fmha_params.prefill_page_indptr,  # Key累积序列长度: (batch_size + 1,) dtype=int32 - 用于索引k/v张量
            fmha_params.max_seqlen_q,  # 批次中最大query序列长度
            fmha_params.max_seqlen_k,  # 批次中最大key序列长度
            dropout_p=0.0,  # Dropout概率 - 评估时应设为0.0
            causal=True,  # 因果注意力掩码 - 用于自回归建模，每个位置只能关注自己和之前的位置
        )

        return res

class AiterMlaDecodeOp:
    def __init__(self, config: GptInitModelParameters,
                 weights: List[Dict[str, torch.Tensor]]):
        self.head_num = config.head_num // config.tp_size
        self.head_num_kv = config.head_num_kv // config.tp_size
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.nope_head_dim
        self.qk_rope_head_dim = config.rope_head_dim
        self.page_size = config.seq_size_per_block
        self.v_head_dim = config.v_head_dim
        self.weights = weights

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        # Create decode parameters using pure Python implementation
        return rocm_fill_mla_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            self.page_size,
        )

    def forward(self, q: torch.Tensor,
                kv_cache: Optional[KVCache],
                fmha_params: Any,
                layer_id: int,
                is_absorb_prefill: bool = False):
        k_weight = self.weights[layer_id].get(W.mla_kc, None)
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        q_nope = torch.bmm(q_nope.transpose(0, 1), k_weight)
        q_nope = q_nope.transpose(0, 1)
        q = torch.cat([q_nope, q_pe], dim=-1)

        max_seqlen_q = fmha_params.max_seq_len
        qo_indptr = fmha_params.qo_indptr
        total_q = qo_indptr[-1].item()

        ckv_cache_shape = kv_cache.k_cache_base.shape
        kv_buffer = kv_cache.k_cache_base.view(ckv_cache_shape[0], ckv_cache_shape[1], 1, ckv_cache_shape[2])
        out_asm = torch.empty((total_q, self.head_num, self.v_head_dim), dtype=torch.bfloat16).fill_(-1)

        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        sm_scale = 1.0 / (qk_head_dim ** 0.5)

        if is_absorb_prefill:
            attn_logits, attn_lse = mla_prefill_fwd(
                q=q,  # shape: [num_seqs, num_heads, head_size]
                kv_buffer=kv_buffer,  # shape: [num_page, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
                o=out_asm,  # shape: [num_seqs, num_heads, v_head_dim]
                qo_indptr=qo_indptr,
                kv_indptr=fmha_params.prefill_page_indptr,
                kv_indices=fmha_params.batch_indice,
                kv_last_page_lens=fmha_params.paged_kv_last_page_len,
                max_seqlen_q=max_seqlen_q,
                sm_scale=sm_scale
            )
        else:
            attn_logits, attn_lse = mla_decode_fwd(
                q=q, # shape: [num_seqs, num_heads, head_size]
                kv_buffer=kv_buffer, # shape: [num_page, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
                o=out_asm, # shape: [num_seqs, num_heads, v_head_dim]
                qo_indptr=qo_indptr,
                kv_indptr=fmha_params.prefill_page_indptr,
                kv_indices=fmha_params.batch_indice,
                kv_last_page_lens=fmha_params.paged_kv_last_page_len,
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
        return rocm_fill_mla_params(
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
        rtp_llm_ops.apply_rope_pos_ids_cos_sin_cache(
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

            rtp_llm_ops.append_paged_mla_kv_cache(
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

try:

    class AiterMlaPrefillImpl(FMHAPrefillImplBase):
        def __init__(
            self,
            config: GptInitModelParameters,
            attn_inputs: PyAttentionInputs,
            weights: List[Dict[str, torch.Tensor]],
            cos_sin_cache: torch.Tensor,
            absorb_opt_len: int = 1024
        ) -> None:
            # trt prefill not support reuse cache yet
            super().__init__(
                AiterMlaPrefillOp(
                    config=config,
                    weights=weights
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
                weights=weights,
            )
            self.aborb_fmha.prepare(attn_inputs)

        def compute_prefill_context(
            self,
            q: torch.Tensor,
            k_pe: torch.Tensor,
            compressed_kv: torch.Tensor,
            kv_cache: Optional[KVCache],
            layer_id: int
        ):
            """Compute prefill context with optimized cache reuse logic."""
            if q.size(0) < self.absorb_opt_len and self.has_reuse_cache:
                return self._handle_short_sequence(
                    q, kv_cache, layer_id
                )
            else:
                return self._handle_long_sequence(
                    q, compressed_kv, k_pe, layer_id
                )

        def _handle_long_sequence(
            self,
            q: torch.Tensor,
            compressed_kv: torch.Tensor,
            k_pe: torch.Tensor,
            layer_id: int
        ):
            """Handle long sequences using cache reuse operation."""
            # Handle cache reuse for longer sequences
            return self.fmha_impl.forward(
                q, compressed_kv, k_pe, self.fmha_params, layer_id
            )

        def _handle_short_sequence(
            self,
            q: torch.Tensor,
            kv_cache: Optional[KVCache],
            layer_id: int
        ) -> torch.Tensor:
            """Handle short sequences using absorb operation."""
            # Split query into nope and pe components
            q_nope, q_pe = torch.split(
                q,
                [self.aborb_fmha.qk_nope_head_dim, self.aborb_fmha.qk_rope_head_dim],
                dim=-1,
            )

            return self.aborb_fmha.forward(
                q_nope, kv_cache, self.fmha_params, layer_id, True
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
                q, k_pe, compressed_kv, kv_cache, layer_id
            )

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
                    weights=weights
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
                q, kv_cache, self.fmha_params, layer_id, False
            )
            return res

except ImportError:
    logging.info("AiterMlaDecodeImpl not available, skipped.")
