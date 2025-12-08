import itertools
import math
import random
from typing import Dict, List, Optional
from unittest import SkipTest, TestCase, main

import torch
import torch.nn.functional as F

# CUR_PATH = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(str(CUR_PATH), "../../../"))
device = torch.device(f"cuda")

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models.rotary_embedding.deepseek_rotary_embedding import (
    DeepseekV3YarnRotaryEmbedding,
)

from rtp_llm.models_py.modules.common.mla.mla_attention import MlaAttention
from rtp_llm.models_py.modules.common.mla.mla_attention_ref import MlaAttentionRef
from rtp_llm.models_py.modules.rocm.mla.mla_attention_ops import (
    AiterMlaDecodeImpl, AiterMlaPrefillImpl
)
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W

from aiter import dtypes

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 以下是从test_pa.py移植过来的test_paged_attention函数及其依赖代码

uniform_range = (-1, 1)
STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": dtypes.bf16,
    "float": dtypes.fp32,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}

def get_kv_cache_torch_dtype(
    cache_dtype,
    model_dtype = None,
):
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str):
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in ["half", "bfloat16", "float"]:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        elif cache_dtype == "fp8":
            torch_dtype = torch.uint8
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype

def kv_cache_factory(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype,
    model_dtype = None,
    seed: int = 0,
    device = "cuda",
):

    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    x = 16 // torch_dtype.itemsize
    k_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    k_caches = []
    for _ in range(num_layers):
        k_cache = torch.empty(size=k_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            k_cache.uniform_(*uniform_range)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        k_caches.append(k_cache)

    v_cache_shape = (num_blocks, num_heads, head_size, block_size)
    v_caches = []
    for _ in range(num_layers):
        v_cache = torch.empty(size=v_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            v_cache.uniform_(*uniform_range)
        else:
            raise ValueError(f"Does not support value cache of type {cache_dtype}")
        v_caches.append(v_cache)
    return k_caches, v_caches



def create_cos_sin_cache():
    rotary_emb = DeepseekV3YarnRotaryEmbedding(
        64,
        163840,
        10000,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=0.707,
        mscale_all_dim=0.707,
    )
    half_rope_dim = 64 // 2
    cos_cache = rotary_emb.cos_cached[:, :half_rope_dim]
    sin_cache = rotary_emb.sin_cached[:, :half_rope_dim]
    # cos sin cache must be float32
    cos_sin_cache = (
        torch.cat([cos_cache, sin_cache], dim=-1)
        .contiguous()
        .to(device)
        .to(torch.float32)
    )
    return cos_sin_cache


class MLATest(TestCase):
    NUM_TOKENS = [16]
    BATCH_SIZE = [64]
    HIDDEN_SIZES = [2048]
    PAGE_SIZE = [64]
    IS_PREFILLS = [False]
    PREFIX_LENS = [0]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device(device)

    def _run_mla_test(self, num_tokens: int, batch_size: int, hidden_size: int,
                      page_size: int, is_prefill: bool, prefix_len: int):
        input_lengths = [num_tokens] * batch_size
        mock_page_num = 2048
        page_num = math.ceil(prefix_len + num_tokens + page_size - 1 / page_size)
        block_list = [i for i in range(1, page_num + 1)]
        # print(f"block_list: {block_list}")
        kvcache_block_id = torch.tensor(
            block_list,
            dtype=torch.int32,
            device=torch.device("cpu"),
        )

        self.config = GptInitModelParameters(128, 16, 27, 1024, 102400)
        self.config.head_num = 16
        self.config.hidden_size = hidden_size
        self.config.nope_head_dim = 128
        self.config.rope_head_dim = 64
        self.config.kv_lora_rank = 512
        self.config.v_head_dim = 128
        self.config.q_lora_rank = 0
        self.config.seq_size_per_block = 64
        self.config.softmax_extra_scale = 1.0
        self.config.use_mla = True
        self.config.size_per_head = 192

        torch.manual_seed(0)
        input_lengths_t = torch.tensor(
            input_lengths, dtype=torch.int32, device=torch.device("cpu")
        )
        if is_prefill:
            prefix_lengths = [prefix_len]*batch_size
            prefix_lengths_t = torch.tensor(
                prefix_lengths, dtype=torch.int32, device=torch.device("cpu")
            )
        else:
            prefix_lengths_t = torch.zeros(
                len(input_lengths), dtype=torch.int32, device=torch.device("cpu")
            )

        attn_inputs: PyAttentionInputs = PyAttentionInputs()
        attn_inputs.is_prefill = is_prefill
        attn_inputs.prefix_lengths = prefix_lengths_t
        attn_inputs.sequence_lengths = torch.tensor(
            [], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.input_lengths = input_lengths_t
        attn_inputs.kv_cache_block_id_host = kvcache_block_id

        # print(attn_inputs.prefix_lengths)
        # print(attn_inputs.sequence_lengths)
        # print(attn_inputs.input_lengths)
        # print(attn_inputs.kv_cache_block_id_host)

        weights = {}
        weights[W.mla_fusedqkrope_no_lora_w] = torch.randn(
            [
                self.config.hidden_size,
                self.config.size_per_head * self.config.head_num
                + self.config.kv_lora_rank
                + self.config.rope_head_dim,
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.mla_kv_a_ln_gamma] = torch.randn(
            [self.config.kv_lora_rank], dtype=torch.bfloat16, device=device
        )

        weights[W.mla_kc] = torch.randn(
            [self.config.head_num, self.config.nope_head_dim, self.config.kv_lora_rank],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.mla_vc] = torch.randn(
            [self.config.head_num, self.config.kv_lora_rank, self.config.v_head_dim],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.mla_v_w] = torch.randn(
            [self.config.kv_lora_rank, hidden_size],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.mla_k_nope_w] = torch.randn(
            [self.config.kv_lora_rank, hidden_size],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.attn_o_w] = torch.randn(
            [
                self.config.head_num * self.config.v_head_dim,
                self.config.hidden_size,
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        layer_weights: List[Dict[str, torch.Tensor]] = []
        layer_weights.append(weights)

        if is_prefill:
            fmha_impl = AiterMlaPrefillImpl(
                self.config, attn_inputs, layer_weights, create_cos_sin_cache()
            )
        else:
            fmha_impl = AiterMlaDecodeImpl(
                self.config, attn_inputs, layer_weights, create_cos_sin_cache()
            )
        deepseekv2_mla = MlaAttention(self.config, weights, 0)
        cache = torch.randn(
            [mock_page_num, page_size, self.config.kv_lora_rank + self.config.rope_head_dim],
            dtype=torch.bfloat16,
            device=device,
        )
        kv_cache: Optional[KVCache] = KVCache()
        kv_cache.k_cache_base = cache

        deepseekv2_mla_ref = MlaAttentionRef(self.config, weights, 0)

        hidden = torch.randn(
            [num_tokens, self.config.hidden_size],
            dtype=torch.bfloat16,
            device=device,
        )

        out = deepseekv2_mla(hidden, fmha_impl, kv_cache)
        out_ref = deepseekv2_mla_ref(hidden)

        out_norm = out / (torch.norm(out) + 1e-8)
        out_ref_norm = out_ref / (torch.norm(out_ref) + 1e-8)
        self.assertTrue(torch.allclose(out_norm, out_ref_norm, atol=0.01, rtol=0.01))

        out_flat = out.flatten()
        out_ref_flat = out_ref.flatten()
        # 计算余弦相似度
        cosine_sim = F.cosine_similarity(
            out_flat.unsqueeze(0), out_ref_flat.unsqueeze(0), dim=1
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(1.0).to(device).to(cosine_sim.dtype),
                cosine_sim,
                atol=0.01,
                rtol=0.01,
            )
        )

    def test_mlp(self):
        for params in itertools.product(
            self.NUM_TOKENS, self.BATCH_SIZE, self.HIDDEN_SIZES, self.PAGE_SIZE, self.IS_PREFILLS, self.PREFIX_LENS
        ):
            with self.subTest(
                    num_tokens=params[0], batch_size=params[1], hidden_size=params[2],
                    page_size=params[3], is_prefill=params[3], prefix_len=params[4]
            ):
                self._run_mla_test(*params)


if __name__ == "__main__":
    main()
