import json
import os
from typing import Any, List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig, MoeWeight
from rtp_llm.models.qwen_v2_moe import Qwen2Moe
from rtp_llm.models.qwen_v3_moe import Qwen3Moe, QWenV3MoeWeight
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    convert_down_proj_,
    convert_gate_up_proj_,
    identity,
    transpose,
)


class QWen35VLMoeWeightInfo(QWenV3MoeWeight):
    def __init__(self, **kwargs):
        QWenV3MoeWeight.__init__(self, **kwargs)
        self.bias = False
        self._use_qk_norm = True

    def _process_meta(self, meta_dicts: Any, weight_keys: List[str]):
        super()._process_meta(meta_dicts, weight_keys)
        self._use_stack_weight = False
        for key in weight_keys:
            if "experts.down_proj" in key or "experts.gate_up_proj" in key:
                self._use_stack_weight = True
                break

    def _get_weight_info(self):
        weights = self._get_hf_weight_info()
        return weights

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        moe_config = MoeConfig(
            expert_num=self.expert_num_,
            align_size=self._align_size,
            routed_scaling_factor=1.0,
            weight_stack=True,
        )
        return [
            MoeWeight(
                sub_weights=[
                    MoeAtomicWeight(
                        W.moe_gate,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix + "layers.{i}.mlp.gate.weight",
                                identity,
                            )
                        ],
                        transpose,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w1,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.mlp.experts.gate_up_proj",
                                convert_gate_up_proj_,
                            )
                        ],
                        identity,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w2,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.mlp.experts.down_proj",
                                convert_down_proj_,
                            )
                        ],
                        identity,
                        config=moe_config,
                    ),
                ],
                config=moe_config,
            )
        ]


class QWen35_VL_MOE(Qwen3Moe):
    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.qwen35vl_moe import Qwen35VLMoeModel

        model_config = self.model_config
        parallelism_config = self.parallelism_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        moe_config = self.moe_config
        self.py_model = Qwen35VLMoeModel(
            model_config,
            parallelism_config,
            self.weight,
            moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    @staticmethod
    def get_weight_cls():
        return QWen35VLMoeWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.attn_config.rope_config.dim = 128
        config.activation_type = "SiGLU"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.norm_type = "rmsnorm"
        config.qk_norm = True
        cls._from_hf(config, ckpt_path)
        return config

    @classmethod
    def _from_hf(cls, config: ModelConfig, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")

        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)
        QWen35_VL_MOE._from_config_json(config, config_json)
        Qwen2Moe.load_moe_config(config, config_json["text_config"])
        config.moe_style = 1
        return config

    @staticmethod
    def _from_config_json(config: ModelConfig, config_json: dict):
        config.mm_related_params.special_tokens.update({"default_mm_token": "<img/>"})
        config.mm_model_config.mm_sep_tokens = [
            [config_json["vision_start_token_id"], config_json["vision_end_token_id"]]
        ]

        config_json = config_json["text_config"]
        config.inter_size = config_json["intermediate_size"]
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json.get(
            "num_key_value_heads", config.attn_config.head_num
        )
        config.attn_config.size_per_head = (
            int(config_json.get("head_dim"))
            if "head_dim" in config_json
            else config_json["hidden_size"] // config.attn_config.head_num
        )
        if config_json.get("hidden_size") is not None:
            config.hidden_size = config_json["hidden_size"]
        config.num_layers = config_json["num_hidden_layers"]
        config.attn_config.rope_config.base = config_json.get(
            "rope_theta", config.attn_config.rope_config.base
        )
        config.vocab_size = config_json["vocab_size"]
        config.attn_config.rope_config.dim = config.attn_config.size_per_head
        config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        config.config_dtype = config_json.get("torch_dtype", None)

        config.attn_config.rope_config.style = 7
        mrope_section = config_json["rope_scaling"].get("mrope_section", [16, 24, 24])
        config.attn_config.rope_config.index_factor = len(mrope_section)
        config.attn_config.rope_config.mrope_dim1 = mrope_section[0]
        config.attn_config.rope_config.mrope_dim2 = mrope_section[1]
        config.attn_config.rope_config.mrope_dim3 = mrope_section[2]
        config.mm_model_config.mm_position_ids_style = 2

        config.mm_related_params.config["ckpt_path"] = config.ckpt_path
        config.mm_model_config.is_multimodal = True


register_model("qwen35_vl_moe", QWen35_VL_MOE, ["Qwen3_5MoeForConditionalGeneration"])
