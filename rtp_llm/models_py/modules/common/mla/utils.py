import torch

from rtp_llm.ops.compute_ops import PyAttentionInputs

def check_attention_inputs(attention_inputs: PyAttentionInputs) -> None:
    device = attention_inputs.input_lengths.device
    dtype = torch.int32

    default_tensors = {
        "prefix_lengths": torch.zeros(0, dtype=dtype, device=device),
        "sequence_lengths": torch.zeros(0, dtype=dtype, device=device),
        "kv_cache_block_id_host": torch.zeros(0, dtype=dtype, device=device),
    }

    for attr_name, default_tensor in default_tensors.items():
        if getattr(attention_inputs, attr_name) is None:
            setattr(attention_inputs, attr_name, default_tensor)