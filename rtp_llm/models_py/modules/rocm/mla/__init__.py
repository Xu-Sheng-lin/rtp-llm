from rtp_llm.models_py.modules.common.mla import DECODE_MLA_IMPS, PREFILL_MLA_IMPS
from .mla_attention_ops import AiterMlaDecodeImpl, AiterMlaPrefillImpl

DECODE_MLA_IMPS.append(AiterMlaDecodeImpl)
PREFILL_MLA_IMPS.append(AiterMlaPrefillImpl)
