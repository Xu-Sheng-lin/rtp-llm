#pragma once

#include "rtp_llm/models_py/bindings/rocm/FusedRopeKVCacheOp.h"
#include "rtp_llm/models_py/bindings/rocm/PagedAttn.h"
#include "rtp_llm/cpp/kernels/mla_kernels_rocm/page_hip.h"
#include "rtp_llm/cpp/kernels/mla_kernels_rocm/rope_hip.h"
namespace torch_ext {

void registerMlaAttnOpBindings(py::module& rtp_ops_m) {
    rtp_ops_m.def("apply_rope_pos_ids_cos_sin_cache",
    &apply_rope_pos_ids_cos_sin_cache,
    "apply rope pos ids cos sin cache",
    py::arg("q"),
    py::arg("k"),
    py::arg("q_rope"),
    py::arg("k_rope"),
    py::arg("cos_sin_cache"),
    py::arg("pos_ids"),
    py::arg("interleave"),
    py::arg("cuda_stream") = 0);

    rtp_ops_m.def("append_paged_mla_kv_cache",
    &append_paged_mla_kv_cache,
    "append paged mla kv cache",
    py::arg("append_ckv"),
    py::arg("append_kpe"),
    py::arg("batch_indices"),
    py::arg("positions"),
    py::arg("ckv_cache"),
    py::arg("kpe_cache"),
    py::arg("kv_indices"),
    py::arg("kv_indptr"),
    py::arg("kv_last_page_len"),
    py::arg("cuda_stream") = 0);
}

void registerAttnOpBindings(py::module& rtp_ops_m) {
    rtp_llm::registerFusedRopeKVCacheOp(rtp_ops_m);
    rtp_llm::registerPagedAttnDecodeOp(rtp_ops_m);
    registerMlaAttnOpBindings(rtp_ops_m);
}

}  // namespace torch_ext
