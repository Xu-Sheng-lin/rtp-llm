#include <torch/all.h>
#include <iostream>
#include <fstream>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_bf16.h>
#include "atrexPA.h"
#include "_pa_decode_dot_kernel.h"
#include "_pa_decode_reduce_kernel.h"

#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
// helpers to check for hip errors
#define HIP_CHECK(ans)                                                                                                 \
    { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(hipError_t code, const char* file, int line) {
    if (code != hipSuccess) {
        const char* prefix = "Triton Error [HIP]: ";
        const char* str;
        hipDrvGetErrorString(code, &str);
        char err[1024] = {0};
        strcat(err, prefix);
        strcat(err, str);
        printf("%s\n", err);
        exit(code);
    }
}

#define CALL_PA_DECODE_REDUCE_KERNEL_CASE(max_num_partitions) \
    case max_num_partitions: \
        HIP_CHECK(_pa_decode_reduce_kernel_512_##max_num_partitions( \
            stream, out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, \
            context_lens_ptr, out.stride(0), out.stride(1), \
            exp_sums.stride(0), exp_sums.stride(1), exp_sums.stride(2), \
            tmp_out.stride(0), tmp_out.stride(1), tmp_out.stride(2), \
            tmp_out.stride(3), grid1[0], grid1[1], grid1[2])); \
        break;

static inline uint64_t next_power_of_2(uint64_t n) {
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

void paged_attention_atrex(torch::Tensor&                      out,
                           torch::Tensor&                      exp_sums,
                           torch::Tensor&                      max_logits,
                           torch::Tensor&                      tmp_out,
                           torch::Tensor&                      query,
                           torch::Tensor&                      key_cache,
                           torch::Tensor&                      value_cache,
                           torch::Tensor&                      context_lens,
                           torch::Tensor&                      block_tables,
                           float                               scale,
                           int64_t                             max_context_len,
                           const std::optional<torch::Tensor>& alibi_slopes) {
    int           num_kv_heads        = key_cache.size(1);
    int           num_seqs            = query.size(0);
    int           num_q_heads         = query.size(1);
    int           kv_blk_sz           = value_cache.size(3);
    int           head_sz             = query.size(2);
    int           query_grp_sz        = num_q_heads / num_kv_heads;
    // NOTE: alibi_slopes is optional.
    hipDeviceptr_t alibi_slopes_ptr =
        alibi_slopes ? reinterpret_cast<hipDeviceptr_t>(alibi_slopes.value().data_ptr()) : nullptr;
    float*          exp_sums_ptr     = reinterpret_cast<float*>(exp_sums.data_ptr());
    float*          max_logits_ptr   = reinterpret_cast<float*>(max_logits.data_ptr());
    __hip_bfloat16* tmp_out_ptr      = reinterpret_cast<__hip_bfloat16*>(tmp_out.data_ptr());
    __hip_bfloat16* query_ptr        = reinterpret_cast<__hip_bfloat16*>(query.data_ptr());
    __hip_bfloat16* key_cache_ptr    = reinterpret_cast<__hip_bfloat16*>(key_cache.data_ptr());
    __hip_bfloat16* value_cache_ptr  = reinterpret_cast<__hip_bfloat16*>(value_cache.data_ptr());
    int*            block_tables_ptr = block_tables.data_ptr<int>();
    int*            context_lens_ptr = context_lens.data_ptr<int>();
 
    __hip_bfloat16*      out_ptr = reinterpret_cast<__hip_bfloat16*>(out.data_ptr());
    const hipStream_t    stream  = at::hip::getCurrentHIPStream().stream();
    if (max_context_len <= 256) {
        std::vector<int32_t> grid = {num_seqs, num_kv_heads, 1};
        HIP_CHECK(_pa_decode_dot_kernel_256(stream,
                                            exp_sums_ptr,
                                            max_logits_ptr,
                                            out_ptr,
                                            query_ptr,
                                            key_cache_ptr,
                                            value_cache_ptr,
                                            block_tables_ptr,
                                            context_lens_ptr,
                                            scale,
                                            alibi_slopes_ptr,
                                            exp_sums.stride(0),
                                            exp_sums.stride(1),
                                            exp_sums.stride(2),
                                            tmp_out.stride(0),
                                            tmp_out.stride(1),
                                            tmp_out.stride(2),
                                            tmp_out.stride(3),
                                            query.stride(0),
                                            query.stride(1),
                                            key_cache.stride(0),
                                            key_cache.stride(1),
                                            key_cache.stride(2),
                                            key_cache.stride(3),
                                            value_cache.stride(0),
                                            value_cache.stride(1),
                                            value_cache.stride(2),
                                            block_tables.stride(0),
                                            grid[0],
                                            grid[1],
                                            grid[2]));
    } else if (max_context_len <= 512) {
        std::vector<int32_t> grid = {num_seqs, num_kv_heads, 1};
        HIP_CHECK(_pa_decode_dot_kernel_512(stream,
                                            exp_sums_ptr,
                                            max_logits_ptr,
                                            out_ptr,
                                            query_ptr,
                                            key_cache_ptr,
                                            value_cache_ptr,
                                            block_tables_ptr,
                                            context_lens_ptr,
                                            scale,
                                            alibi_slopes_ptr,
                                            exp_sums.stride(0),
                                            exp_sums.stride(1),
                                            exp_sums.stride(2),
                                            tmp_out.stride(0),
                                            tmp_out.stride(1),
                                            tmp_out.stride(2),
                                            tmp_out.stride(3),
                                            query.stride(0),
                                            query.stride(1),
                                            key_cache.stride(0),
                                            key_cache.stride(1),
                                            key_cache.stride(2),
                                            key_cache.stride(3),
                                            value_cache.stride(0),
                                            value_cache.stride(1),
                                            value_cache.stride(2),
                                            block_tables.stride(0),
                                            grid[0],
                                            grid[1],
                                            grid[2]));
    } else {
        constexpr int _SEQ_PARTITION_SIZE = 512;
        int max_num_partitions = (max_context_len + _SEQ_PARTITION_SIZE - 1) / _SEQ_PARTITION_SIZE;
        std::vector<int32_t> grid = {num_seqs, num_kv_heads, max_num_partitions};
        HIP_CHECK(_pa_decode_dot_kernel_512(stream,
                                            exp_sums_ptr,
                                            max_logits_ptr,
                                            tmp_out_ptr,
                                            query_ptr,
                                            key_cache_ptr,
                                            value_cache_ptr,
                                            block_tables_ptr,
                                            context_lens_ptr,
                                            scale,
                                            alibi_slopes_ptr,
                                            exp_sums.stride(0),
                                            exp_sums.stride(1),
                                            exp_sums.stride(2),
                                            tmp_out.stride(0),
                                            tmp_out.stride(1),
                                            tmp_out.stride(2),
                                            tmp_out.stride(3),
                                            query.stride(0),
                                            query.stride(1),
                                            key_cache.stride(0),
                                            key_cache.stride(1),
                                            key_cache.stride(2),
                                            key_cache.stride(3),
                                            value_cache.stride(0),
                                            value_cache.stride(1),
                                            value_cache.stride(2),
                                            block_tables.stride(0),
                                            grid[0],
                                            grid[1],
                                            grid[2]));
        std::vector<int32_t> grid1 = {num_seqs, num_kv_heads, 1};
        const auto max_num_partitions_pow_2 = next_power_of_2(max_num_partitions);
        switch(max_num_partitions_pow_2) {
            CALL_PA_DECODE_REDUCE_KERNEL_CASE(2)
            CALL_PA_DECODE_REDUCE_KERNEL_CASE(4)
            CALL_PA_DECODE_REDUCE_KERNEL_CASE(8)
            CALL_PA_DECODE_REDUCE_KERNEL_CASE(16)
            CALL_PA_DECODE_REDUCE_KERNEL_CASE(32)
            default:
                throw std::runtime_error("max_context_len: %d not supported!");
        }
    }
}