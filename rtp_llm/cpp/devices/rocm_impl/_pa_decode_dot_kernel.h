#pragma once

#include <hip/hip_runtime.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

hipError_t _pa_decode_dot_kernel_256(hipStream_t stream, hipDeviceptr_t exp_sums_ptr, hipDeviceptr_t max_logits_ptr, hipDeviceptr_t logits_ptr, hipDeviceptr_t q_ptr, hipDeviceptr_t k_cache_ptr, hipDeviceptr_t v_cache_ptr, hipDeviceptr_t blk_tables_ptrs, hipDeviceptr_t seq_lens_ptr, float scale, hipDeviceptr_t alibi_slopes, int32_t stride_max_logits_s, int32_t stride_max_logits_nh, int32_t stride_max_logits_p, int32_t stride_logits_s, int32_t stride_logits_nh, int32_t stride_logits_p, int32_t stride_logits_g, int32_t stride_q_s, int32_t stride_q_nh, int32_t stride_k_b, int32_t stride_k_nh, int32_t stride_k_hz, int32_t stride_k_bz, int32_t stride_v_b, int32_t stride_v_nh, int32_t stride_v_hz, int32_t stride_bt_s, int32_t gX, int32_t gY, int32_t gZ);
hipError_t _pa_decode_dot_kernel_512(hipStream_t stream, hipDeviceptr_t exp_sums_ptr, hipDeviceptr_t max_logits_ptr, hipDeviceptr_t logits_ptr, hipDeviceptr_t q_ptr, hipDeviceptr_t k_cache_ptr, hipDeviceptr_t v_cache_ptr, hipDeviceptr_t blk_tables_ptrs, hipDeviceptr_t seq_lens_ptr, float scale, hipDeviceptr_t alibi_slopes, int32_t stride_max_logits_s, int32_t stride_max_logits_nh, int32_t stride_max_logits_p, int32_t stride_logits_s, int32_t stride_logits_nh, int32_t stride_logits_p, int32_t stride_logits_g, int32_t stride_q_s, int32_t stride_q_nh, int32_t stride_k_b, int32_t stride_k_nh, int32_t stride_k_hz, int32_t stride_k_bz, int32_t stride_v_b, int32_t stride_v_nh, int32_t stride_v_hz, int32_t stride_bt_s, int32_t gX, int32_t gY, int32_t gZ);
