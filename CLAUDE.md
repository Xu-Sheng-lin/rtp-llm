# Qwen3.5-9B 精度修复项目 — 接手指南

## 背景

本仓库是 rtp-llm 的工作副本，包含 Qwen3.5-9B batch-induced 精度问题的排查与修复代码。
代码当前处于 **v1 修复状态**（精度验证通过，性能有损），v2 优化方案待验证。

## 快速上手

### 1. 读完整工作日志（最重要）

```
qwen35_9b_precision_worklog.md
```

包含：问题描述、双根因分析、v1/v2 方案细节、性能对比数据、所有环境开关、恢复指南。

### 2. 关键源码文件

| 文件 | 内容 |
|------|------|
| `rtp_llm/models_py/modules/factory/linear/impl/rocm/f16_linear.py` | Fix A：FINDALL 固定 GEMM sol |
| `rtp_llm/models_py/model_desc/qwen3_next.py` | Fix B：PyTorch decode fallback + 精度探针 |
| `rtp_llm/models_py/triton_kernels/fla/fused_sigmoid_gating_recurrent.py` | Triton kernel（根因 B 所在） |
| `rtp_llm/models_py/modules/factory/attention/rocm_impl/aiter.py` | attention hash dump |

### 3. 启动服务（v1 精度正确方案）

```bash
cd /home/claudeuser/rtp-llm

KV_CACHE_MEM_MB=6000 HIP_VISIBLE_DEVICES=0,1 REUSE_CACHE=0 \
SEQ_SIZE_PER_BLOCK=1024 KERNEL_SEQ_SIZE_PER_BLOCK=16 WARM_UP=0 \
CONCURRENCY_LIMIT=128 ENABLE_CUDA_GRAPH=0 LOAD_PYTHON_MODEL=1 USE_ASM_PA=0 \
WORLD_SIZE=2 DP_SIZE=1 TP_SIZE=2 EP_SIZE=1 \
DEVICE_RESERVE_MEMORY_BYTES=-30368709120 RESERVER_RUNTIME_MEM_MB=40960 \
AITER_ASM_DIR=/opt/conda310/envs/qwen35dm/lib/python3.10/site-packages/aiter_meta/hsa/ \
MAX_SEQ_LEN=8192 START_PORT=8066 ACT_TYPE=bf16 \
TOKENIZER_PATH=/root/models/Qwen3.5-9B \
CHECKPOINT_PATH=/root/models/Qwen3.5-9B/ \
MODEL_TYPE=qwen35_dense FT_SERVER_TEST=1 \
ROCM_DISABLE_CUSTOM_AG=True FT_DISABLE_CUSTOM_AR=True \
RTP_F16_LINEAR_FINDALL=1 \
RTP_FLA_DECODE_TORCH=1 \
/opt/conda310/envs/qwen35dm/bin/python3.10 -m rtp_llm.start_server
```

注意：`HIP_VISIBLE_DEVICES=0,1` 根据实际 GPU 编号调整。

### 4. 验证精度 / 性能

```bash
/opt/conda310/envs/qwen35dm/bin/python test_taobao_full_verify.py   # 期望 9/9 MATCH
/opt/conda310/envs/qwen35dm/bin/python decode_perf_bench.py          # 测 decode tps
```

### 5. 下一步工作

验证两个 v2 性能优化方案（详见 worklog 第六节）：
- **Fix A v2**：修改 `_pick_findall_sol` 为 lock-after-first-M
- **Fix B v2**：去掉 Triton kernel 的 `do_not_specialize=["N", "T"]`

## 注意事项

- Python 必须用 `/opt/conda310/envs/qwen35dm/bin/python`
- MODEL_TYPE 必须用 `qwen35_dense`（不是 `qwen3_next`）
- 改 Triton kernel 后需 `rm -rf ~/.triton/cache/*`
- 模型在 `/root/models/Qwen3.5-9B`，需要读权限
