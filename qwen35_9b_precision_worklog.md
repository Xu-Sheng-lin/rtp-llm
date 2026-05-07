# Qwen3.5-9B @ rtp-llm 精度问题排查与修复 — 完整工作日志

> 环境：2×MI308X / TP=2 / bf16 / rtp-llm `qwen-0401-dm` 分支
> 排查周期：2026-04-23 ~ 2026-04-30

---

## 一、问题描述

Qwen3.5-9B（hybrid 架构：linear_attn + full_attn）在 rtp-llm 上部署后，**并发请求导致输出不确定**：

- 同一 prompt、temperature=0，串行请求（bs=1）输出正确且稳定
- 并发请求（bs≥2）时，部分 slot 输出格式抖动、答案错误、甚至出现退化复读
- 用淘宝客服真实 prompt（~7600 tokens）和 GSM8K 数学题均可复现

本质是 **batch-induced non-determinism**：不同 batch size 下同一 prompt 的计算路径产生 bit-level 差异，经过多层累积后导致输出语义偏移。

---

## 二、根因定位

通过 hash 逐层回溯 + 全量插桩（32 层 × 8 checkpoint = 256 个探针点一次采集），定位到**两个独立根因**：

### 根因 A：hipBLASLt autoselect（prefill 阶段）

**位置**：`rtp_llm/models_py/modules/factory/linear/impl/rocm/f16_linear.py` 中的 `hipb_mm(x, W, solution_index=-1)`
**首发 diverge 点**：GatedDeltaNet 层的 `in_proj_qkvz` Linear（[M, 4096] × [4096, 6144]）

**机制**：`solution_index=-1` 让 hipBLASLt 自动选择 GEMM 算法。不同 M（token 数，由 batch 拼接决定）选不同的 tile/split-K 算法，不同算法的浮点 reduction order 不同，产生 bit-different 输出。完全绕过 PyTorch，`torch.use_deterministic_algorithms(True)` 无效。

### 根因 B：Triton kernel batch-dependent JIT（decode 阶段）

**位置**：`rtp_llm/models_py/triton_kernels/fla/fused_sigmoid_gating_recurrent.py`
**kernel**：`fused_sigmoid_gating_delta_rule_update_kernel`

**机制**：装饰器 `@triton.jit(do_not_specialize=["N", "T"])` 标记 N（batch size）和 T（seq length）不参与 kernel 特化。但 grid 参数 `(NK, triton.cdiv(V, META["BV"]), N * HV)` 依赖 N，导致 N=1 vs N=3 产生不同 grid → 不同 JIT 行为 → 不同浮点 reduction order。

---

## 三、排除的干扰因素

| 实验 | 结果 | 排除了什么 |
|------|------|-----------|
| `TP_SIZE=1` 单卡 | 仍复现 | TP 通信 / AllReduce |
| `torch.matmul` 替代 `hipb_mm` | 仍不确定 | "标准库一定精确" 的假设（torch 在 ROCm 底层也走 hipBLASLt） |
| fp32 accumulation | hash 变了但仍不一致 | "精度不够" 的假设（是 reduction order 问题，不是累加精度问题） |

---

## 四、已验证的修复方案（v1，精度通过）

### Fix A（v1）：`RTP_F16_LINEAR_FINDALL=1`

在 `f16_linear.py` 中增加 `_pick_findall_sol()` 方法：
1. 首次遇到某个 M 值时，调用 `hipb_findallsols()` 枚举所有可行 GEMM 算法
2. 对返回的 sol 列表排序（`sorted()`），跨不同 M 值求交集
3. 从交集中取 `min()` 作为固定的 `solution_index`
4. 后续所有 M 值统一使用同一 algo → 消除 autoselect 引入的 non-determinism

### Fix B（v1）：`RTP_FLA_DECODE_TORCH=1`

decode 阶段（T=1）时，用纯 PyTorch recurrence（`_torch_fla_decode_recurrence()`）替代 Triton kernel。T=1 计算量极小，理论上无性能损失。

### 启动方式

```bash
RTP_F16_LINEAR_FINDALL=1 \
RTP_FLA_DECODE_TORCH=1 \
ENABLE_CUDA_GRAPH=0 \
MODEL_TYPE=qwen35_dense \
CHECKPOINT_PATH=/root/models/Qwen3.5-9B \
TP_SIZE=2 \
MAX_SEQ_LEN=32768 \
KV_CACHE_MEM_MB=20000 \
SEQ_SIZE_PER_BLOCK=1 \
DEVICE_RESERVE_MEMORY_BYTES=2147483648 \
CONCURRENCY_LIMIT=32 \
PORT=8066 \
/opt/conda310/envs/qwen35dm/bin/python -m rtp_llm.start_server
```

### 精度验证结果

使用淘宝客服真实 prompt（system 5758 字 + user 6893 字，约 7658 tokens），char-level exact match：

| 测试组 | 结果 |
|--------|------|
| bs=1 baseline | 输出正常，len=1813 |
| bs=2 sync barrier | 2/2 MATCH |
| bs=3 sync barrier | 3/3 MATCH |
| bs=4 sync barrier | 4/4 MATCH |
| Serial (conc=1) ×3 | 3/3 PASS，hash 全为 `8b5ec4f2` |
| Concurrent bs=3 ×3 | 3/3 PASS |
| Concurrent bs=4 ×2 | 2/2 PASS |
| **总计** | **17/17 PASS** |

---

## 五、性能影响（v1 方案的代价）

通过正规 A/B 对比测试（baseline 无 fix vs. 开启 fix），使用 `decode_perf_bench.py` 测量：

### Baseline 说明

Qwen3.5-9B hybrid 模型的 decode 路径包含 `causal_conv1d_update`、ssm state 的 data-dependent block_map indexing 等操作，原版就不支持 CUDA Graph（开启后 SIGABRT）。因此 Baseline 和 Fixed 均为无 Graph 模式。

### 各组合的性能对比

| 配置 | bs=1 tps | bs=4 agg tps | bs=8 agg tps |
|------|----------|-------------|-------------|
| Baseline（无 fix） | ~27.6 | ~57 | ~68 |
| FINDALL only | ~26.4 (-4%) | ~50 (-12%) | ~30 (-56%) |
| FLA_DECODE_TORCH only | ~19.3 (-30%) | ~46 (-19%) | ~58 (-15%) |
| Both（v1 完整方案） | ~19.8 (-28%) | ~39 (-32%) | ~26 (-62%) |

### 性能瓶颈分析

- **Fix B（FLA_DECODE_TORCH）是 bs=1 的主要瓶颈**：-30%，PyTorch recurrence 的 Python dispatch overhead 远超 Triton fused kernel
- **Fix A（FINDALL）是高并发的主要瓶颈**：bs=8 时 -56%，原因是 `hipb_findallsols` 在推理热路径上被反复调用——每遇到新的 M 值就要对所有 Linear 层实例（64+ 个）各调用一次 findallsols，每次耗时数百 ms

---

## 六、性能优化替代方案探索（v2，未完成验证）

### Fix A v2：lock-after-first-M

**思路**：第一次调用 `hipb_findallsols` 选定 sol 后立即锁定 `_chosen_sol`，后续所有 M 值直接复用，不再为新 M 调用 findallsols。

**预期效果**：消除 bs=8 时 -56% 的性能损失（那个损失是 findallsols 在推理路径上被每个新 M 反复触发造成的）

**代码变更**（已写入 f16_linear.py，需恢复到 v1 测试时再切换）：
```python
def _pick_findall_sol(self, input, bpreshuffle=False):
    if self._chosen_sol is not None:
        return self._chosen_sol        # 锁定后直接返回
    if torch.cuda.is_current_stream_capturing():
        return -1
    sols = sorted(hipb_findallsols(input, self.weight, ...))
    self._chosen_sol = min(sols) if sols else -1
    return self._chosen_sol
```

**风险**：假设第一个 M 值的 sol 集合对其他 M 值也有效。根据实测数据（2911 个 sol 在所有测试 M 值上交集未坍缩），这个假设大概率成立，但需要验证。

### Fix B v2：去掉 do_not_specialize

**思路**：将 Triton kernel 装饰器从 `@triton.jit(do_not_specialize=["N", "T"])` 改为 `@triton.jit`，让 Triton 为每个 (N, T) 组合独立编译 kernel，而不是用 PyTorch 回退。

**预期效果**：恢复 Triton kernel 的 decode 性能（消除 -30% 的 bs=1 损失），同时通过 kernel 特化保证每个 (N,T) 的结果确定性。

**已知风险**：
1. 如果 non-determinism 的真正根因是 ROCm 运行时的 GPU scheduling 差异（而非 JIT 编译差异），去掉 `do_not_specialize` 不会修复问题
2. 去掉后 Triton 需要为每个新 (N, T) 组合编译 kernel，会增加首次推理延迟（每个 N 值需要编译一次）
3. 需要清理 Triton cache（`rm -rf /root/.triton/cache/*`）才能生效

**验证状态**：已写入代码并启动服务，但服务启动 + 模型加载 + Triton 首次编译耗时较长，未等到服务就绪即切换任务。**精度和性能均未验证。**

---

## 七、已知局限

### 1. FINDALL 交集坍缩

当模型有多种 (K, N) 形状的 Linear 且遇到新 M 值时，兼容 sol 交集可能为空 → 回退 autoselect (-1)。当前 Qwen3.5-9B 有两种形状（[4096, 6144] 和 [4096, 32]），实测 2911 个 sol 的交集未坍缩，但不保证所有场景。

### 2. 跨冷启动一致性

`hipb_findallsols` 返回的可行 sol 集合可能因 GPU 初始状态不同而跨冷启动变化。当前用 `sorted() + min()` 缓解，但如果某次冷启动的 sol 集合恰好缺少 min sol，选出的 sol 会不同 → 跨冷启动输出可能不一致（同一次启动内是确定的）。

### 3. 验证范围

- 仅验证了 Qwen3.5-9B，其他模型需要单独验证
- 仅验证了 taobao 客服 prompt（~7600 tokens），更长或更短的 prompt 是否触发新的 M 值需关注
- Prefill 阶段的 attention kernel 在当前测试中 deterministic，但未排除所有 edge case

### 4. CUDA Graph 不兼容

Qwen3.5-9B hybrid 模型的 decode 路径包含 `causal_conv1d_update`、ssm state 的 data-dependent block_map indexing 等操作，**原版就不支持 CUDA Graph**（开启后 SIGABRT）。这是模型架构限制，与精度 fix 无关。

---

## 八、文件清单与环境开关

### 修改的源码文件（4 个）

| 文件 | 改动内容 |
|------|---------|
| `rtp_llm/models_py/modules/factory/linear/impl/rocm/f16_linear.py` | Fix A：增加 `_pick_findall_sol` + env 开关 |
| `rtp_llm/models_py/model_desc/qwen3_next.py` | Fix B：增加 `_torch_fla_decode_recurrence` + 精度探针 |
| `rtp_llm/models_py/triton_kernels/fla/fused_sigmoid_gating_recurrent.py` | Fix B v2（未验证）：去掉 `do_not_specialize` |
| `rtp_llm/models_py/modules/factory/attention/rocm_impl/aiter.py` | 调试用：attention hash dump |

### 环境开关

| 开关 | 默认 | 作用 |
|------|------|------|
| `RTP_F16_LINEAR_FINDALL=1` | 0 | 启用 findallsols 固定 GEMM sol（Fix A） |
| `RTP_F16_LINEAR_FINDALL_LOG=1` | 0 | 打印 findallsols 选择日志 |
| `RTP_FLA_DECODE_TORCH=1` | 0 | decode 用 PyTorch 替代 Triton kernel（Fix B v1） |
| `RTP_F16_LINEAR_TORCH=1` | 0 | 全部用 torch.matmul 替代 hipb_mm（调试用） |
| `RTP_F16_LINEAR_FP32_ACC=1` | 0 | fp32 累加（调试用） |
| `RTP_F16_LINEAR_FIXED_SOL=N` | 空 | 强制使用指定 sol index（调试用） |
| `RTP_DEBUG_PRECISION=1` | 0 | 启用逐层精度探针 |
| `RTP_DEBUG_ATTN_HASH=1` | 0 | 启用 attention hash dump |
| `RTP_FORCE_DETERMINISTIC=1` | 0 | 调用 `torch.use_deterministic_algorithms` |
| `ENABLE_CUDA_GRAPH=0` | - | 必须关闭（模型架构不支持） |

### 测试/验证脚本

| 脚本 | 用途 |
|------|------|
| `test_taobao_full_verify.py` | 精度测试：bs=1/2/3/4 sync barrier，char-level exact match |
| `taobao_precision_verify.py` | 多 trial 精度测试：conc=1/3/4，hash 比对 |
| `decode_perf_bench.py` | 性能测试：stream 模式，bs=1/4/8，测 decode tps |
| `taobao_messages.json` | 淘宝客服真实 prompt（system 5758 字 + user 6893 字） |

---

## 九、恢复工作指南

### 当前代码状态

代码已恢复到 **v1（精度验证通过）** 状态。v2 优化方案的变更已保存在本文档中，未合入代码。

### 快速启动（v1 精度正确方案）

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

### 验证精度

```bash
cd /home/claudeuser/rtp-llm
/opt/conda310/envs/qwen35dm/bin/python test_taobao_full_verify.py
```

期望结果：bs=1/2/3/4 全部 MATCH。

### 验证性能

```bash
cd /home/claudeuser/rtp-llm
/opt/conda310/envs/qwen35dm/bin/python decode_perf_bench.py
```

### 继续优化（下一步工作）

1. **验证 Fix A v2（lock-after-first-M）**：修改 `f16_linear.py` 中 `_pick_findall_sol` 为简化版（见第六节代码），测精度 + 性能
2. **验证 Fix B v2（去 do_not_specialize）**：修改 Triton kernel 装饰器为 `@triton.jit`（去掉 `do_not_specialize=["N", "T"]`），清 Triton cache，测精度 + 性能
3. 两个 v2 方案独立验证，通过后再组合测试
4. 注意事项：
   - 改 Triton kernel 后需 `rm -rf /root/.triton/cache/*`
   - MODEL_TYPE 必须用 `qwen35_dense`（不是 `qwen3_next`）
   - Python 必须用 `/opt/conda310/envs/qwen35dm/bin/python`

---

## 十、排查方法论总结

1. **全量插桩一轮定位**：不要猜，用 hash dump / precision_probe 逐层回溯找首发 diverge 点
2. **固定 sol_idx 测 bit-exact**：判断 GEMM 是否 autoselect 导致非确定的金标准
3. **每个实验必须能给出"排除了什么"的明确信号**：如 TP=1 排除通信、关 Graph 排除 capture
4. **定位 diverge ≠ 下结论**：必须走通"症状 → 定位 → 证据 → mitigation 验证有效"全链路
5. **修复后必须同时测精度和性能**：只测一头会导致生产决策失误

---

## 十一、后续可探索的优化方向

| 方向 | 思路 | 难度 | 状态 |
|------|------|------|------|
| Fix A v2: lock-after-first-M | 首次 findallsols 后锁定 sol，不再为新 M 调用 | 低 | 代码已写，精度未验证 |
| Fix B v2: 去 do_not_specialize | 让 Triton 为每个 (N,T) 编译独立 kernel | 低 | 代码已写，精度未验证 |
| Sol 持久化 | warmup 后把 (M, K, N) → sol_id 序列化到文件 | 低 | 未开始 |
| FINDALL 选最快 sol | 从兼容交集中 benchmark 选最快而非 min | 低~中 | 未开始 |
| 按形状独立维护 sol 交集 | 避免不同 Linear 形状互相挤压兼容集 | 低 | 未开始 |
| hipBLASLt deterministic API | 类似 cuBLAS CUBLAS_WORKSPACE_CONFIG | 依赖上游 | 未开始 |
