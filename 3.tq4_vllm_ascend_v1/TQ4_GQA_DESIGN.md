# TQ4 GQA 压缩/解压缩 — Qwen3 系列设计方案

## 1. 背景

### 1.1 目标模型

本方案支持两个 Qwen 系列 GQA 模型：

| | **Qwen3-30B-A3B** | **Qwen3.6-35B-A3B** |
|---|---|---|
| 路径 | `/models/Qwen3-30B-A3B` | `/models/Qwen3.6-35B-A3B` |
| model_type | `qwen3_moe` | `qwen3_5_moe` |
| architecture | `Qwen3MoeForCausalLM` | `Qwen3_5MoeForConditionalGeneration` |
| 模态 | 纯文本 | 多模态（vision + video） |

### 1.2 现状

vLLM-Ascend 已实现 TQ4（TurboQuant 4-bit）KV Cache 压缩，但**仅支持 MLA 架构**（DeepSeek V3/V4、GLM DSA），不适用于 Qwen 等 GQA 架构的模型。

当前 TQ4 方案的核心局限：

| 维度 | 当前 TQ4 (MLA only) | 不可用于 GQA 的原因 |
|------|-------------------|--------------------|
| 压缩目标 | 单个 KV latent（kv_lora_rank=512，K=V 同一压缩表示） | GQA 的 K 和 V 是独立张量 |
| 注意力模式 | MLA-absorb（K=V，dequant 后的 nope 即 K 又即 V） | GQA 的 K ≠ V |
| 注意力内核 | 融合 dequant + sparse flash attention（仅 MLA_ABSORB 模式） | GQA 使用 FIA，不用 SFA |
| KV cache 格式 | 融合 slot = nibbles(256B) + rope(128B) + scale(2B) | GQA 无 rope 嵌入 slot |
| Slots 格式 | fused_slot_size = packed + rope + scale | base_slot_size = packed + scale |
| Codebook | N(0, 1/512) 训练 | head_dim=128/256，K/V 分布不同 |
| Hadamard | query/key 在 Hadamard 空间（MLA latent 重建需要） | GQA 不需要 Hadamard 变换 |

### 1.3 设计原则

1. **不修改现有算子和代码**：新增独立的压缩/解压缩 AscendC 算子，新增独立的 GQA attention 实现类
2. **算子完全解耦注意力**：压缩和解压缩是独立算子，不与任何注意力内核融合
3. **复用现有 FIA**：解压缩后的 K/V 直接喂给 `AscendAttentionBackendImpl`，不写新的 attention kernel
4. **GQA 标识符隔离**：新增算子、类、文件均带 `gqa` 标识，与现有 MLA TQ4 代码完全隔离

---

## 2. 模型架构分析

### 2.1 Qwen3-30B-A3B（纯 GQA，48 层全标准 attention）

**模型参数**：

```json
{
  "architectures": ["Qwen3MoeForCausalLM"],
  "model_type": "qwen3_moe",
  "num_hidden_layers": 48,
  "num_attention_heads": 32,
  "num_key_value_heads": 4,
  "head_dim": 128,
  "hidden_size": 2048,
  "rope_theta": 1000000.0,
  "rope_scaling": {"type": "linear", "factor": 2.0},
  "max_position_embeddings": 40960,
  "sliding_window": null,
  "num_experts": 128,
  "num_experts_per_tok": 8,
  "moe_intermediate_size": 768,
  "intermediate_size": 6144,
  "torch_dtype": "bfloat16"
}
```

**关键特征**：

| 特征 | 值 |
|------|-----|
| Attention 类型 | 纯 GQA（Q:K:V = 32:4:4, ratio = 8:1），**全部 48 层** |
| head_dim | 128 |
| num_kv_heads | 4 |
| Sliding Window | 无 |
| 现有 attention 后端 | `AscendAttentionBackendImpl`（`CUSTOM/ASCEND`） |
| 现有 KV cache | `(k_cache, v_cache)`, shape `(num_blocks, block_size, 4, 128)`, bf16 |
| **TQ4 GQA 适用范围** | **全部 48 层** |

### 2.2 Qwen3.6-35B-A3B（Hybrid Attention，40 层混合架构）

**模型参数**：

```json
{
  "architectures": ["Qwen3_5MoeForConditionalGeneration"],
  "model_type": "qwen3_5_moe",
  "text_config": {
    "model_type": "qwen3_5_moe_text",
    "num_hidden_layers": 40,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "hidden_size": 2048,
    "max_position_embeddings": 262144,
    "rope_parameters": {
      "rope_theta": 10000000,
      "rope_type": "default",
      "partial_rotary_factor": 0.25,
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10]
    },
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "attn_output_gate": true,
    "full_attention_interval": 4,
    "layer_types": [
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      ... (每 4 层重复：3 个 linear + 1 个 full)
    ],
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_value_head_dim": 128,
    "linear_num_value_heads": 32,
    "linear_conv_kernel_dim": 4,
    "mtp_num_hidden_layers": 1
  },
  "vision_config": { ... }
}
```

**关键特征**：

| 特征 | 值 |
|------|-----|
| **两层架构** | **"linear_attention" × 30** + **"full_attention" × 10**（每 4 层 1 个 full） |
| full_attention 层 GQA | Q:K:V = 16:2:2, ratio = 8:1 |
| full_attention head_dim | **256**（是 30B 的 2 倍） |
| full_attention num_kv_heads | **2** |
| linear_attention 层 | Gated DeltaNet（Mamba/SSM 风格），**独立的 KV state 管理** |
| max_position_embeddings | **262,144**（是 30B 的 6.4 倍） |
| RoPE | partial_rotary_factor=0.25 + MRoPE interleaved |
| attn_output_gate | 是（attention 输出有过 sigmoid gate） |
| 现有 attention 后端 | full_attention 用 `AscendQwen3NextAttention`；linear_attention 用 `AscendGatedDeltaNetAttention` |
| **TQ4 GQA 适用范围** | **仅 full_attention 的 10 层**。linear_attention 层使用 SSM state cache，不走 KV cache，不适用 TQ4 |

### 2.3 两模型差异总结及其对 TQ4 GQA 设计的影响

| 维度 | Qwen3-30B-A3B | Qwen3.6-35B-A3B | 设计影响 |
|------|-------------|----------------|---------|
| TQ4 适用层数 | **全部 48 层** | **仅 10 层** (full_attention) | 35B 的 TQ4 收益较小（内存节省约 25% vs 50% 全模型） |
| head_dim | **128** | **256** | 需要支持 head_dim=256 的 slot 格式 |
| num_kv_heads | 4 | 2 | 不影响 |
| GQA ratio | 8:1 | 8:1 | 相同 |
| slot_bytes (per head) | **128B** | **192B** | `align(256/2+2, 64) = 192` |
| RoPE | 标准 rotary | partial_rotary + MRoPE | TQ4 压缩的是去 RoPE 后的 nope 分量，RoPE 不影响 |
| MoE | 128 experts | 256 experts | 无关 |
| Codebook | 需 head_dim=128 训练 | 需 head_dim=256 训练 | **需要两个独立的 codebook** |
| vllm-ascend 支持 | ✅ 已支持 | ✅ 已支持（通过 qwen3_5 patches） | 都已有基础 attention 支持 |

---

## 3. TQ4 压缩参数计算

### 3.1 head_dim=128（Qwen3-30B-A3B）

```
head_dim              = 128
packed_nibble_bytes   = head_dim / 2                 = 64  bytes
scale_bytes           = sizeof(fp16)                 = 2   bytes
base_slot_bytes       = align_up(64 + 2, 64)         = 128 bytes
```

### 3.2 head_dim=256（Qwen3.6-35B-A3B full_attention 层）

```
head_dim              = 256
packed_nibble_bytes   = head_dim / 2                 = 128 bytes
scale_bytes           = sizeof(fp16)                 = 2   bytes
base_slot_bytes       = align_up(128 + 2, 64)        = 192 bytes
```

### 3.3 内存节省

| 模型 | 层数 | 每层 per-token | 原始 KV | TQ4 KV | 节省 |
|------|------|---------------|---------|--------|------|
| **30B（48 层全 GQA）** | 48 | 4×128×2=1024B | **48KB** | **24KB** | **50%** |
| **35B（10 层 GQA）** | 10 | 2×256×2=1024B | **10KB** | **3.75KB** | **62.5%** |
| 35B（30 层 linear） | 30 | SSM state (不适用) | N/A | N/A | N/A |
| **35B（合计 TQ4 受益）** | 10/40 | - | 10KB | 3.75KB | **25%** |

> **注意**：35B 模型的 linear_attention 层使用 Mamba/SSM 风格的状态缓存，不经过 KV cache，因此 TQ4 GQA 的**全模型内存节省仅为 25%**（而非 50%）。但 per-GQA-layer 的节省率 (62.5%) 实际上更高，因为 head_dim=256 时 bf16 到 4-bit 的压缩比更优。

---

## 4. TQ4 压缩方案

### 4.1 算法描述

沿用现有 TQ4 的 4-bit codebook 量化方案，适配 GQA 的 K/V 张量：

```
输入: K 或 V 张量 x ∈ R^{N×H}  （H = head_dim, N = token 数）
     ★ IMPORTANT: 输入是 raw fp32/bf16 张量，不需要预先 L2 归一化 ★
     归一化由 AscendC 内核内部完成。

输出: slot ∈ uint8^{N×S}  （S = base_slot_size）

内核算法 (AscendC tq_gqa_compress):
  1. L2 归一化:   norm = ||x_i||₂,  u = x_i / norm       (内核内部)
  2. 边界量化:     nib[d] = count { bnd[k] ≤ u[d] }      (共 15 个中点分界)
  3. int4b_t 打包: nib ∈ [0,15] → signed s = (nib<8) ? nib : nib-16 → int4b_t (low-first)
  4. 写入 vecNorm: slot[packed:packed+2] = fp16(norm) 的 uint8 视图
  5. 对齐填充:     rest = 0
```

**量化方式**：基于 Lloyd-Max 质心，用相邻质心中点作为 decision boundary，逐元素计数比较（而非 argmin 距离搜索），在 AscendC 上更高效。

**Codebook**：16 级 Lloyd-Max 质心。每个 head_dim 需要独立训练的 codebook：
- `head_dim=128`：已训练（见 `TRAINING_RESULT.md`）
- `head_dim=256`：待训练

### 4.2 Codebook 与数据流

```
phase1_codebook/train_codebook.py
  ├── 训练：Lloyd-Max 迭代 → centroids [16] fp32 sorted
  ├── centSigned = centroids rotated by 8 (仅用于 AscendC 内核的 int4b_t 解包阶段)
  │
  ├── 输出: phase1_codebook/tq4_centroids_gqa.pt  → Python 端加载，tq_gqa.py 传给 kernel
  │         └─ {"centroids": [16] fp32, "cent_signed": [16] fp32}
  │         ★ 未来需扩展到 head_dim=256，产出 tq4_centroids_gqa_256.pt
  │
  └── 输出: phase1_codebook/tq4_centroids_gqa.h   → 编译时可选 include，供内核静态链接

Python → kernel 调用链:
  tq_gqa.compress_kernel(x)
    → cent = get_centroids(head_dim, device)   # 从 .pt 加载原始 centroids（sorted）
    → torch.ops._C_ascend.tq_gqa_compress(x.float(), cent)
       # kernel 接收原始 centroids（非 centSigned），内部自行计算边界中点
```

**关键澄清**：
- Python 传入 kernel 的 `centroids` 参数是**原始 Lloyd-Max 质心**（sorted），不是 centSigned
- Kernel 内部用边界中点 `bnd[i] = (cent[i] + cent[i+1]) * 0.5` 做量化
- centSigned 旋转仅在 int4b_t 打包/解包时使用（signed nibble 与 unsigned 索引间的映射）

### 4.3 解压缩算法

```
输入: slot ∈ uint8^{N×S}
输出: x̂ ∈ bf16^{N×H}

算法:
  1. 解包 nibbles:    slot[0:packed] → H 个 4-bit signed 值
  2. 查表质心:        cent[nibble + 8]（signed → unsigned 索引偏移 8）
  3. 范数修正:        norm_corr = 1 / sqrt(Σ cent²)
  4. 乘以 vecNorm:    x̂ = cent * norm_corr * fp16→f32(slot[packed:packed+2])
```

---

## 5. 文件结构

### 5.1 新增文件

```
vllm-ascend/
├── csrc/
│   ├── tq_gqa_compress/                  # 压缩 AscendC 算子（✅ 已实现 v0）
│   │   ├── op_kernel/
│   │   │   ├── tq_gqa_compress.h         # 内核：归一化 + 边界量化 + int4b_t 打包
│   │   │   └── tq_gqa_compress.cpp
│   │   ├── op_host/                      # 虚拟化/tiling
│   │   │   ├── CMakeLists.txt
│   │   │   ├── tq_gqa_compress_def.cpp
│   │   │   ├── tq_gqa_compress_infershape.cpp
│   │   │   ├── tq_gqa_compress_tiling.cpp
│   │   │   └── tq_gqa_compress_tiling.h
│   │   └── tq_gqa_compress_torch_adpt.h  # PyTorch 绑定
│   │
│   ├── tq_gqa_decompress/                # [新建] 解压缩 AscendC 算子
│   │   ├── op_kernel/
│   │   │   ├── tq_gqa_decompress.h
│   │   │   └── tq_gqa_decompress.cpp
│   │   ├── op_host/
│   │   │   ├── CMakeLists.txt
│   │   │   ├── tq_gqa_decompress_def.cpp
│   │   │   ├── tq_gqa_decompress_infershape.cpp
│   │   │   ├── tq_gqa_decompress_tiling.cpp
│   │   │   └── tq_gqa_decompress_tiling.h
│   │   └── tq_gqa_decompress_torch_adpt.h
│   │
│   ├── torch_binding.cpp                 # [修改] 注册新算子
│   ├── torch_binding_meta.cpp            # [修改] meta 注册
│   ├── cmake/func.cmake                  # [修改] 添加编译路径
│   └── build_aclnn.sh                    # [修改] 添加编译目标
│
├── vllm_ascend/
│   ├── turboquant/
│   │   └── tq_gqa.py                     # GQA TQ4 Python 封装（✅ 已实现 v0，待扩展 head_dim=256）
│   │
│   └── attention/
│       └── tq4_gqa_v1.py                 # [新建] GQA TQ4 Attention Backend + Impl + Spec
│
└── tests/
    └── ut/
        └── ops/
            ├── test_tq_gqa_compress.py    # [新建] 压缩算子单元测试（需覆盖 128/256）
            └── test_tq_gqa_decompress.py  # [新建] 解压缩算子单元测试（需覆盖 128/256）
```

### 5.2 部署目录（当前 repo）

```
3.tq4_vllm_ascend_v1/
├── TQ4_GQA_DESIGN.md           # 本文件
├── deploy_30B.sh                # 30B 模型部署脚本
├── deploy_35B.sh                # 35B 模型部署脚本
├── benchmark.sh                 # 性能 benchmark
├── smoke_test.sh                # 冒烟测试
├── tq_gqa.py                    # GQA TQ4 Python 封装
├── test_tq4_ops.py              # 算子测试
├── test_tq_gqa_unittest.py      # 单元测试
├── build.sh / install.sh        # 编译安装脚本
│
└── phase1_codebook/             # Phase 1 — Codebook 训练
    ├── TRAINING_RESULT.md       # 训练结果（当前仅 head_dim=128）
    ├── train_codebook.py        # 训练脚本（需扩展支持 head_dim=256）
    ├── tq4_centroids_gqa.pt     # head_dim=128 码本权重
    └── tq4_centroids_gqa.h      # head_dim=128 AscendC 常量头文件
```

### 5.3 修改文件（最小化改动）

```
vllm-ascend/
├── vllm_ascend/attention/attention_v1.py   # 不直接修改；TQ4 子类覆盖 forward
├── vllm_ascend/worker/model_runner_v1.py   # [修改] KV cache dtype/size 适配
└── vllm_ascend/patch/platform/
    └── patch_kv_cache_interface.py          # [修改] 新增 GQA TQ4 attention spec
```

---

## 6. 算子接口设计

### 6.1 压缩算子 `tq_gqa_compress`（✅ 已实现）

**PyTorch 接口**：
```python
torch.ops._C_ascend.tq_gqa_compress(
    latent: Tensor,      # [N, head_dim] fp32 — ★ raw K or V，未归一化 ★
    centroids: Tensor,   # [16] fp32 — Lloyd-Max 原始质心（sorted），非 centSigned
) -> Tensor              # [N, base_slot_size] uint8
```

**AscendC kernel 输入**：
- `latent [N, head_dim] fp32`: 原始 K/V 数据，**内核内部完成 L2 归一化**
- `centroids [16] fp32`: 原始 Lloyd-Max 质心，从 `.pt` 文件加载后由 Python 传入
- `head_dim`: 从 tiling data 传入（支持 64/128/256 等 2 的幂次）

**AscendC kernel 输出**：
- `slot [N, base_slot_size] uint8`

**内核内部流程**（参考 `tq_gqa_compress.h`）：
1. 读取 raw fp32 → 计算 `norm = sqrt(Σx²)` → 归一化 `u = x / norm`
2. 对 15 个边界中点做 `CompareScalar`（CANN 9.0 中已更名为 `Compares`）计数 → nibble index
3. nibble → signed → int4b_t 打包（low-first）
4. 写入 `fp16(norm)` 到 packed_bytes 偏移处
5. 剩余字节填零

> **CANN 9.0.0 兼容性**：现有 kernel 使用 `CompareScalar`，在 CANN 9.0.0 中需替换为 `Compares`。其他 API（`Duplicate`, `Mul`, `Muls`, `Add`, `ReduceSum`, `DataCopy`, `ReinterpretCast`, `Cast`, `Select`, `PipeBarrier`）均无变化。详见 §12 CANN 9.0.0 对齐。

**base_slot_size 计算**：
```cpp
uint32_t packed_bytes = head_dim / 2;
uint32_t base_slot_size = ((packed_bytes + 2 + 63) / 64) * 64;  // 对齐到 64B
```

### 6.2 解压缩算子 `tq_gqa_decompress`（❌ 待实现）

**PyTorch 接口**：
```python
torch.ops._C_ascend.tq_gqa_decompress(
    slots: Tensor,       # [N, base_slot_size] uint8, TQ4 压缩的 slots
    centroids: Tensor,   # [16] fp32, Lloyd-Max 原始质心（sorted）
    head_dim: int,       # 解压后的维度
) -> Tensor              # [N, head_dim] bf16
```

**AscendC kernel 输入**：
- `slots [N, base_slot_size] uint8`
- `centroids [16] fp32`（原始质心，sorted，**注意**：内核内部需做 centSigned 旋转用于查表）
- `head_dim`: 运行时参数（需同时支持 128 和 256）

**AscendC kernel 输出**：
- `output [N, head_dim] bf16`

**内核应实现的流程**：
1. 从 slot 解包 int4b_t → signed nibbles → unsigned index = s + 8
2. 用 centSigned 查表：`cent_signed[(unsigned)nib]`
3. 计算 `norm_corr = 1/sqrt(Σcent²)`
4. 读取 fp16 vecNorm → `output = cent_vals * norm_corr * vecNorm` → cast 到 bf16

**head_dim=256 适配**：decompress kernel 不硬编码 head_dim，与 compress kernel 一样从 tiling data 接收。

### 6.3 与现有算子对比

| 对比维度 | 现有 `tq_compress_latent` | `tq_gqa_compress` | `tq_gqa_decompress` |
|---------|--------------------------|-------------------|---------------------|
| 适用范围 | MLA latent only | GQA K/V（head_dim=128/256） | GQA K/V（head_dim=128/256） |
| head_dim | 硬编码 512 | 运行时传入 | 运行时传入 |
| centroids | 硬编码 16 个 | 运行时传入 | 运行时传入 |
| 输入归一化 | 否（内核内部归一化） | 否（内核内部归一化） | - |
| 额外变换 | Hadamard（Python 端 `@ _PIT`） | 无 | 无 |
| slot_bytes (128) | 320B | 128B | - |
| slot_bytes (256) | N/A | 192B | - |

---

## 7. GQA TQ4 Attention 实现设计

### 7.1 架构总览

```
  ┌──── Qwen3-30B-A3B (48 层全 GQA, head_dim=128) ────┐
  │                                                     │
  │  Prefill: K,V → compress → KV Cache (int8 slots)   │
  │           └→ raw K,V → FIA attention               │
  │                                                     │
  │  Decode:  KV Cache → decompress active blocks      │
  │           └→ bf16 K,V → FIA attention              │
  └─────────────────────────────────────────────────────┘

  ┌──── Qwen3.6-35B-A3B (40 层混合) ──────────────────┐
  │                                                     │
  │  full_attention 层 ×10 (head_dim=256):              │
  │    → TQ4 GQA 路径（同 30B，但 head_dim=256）        │
  │                                                     │
  │  linear_attention 层 ×30 (GDN/Mamba):               │
  │    → 不适用 TQ4（使用 SSM state cache）             │
  └─────────────────────────────────────────────────────┘
```

### 7.2 实现方式

**方案**：新建 `AscendTQ4GQAAttentionBackendImpl`，继承 `AscendAttentionBackendImpl`，**仅** override `forward()` 方法。与 `AscendC8AttentionBackendImpl` 的模式一致。

**Qwen3.6 的 full_attention 层兼容性**：
- 35B 的 full_attention 层使用 `Qwen3NextAttention`（已由 `AscendQwen3NextAttention` patch 处理）
- `attn_output_gate` 已由 vllm-ascend patch 处理（`patch_qwen3_5.py:70-73`）
- TQ4 GQA Backend 在 full_attention 层插入 compress/decompress，不影响现有的 attention output gate 逻辑

**linear_attention 层的处理**：
- TQ4 GQA Backend **不覆盖** linear_attention 层（这些层使用 `AscendGatedDeltaNetAttention`）
- 通过 `layer_types` 判断当前层是否为 full_attention，仅在 full_attention 层激活 TQ4

### 7.3 forward 伪代码

```
forward(query, key, value, kv_cache, attn_metadata, output):

  # ── 仅处理 full_attention 层（linear_attention 层不走此路径）──

  # ── Prefill/Chunked-Prefill 阶段 ──
  if key is not None and value is not None:
    # 保存 raw K,V 用于 attention 计算（不等压缩后的值）
    raw_key, raw_value = key, value

    # [注入点 1] 压缩并写入 KV cache
    k_slots = tq_gqa_compress(key.float(), centroids)
    v_slots = tq_gqa_compress(value.float(), centroids)
    write_int8_slots_to_cache(k_slots, v_slots, kv_cache, slot_mapping)

    # 用 raw K,V 做 attention（无压缩回环）
    return parent.forward_impl(query, raw_key, raw_value, kv_cache, ...)

  # ── Decode 阶段 ──
  if attn_metadata.attn_state == DecodeOnly:
    # [注入点 2] 只解压 block_table 中活跃的 block
    active_key   = decompress_active_blocks(kv_cache[0], block_table, centroids, head_dim)
    active_value = decompress_active_blocks(kv_cache[1], block_table, centroids, head_dim)

    # [注入点 3] Decompressed K,V → FIA（不用 PA）
    return parent.forward_fused_infer_attention(
        query, active_key, active_value, ...)
```

### 7.4 注入点详解

#### 注入点 1：Prefill 压缩写入 KV Cache

```python
def _tq4_compress_and_cache(self, key, value, kv_cache, slot_mapping):
    """
    key, value: [num_tokens, num_kv_heads, head_dim] bf16 — raw K/V
    kv_cache: tuple of tensors, int8, [num_blocks, block_size, num_kv_heads, slot_bytes]
    """
    # 1. TQ4 压缩：内核内部处理 L2 归一化，传入 raw fp32
    k_slots = torch.ops._C_ascend.tq_gqa_compress(key.float().contiguous(), self._centroids)
    v_slots = torch.ops._C_ascend.tq_gqa_compress(value.float().contiguous(), self._centroids)

    # 2. 写入 KV cache（通过 slot_mapping scatter）
    key_cache = kv_cache[0]   # [num_blocks, block_size, num_kv_heads, slot_bytes], int8
    value_cache = kv_cache[1]

    key_cache_flat = key_cache.view(-1, key_cache.shape[-1])
    value_cache_flat = value_cache.view(-1, value_cache.shape[-1])
    key_cache_flat[slot_mapping] = k_slots.view(torch.int8)
    value_cache_flat[slot_mapping] = v_slots.view(torch.int8)
```

**head_dim 无关性**：上述代码不依赖 head_dim 的硬编码值。`slot_bytes` 从 `kv_cache.shape[-1]` 或 `self._slot_bytes` 获取，对 128 和 256 都适用。

#### 注入点 2：Decode 解压缩活跃 Block

```python
def _decompress_active_blocks(self, cache_slots, block_table, centroids, head_dim):
    """
    cache_slots: [num_blocks, block_size, num_kv_heads, slot_bytes] int8
    block_table: [batch_size, max_blocks_per_seq] int32
    centroids: [16] fp32 — 针对当前 head_dim 训练的码本
    head_dim: int — 解压目标维度

    Returns: bf16 tensor [num_active_tokens, num_kv_heads, head_dim]
    """
    # 只解压活跃 block（不解压全量 cache）
    active_block_ids = block_table.unique()
    active_slots = cache_slots[active_block_ids]

    N = active_slots.numel() // active_slots.shape[-1]
    slots_flat = active_slots.reshape(-1, self._slot_bytes)

    decompressed = torch.ops._C_ascend.tq_gqa_decompress(
        slots_flat, centroids, head_dim
    )
    return decompressed.view(-1, self.num_kv_heads, head_dim).bfloat16()
```

#### 注入点 3：复用父类 FIA，不使用 PA

**TQ4 GQA Decode 阶段统一走 FIA**，原因：
- FIA 接受显式 key/value 参数，可以传入解压后的 bf16 张量
- PA 直接读 `self.key_cache`（此时是 int8 slot 格式，PA 不认识 int8 slots）
- 后续可考虑实现 int8-aware paged attention 来支持 PA 路径

```python
output = self.forward_fused_infer_attention(
    query, decompressed_key, decompressed_value, attn_metadata, output, kv_cache
)
```

### 7.5 Backend 注册

```python
# vllm_ascend/attention/tq4_gqa_v1.py

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND_TQ4_GQA")
class AscendTQ4GQABackend(AscendAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_TQ4_GQA"

    @staticmethod
    def get_impl_cls():
        return AscendTQ4GQAAttentionBackendImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int, block_size: int,
        num_kv_heads: int, head_size: int,
        cache_type: str = "",
    ) -> tuple[int, ...]:
        slot_bytes = get_slot_bytes(head_size)  # 自动处理 128→128, 256→192
        return (2, num_blocks, block_size, num_kv_heads, slot_bytes)
```

### 7.6 KV Cache 适配

#### Attention Spec

```python
@dataclass(frozen=True)
class AscendGQATQ4AttentionSpec(AttentionSpec):
    head_dim: int         # 128 or 256
    slot_bytes: int       # derived: align(head_dim/2+2, 64)

    @property
    def page_size_bytes(self) -> int:
        k_bytes = self.block_size * self.num_kv_heads * self.slot_bytes
        v_bytes = k_bytes
        return k_bytes + v_bytes
```

#### 对 Qwen3.6 的特殊处理

35B 模型的 `layer_types` 中，只有标记为 `"full_attention"` 的层使用 TQ4 GQA Backend。`"linear_attention"` 层使用现有的 `AscendGatedDeltaNetAttention`，不经过 KV cache 压缩路径。

这要求 attention backend 选择层粒度（layer-granular），而非模型粒度：
```python
if layer_type == "full_attention" and envs.VLLM_ASCEND_ENABLE_TQ4_GQA:
    backend = "ASCEND_TQ4_GQA"
elif layer_type == "full_attention":
    backend = "ASCEND"
elif layer_type == "linear_attention":
    backend = "GDN"  # 现有路径
```

---

## 8. 配置与启用

### 8.1 环境变量

```python
# vllm_ascend/envs.py
"VLLM_ASCEND_ENABLE_TQ4_GQA": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_TQ4_GQA", 0)),
```

### 8.2 启动示例

```bash
# Qwen3-30B-A3B
export VLLM_ASCEND_ENABLE_TQ4_GQA=1
vllm serve /models/Qwen3-30B-A3B \
    --dtype bfloat16 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager

# Qwen3.6-35B-A3B（仅 full_attention 的 10 层走 TQ4）
export VLLM_ASCEND_ENABLE_TQ4_GQA=1
vllm serve /models/Qwen3.6-35B-A3B \
    --dtype bfloat16 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager
```

说明：MVP 阶段 `--enforce-eager` 必须开启，TQ4 路径暂不支持图模式捕获（见 10.4）。

---

## 9. 测试方案

### 9.1 算子单元测试：compress ↔ decompress roundtrip

```python
def test_tq_gqa_roundtrip(head_dim=128):
    """
    验证：raw fp32 → compress → decompress → cosine similarity > 0.99

    注意：compress 内核期望 raw 输入（未归一化），
    所以 roundtrip 的参考值是 raw tensor，不是手动归一化后的值。
    """
    N = 64
    centroids = tq_gqa.get_centroids(head_dim)  # 从 .pt 加载原始 centroids

    x_raw = torch.randn(N, head_dim).bfloat16()
    slots = torch.ops._C_ascend.tq_gqa_compress(x_raw.float(), centroids)
    x_hat = torch.ops._C_ascend.tq_gqa_decompress(slots, centroids, head_dim)

    cos_sim = torch.nn.functional.cosine_similarity(
        x_raw.float().view(-1).unsqueeze(0), x_hat.float().view(-1).unsqueeze(0)
    ).item()
    assert cos_sim > 0.99

# 两个 head_dim 都测
test_tq_gqa_roundtrip(head_dim=128)
test_tq_gqa_roundtrip(head_dim=256)
```

### 9.2 GQA Attention 功能测试

```python
def test_tq4_gqa_prefill_decode():
    """Prefill 写入压缩 KV cache，Decode 解压缩后 attention 输出一致"""
    # 对比标准路径（bf16 KV cache + FIA）vs TQ4 路径
    # attention output cosine similarity > 0.99
```

### 9.3 35B Hybrid 模型测试

```python
def test_qwen3_6_hybrid_tq4():
    """验证 linear_attention 层不受 TQ4 影响，full_attention 层正常压缩/解压"""
    # 1. 检查 linear_attention 层走正常 GDN 路径
    # 2. 检查 full_attention 层走 TQ4 路径
    # 3. 端到端 smoke test
```

### 9.4 冒烟测试

```bash
bash smoke_test.sh
```

---

## 10. 实现顺序

| 阶段 | 任务 | 产出物 | 状态 | 备注 |
|------|------|--------|------|------|
| Phase 1 | Codebook 训练 (128) | `phase1_codebook/tq4_centroids_gqa.pt`, `.h` | ✅ 已完成 | `phase1_codebook/train_codebook.py` |
| Phase 1b | Codebook 训练 (256) | `tq4_centroids_gqa_256.pt`, `.h` | ❌ 待实现 | 扩展 train_codebook.py 支持自定义 head_dim |
| Phase 2 | `tq_gqa_compress` AscendC 算子 | `csrc/tq_gqa_compress/` | ✅ 已完成 | 已支持 head_dim=128；验证 head_dim=256 |
| Phase 2b | compress kernel CANN 9.0.0 迁移 | `tq_gqa_compress.h` | ❌ 待实现 | `CompareScalar` → `Compares` |
| Phase 3 | `tq_gqa_decompress` AscendC 算子 | `csrc/tq_gqa_decompress/` | ❌ 待实现 | **Decode 路径关键依赖**；按 9.0.0 int4b_t 规范实现 |
| Phase 4 | Torch 绑定 & Python 封装 | `tq_gqa.py`, `torch_binding.cpp` | 🔶 部分完成 | compress 已有；decompress 待加；扩展 256 支持 |
| Phase 5 | `AscendTQ4GQAAttentionBackendImpl` | `tq4_gqa_v1.py` | ❌ 待实现 | 继承父类，override forward；需处理 layer_types 分流 |
| Phase 6 | KV cache layout 适配 | `patch_kv_cache_interface.py`, `model_runner_v1.py` | ❌ 待实现 | int8 view, spec, coordinator |
| Phase 7 | 单元测试 & 冒烟测试 | `tests/` | ❌ 待实现 | 需覆盖 128/256，30B/35B 两个模型 |
| Phase 8 | 端到端验证 & 精度/性能 bench | 报告 | ❌ 待实现 | 分别对 30B 和 35B 做 |

---

## 11. 风险与注意事项

### 11.1 Codebook 分布依赖

当前 codebook（head_dim=128）基于合成高斯分布 N(0, 1/sqrt(128)) 训练。head_dim=256 的 codebook 同样需要训练。两个 head_dim 的 K/V 归一化后的分布可能不同，Phase 8 时应用真实模型 K/V 输出验证并重训练。

### 11.2 Qwen3.6-35B-A3B 的 TQ4 覆盖范围有限

35B 模型仅有 **10/40** 层走 TQ4 GQA（full_attention 层），全模型 KV cache 内存节省约 **25%**（vs 30B 的 50%）。这是架构层面的限制，不是 TQ4 实现的问题。后续如需进一步优化，需要研究如何对 GDN/linear_attention 层的 SSM state 做压缩。

### 11.3 35B 的 attention 层粒度选择

35B 模型的 `layer_types` 要求在**层粒度**选择 backend，而非模型粒度。需要确保 model_runner/attention 初始化代码支持 per-layer backend dispatch。

### 11.4 Decode 阶段解压缩开销

解压缩在 decode 阶段每次 forward 都会执行，遵循"只解压活跃 block"原则：
- 活跃 block 数量 ≤ batch_size × max_blocks_per_seq，远小于 num_blocks
- head_dim=256 时每个 slot 更大（192B），但 num_kv_heads=2 且仅 10 层，实际开销可控

后续优化方向：
- Page-level lazy decompress（缓存最近解压的 page）
- Fuse decompress + FIA（类似 MLA 的 `turboquant_sparse_flash_attention`）

### 11.5 FIA 兼容性与 PA 不可用

解压后的 K/V 是 bf16 dense tensor，与 FIA TND layout 兼容。**TQ4 GQA Decode 阶段不能使用 Paged Attention**，因为 PA 直接读 `self.key_cache`（此时是 int8 slot 格式）。Decode 统一走 FIA。

### 11.6 Graph Capture

TQ4 的 compress/decompress 算子需要确认是否支持 ACL 图捕获：
- **compress**：在 prefill 阶段调用，prefill 通常不走图模式 → 风险较低
- **decompress**：在 decode 阶段调用，decode 走图模式 → 算子必须注册 meta 函数

**MVP 策略**：开启 `--enforce-eager`，待算子验证通过后再适配图模式。

### 11.7 centroids 的双重维护

- `phase1_codebook/tq4_centroids_gqa.pt`（和未来的 `_256.pt`）：Python 端加载，运行时传给 kernel（权威来源）
- `phase1_codebook/tq4_centroids_gqa.h`（和未来的 `_256.h`）：C++ 编译时常量，仅供 kernel 静态链接参考

两者由 `phase1_codebook/train_codebook.py` 同一次运行产出，重训练时必须同时更新。

### 11.8 两个模型的 head_dim 不同 → 需要两个 Codebook

| head_dim | 模型 | slot_bytes | centroids 文件 |
|----------|------|-----------|---------------|
| 128 | 30B (48 层) | 128B | `phase1_codebook/tq4_centroids_gqa.pt` ✅ |
| 256 | 35B (10 层) | 192B | `tq4_centroids_gqa_256.pt` ❌ |

`get_centroids(head_dim, device)` 已支持按 head_dim key 缓存，自动加载对应 `.pt`。需添加 `_256.pt` 的路径映射和 fallback 训练逻辑到 `tq_gqa.py`。

---

## 12. CANN 9.0.0 对齐

### 12.1 概述

本文档设计基于 CANN 8.5.0 Ascend C 算子开发指南。CANN 9.0.0 已于 2026-06 发布（参考 `/docs/CANN_9.0.0_ops_doc/`），以下是与本方案相关的变更与对齐策略。

### 12.2 API 名称变更

CANN 9.0.0 重命名了部分 API。现有 `tq_gqa_compress` kernel 中使用到的 API 中，**仅有 1 个受影响**：

| 旧名 (8.5.0) | 新名 (9.0.0) | 使用位置 | 迁移方式 |
|-------------|-------------|---------|---------|
| `CompareScalar` | `Compares` | `tq_gqa_compress.h:122, 132` | 全局替换 |

**不受影响**的 API（`tq_gqa_compress.h` 中使用的）：

`Duplicate`, `Mul`, `Muls`, `Add`, `ReduceSum`, `DataCopy`, `ReinterpretCast`, `Cast`, `Select`, `PipeBarrier`, `AllocTensor`, `FreeTensor`, `EnQue`, `DeQue`, `GetBlockIdx`, `TPipe`, `TQue`, `TBuf`, `GlobalTensor`, `LocalTensor`

> 详细变更列表见 `ops_section_6.6.3.md`（CANN 9.0.0 API 参考 → 接口变更说明）。

### 12.3 int4b_t 使用验证

CANN 9.0.0 文档明确了 `int4b_t` 的推荐使用模式（`ops_section_6.2.3.3.5.1.md`）：

```cpp
// int4b_t 打包（拆分为 half→int4b_t）
Cast<half, float>(tmpHalf, srcFloat, RoundMode::CAST_RINT, length);
Cast<int4b_t, half>(dstInt4, tmpHalf, RoundMode::CAST_RINT, length);

// int4b_t 解包（int4b_t→half→float）
Cast<half, int4b_t>(tmpHalf, srcInt4, RoundMode::CAST_NONE, length * 2);
Cast<float, half>(dstFloat, tmpHalf, RoundMode::CAST_NONE, length * 2);
```

**对齐状态**：✅ 当前 `tq_gqa_compress` 的实现（`tq_gqa_compress.h:138-141`）已遵循此模式，无需修改：

```cpp
Cast(packHalf, tmp, RoundMode::CAST_RINT, headDim_);           // float → half
LocalTensor<int4b_t> i4 = slot.ReinterpretCast<int4b_t>();     
Cast(i4, packHalf, RoundMode::CAST_RINT, headDim_);             // half → int4b_t
```

**decompress kernel 设计参考**：解包时按相反路径：int4b_t → half → float → bf16。

### 12.4 架构兼容性（351x）

CANN 9.0.0 正式支持 351x 架构（Atlas A2/A3/350）。关键变更：

| 变更 | 对本方案影响 |
|------|------------|
| Cube 不支持 int4b_t | ⚠️ 无影响：compress/decompress 是纯 Vector 操作，不使用 Cube/Matmul |
| UB 容量增大、bank 结构变化 | ⚠️ 无影响：现有 tiling 基于运行时参数，不硬编码 UB 布局 |
| ND-DMA 多维度拷贝 | 🔷 可选优化：可用于未来优化 slot 排列的 stride 写入 |
| L1→GM 路径删除 | ⚠️ 无影响：当前 kernel 只做 GM↔UB 搬运 |
| RegBase Vector Core | 🔷 可选优化：静态 Tensor 编程可进一步降低延迟 |

**结论**：当前 compress kernel 设计与 351x 完全兼容。decompress kernel 设计时应遵循相同模式。

### 12.5 可选优化方向（非 MVP）

CANN 9.0.0 引入的新特性可作为后续性能优化方向：

| 特性 | 章节 | 适用场景 | 预期收益 |
|------|------|---------|---------|
| **静态 Tensor 编程** | `ops_section_2.2.3.3.2.md` | 纯 Vector 算子的 compress/decompress | 减少 ~数百纳秒/kernel Launch overhead |
| **ND-DMA** | 6.2.3.3 | 带 stride 的 slot 读写 | 简化数据搬运，减少显式循环 |
| **Tiling 模板编程** | `ops_section_2.10.2.5.5.md` | 多 head_dim 变体（128/256） | 用模板替代多套 tiling struct |

### 12.6 实现顺序更新

| 阶段 | 任务 | 备注 |
|------|------|------|
| Phase 2b | `tq_gqa_compress` API 迁移 | `CompareScalar` → `Compares`，适配 CANN 9.0.0 |
| Phase 3 | `tq_gqa_decompress` AscendC 算子 | 按 CANN 9.0.0 int4b_t 解包模式实现 |
