# TQ4 GQA 压缩/解压缩 — Qwen3 系列设计方案

## 1. 背景

### 1.1 现状

vLLM-Ascend 已实现 TQ4（TurboQuant 4-bit）KV Cache 压缩，但**仅支持 MLA 架构**（DeepSeek V3/V4、GLM DSA），不适用于 Qwen 等 GQA 架构的模型。

当前 TQ4 方案的核心局限：

| 维度 | 当前 TQ4 (MLA only) | 不可用于 GQA 的原因 |
|------|-------------------|--------------------|
| 压缩目标 | 单个 KV latent（kv_lora_rank=512，K=V 同一压缩表示） | GQA 的 K 和 V 是独立张量 |
| 注意力模式 | MLA-absorb（K=V，dequant 后的 nope 即 K 又即 V） | GQA 的 K ≠ V |
| 注意力内核 | 融合 dequant + sparse flash attention（仅 MLA_ABSORB 模式） | GQA 使用 FIA，不用 SFA |
| KV cache 格式 | 融合 slot = nibbles(256B) + rope(128B) + scale(2B) | GQA 无 rope 嵌入 slot |
| Slots 格式 | fused_slot_size = packed + rope + scale | base_slot_size = packed + scale |
| Codebook | N(0, 1/512) 训练 | head_dim=128，K/V 分布不同 |
| Hadamard | query/key 在 Hadamard 空间（MLA latent 重建需要） | GQA 不需要 Hadamard 变换 |

### 1.2 设计原则

1. **不修改现有算子和代码**：新增独立的压缩/解压缩 AscendC 算子，新增独立的 GQA attention 实现类
2. **算子完全解耦注意力**：压缩和解压缩是独立算子，不与任何注意力内核融合
3. **复用现有 FIA**：解压缩后的 K/V 直接喂给 `AscendAttentionBackendImpl`，不写新的 attention kernel
4. **GQA 标识符隔离**：新增算子、类、文件均带 `gqa` 标识，与现有 MLA TQ4 代码完全隔离

---

## 2. 目标模型分析：Qwen3-30B-A3B

### 2.1 模型参数

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
  "sliding_window": null,
  "num_experts": 128,
  "num_experts_per_tok": 8,
  "moe_intermediate_size": 768,
  "torch_dtype": "bfloat16"
}
```

### 2.2 关键特征

| 特征 | 值 |
|------|-----|
| Attention 类型 | GQA（Grouped Query Attention，Q:K:V = 32:4:4, ratio=8:1） |
| head_dim | 128 |
| num_kv_heads | 4 |
| 是否有 Sliding Window | 否 |
| 是否有 MLA | 否（标准 attention，q_proj/k_proj/v_proj/o_proj） |
| 是否 MoE | 是（128 experts, top-8 routing） |
| 现有 attention 后端 | `AscendAttentionBackendImpl`（注册名 `CUSTOM/ASCEND`） |
| KV cache 当前格式 | `(k_cache, v_cache)`，各 shape `(num_blocks, block_size, 4, 128)`，dtype bf16 |

### 2.3 TQ4 压缩参数计算

```
head_dim              = 128
packed_nibble_bytes   = head_dim / 2                 = 64  bytes
scale_bytes           = sizeof(fp16)                 = 2   bytes
base_slot_bytes       = align_up(64 + 2, 64)         = 128 bytes
```

### 2.4 内存节省

| 项目 | 原始 (bf16) | TQ4 压缩后 | 节省 |
|------|-----------|-----------|------|
| 单 head K（per token） | 128×2 = 256B | 128B | 50% |
| 单 head V（per token） | 128×2 = 256B | 128B | 50% |
| 单层 KV（per token，4 heads） | 2048B | 1024B | 50% |
| 全模型 KV（per token，48 层） | 96KB | 48KB | 50% |

**对 30B 模型的影响**：在 block_size=128 的情况下，KV cache 总内存减半，等效于并发容量翻倍。

---

## 3. TQ4 压缩方案

### 3.1 算法描述

沿用现有 TQ4 的 4-bit codebook 量化方案，适配 GQA 的 K/V 张量：

```
输入: K 或 V 张量 x ∈ R^{N×H} （H = head_dim = 128, N = 总 token 数）
输出: slot ∈ uint8^{N×S} （S = base_slot_size, 每 slot 128 字节）

算法:
  1. 归一化:         norm = ||x_i||₂,  u = x_i / norm
  2. Hadamard:       跳过（GQA 不需要）
  3. 标量量化:       nib[d] = argmin_k |u[d] - cent[k]|  →  4-bit signed 索引
  4. 打包:           每 4 个 nibble → 2 字节（int4b_t low-first 打包）
  5. 写入 vecNorm:   slot[64:66] = uint8_view(fp16(norm))
  6. 对齐填充:       slot[66:128] = 0
```

**Codebook**：16 级 Lloyd-Max 质心，针对 head_dim=128 的 K/V 分布重新训练（见 3.2 节）。

### 3.2 Codebook 训练

现有 TQ4 codebook 针对 N(0, 1/512) 分布训练，存储在 `tq4_centroids.h`。Qwen GQA 需要新的 codebook：

```python
# 训练方式（离线，一次性）
# 1. 从 Qwen3-30B-A3B 取 1000 个样本的 K 和 V 输出
# 2. 归一化到单位球面
# 3. Lloyd-Max 迭代（15 个分界点 → 16 个质心）
# 4. 保存到 cent_qwen3_128_fp32
```

**CentSigned 表示**（供 AscendC 内核直接使用）：质心旋转 8 位：
```
centSigned[s] = cent[(s + 8) % 16] for s in 0..15
```
使内核中 `int4b_t` 解码得到的 signed nibble + 8 即码本索引。

### 3.3 解压缩算法

```
输入: slot ∈ uint8^{N×128}
输出: x̂ ∈ bf16^{N×128}

算法:
  1. 解包 nibbles:    slot[0:64] → 128 个 4-bit signed 值
  2. 查表质心:        cent[nibble]（码本）
  3. 范数修正:        norm_corr = 1 / sqrt(Σ cent²)
  4. 乘以 vecNorm:    x̂ = cent * norm_corr * fp16→f32(slot[64:66])
```

---

## 4. 文件结构

### 4.1 新增文件

```
vllm-ascend/
├── csrc/
│   ├── tq_gqa_compress/                  # [新建] GQA TQ4 压缩 AscendC 算子
│   │   ├── op_kernel/
│   │   │   ├── tq_gqa_compress.h         # 压缩内核头文件
│   │   │   └── tq_gqa_compress.cpp       # 压缩内核入口
│   │   ├── op_host/
│   │   │   ├── CMakeLists.txt
│   │   │   ├── tq_gqa_compress_def.cpp
│   │   │   ├── tq_gqa_compress_infershape.cpp
│   │   │   ├── tq_gqa_compress_tiling.cpp
│   │   │   └── tq_gqa_compress_tiling.h
│   │   └── tq_gqa_compress_torch_adpt.h  # PyTorch 绑定
│   │
│   ├── tq_gqa_decompress/                # [新建] GQA TQ4 解压缩 AscendC 算子
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
│   │   ├── tq_gqa.py                     # [新建] GQA TQ4 Python 封装（compress/decompress/centroids）
│   │   └── tq4_centroids_gqa.h           # [新建] GQA Codebook 常量头文件
│   │
│   └── attention/
│       └── tq4_gqa_v1.py                 # [新建] GQA TQ4 Attention 接口层
│
└── tests/
    └── ut/
        └── ops/
            ├── test_tq_gqa_compress.py    # [新建] 压缩算子单元测试
            └── test_tq_gqa_decompress.py  # [新建] 解压缩算子单元测试
```

### 4.2 修改文件（最小化改动）

```
vllm-ascend/
├── vllm_ascend/attention/attention_v1.py   # [修改] 在 forward 中增加 TQ4 条件分支
├── vllm_ascend/worker/model_runner_v1.py   # [修改] KV cache dtype/size 适配
└── vllm_ascend/patch/platform/
    └── patch_kv_cache_interface.py          # [修改] 新增 GQA TQ4 attention spec
```

---

## 5. 算子接口设计

### 5.1 压缩算子 `tq_gqa_compress`

**PyTorch 接口**：
```python
torch.ops._C_ascend.tq_gqa_compress(
    latent: Tensor,      # [N, head_dim] fp32, L2 归一化后的 K 或 V
    centroids: Tensor,   # [16] fp32, Lloyd-Max 质心（Python 端计算 centSigned 后传入）
) -> Tensor              # [N, base_slot_size] uint8
```

**AscendC kernel 输入**：
- `latent [N, head_dim] fp32`: 已归一化的原始数据（z = x / ||x||）
- `centroids [16] fp32`: 码本质心（由 Python 端传入，不硬编码）
- `head_dim`: 运行时参数（支持 64/128/256 等 2 的幂次）

**AscendC kernel 输出**：
- `slot [N, base_slot_size] uint8`

**base_slot_size 计算**：
```cpp
uint32_t packed_bytes = head_dim / 2;
uint32_t base_slot_size = ((packed_bytes + 2 + 63) / 64) * 64;  // 对齐到 64B
```

### 5.2 解压缩算子 `tq_gqa_decompress`

**PyTorch 接口**：
```python
torch.ops._C_ascend.tq_gqa_decompress(
    slots: Tensor,       # [N, base_slot_size] uint8, TQ4 压缩的 slots
    centroids: Tensor,   # [16] fp32, Lloyd-Max 质心
    head_dim: int,       # 解压后的维度
) -> Tensor              # [N, head_dim] bf16
```

**AscendC kernel 输入**：
- `slots [N, base_slot_size] uint8`
- `centroids [16] fp32`
- `head_dim`: 运行时参数

**AscendC kernel 输出**：
- `output [N, head_dim] bf16`, 解压后的 K 或 V 张量

### 5.3 与现有算子对比

| 对比维度 | 现有 `tq_compress_latent` | 新 `tq_gqa_compress` | 新 `tq_gqa_decompress` |
|---------|--------------------------|---------------------|----------------------|
| 适用范围 | MLA latent only | GQA K/V（任意 head_dim） | GQA K/V（任意 head_dim） |
| head_dim | 硬编码 512 | 运行时传入 | 运行时传入 |
| centroids | 硬编码 16 个 | 运行时传入 | 运行时传入 |
| 输入是否归一化 | 否（内核内部归一化） | 是（外部 L2 norm → 传 z=x/norm） | - |
| 输出包含 rope | 否（rope 在 Python 端拼接） | 否（纯 nope 压缩） | - |
| slot 格式 | 320B (256+2+62pad) | base_slot_size(128B for head_dim=128) | - |

**设计决策**：为了复用现有 `tq_compress_latent` 的 AscendC 内核逻辑（归一化 + 量化 + 打包），新算子也可以在内部完成归一化。待实现时根据性能测试决定归一化放在 Python 端还是内核端。

---

## 6. GQA TQ4 Attention 实现设计

### 6.1 架构总览

```
                    Prefill                                 Decode
            ┌─────────────────┐                    ┌─────────────────┐
            │  K,V = fwd(q/k/v) │                    │  K,V = decompress │
            │  (标准 forward)    │                    │  (从 slot 反量化)   │
            └────────┬─────────┘                    └────────┬─────────┘
                     │      │                                │
                     ▼      ▼                                │
            ┌────────┐  ┌─────────┐                          │
            │compress│  │compress │                          │
            │   K    │  │    V    │                          │
            └───┬────┘  └────┬────┘                          │
                │            │                                │
                ▼            ▼                                ▼
            ┌─────────────────────────┐            ┌─────────────────────────┐
            │   KV Cache (int8)       │            │   FIA attention          │
            │   k_slots, v_slots      │ ──────►    │   (npu_fused_infer_attn) │
            └─────────────────────────┘            └─────────────────────────┘
```

### 6.2 实现方式

**方案**：新建 `AscendTQ4GQAAttentionBackendImpl`，继承 `AscendAttentionBackendImpl`，**仅** override `forward()` 方法，在 3 个注入点插入 compress/decompress：

```
forward(key, value, kv_cache, ...):
  if key is not None and value is not None:      # Prefill 阶段（有新 KV 写入）
    ┌─────────────────────────────────────────────┐
    │  [注入点 1] compress:                       │
    │    k_slots = tq_gqa_compress(k_normed)      │
    │    v_slots = tq_gqa_compress(v_normed)      │
    │  // 将 int8 slots 写入 kv_cache[0], [1]    │
    │  DeviceOperator.reshape_and_cache(           │
    │      k_slots, v_slots, ...)                 │
    └─────────────────────────────────────────────┘

  # 注意力计算（Prefill 用原始 K/V，Decode 用解压 K/V）
  if DecodeOnly:
    ┌─────────────────────────────────────────────┐
    │  [注入点 2] decompress:                     │
    │    k_f32 = tq_gqa_decompress(kv_cache[0])   │
    │    v_f32 = tq_gqa_decompress(kv_cache[1])   │
    └─────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────┐
    │  [注入点 3] FIA attention:                  │
    │    复用父类 forward_fused_infer_attention() │
    │    或 forward_paged_attention()             │
    └─────────────────────────────────────────────┘
```

### 6.3 注入点详解

#### 注入点 1：Prefill 压缩写入 KV Cache

位置对应现有 `reshape_and_cache()`（`attention_v1.py:1226`）。在调用 `DeviceOperator.reshape_and_cache` 之前：

```python
def _tq4_compress_and_cache(self, key, value, kv_cache, attn_metadata):
    # 1. L2 归一化
    k_norm = key.float()
    k_scales = k_norm.norm(dim=-1, keepdim=True)
    k_normed = k_norm / (k_scales + 1e-8)

    v_norm = value.float()
    v_scales = v_norm.norm(dim=-1, keepdim=True)
    v_normed = v_norm / (v_scales + 1e-8)

    # 2. TQ4 压缩
    k_slots = torch.ops._C_ascend.tq_gqa_compress(k_normed, self._centroids)
    v_slots = torch.ops._C_ascend.tq_gqa_decompress(v_normed, self._centroids)

    # 3. 写入 KV cache（k_slots/v_slots 是 uint8，需 view 为 int8 写入 cache）
    key_cache = kv_cache[0]   # shape: [num_blocks, block_size, num_kv_heads, slot_bytes], dtype int8
    value_cache = kv_cache[1]

    slots = attn_metadata.slot_mapping
    key_cache_flat = key_cache.view(-1, key_cache.shape[-1])
    value_cache_flat = value_cache.view(-1, value_cache.shape[-1])
    key_cache_flat[slots] = k_slots.view(torch.int8)
    value_cache_flat[slots] = v_slots.view(torch.int8)
```

#### 注入点 2：Decode 解压缩

在 `_get_fia_params()` 返回后、FIA 调用前：

```python
if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
    # 从 KV cache 读取压缩的 slots
    key_slots = self.key_cache.view(num_block * block_size, ...)
    value_slots = self.value_cache.view(num_block * block_size, ...)

    # TQ4 解压缩
    key = torch.ops._C_ascend.tq_gqa_decompress(
        key_slots.reshape(-1, self._slot_bytes), self._centroids, self.head_size
    ).view(num_block, block_size, self.num_kv_heads, self.head_size)

    value = torch.ops._C_ascend.tq_gqa_decompress(
        value_slots.reshape(-1, self._slot_bytes), self._centroids, self.head_size
    ).view(num_block, block_size, self.num_kv_heads, self.head_size)
```

#### 注入点 3：复用父类 FIA

```python
# 解压后的 K/V 直接喂给父类的 FIA attention
return super().forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)
```

### 6.4 注册和激活

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
```

### 6.5 KV Cache 适配

#### Attention Spec

```python
@dataclass(frozen=True)
class AscendGQATQ4AttentionSpec(AttentionSpec):
    """GQA TQ4 KV cache spec."""
    cache_tq4_gqa: bool = False
    head_dim: int = 128
    slot_bytes: int = 128  # base_slot_size

    @property
    def page_size_bytes(self) -> int:
        if self.cache_tq4_gqa:
            # K cache: num_kv_heads * head_dim → num_kv_heads * slot_bytes (int8)
            # V cache: same
            k_bytes = self.block_size * self.num_kv_heads * self.slot_bytes
            v_bytes = self.block_size * self.num_kv_heads * self.slot_bytes
            return k_bytes + v_bytes
        else:
            return super().page_size_bytes
```

#### KV Cache Allocation（model_runner_v1.py）

```python
# 在 kv_cache 创建时检测 cache_tq4_gqa 属性
if getattr(current_kv_cache_spec, "cache_tq4_gqa", False):
    slot_bytes = current_kv_cache_spec.slot_bytes
    # K cache: int8 dtype, [num_blocks, block_size, num_kv_heads, slot_bytes]
    k_cache = raw_k_tensor.view(torch.int8).view(
        num_blocks, block_size, num_kv_heads, slot_bytes
    )
    v_cache = raw_v_tensor.view(torch.int8).view(
        num_blocks, block_size, num_kv_heads, slot_bytes
    )
```

---

## 7. 配置与启用

### 7.1 环境变量

```python
# vllm_ascend/envs.py
"VLLM_ASCEND_ENABLE_TQ4_GQA": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_TQ4_GQA", 0)),
```

### 7.2 启动示例

```bash
export VLLM_ASCEND_ENABLE_TQ4_GQA=1
vllm serve /models/Qwen3-30B-A3B \
    --dtype bfloat16 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager
```

### 7.3 attention backend 选择逻辑

```python
# vllm_ascend/platform.py 或 attention 初始化处
if envs.VLLM_ASCEND_ENABLE_TQ4_GQA and model_config.is_gqa:
    attn_backend = "ASCEND_TQ4_GQA"
else:
    attn_backend = "ASCEND"
```

---

## 8. 测试方案

### 8.1 算子单元测试

```python
# tests/ut/ops/test_tq_gqa_compress.py
def test_tq_gqa_compress_roundtrip():
    """压缩后解压缩，与原数据对比精度（cosine similarity）"""
    head_dim = 128
    N = 64
    centroids = _build_centroids(head_dim)  # Lloyd-Max codebook

    x = torch.randn(N, head_dim).bfloat16()
    x_normed = x.float()
    x_normed = x_normed / (x_normed.norm(dim=-1, keepdim=True) + 1e-8)

    slots = torch.ops._C_ascend.tq_gqa_compress(x_normed, centroids)
    x_hat = torch.ops._C_ascend.tq_gqa_decompress(slots, centroids, head_dim)

    cos_sim = torch.nn.functional.cosine_similarity(x.float(), x_hat.float()).mean()
    assert cos_sim > 0.99  # 预期高保真
```

### 8.2 GQA Attention 功能测试

```python
# tests/e2e/.../test_tq4_gqa_attention.py
def test_tq4_gqa_prefill_decode():
    """Prefill 写入压缩 KV cache，Decode 解压缩后 attention 输出一致"""
    # 对比：
    #   1. 标准路径（bf16 KV cache + FIA）
    #   2. TQ4 路径（int8 KV cache + decompress + FIA）
    # 两者 attention output cosine similarity > 0.99
```

### 8.3 冒烟测试

```bash
# 启动服务后发送简单请求验证输出合理性
python smoke_test.sh
```

---

## 9. 实现顺序

| 阶段 | 任务 | 产出物 | 预估工时 |
|------|------|--------|---------|
| Phase 1 | Codebook 训练 | cent_qwen3_128 码本 | 0.5d |
| Phase 2 | `tq_gqa_compress` AscendC 算子 | csrc/tq_gqa_compress/ | 1d |
| Phase 3 | `tq_gqa_decompress` AscendC 算子 | csrc/tq_gqa_decompress/ | 1d |
| Phase 4 | Torch 绑定 & Python 封装 | tq_gqa.py, torch_binding.cpp | 0.5d |
| Phase 5 | `AscendTQ4GQAAttentionBackendImpl` | tq4_gqa_v1.py | 1d |
| Phase 6 | KV cache layout 适配 | patch_kv_cache_interface.py, model_runner_v1.py | 0.5d |
| Phase 7 | 单元测试 & 冒烟测试 | tests/ | 1d |
| Phase 8 | 端到端验证 & 精度/性能 benchmark | 报告 | 0.5d |
| **合计** | | | **~6d** |

---

## 10. 风险与注意事项

1. **Codebook 分布依赖**：当前 codebook 针对 N(0, 1/512) 训练。Qwen3 GQA 的 K/V 归一化后分布可能不同，需要实际采样训练并验证。

2. **Decode 阶段解压缩开销**：解压缩在 decode 阶段每次 forward 都会执行，需要测量 AscendC 内核的耗时。如果成为瓶颈，后续可考虑 cache 整页解压或预解压。

3. **FIA 兼容性**：解压后的 K/V 是 bf16 dense tensor，与现有 FIA 接口完全兼容。但需确认 FIA 的 `input_layout` 在 Decode 阶段（`num_block * block_size`）下的 shape 是否正确。

4. **Graph Capture**：`AscendAttentionBackendImpl` 的 FIA 路径使用图模式。TQ4 的 compress/decompress 需要确认是否支持图捕获，如不支持需在捕获期和运行时分别处理。

5. **Prefill 阶段 K/V 处理**：Prefill 时的新 token K/V 可以直接用未压缩的原值做 attention（不需要压缩—再解压），压缩仅用于写入 cache，不影响 Prefill attention 计算。
