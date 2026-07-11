# flower-bridge

vLLM-Ascend 推理优化集成项目。面向昇腾 NPU (CANN 9.0.0)，将以下三项优化能力集成到统一镜像包中，并提供参考配置脚本。

## 项目结构

```
flower-bridge/
├── 0.baseline_v1/              # 基线部署 — 无优化，性能参照点
├── 1.mtp/                      # 投机推理 — MTP (Multi-Token Prediction) 加速 Decode
├── 2.mookcake_v1/              # Mooncake 池化 — 跨实例共享 KV Cache，降低 TTFT
├── 3.tq4_vllm_ascend_v1/       # KV Cache TQ4 压缩 — 4-bit 量化，GQA 模型内存减半
└── docs/                       # CANN 参考文档 (8.5.0 / 9.0.0)
```

## 优化能力

| 目录 | 能力 | 目标模型 | 收益 |
|------|------|---------|------|
| `1.mtp` | **投机推理 (MTP/Eagle)** — 小模型 draft + 大模型 verify，提升 Decode 吞吐 | Qwen3-30B-A3B | Decode 加速 |
| `2.mookcake_v1` | **Mooncake KV Cache 池化** — 跨 vLLM 实例共享 KV Cache，复用 Prefill 结果 | Qwen3-30B-A3B | TTFT 降低 |
| `3.tq4_vllm_ascend_v1` | **TQ4 4-bit KV Cache 压缩** — GQA 模型 K/V 量化为 4-bit，内存减半 | Qwen3-30B-A3B / Qwen3.6-35B-A3B | 并发容量翻倍 |

## 各模块状态

### 0. baseline_v1 — 基线 ✅

- 模型: Qwen3-30B-A3B, bf16, TP=2
- 冒烟测试 ✅ / Bench 63K/1K 通过 ✅
- 提供 `baseline_v1.sh` 参考启动脚本

### 1. mtp — 投机推理 🔶

- MTP patch（MAX_MTP 8→16）已应用
- 基准冒烟（无投机）通过 ✅
- MTP 投机模式功能测试待完成

### 2. mookcake_v1 — Mooncake 池化 🔶

- 三个部署场景均已搭建 + 冒烟通过 ✅
  1. 单机自池化 (standalone)
  2. 多实例共享池 (multi-instance)
  3. PD 分离池 (producer/consumer)
- KV 池化功能验证 + 性能 Bench 待完成

### 3. tq4_vllm_ascend_v1 — KV Cache TQ4 🔶

- 目标: Qwen3-30B-A3B (head_dim=128) + Qwen3.6-35B-A3B (head_dim=256, hybrid attention)
- Codebook 训练 ✅ (head_dim=128)
- `tq_gqa_compress` AscendC kernel ✅
- `tq_gqa_decompress` kernel + Attention Backend + 全链路验证 待实现
- 详见 `TQ4_GQA_DESIGN.md`

## 启动参考

```bash
# 基线部署
cd 0.baseline_v1 && bash baseline_v1.sh

# MTP 投机推理
cd 1.mtp && bash mtp_without_draft_v1.sh

# Mooncake 池化
cd 2.mookcake_v1/1.scenario_standalone_kvboth && bash mooncake.sh

# TQ4 KV Cache 压缩 (30B)
cd 3.tq4_vllm_ascend_v1 && bash deploy_30B.sh
```

## 目标

将上述三项优化能力 + 基线部署集成为统一镜像包，支持 Qwen3 系列模型在昇腾 NPU 上的高性能推理服务。
