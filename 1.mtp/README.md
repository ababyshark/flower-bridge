# 0.1-mtp

基于 baseline_v1，添加 vllm-ascend MTP（Multi-Token Prediction / 投机推理）补丁。MTP（亦称 eagle/eagle3 投机解码）通过一个小型 draft 模型预测后续 token，主模型批量验证，在保证精度的前提下提升 decode 吞吐。

## 基础配置

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-30B-A3B |
| 精度 | bfloat16 |
| TP | 2 |
| NPU | 12,13 |
| 端口 | 8002 |
| max_model_len | 40960 |
| enforce_eager | true |

## 补丁说明

补丁基于 vllm-ascend baseline commit: `367b8e62`（Reduce sampling reconstructed, 已内置 spec_decode 投机解码框架）。

在此 baseline 之上新增 1 个 commit（当前 HEAD `9124d402`）：

| Commit | 说明 |
|--------|------|
| `9124d402` | 将 MAX_MTP 从 8 扩大到 16（recurrent_gated_delta_rule kernel） |

### 补丁文件

- `patch/0001-mtp-kernel.patch` — 666 行，`367b8e62..HEAD` diff

## 启动

### 无投机推理（验证补丁未破坏基础推理）

```bash
bash mtp_without_draft_v1.sh
# 默认端口 8002，NPU 12,13
```

### 带投机推理（验证 MTP 生效）

（待补充启动脚本）

## 投机推理开关

补丁集成后，MTP 投机推理需在启动时显式开启（添加对应 flag），未开启时等价 baseline。

- **`mtp_without_draft_v1.sh`** — 关闭投机推理，验证基础功能正确 ✅ 已通过
- **待创建** — 开启投机推理，验证 MTP 生效后精度正确
