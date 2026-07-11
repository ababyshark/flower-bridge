# Baseline v1

最原始基准部署，无 Mooncake KV 池化，用于对比后续方案的性能基线。

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-30B-A3B |
| 精度 | bfloat16 |
| TP | 2 |
| NPU | 10,11 |
| 端口 | 8001 |
| max_model_len | 65536 |
| max_num_seqs | 128 |
| max_num_batched_tokens | 16384 |
| enforce_eager | true |

## 启动

```bash
bash baseline_v1.sh
```

## 冒烟测试

```bash
bash smoke_test.sh
```

或指定端口：

```bash
PORT=8001 bash smoke_test.sh
```
