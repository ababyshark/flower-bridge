# Mooncake KV Pool 部署方案

## 概述

基于 vLLM + vLLM-Ascend + Mooncake 实现 KV Cache 池化，通过 `AscendStoreConnector` 将 vLLM 的 KV cache 存入 mooncake 共享存储池，后续请求可直接复用，跳过重复的 prompt 计算，大幅降低 TTFT。

## 硬件要求

- **硬件**：Ascend 910B (A3)
- **模型**：Qwen3-30B-A3B (bf16, TP=2, 单实例需 2 芯片)
- **环境**：vLLM + vLLM-Ascend + CANN 9.0.0 + Mooncake

## 编译构建 Mooncake

基于 [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) v0.3.11-rc1，分支 `tq4-rc1`。

### 编译

```bash
bash build_mooncake.sh
```

构建选项说明：

| CMake 选项 | 值 | 说明 |
|-----------|-----|------|
| `USE_ASCEND_DIRECT` | ON | Ascend NPU direct 传输（HIXL） |
| `USE_ETCD` | OFF | 不需要 etcd（单机 P2PHANDSHAKE 模式） |
| `USE_REDIS` | OFF | 不需要 Redis |
| `WITH_STORE_GO` | OFF | 不需要 Go 客户端绑定（我们只用 C++/Python） |
| `WITH_STORE_RUST` | OFF | 不需要 Rust 绑定 |

依赖项：`build-essential cmake libjsoncpp-dev libcurl4-openssl-dev wget unzip`

### 安装

```bash
bash install_mooncake.sh
```

安装后验证：

```bash
python3 -c "import mooncake; print(mooncake.__version__)"
```

## 三个方案对比

| | 1. 单实例自池 | 2. 多实例共享池 | 3. PD 分离 |
|---|---|---|---|
| **NPU 芯片** | 2 | 4 | 4 |
| **vllm 实例数** | 1 | 2 | 2 (P+D) |
| **跨实例共享** | ✗ | ✓ | ✓ |
| **角色分工** | `kv_both` | `kv_both` × 2 | `kv_producer` + `kv_consumer` |
| **复杂度** | 低 | 中 | 高 |
| **适用场景** | 单服务复用自身历史 | 多服务共享缓存 | Prefill/Decode 独立扩缩 |

### 核心理念

```
┌─────────────┐     ┌─────────────┐
│  vllm-1     │     │  vllm-2     │
│  (计算 KV)  │     │  (复用 KV)  │
└──────┬──────┘     └──────┬──────┘
       │ 存入               │ 取出
       └────────┬──────────┘
           ┌────┴────┐
           │ mooncake │  ← KV Cache 共享池
           │  store   │
           └────┬────┘
                │
        ┌───────┴───────┐
        │ mooncake_master│  ← 元数据管理 (纯 CPU)
        └───────────────┘
```

---

## 1. 单实例自池 (`1.scenario_standalone_kvboth`)

### 架构

```
vllm (NPU 14,15) ─── mooncake pool
       ↑ kv_both (自己存, 自己取)
```

### 启动

```bash
bash 1.scenario_standalone_kvboth/mooncake.sh
```

### 端口

| 服务 | 端口 |
|------|------|
| vllm | 8010 |
| mooncake_master | 50088 |

### 冒烟

```bash
bash 1.scenario_standalone_kvboth/smoke_test.sh
```

---

## 2. 多实例共享池 (`2.scenario_multi_instance_pool`)

### 架构

```
vllm-A (NPU 8,9, port 8011)   ──┐
                                  ├── mooncake pool
vllm-B (NPU 10,11, port 8012) ──┘
       ↑ 均为 kv_both
```

### 启动

```bash
bash 2.scenario_multi_instance_pool/mooncake.sh
```

### 端口

| 服务 | 端口 |
|------|------|
| vllm 实例 A | 8011 |
| vllm 实例 B | 8012 |
| mooncake_master | 50088 |

### 验证池化

1. 向实例 A 发请求，KV cache 存入 pool
2. 向实例 B 发**相同**请求
3. 观察 B 日志：`External prefix cache hit rate` 应 > 0%

```bash
# 请求 A (存 KV)
curl -s http://localhost:8011/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/models/Qwen3-30B-A3B","messages":[{"role":"user","content":"请用中文介绍人工智能"}],"max_tokens":200}'

# 相同请求 B (应从 pool 取 KV)
curl -s http://localhost:8012/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/models/Qwen3-30B-A3B","messages":[{"role":"user","content":"请用中文介绍人工智能"}],"max_tokens":200}'

# 查看 B 的 cache hit rate
tail -f logs/instance_B_port8012.log | grep "External prefix cache hit rate"
```

---

## 3. PD 分离 (`3.scenario_pd_disaggregation`)

### 架构

```
Prefill (NPU 12,13, port 8015) ─── 存 KV ──→ mooncake pool
  ↑ kv_producer                                   │
                                                   │ 取 KV
Decode  (NPU 14,15, port 8016) ←──────────────────┘
  ↑ kv_consumer
```

### 启动

```bash
bash 3.scenario_pd_disaggregation/mooncake.sh
```

### 端口

| 服务 | 端口 | 角色 |
|------|------|------|
| Prefill | 8015 | kv_producer (计算 KV, 存入 pool) |
| Decode | 8016 | kv_consumer (从 pool 取 KV, 直接生成) |
| mooncake_master | 50088 | 元数据管理 |

### 验证池化

1. 向 Prefill 发请求，完成 prompt 计算并存入 pool
2. 向 Decode 发**相同**请求，从 pool 取 KV 直接生成

```bash
# 请求 Prefill (计算 KV 并存入 pool)
curl -s http://localhost:8015/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/models/Qwen3-30B-A3B","messages":[{"role":"user","content":"请用中文介绍人工智能"}],"max_tokens":200}'

# 相同请求 Decode (从 pool 取 KV)
curl -s http://localhost:8016/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/models/Qwen3-30B-A3B","messages":[{"role":"user","content":"请用中文介绍人工智能"}],"max_tokens":200}'

# 查看 Decode 的 cache hit rate
tail -f logs/decode_port8016.log | grep "External prefix cache hit rate"
```

---

## mooncake.json 配置说明

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "ascend",
  "device_name": "",
  "master_server_address": "127.0.0.1:50088",
  "global_segment_size": "300GB",
  "preferred_segment": false,
  "prefer_alloc_in_same_node": true
}
```

| 参数 | 说明 |
|------|------|
| `metadata_server` | P2P 握手模式 |
| `protocol` | ascend 协议 (NPU) |
| `master_server_address` | mooncake_master 地址 |
| `global_segment_size` | 每卡注册到 pool 的内存大小 |

## 环境变量

| 变量 | 值 | 说明 |
|------|-----|------|
| `PYTHONHASHSEED` | 0 | 统一 hash，多节点必须一致 |
| `ASCEND_BUFFER_POOL` | 4:8 | A3 硬件，4 个 8MB buffer |
| `HCCL_RDMA_TIMEOUT` | 17 | RDMA 超时 |
| `ASCEND_CONNECT_TIMEOUT` | 10000 | 连接超时 (ms) |
| `ASCEND_TRANSFER_TIMEOUT` | 10000 | 传输超时 (ms) |
| `MOONCAKE_CONFIG_PATH` | mooncake.json 路径 | mooncake 配置文件 |

## KV 连接器参数

| 参数 | 说明 |
|------|------|
| `kv_connector` | `AscendStoreConnector` |
| `kv_role` | `kv_both` / `kv_producer` / `kv_consumer` |
| `lookup_rpc_port` | 查找端口，0=自动，多实例需设不同值 |
| `backend` | 存储后端，固定 `mooncake` |
| `load_async` | 是否异步加载 (PD 分离 Decode 建议开启) |
