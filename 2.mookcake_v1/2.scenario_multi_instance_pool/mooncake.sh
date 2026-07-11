#!/bin/bash
set -euo pipefail

# =============================================
# 2.scenario_multi_instance_pool
# 2 个 vllm 实例共享同一个 mooncake KV pool
# 实例 A: NPU 4,5    port 8011
# 实例 B: NPU 6,7  port 8012
# =============================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=${MODEL:-/models/Qwen3-30B-A3B}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MOONCAKE_MASTER_PORT=${MOONCAKE_MASTER_PORT:-50088}
LOG_DIR="${SCRIPT_DIR}/logs"

export PATH=/usr/local/python3.11.15/bin:$PATH
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3

# 当前仅支持 A3
export PYTHONHASHSEED=0
export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export ASCEND_BUFFER_POOL=4:8

MOONCAKE_CONFIG="${SCRIPT_DIR}/mooncake.json"
export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG}"

if [ ! -f "${MOONCAKE_CONFIG}" ]; then
    echo "ERROR: mooncake config not found at ${MOONCAKE_CONFIG}"
    exit 1
fi

mkdir -p "${LOG_DIR}"

echo "============================================"
echo " 2. 多实例 KV Pool"
echo "============================================"
echo "  Model:            ${MODEL}"
echo "  TP:               2"
echo "  实例 A:           NPU 4,5    port 8011"
echo "  实例 B:           NPU 6,7  port 8012"
echo "  Mooncake master:  127.0.0.1:${MOONCAKE_MASTER_PORT}"
echo ""

# ---------- 启动 mooncake_master ----------
echo "=== Starting mooncake_master ==="
if pgrep -f "mooncake_master.*${MOONCAKE_MASTER_PORT}" > /dev/null; then
    echo "mooncake_master already running"
else
    nohup /usr/local/bin/mooncake_master \
        --port "${MOONCAKE_MASTER_PORT}" \
        --eviction_high_watermark_ratio 0.9 \
        --eviction_ratio 0.05 \
        --default_kv_lease_ttl 11000 \
        > "${LOG_DIR}/mooncake_master.log" 2>&1 &
    echo "mooncake_master started (PID: $!)"
    sleep 2
fi

# ---------- 清理旧进程 ----------
for p in 8011 8012; do
    PID=$(lsof -ti :${p} 2>/dev/null || true)
    if [ -n "${PID}" ]; then
        echo "Killing old vllm on port ${p} (PID: ${PID})"
        kill -9 ${PID} 2>/dev/null || true
        sleep 1
    fi
done

cd /vllm-workspace/vllm

KV_CONFIG='{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "lookup_rpc_port": "0",
        "backend": "mooncake"
    }
}'

# ---------- 实例 A ----------
echo "=== Starting instance A (port 8011, NPU 4,5) ==="
ASCEND_RT_VISIBLE_DEVICES=4,5 python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --trust-remote-code \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --port 8011 \
    --kv-transfer-config "${KV_CONFIG}" \
    > "${LOG_DIR}/instance_A_port8011.log" 2>&1 &
PID_A=$!
echo "instance A PID: ${PID_A}"

# ---------- 实例 B ----------
echo "=== Starting instance B (port 8012, NPU 6,7) ==="
ASCEND_RT_VISIBLE_DEVICES=6,7 python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --trust-remote-code \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --port 8012 \
    --kv-transfer-config "${KV_CONFIG}" \
    > "${LOG_DIR}/instance_B_port8012.log" 2>&1 &
PID_B=$!
echo "instance B PID: ${PID_B}"

# ---------- 等待就绪 ----------
wait_for() {
    local port=$1 pid=$2 name=$3
    echo "=== Waiting for ${name} (port ${port}, up to 600s) ==="
    for i in $(seq 1 300); do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "${name} ready on port ${port}"
            return 0
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: ${name} exited prematurely"
            tail -30 "${LOG_DIR}/instance_${name#instance }.log" 2>/dev/null || true
            return 1
        fi
        echo -n "."
        sleep 2
    done
    echo "ERROR: ${name} timeout"
    return 1
}

wait_for 8011 "${PID_A}" "instance_A"
echo ""
wait_for 8012 "${PID_B}" "instance_B"

echo ""
echo "============================================"
echo " Both instances ready!"
echo ""
echo "  实例 A: http://localhost:8011"
echo "  实例 B: http://localhost:8012"
echo ""
echo "  验证池化: 先在 A 上发请求存 KV, 再在 B 上发相同请求查 External prefix cache hit rate"
echo "============================================"
