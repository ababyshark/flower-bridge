#!/bin/bash
set -euo pipefail

# =============================================
# 3.scenario_pd_disaggregation
# PD 分离: Prefill 算 KV 存入 pool, Decode 取 KV 直接生成
# Prefill: NPU 4,5  port 8015  kv_producer
# Decode:  NPU 6,7  port 8016  kv_consumer
#
# 测试方法:
#   1. 请求先发到 Prefill (8015), 其计算 KV 并存入 pool
#   2. 相同请求发到 Decode (8016), 从 pool 取 KV 直接生成
#   3. 观察 Decode 日志的 "External prefix cache hit rate"
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
echo " 3. PD 分离"
echo "============================================"
echo "  Model:            ${MODEL}"
echo "  TP:               2"
echo "  Prefill:          NPU 4,5  port 8015 (kv_producer)"
echo "  Decode:           NPU 6,7  port 8016 (kv_consumer)"
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
for p in 8015 8016; do
    PID=$(lsof -ti :${p} 2>/dev/null || true)
    if [ -n "${PID}" ]; then
        echo "Killing old vllm on port ${p} (PID: ${PID})"
        kill -9 ${PID} 2>/dev/null || true
        sleep 1
    fi
done

cd /vllm-workspace/vllm

# ---------- Prefill (kv_producer) ----------
PREFILL_CONFIG='{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "lookup_rpc_port": "1",
        "backend": "mooncake"
    }
}'

echo "=== Starting Prefill (port 8015, NPU 4,5) ==="
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
    --port 8015 \
    --kv-transfer-config "${PREFILL_CONFIG}" \
    > "${LOG_DIR}/prefill_port8015.log" 2>&1 &
PID_P=$!
echo "Prefill PID: ${PID_P}"

# ---------- Decode (kv_consumer) ----------
DECODE_CONFIG='{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
        "lookup_rpc_port": "2",
        "backend": "mooncake",
        "load_async": true
    }
}'

echo "=== Starting Decode (port 8016, NPU 6,7) ==="
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
    --port 8016 \
    --kv-transfer-config "${DECODE_CONFIG}" \
    > "${LOG_DIR}/decode_port8016.log" 2>&1 &
PID_D=$!
echo "Decode PID: ${PID_D}"

# ---------- 等待就绪 ----------
wait_for() {
    local port=$1 pid=$2 name=$3 log=$4
    echo "=== Waiting for ${name} (port ${port}, up to 600s) ==="
    for i in $(seq 1 300); do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "${name} ready on port ${port}"
            return 0
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: ${name} exited prematurely"
            tail -30 "${log}" 2>/dev/null || true
            return 1
        fi
        echo -n "."
        sleep 2
    done
    echo "ERROR: ${name} timeout"
    return 1
}

wait_for 8015 "${PID_P}" "Prefill" "${LOG_DIR}/prefill_port8015.log"
echo ""
wait_for 8016 "${PID_D}" "Decode"  "${LOG_DIR}/decode_port8016.log"

echo ""
echo "============================================"
echo " PD 分离就绪!"
echo ""
echo "  Prefill:  http://localhost:8015 (算 KV, 存入 pool)"
echo "  Decode:   http://localhost:8016 (从 pool 取 KV)"
echo ""
echo "  测试方法:"
echo "    1. 发请求到 Prefill:  curl -s http://localhost:8015/v1/chat/completions ..."
echo "    2. 发相同请求到 Decode: curl -s http://localhost:8016/v1/chat/completions ..."
echo "    3. 观察 Decode 日志: tail -f ${LOG_DIR}/decode_port8016.log | grep 'External prefix cache hit rate'"
echo "============================================"
