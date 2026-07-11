#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=${MODEL:-/models/Qwen3-30B-A3B}
PORT=${PORT:-8010}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
TP_SIZE=${TP_SIZE:-2}
MOONCAKE_MASTER_PORT=${MOONCAKE_MASTER_PORT:-50088}
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/mooncake_vllm_$(date +%Y%m%d_%H%M%S).log"
MASTER_LOG="${LOG_DIR}/mooncake_master.log"

export PATH=/usr/local/python3.11.15/bin:$PATH
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3

# ---------- NPU selection ----------
# devices 14,15 for TP=2
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-6,7}
export ASCEND_RT_VISIBLE_DEVICES

# ---------- Env vars for mooncake ----------
# 当前仅支持 A3
export PYTHONHASHSEED=0
export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export ASCEND_BUFFER_POOL=4:8

# ---------- Mooncake config ----------
MOONCAKE_CONFIG="${SCRIPT_DIR}/mooncake.json"
export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG}"

if [ ! -f "${MOONCAKE_CONFIG}" ]; then
    echo "ERROR: mooncake config not found at ${MOONCAKE_CONFIG}"
    exit 1
fi

mkdir -p "${LOG_DIR}"

echo "=== Mooncake vLLM Deployment ==="
echo "  Model:            ${MODEL}"
echo "  Port:             ${PORT}"
echo "  TP:               ${TP_SIZE}"
echo "  NPU devices:      ${ASCEND_RT_VISIBLE_DEVICES}"
echo "  Max model len:    ${MAX_MODEL_LEN}"
echo "  Max num seqs:     ${MAX_NUM_SEQS}"
echo "  Mooncake master:  127.0.0.1:${MOONCAKE_MASTER_PORT}"
echo "  Log:              ${LOG_FILE}"
echo "  Master log:       ${MASTER_LOG}"
echo ""

# ---------- Step 1: Kill the mooncake version service on port if already running ----------
echo "=== Step 1: Cleanup previous mooncake vllm on port ${PORT} ==="
EXISTING_VLLM=$(lsof -ti :${PORT} 2>/dev/null || true)
if [ -n "${EXISTING_VLLM}" ]; then
    echo "Killing existing vllm on port ${PORT} (PID: ${EXISTING_VLLM})..."
    kill -9 ${EXISTING_VLLM} 2>/dev/null || true
    sleep 2
    echo "Killed."
fi

# ---------- Step 2: Start mooncake_master ----------
echo "=== Step 2: Starting mooncake_master ==="
EXISTING_MASTER=$(pgrep -f "mooncake_master.*${MOONCAKE_MASTER_PORT}" 2>/dev/null || true)
if [ -z "${EXISTING_MASTER}" ]; then
    nohup /usr/local/bin/mooncake_master \
        --port "${MOONCAKE_MASTER_PORT}" \
        --eviction_high_watermark_ratio 0.9 \
        --eviction_ratio 0.05 \
        --default_kv_lease_ttl 11000 \
        > "${MASTER_LOG}" 2>&1 &
    MASTER_PID=$!
    echo "mooncake_master started (PID: ${MASTER_PID})"
    sleep 2
    if ! kill -0 "${MASTER_PID}" 2>/dev/null; then
        echo "ERROR: mooncake_master failed to start. Check ${MASTER_LOG}"
        tail -20 "${MASTER_LOG}"
        exit 1
    fi
else
    echo "mooncake_master already running on port ${MOONCAKE_MASTER_PORT} (PID: ${EXISTING_MASTER})"
    MASTER_PID="${EXISTING_MASTER}"
fi

# ---------- Step 3: Start vllm serve with mooncake AscendStoreConnector ----------
echo "=== Step 3: Starting vllm serve with mooncake KV pool ==="
cd /vllm-workspace/vllm

KV_CONFIG='{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "lookup_rpc_port": "0",
        "backend": "mooncake"
    }
}'

python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --trust-remote-code \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --enforce-eager \
    --tensor-parallel-size "${TP_SIZE}" \
    --port "${PORT}" \
    --kv-transfer-config "${KV_CONFIG}" \
    > "${LOG_FILE}" 2>&1 &

SERVER_PID=$!
echo "vllm server started (PID: ${SERVER_PID})"

# ---------- Step 4: Wait for server to be ready ----------
echo "=== Step 4: Waiting for server to be ready (up to 600s) ==="
for i in $(seq 1 300); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo ""
        echo "Server is ready on port ${PORT}!"
        echo ""
        echo "========== KV Cache Memory Info =========="
        grep -i -E "blocks|kv cache|gpu blocks|cpu blocks|block_size|memory|Mooncake|mooncake|AscendStore|Registering KV" "${LOG_FILE}" | tail -30
        echo "==========================================="
        echo ""
        echo "========== Smoke Test =========="
        curl -s "http://localhost:${PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello, what is AI?\"}],\"max_tokens\":50,\"temperature\":0.0}" \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print('Response:', d.get('choices',[{}])[0].get('message',{}).get('content','ERROR: No response')[:200])" 2>&1
        echo "================================"
        exit 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo ""
        echo "ERROR: Server exited prematurely."
        echo "=== Last 80 lines of log ==="
        tail -80 "${LOG_FILE}"
        exit 1
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "ERROR: Server did not become ready within 600 seconds"
echo "=== Last 80 lines of log ==="
tail -80 "${LOG_FILE}"
exit 1
