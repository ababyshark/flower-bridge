#!/bin/bash
set -euo pipefail

# =============================================
# 3.tq4_vllm_ascend_v1 — TQ4 GQA 部署 (turboquant-vllm-npu plugin)
# Qwen3.6-35B-A3B, NPU 8,9, port 8013
# activate: VLLM_ASCEND_ENABLE_TQ4_GQA=1
# =============================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=${MODEL:-/models/Qwen3.6-35B-A3B/}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-81920}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
LOG_FILE="${SCRIPT_DIR}/tq4_35B_$(date +%Y%m%d_%H%M%S).log"

export PATH=/usr/local/python3.11.15/bin:$PATH
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

export http_proxy=http://p_test123:Huawei%40123@proxy.huawei.com:8080
export https_proxy=http://p_test123:Huawei%40123@proxy.huawei.com:8080
export NO_PROXY="10.246.63.56,10.246.63.32,localhost,127.0.0.1,192.168.0.0/16,.local,10.0.0.0/8,172.16.0.0/12,openclaw-gateway,mirrors.tools.huawei.com,.huawei.com"

export VLLM_ASCEND_ENABLE_TQ4_GQA=1
export TQ4_PROFILE=${TQ4_PROFILE:-0}
export TQ4_PROFILE_SYNC=${TQ4_PROFILE_SYNC:-1}
export TQ4_PROFILE_INTERVAL=${TQ4_PROFILE_INTERVAL:-100}
export ASCEND_RT_VISIBLE_DEVICES=8,9
source /usr/local/Ascend/cann-8.5.1/opp/vendors/vllm-ascend/bin/set_env.bash 2>/dev/null || true

echo "============================================"
echo " TQ4 GQA 35B (turboquant-vllm-npu plugin)"
echo "============================================"
echo "  Model:            ${MODEL}"
echo "  Port:             8013"
echo "  TP:               2"
echo "  NPU devices:      8,9"
echo "  Max model len:    ${MAX_MODEL_LEN}"
echo "  Max num seqs:     ${MAX_NUM_SEQS}"
echo "  Log:              ${LOG_FILE}"
echo ""

echo "=== Cleanup old vllm on port 8013 ==="
EXISTING=$(lsof -ti :8013 2>/dev/null || true)
if [ -n "${EXISTING}" ]; then
    kill -9 ${EXISTING} 2>/dev/null || true
    sleep 2
fi

echo "=== Starting vllm serve ==="
cd /vllm-workspace/vllm
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --trust-remote-code --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.85 --dtype bfloat16 \
    --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enforce-eager --tensor-parallel-size 2 --port 8013 \
    > "${LOG_FILE}" 2>&1 &
SERVER_PID=$!
echo "PID: ${SERVER_PID}"

# 等待就绪
echo "=== Waiting for server to be ready ==="
for i in $(seq 1 300); do
    if curl -s http://localhost:8013/health > /dev/null 2>&1; then
        echo "Server ready on port 8013"
        grep -i -E "kv cache|blocks|memory|TQ4|GQA|turboquant" "$LOG_FILE" | tail -20
        exit 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: Server exited prematurely"
        tail -80 "$LOG_FILE"
        exit 1
    fi
    echo -n "."
    sleep 2
done
echo ""
echo "ERROR: Server did not start within 10 minutes"
tail -50 "$LOG_FILE"
exit 1
