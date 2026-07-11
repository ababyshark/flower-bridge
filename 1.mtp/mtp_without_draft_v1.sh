#!/bin/bash
set -eo pipefail; export TMPDIR=/dev/shm
MODEL=/models/Qwen3-30B-A3B
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
LOG_FILE=${LOG_FILE:-/vllm-workspace/deploy/0.1-mtp/0.1-mtp.log}

export ASCEND_SLOG_PRINT_TO_STDOUT=0; export ASCEND_GLOBAL_LOG_LEVEL=3

export PATH=/usr/local/python3.11.15/bin:$PATH

export ASCEND_RT_VISIBLE_DEVICES=12,13

echo "=== Baseline :8002 (NPU 12,13, max_num_seqs=${MAX_NUM_SEQS}, max_model_len=${MAX_MODEL_LEN}, max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}==="
cd /vllm-workspace/vllm
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --trust-remote-code --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.85 --dtype bfloat16 \
    --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enforce-eager --tensor-parallel-size 2 --port 8002 \
    > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "PID: ${SERVER_PID} | log: ${LOG_FILE}"

(
  for i in $(seq 1 600); do
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
      echo ""
      echo "========== KV Cache Memory Info =========="
      grep -i -E "blocks|kv cache|gpu blocks|cpu blocks|block_size|memory" "$LOG_FILE" | tail -20
      echo "==========================================="
      exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "ERROR: Server exited. tail -100 ${LOG_FILE}"
      exit 1
    fi
    sleep 2
  done
) >> "$LOG_FILE" 2>&1 &

echo "Server starting in background. Check status:"
echo "  curl http://localhost:8002/health"
echo "  tail -f ${LOG_FILE}"
