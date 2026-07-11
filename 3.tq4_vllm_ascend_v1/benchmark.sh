#!/bin/bash
set -euo pipefail

# =============================================
# TQ4 Benchmark: 63K/1K, parallel 1/8/16/32
# =============================================

MODEL=${MODEL:-/models/Qwen3-30B-A3B}
PORT=${PORT:-8013}
PROMPT_TOKENS=${PROMPT_TOKENS:-63488}
OUTPUT_TOKENS=${OUTPUT_TOKENS:-1024}

echo "=== TQ4 Benchmark port ${PORT} (63K/1K) ==="
LOG_FILE="${SCRIPT_DIR:-$(dirname "$0")}/evalscope_perf_${PORT}_tq4.log"
nohup evalscope perf \
    --log-every-n-query 1 \
    --model "$MODEL" \
    --url "http://localhost:${PORT}/v1/chat/completions" \
    --api openai \
    --dataset random \
    --min-prompt-length "$PROMPT_TOKENS" \
    --max-prompt-length "$PROMPT_TOKENS" \
    --min-tokens "$OUTPUT_TOKENS" \
    --max-tokens "$OUTPUT_TOKENS" \
    --parallel 1 8 16 32 \
    --number 4 32 64 128 \
    --stream \
    --tokenizer-path "$MODEL" \
    --extra-args '{"ignore_eos": true}' \
    > "$LOG_FILE" 2>&1 &
echo "PID=$!"
echo "tail -f ${LOG_FILE}"
