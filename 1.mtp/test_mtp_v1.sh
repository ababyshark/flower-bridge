#!/bin/bash
set -euo pipefail

MODEL=${MODEL:-/models/Qwen3-30B-A3B}
PORT=${PORT:-8001}
PROMPT_TOKENS=${PROMPT_TOKENS:-63488}
OUTPUT_TOKENS=${OUTPUT_TOKENS:-1024}

echo "=== Starting Baseline v1 benchmark on port ${PORT} (64K/1K) ==="
LOG_FILE="evalscope_perf_${PORT}_baseline_v1.log"
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
echo "  PID=$!"
echo "tail -f evalscope_perf_${PORT}_baseline_v1.log"
