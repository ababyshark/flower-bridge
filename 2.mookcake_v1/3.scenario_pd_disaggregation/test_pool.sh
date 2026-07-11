#!/bin/bash
set -euo pipefail

# ============================================
# PD 分离池化验证: Prefill 存 KV, Decode 取 KV
# ============================================

PORT_PREFILL=${1:-8015}
PORT_DECODE=${2:-8016}
MODEL=${MODEL:-/models/Qwen3-30B-A3B}
PROMPT="人工智能是计算机科学的一个重要分支，它主要研究如何让计算机模拟人类的智能行为。近年来，随着深度学习技术的快速发展，人工智能在医疗诊断、自动驾驶、自然语言处理等领域取得了突破性进展。在医疗领域，AI可以帮助医生更准确地分析医学影像，辅助诊断疾病，提高诊疗效率。在自动驾驶领域，AI技术使汽车能够感知周围环境，做出安全的驾驶决策。在自然语言处理方面，大型语言模型能够理解和生成人类语言，为人们提供智能对话、翻译、写作等服务。随着算力的不断提升和数据的日益丰富，人工智能正在深刻改变着我们的生活方式和社会形态。"

echo "=========================================="
echo " PD 分离池化验证"
echo "  Prefill: port ${PORT_PREFILL} (存 KV)"
echo "  Decode:  port ${PORT_DECODE}  (取 KV)"
echo "=========================================="

# 1. 请求 Prefill — 计算 KV 并存入 pool
echo ""
echo "=== Step 1: 请求 Prefill (${PORT_PREFILL}), 计算并存入 KV ==="
curl -s "http://localhost:${PORT_PREFILL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":50,\"temperature\":0.0}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('Prefill prompt_tokens:', d.get('usage',{}).get('prompt_tokens','?'))" 2>/dev/null
sleep 30

# 2. 请求 Decode — 从 pool 取 KV
echo ""
echo "=== Step 2: 请求 Decode (${PORT_DECODE}), 从 pool 取 KV ==="
curl -s "http://localhost:${PORT_DECODE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":50,\"temperature\":0.0}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('Decode prompt_tokens:', d.get('usage',{}).get('prompt_tokens','?'))" 2>/dev/null

# 3. 检查 Decode 的 External prefix cache hit rate
echo ""
echo "=== Step 3: Decode 日志中 External prefix cache hit rate ==="
D_LOG="$(ls -t /vllm-workspace/deploy/2.mookcake_v1/3.scenario_pd_disaggregation/logs/decode_port${PORT_DECODE}*.log 2>/dev/null | head -1)"
if [ -z "${D_LOG}" ]; then
    echo "  WARNING: 找不到 Decode 日志"
else
    echo "  log: ${D_LOG}"
    grep -i "External prefix cache hit rate" "${D_LOG}" | tail -5 || echo "  未出现 External prefix cache hit rate 行"
fi

echo ""
echo "=========================================="
echo " External prefix cache hit rate > 0% → 池化生效 ✅"
echo "=========================================="
