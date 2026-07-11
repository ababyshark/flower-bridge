#!/bin/bash
set -euo pipefail

# ============================================
# 池化验证: 实例 A 存 KV, 实例 B 取 KV
# 预期: B 日志中出现 External prefix cache hit rate > 0
# ============================================

PORT_A=${1:-8011}
PORT_B=${2:-8012}
MODEL=${MODEL:-/models/Qwen3-30B-A3B}
PROMPT="人工智能是计算机科学的一个重要分支，它主要研究如何让计算机模拟人类的智能行为。近年来，随着深度学习技术的快速发展，人工智能在医疗诊断、自动驾驶、自然语言处理等领域取得了突破性进展。在医疗领域，AI可以帮助医生更准确地分析医学影像，辅助诊断疾病，提高诊疗效率。在自动驾驶领域，AI技术使汽车能够感知周围环境，做出安全的驾驶决策。在自然语言处理方面，大型语言模型能够理解和生成人类语言，为人们提供智能对话、翻译、写作等服务。随着算力的不断提升和数据的日益丰富，人工智能正在深刻改变着我们的生活方式和社会形态。"

echo "=========================================="
echo " 池化验证: A(存KV) -> B(取KV)"
echo " 实例 A: port ${PORT_A}"
echo " 实例 B: port ${PORT_B}"
echo "=========================================="

# 1. 请求实例 A — 存 KV
echo ""
echo "=== Step 1: 请求实例 A (${PORT_A}), 计算并存入 KV ==="
START_A=$(date +%s%3N)
curl -s "http://localhost:${PORT_A}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":50,\"temperature\":0.0}" \
  > /tmp/pool_test_A.json
END_A=$(date +%s%3N)
TTFT_A=$(python3 -c "import json; d=json.load(open('/tmp/pool_test_A.json')); print(d.get('usage',{}).get('prompt_tokens',0))" 2>/dev/null || echo "?")
echo "  prompt_tokens: ${TTFT_A}"
echo "  latency: $((END_A - START_A))ms"
sleep 30

# 2. 请求实例 B — 应从 pool 取 KV
echo ""
echo "=== Step 2: 请求实例 B (${PORT_B}), 应从 pool 取 KV ==="
START_B=$(date +%s%3N)
curl -s "http://localhost:${PORT_B}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":50,\"temperature\":0.0}" \
  > /tmp/pool_test_B.json
END_B=$(date +%s%3N)
TTFT_B=$(python3 -c "import json; d=json.load(open('/tmp/pool_test_B.json')); print(d.get('usage',{}).get('prompt_tokens',0))" 2>/dev/null || echo "?")
echo "  prompt_tokens: ${TTFT_B}"
echo "  latency: $((END_B - START_B))ms"

# 3. 检查 External prefix cache hit rate
echo ""
echo "=== Step 3: B 实例日志中 External prefix cache hit rate ==="
B_LOG="$(ls -t /vllm-workspace/deploy/mookcake_v1/2.scenario_multi_instance_pool/logs/instance_B_port${PORT_B}*.log 2>/dev/null | head -1)"
if [ -z "${B_LOG}" ]; then
    echo "  WARNING: 找不到 B 实例日志"
else
    echo "  log: ${B_LOG}"
    grep -i "External prefix cache hit rate" "${B_LOG}" | tail -5 || echo "  未出现 External prefix cache hit rate 行"
fi

echo ""
echo "=========================================="
echo " 如果 B 的 latency < A, 且日志中出现"
echo " External prefix cache hit rate > 0%"
echo " 说明池化生效 ✅"
echo "=========================================="
