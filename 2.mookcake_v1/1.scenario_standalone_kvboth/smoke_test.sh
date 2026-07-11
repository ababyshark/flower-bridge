#!/bin/bash
set -euo pipefail
PORT=${1:-8010}
API_BASE="http://localhost:${PORT}"
URL="${API_BASE}/v1/chat/completions"
MODEL_NAME="$(curl -sf "${API_BASE}/v1/models" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "/models/Qwen3-30B-A3B")"

echo "=========================================="
echo " Smoke Test - Port ${PORT}"
echo "=========================================="

tests=(
  "Hello:Hello, how are you?"
  "自我介绍:请用中文做一个简短的自我介绍。"
  "法国首都:The capital of France is"
  "1+1:1+1等于几？请直接说出答案。"
  "写诗:请用中文写一首关于春天的五言绝句。"
  "翻译:请把'Artificial intelligence is transforming the world'翻译成中文。"
  "常识:地球绕太阳转一圈需要多长时间？"
  "代码:用Python写一个计算斐波那契数列前10个数的函数。"
  "长文本总结:请用一句话总结：'深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的层次化表示。近年来，随着计算能力的提升和大规模数据集的出现，深度学习在图像识别、自然语言处理和语音识别等领域取得了突破性进展。'"
)

for item in "${tests[@]}"; do
  title="${item%%:*}"
  prompt="${item#*:}"
  echo ""
  echo ">>> [${title}]"
  echo ">>> ${prompt}"
  echo "---"

  response=$(curl -s "$URL" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg m "$MODEL_NAME" --arg p "$prompt" '{
      model: $m,
      messages: [{role: "user", content: $p}],
      max_tokens: 1024,
      temperature: 0
    }')")

  content=$(echo "$response" | jq -r '.choices[0].message.content // "ERROR: no content"')
  usage=$(echo "$response" | jq -r '.usage // empty')

  echo "$content"
  if [ -n "$usage" ]; then
    echo "(tokens: $(echo "$usage" | jq -c .))"
  fi
done

echo ""
echo "=========================================="
echo " Smoke Test Complete"
echo "=========================================="
