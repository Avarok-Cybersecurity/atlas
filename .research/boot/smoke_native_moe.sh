#!/bin/bash
# temp=0 smoke tests against the native-MoE serve on :8890.
set -u
OUT=/home/ms/.claude/jobs/5a7bd33d/tmp/boot-smoke
mkdir -p "$(dirname "$OUT")"

echo "=== test 1: 2+2 (decode arm, tiny prompt) ==="
curl -s http://127.0.0.1:8890/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen4exp-exl3","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"temperature":0,"max_tokens":32}' \
  | tee "$OUT-1.json"
echo
echo "=== test 2: ~200-token prompt (prefill arm) ==="
LONG="The Industrial Revolution was a period of global transition of the human economy towards more widespread, efficient and stable manufacturing processes that succeeded the Agricultural Revolution. Beginning in Great Britain, the Industrial Revolution spread to continental Europe and the United States, during the period from around 1760 to about 1840. This transition included going from hand production methods to machines; new chemical manufacturing and iron production processes; the increasing use of water power and steam power; the development of machine tools; and the rise of the mechanized factory system. Output increased greatly, and the result was an unprecedented rise in population and the rate of population growth. The textile industry was the first to use modern production methods, and textiles became the dominant industry in terms of employment, value of output, and capital invested. Many technological and architectural innovations were British in origin. By the mid-18th century, Britain was the leading commercial nation, controlling a global trading empire with colonies in North America and the Caribbean. Britain had major military and political hegemony on the Indian subcontinent. In one short sentence, what was the first industry to use modern production methods according to this passage?"
curl -s http://127.0.0.1:8890/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"qwen4exp-exl3\",\"messages\":[{\"role\":\"user\",\"content\":\"$LONG\"}],\"temperature\":0,\"max_tokens\":64}" \
  | tee "$OUT-2.json"
echo
echo "=== test 3: 64-token generation ==="
curl -s http://127.0.0.1:8890/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen4exp-exl3","messages":[{"role":"user","content":"List the first eight planets of the solar system in order from the sun, one per line."}],"temperature":0,"max_tokens":64}' \
  | tee "$OUT-3.json"
echo
