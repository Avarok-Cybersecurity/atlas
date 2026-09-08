#!/bin/bash
# One A/B arm: launch the fixed binary (env passthrough), wait ready, measure, save greedy sample, stop.
# $1 = arm name (used for file names). Extra env (e.g. ATLAS_EXL3_SHARED_PREFILL_GEMM=1) is inherited.
set -u
ARM=$1
D=/home/ms/.claude/jobs/5a7bd33d/tmp/exl3bench
LOG=$D/serve_${ARM}.log
SERVE=$D/serve_exl3_fix.sh; [ "${MTP:-0}" = 1 ] && SERVE=$D/serve_exl3_fix_agentic.sh
[ "${NOLAUNCH:-0}" = 1 ] || setsid $SERVE 8888 > "$LOG" 2>&1 < /dev/null &
for i in $(seq 1 900); do
  if curl -s -m 2 http://127.0.0.1:8888/v1/models >/dev/null 2>&1; then echo "READY after ~${i}s"; break; fi
  if ! pgrep -f "serv[e] --model-from-path.*8888" >/dev/null; then echo "SERVER EXITED"; tail -20 "$LOG" | cut -c1-200; exit 1; fi
  sleep 1
done
python3 -u $D/measure_decode.py --port 8888 --repeats 3 --max-tokens 300 > $D/measure_${ARM}.txt 2>&1
cat $D/measure_${ARM}.txt
# greedy sample for output-equality check across arms (non-streaming, 200 tokens)
curl -s -m 600 http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"qwen3.8-flash-next","temperature":0.0,"max_tokens":200,
  "chat_template_kwargs":{"reasoning_effort":"low"},
  "messages":[{"role":"user","content":"Write a complete Rust implementation of an LRU cache with generic key and value types, using a HashMap and a doubly linked list of indices into a Vec arena. Include get, put, capacity handling, and unit tests. Explain each design decision briefly in comments."}]}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print(m.get("reasoning_content") or m.get("reasoning") or ""); print("=====CONTENT====="); print(m.get("content"))' > $D/sample_${ARM}.txt 2>&1
wc -c $D/sample_${ARM}.txt; sha256sum $D/sample_${ARM}.txt | cut -c1-16
grep -aE "shared|GEMV|gemv" "$LOG" | grep -ai exl3 | head -3 | cut -c1-200
pkill -f "serv[e] --model-from-path.*8888"; sleep 6; pgrep -af "spark serv[e]" | cut -c1-40; echo "arm $ARM done"
