#!/bin/bash
set -u; cd /workspace/endpoints-fresh
BIN=/workspace/.wt-decode-fold/target/release/spark
sudo docker rm -f atlas-fp8kv-conglom >/dev/null 2>&1; sleep 3
sudo docker run -d --name atlas-fp8kv-conglom --network host --gpus all --ipc=host \
  -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0 \
  -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 \
  -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" -v "$BIN:/usr/local/bin/spark:ro" \
  atlas-gb10:followups serve centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf --host 0.0.0.0 --port 8888 \
  --model-name centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf --max-seq-len 32768 --max-batch-size 1 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.70 --enable-prefix-caching --ssm-cache-slots 128 \
  --ssm-checkpoint-interval 32 --speculative --num-drafts 2 --mtp-quantization bf16 \
  --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking >/dev/null 2>&1
for i in $(seq 1 150); do curl -sf -m4 http://localhost:8888/v1/models 2>/dev/null|grep -q Qwen&&break
  sudo docker ps --format '{{.Names}}'|grep -q atlas-fp8kv-conglom||{ echo SERVE_DIED; exit 1;}; sleep 5; done
echo "serve up ($((i*5))s); smoke:"; curl -sf -m30 http://localhost:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf","messages":[{"role":"user","content":"Say hi in 3 words."}],"max_tokens":20,"temperature":0}' 2>/dev/null | head -c 200; echo
./.venv/bin/inference-endpoint benchmark from-config -c fp8kv_conglom.yaml --mode both -v
echo FP8KV_CONGLOM_DONE
