#!/bin/bash
set -u
IMG=atlas-gb10:followups; BIN=/workspace/.wt-decode-fold/target/release/spark
MODEL=centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf; HF=/workspace/.cache/huggingface; PORT=8888
OUT=/workspace/.wt-decode-fold/kv_ab; mkdir -p "$OUT"
ENV=(-e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0 -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 -e ATLAS_SSM_TAIL_PROTECT=1 -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1)
leg(){
  local tag="$1"; local kv="$2"; local CN="atlas-kv-$tag"
  for c in $(sudo docker ps -q --filter name=atlas-kv-); do sudo docker rm -f "$c">/dev/null 2>&1; done; sleep 4
  sudo docker run -d --name "$CN" --network host --gpus all --ipc=host "${ENV[@]}" \
    -v "$HF:/root/.cache/huggingface:ro" -v "$BIN:/usr/local/bin/spark:ro" "$IMG" serve "$MODEL" \
    --host 0.0.0.0 --port $PORT --model-name qwen --max-seq-len 32768 --max-batch-size 1 \
    --kv-cache-dtype $kv --gpu-memory-utilization 0.70 --enable-prefix-caching --ssm-cache-slots 128 \
    --ssm-checkpoint-interval 32 --speculative --num-drafts 2 --mtp-quantization bf16 \
    --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking >/dev/null 2>&1
  for i in $(seq 1 150); do curl -sf -m4 http://0.0.0.0:$PORT/v1/models 2>/dev/null|grep -q qwen&&break
    sudo docker ps --format '{{.Names}}'|grep -q "^$CN$"||{ echo "[$tag] DIED"; sudo docker logs "$CN" 2>&1|tail -20; return 1;}; sleep 5; done
  echo "== $tag ($kv) up =="; python3 /workspace/.wt-decode-fold/ab_probe.py $PORT "$tag" "$OUT/$tag.json"
  sudo docker rm -f "$CN">/dev/null 2>&1; sleep 4; }
echo "### fp8-KV A/B $(date)"; leg bf16kv bf16||exit 1; leg fp8kv fp8||exit 1
python3 - <<'PY'
import json
a=json.load(open("/workspace/.wt-decode-fold/kv_ab/bf16kv.json")); b=json.load(open("/workspace/.wt-decode-fold/kv_ab/fp8kv.json"))
print("\n== FP8-KV SUMMARY =="); print(f"  bf16-KV TPOT {a['tpot_med_warm']:.2f}ms  fp8-KV TPOT {b['tpot_med_warm']:.2f}ms")
d=(b['tpot_med_warm']-a['tpot_med_warm'])/a['tpot_med_warm']*100 if a['tpot_med_warm'] else 0
print(f"  delta {d:+.1f}%  {'fp8-KV FASTER' if d<-1 else 'no win'}  (accuracy needs IoU/BFCL e2e; prev golden_fp8kv=86.33 PASS)")
PY
echo KV_AB_DONE
