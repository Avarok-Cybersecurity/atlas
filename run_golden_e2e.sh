#!/bin/bash
# FINAL e2e for submission handoff: golden/frozen c2final config, full both-phase
# MLCommons run (1007 perf + 995 BFCL), --num-drafts = winner. Matches the
# defaults_20260721/dgx2-baseline serve exactly except ND.
set -u
ND="${ND:?winner num-drafts}"
BIN="${BIN:-/workspace/.wt-decode-fold/target/release/spark}"
cd /workspace/endpoints-fresh
TS=$(date +%Y%m%d_%H%M%S); RD="results/chainK_golden_e2e_${TS}"
python3 - "$RD" <<'PY'
import sys, yaml
c = yaml.safe_load(open("results/defaults_20260721_173342/config.yaml"))
c["report_dir"] = sys.argv[1]
c.setdefault("endpoint_config", {})["endpoints"] = ["http://localhost:8888"]
yaml.safe_dump(c, open("/workspace/.wt-decode-fold/golden_e2e.yaml", "w"), sort_keys=False)
print("e2e config ->", sys.argv[1])
PY
sudo docker rm -f atlas-golden-e2e >/dev/null 2>&1; sleep 3
sudo docker run -d --name atlas-golden-e2e --network host --gpus all --ipc=host \
  -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0 \
  -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 -e ATLAS_SSM_TAIL_PROTECT=1 \
  -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v "$BIN:/usr/local/bin/spark:ro" \
  atlas-gb10:followups serve centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf \
  --host 0.0.0.0 --port 8888 --model-name centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf \
  --max-seq-len 32768 --max-batch-size 1 --kv-cache-dtype bf16 --gpu-memory-utilization 0.70 \
  --enable-prefix-caching --ssm-cache-slots 128 --ssm-checkpoint-interval 32 \
  --speculative --num-drafts "$ND" --mtp-quantization bf16 \
  --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking >/dev/null 2>&1
for i in $(seq 1 150); do curl -sf -m4 http://localhost:8888/v1/models 2>/dev/null | grep -q Qwen && break
  sudo docker ps --format '{{.Names}}' | grep -q atlas-golden-e2e || { echo SERVE_DIED; exit 1; }; sleep 5; done
echo "golden e2e serve up (nd=$ND); smoke:"
curl -sf -m30 http://localhost:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"temperature":0}' | head -c 150; echo
./.venv/bin/inference-endpoint benchmark from-config -c /workspace/.wt-decode-fold/golden_e2e.yaml --mode both -v
echo "GOLDEN_E2E_DONE rd=$RD"
