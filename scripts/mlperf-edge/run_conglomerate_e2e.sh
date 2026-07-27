#!/bin/bash
# Conglomerate e2e: build folded-wins binary, serve frozen c2final K=3, run full MLCommons
# (1007 perf + 995 BFCL, temp0/seed42). Run ON the e2e box (dgx2). Resilient (setsid).
set -u
BOX="${BOX:-dgx2}"; SHA="${SHA:?set SHA=<folded-branch-tip>}"
WT="${WT:-/home/claude/atlas-conglom}"; TS=$(date +%Y%m%d_%H%M%S)
ART="/home/claude/e2e_conglom_${SHA:0:8}_${TS}"
echo "[conglom] box=$BOX sha=$SHA art=$ART"
# 1. build folded-wins binary at SHA (correct target)
git -C "$WT" fetch origin 2>/dev/null; git -C "$WT" checkout "$SHA" 2>/dev/null || { echo FETCH_FIRST; exit 1; }
( cd "$WT" && PATH=/usr/local/cuda/bin:$PATH ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.6-27b \
   cargo build --release -p spark-server --bin spark --features cuda ) || { echo BUILD_FAIL; exit 1; }
grep -m1 'compiled .* kernels for target' "$WT"/build*.log 2>/dev/null || true
# 2. serve (frozen c2final, ARM=bare/K=3) — mount fresh binary
sudo docker rm -f atlas-conglom >/dev/null 2>&1; sleep 3
sudo docker run -d --name atlas-conglom --network host --gpus all --ipc=host \
  -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0 \
  -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 -e ATLAS_SSM_TAIL_PROTECT=1 \
  -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1 ${EXTRA_ENV:-} \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v "$WT/target/release/spark:/usr/local/bin/spark:ro" atlas-gb10:followups \
  serve centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf --host 0.0.0.0 --port 8888 \
  --model-name centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf --max-seq-len 32768 --max-batch-size 1 \
  --kv-cache-dtype bf16 --gpu-memory-utilization 0.70 --enable-prefix-caching \
  --ssm-cache-slots 128 --ssm-checkpoint-interval 32 --speculative --num-drafts 2 \
  --mtp-quantization bf16 --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking >/dev/null 2>&1
for i in $(seq 1 150); do curl -sf -m4 http://0.0.0.0:8888/v1/models 2>/dev/null | grep -q Qwen && break; sleep 5; done
# 3. MLCommons run (edge-agentic-full-run config), resilient
mkdir -p "$ART"
echo "[conglom] serve up; launch MLCommons harness into $ART (see endpoints-fresh runner)"
