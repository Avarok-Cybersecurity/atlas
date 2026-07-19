#!/bin/bash
# verify_win.sh — verify a performance win WHILE confirming accuracy is preserved/recovered.
# Two tiers (cheap — no full 2.7h e2e per variant):
#   MODE=equiv  : for numerics-equivalent changes (in-place GDN, midchunk GDN tail capture).
#                 byte-identical output check (identical outputs => identical BFCL score =>
#                 accuracy PROVABLY preserved) + speed probe. Strongest + cheapest.
#   MODE=numerics: for numerics-changing changes (fp8 KV — quant drifts outputs).
#                 BFCL subset A/B (category_sample_pct cut + --accuracy-only, ~5 min) for the
#                 accuracy delta + speed probe. Fold only if opt accuracy >= baseline.
#
# Usage: MODE=equiv BASE_IMG=atlas-gb10:mtp-reverify BASE_KV=bf16 \
#        OPT_IMG=atlas-gb10:midchunk OPT_KV=bf16 MODEL=nvidia/Qwen3.6-27B-NVFP4 K=3 PORT=8880 \
#        bash verify_win.sh
set +e
MODE="${MODE:-equiv}"
BASE_IMG="${BASE_IMG:?}"; BASE_KV="${BASE_KV:-bf16}"
OPT_IMG="${OPT_IMG:?}"; OPT_KV="${OPT_KV:-bf16}"
MODEL="${MODEL:?}"; K="${K:-3}"; PORT="${PORT:-8880}"
DRAFTS="${K}"  # num-drafts
STAMP=$(date +%Y%m%d_%H%M%S)
MLBL=$(echo "$MODEL"|sed 's#^.*/##;s#[/.]#_#g')
LOG=/workspace/verify_win_${MODE}_${MLBL}_${STAMP}.log
exec > "$LOG" 2>&1
ROOT=/workspace/endpoints-fresh; EXDIR=examples/11_Edge_Agentic_Example

serve() { # $1=cn $2=img $3=kv
  sudo docker rm -f "$1" 2>/dev/null; sleep 5
  sudo docker run -d --name "$1" --network host --gpus all --ipc=host \
    -e ATLAS_MTP_DRAFTER_PREFILL="${DRAFTS:+1}" \
    -v /workspace/.cache/huggingface:/root/.cache/huggingface \
    "$2" serve "$MODEL" --host 0.0.0.0 --port $PORT --model-name "$MODEL" \
      --max-seq-len 32768 --max-batch-size 1 --kv-cache-dtype "$3" --gpu-memory-utilization 0.70 \
      --enable-prefix-caching --ssm-cache-slots 128 --ssm-checkpoint-interval 32 \
      --speculative --num-drafts "$DRAFTS" --mtp-quantization bf16 \
      --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking 2>&1 | tail -1
}
wait_ready() { for i in $(seq 1 48); do curl -sf -m4 "http://localhost:$PORT/v1/models" 2>/dev/null|grep -q Qwen && { echo "READY ($((i*5))s)"; return 0; }; sudo docker ps --format '{{.Names}}'|grep -q "^$1$" || { echo "DIED"; sudo docker logs --tail 20 "$1"; return 1; }; sleep 5; done; echo "NOT READY"; return 1; }

bfcl_subset() { # $1=label $2=cn  — tiny BFCL accuracy subset (~5 min)
  wait_ready "$2" || return 1
  local DST="$ROOT/$EXDIR/verify_win_subset_${1}_${STAMP}.yaml" RPT="$ROOT/results/verify_win_subset_${1}_${STAMP}"
  python3 - "$EXDIR/online_edge_full_run.yaml" "$DST" "$MODEL" "$PORT" "$RPT" <<'PY'
import re,sys
src,dst,served,port,rpt=sys.argv[1:6]
s=open(src).read()
s=re.sub(r'(\n  name: )"Qwen3\.6-27B-Q4_K_M".*',r'\1"%s"'%served,s,count=1)
s=re.sub(r'(\n    - )"http://localhost:8080".*',r'\1"http://localhost:%s"'%port,s,count=1)
s=re.sub(r'(\nreport_dir: ).*',r'\1%s/'%rpt,s,count=1)
s=s.replace("        non_live: 62","        non_live: 8").replace("        live: 10","        live: 2").replace("        hallucination: 10","        hallucination: 2").replace("      subset_floor: 25","      subset_floor: 3")
open(dst,"w").write(s)
PY
  cd "$ROOT"; source "$ROOT/.venv/bin/activate" 2>/dev/null
  inference-endpoint benchmark from-config --config "$DST" --accuracy-only > "/workspace/verify_win_subset_${1}.log" 2>&1
  grep -aE "Score for bfcl|Errors:|samples evaluated|Completed in" "/workspace/verify_win_subset_${1}.log" 2>/dev/null | tail -3
  cd /workspace
}

echo "=== verify_win  MODE=$MODE  base=$BASE_IMG($BASE_KV)  opt=$OPT_IMG($OPT_KV)  model=$MODEL K=$K  $(date '+%F %T') ==="

echo "=== [1/2] BASELINE ($BASE_IMG $BASE_KV) ==="
serve atlas-vw-base "$BASE_IMG" "$BASE_KV"
wait_ready atlas-vw-base || { echo "base serve failed"; exit 1; }
sudo docker logs atlas-vw-base 2>&1 | grep -aiE "KV cache dtype" | head -1
python3 /workspace/probe_byteidentical.py "localhost:$PORT" "$MODEL" "base_vw" 2>&1 | tail -3
python3 /workspace/probe_profile.py "localhost:$PORT" "$MODEL" "base_vw_profile" 4 2>&1 | tail -7
[ "$MODE" = numerics ] && bfcl_subset base_vw atlas-vw-base
sudo docker rm -f atlas-vw-base 2>/dev/null

echo "=== [2/2] OPT ($OPT_IMG $OPT_KV) ==="
serve atlas-vw-opt "$OPT_IMG" "$OPT_KV"
wait_ready atlas-vw-opt || { echo "opt serve failed"; exit 1; }
sudo docker logs atlas-vw-opt 2>&1 | grep -aiE "KV cache dtype" | head -1
python3 /workspace/probe_byteidentical.py "localhost:$PORT" "$MODEL" "opt_vw" 2>&1 | tail -3
python3 /workspace/probe_profile.py "localhost:$PORT" "$MODEL" "opt_vw_profile" 4 2>&1 | tail -7
[ "$MODE" = numerics ] && bfcl_subset opt_vw atlas-vw-opt
sudo docker rm -f atlas-vw-opt 2>/dev/null

echo "=== VERDICT ==="
if [ "$MODE" = equiv ]; then
  if diff -u /workspace/base_vw_outputs.json /workspace/opt_vw_outputs.json > /workspace/vw_diff.txt 2>&1; then
    echo "BYTE_IDENTICAL: accuracy PROVABLY preserved (outputs match exactly)"
  else
    echo "NOT_BYTE_IDENTICAL: outputs differ — accuracy may have changed (DO NOT fold)"
    head -30 /workspace/vw_diff.txt
  fi
  echo "--- speed (TTFT/decode) baseline vs opt ---"
  python3 - <<'PY'
import json
b=json.load(open("/workspace/base_vw_profile_profile.json"))["summary"] if __import__('os').path.exists("/workspace/base_vw_profile_profile.json") else json.load(open("/workspace/base_vw_profile.json"))["summary"]
o=json.load(open("/workspace/opt_vw_profile_profile.json"))["summary"] if __import__('os').path.exists("/workspace/opt_vw_profile_profile.json") else json.load(open("/workspace/opt_vw_profile.json"))["summary"]
print(f"{'bucket':12}{'base_dec':>9}{'opt_dec':>9}{'d_dec':>7}{'base_ttft':>10}{'opt_ttft':>10}")
for bk in b:
    bv=b[bk];ov=o.get(bk,{})
    bd,od=bv.get("decode_tok_s_med"),ov.get("decode_tok_s_med"); bt,ot=bv.get("ttft_med_ms"),ov.get("ttft_med_ms")
    print(f"{bk:12}{('%.1f'%bd) if bd else '-':>9}{('%.1f'%od) if od else '-':>9}{f'{(od-bd):+.1f}' if (bd and od) else '-':>7}{('%d'%bt) if bt else '-':>10}{('%d'%ot) if ot else '-':>10}")
PY
elif [ "$MODE" = numerics ]; then
  echo "--- BFCL subset accuracy: base vs opt ---"
  echo "base:"; grep -aE "Score for bfcl|Errors:" /workspace/verify_win_subset_base_vw.log 2>/dev/null | tail -2
  echo "opt:";  grep -aE "Score for bfcl|Errors:" /workspace/verify_win_subset_opt_vw.log 2>/dev/null | tail -2
  echo "--- speed (cold-prefill TTFT) base vs opt ---"
  python3 - <<'PY'
import json,os
def ld(p): return json.load(open(p))["summary"] if os.path.exists(p) else {}
b=ld("/workspace/base_vw_profile_profile.json") or ld("/workspace/base_vw_profile.json")
o=ld("/workspace/opt_vw_profile_profile.json") or ld("/workspace/opt_vw_profile.json")
print(f"{'bucket':12}{'base_ttft':>10}{'opt_ttft':>10}{'d_ttft':>9}{'base_dec':>9}{'opt_dec':>9}")
for bk in b:
    bv=b[bk];ov=o.get(bk,{})
    bt,ot=bv.get("ttft_med_ms"),ov.get("ttft_med_ms"); bd,od=bv.get("decode_tok_s_med"),ov.get("decode_tok_s_med")
    print(f"{bk:12}{('%d'%bt) if bt else '-':>10}{('%d'%ot) if ot else '-':>10}{f'{(ot-bt):+.0f}ms' if (bt and ot) else '-':>9}{('%.1f'%bd) if bd else '-':>9}{('%.1f'%od) if od else '-':>9}")
PY
fi
echo "VERIFY_WIN_DONE  mode=$MODE"
