# Recommended serve config — FOLD: fp8-KV (decode-gap campaign, 2026-07-24)

**Change vs frozen c2final: `--kv-cache-dtype bf16` → `--kv-cache-dtype fp8`.** The ONE foldable win
from the decode-gap campaign. Confirmed by full MLCommons e2e (1007 perf + 995 BFCL, temp0/seed42):

| config | wall | TTFT p50 | TPOT p50 | TPS | BFCL | IoU |
|---|---|---|---|---|---|---|
| bf16-KV (baseline) | ~4984s | ~1557ms | ~40ms | 15.9 | ~87 | 0.6285 |
| **fp8-KV (FOLD)** | **4534.9s** | **1271ms** | 39.08ms | **17.08** | **87.54** | 0.6223 |
| confirmed vLLM | 5361s | — | — | 14.6 | 86.43 | 0.6269 |

fp8-KV: **wall −9% / TTFT −18% / TPS +7% / BFCL +0.5** vs bf16 baseline; **wall −15% / TPS +17% /
BFCL +1.1** vs confirmed vLLM. IoU 0.6223 = −0.006 vs bf16, −0.005 vs vLLM — **within the ~0.022 IoU
MDE (tie)**. Accuracy (BFCL) IMPROVED. The win is TTFT/wall (fp8 halves KV-cache traffic → prefill);
raw TPOT is ~unchanged (roofline-bound). ⚠ OWNER CONFIRM: IoU nominally −0.006 (MDE-tie); if the thin
IoU margin matters for the official submission, keep bf16-KV. For throughput, fp8-KV is the win.

## Full recommended serve (dense-27B, K=3)
```
--kv-cache-dtype fp8   # <-- the fold (was bf16)
--max-seq-len 32768 --max-batch-size 1 --gpu-memory-utilization 0.70 --enable-prefix-caching
--ssm-cache-slots 128 --ssm-checkpoint-interval 32 --speculative --num-drafts 2 --mtp-quantization bf16
--tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking
ENV: ATLAS_NO_FFN_NVFP4_MMQ=1 ATLAS_SSM_TAIL_MIDCHUNK=0 ATLAS_MTP_CATCHUP=0 ATLAS_MTP_DRAFT_CONF=0.0
     ATLAS_MTP_GATE_FORCE=1 ATLAS_SSM_TAIL_PROTECT=1 ATLAS_SSM_TAIL_LEASE_TTL=128 ATLAS_BF16_TC_PREFILL=1
```
