# Qwen3.8-27B NVFP4 concurrency ladder — Atlas vs latest vLLM (2026-08-16)

**Status: campaign in progress — 6/8 rungs won. PRELIMINARY; not yet gate-certified.**

## Fingerprint

- Box: dgx2 (spark-43fa, GB10 121.7 GB), same box/checkpoint/client for both engines, back-to-back.
- Checkpoint: `unsloth/Qwen3.8-27B-NVFP4` (dense 27B hybrid, 48 GDN + 16 attn layers).
- Harness: `w55_conc_ladder.py` (sha256 `6412b12d…`), ISL 128 (~200 rendered prompt tokens),
  OSL 1024, temp 0.0, seed 42, 3 reps/rung, 1 warmup.
- vLLM: `vllm/vllm-openai:latest`
  (`sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`),
  `--max-model-len 4096 --max-num-seqs 128 --gpu-memory-utilization 0.85
  --enable-prefix-caching --dtype bfloat16 --kv-cache-dtype bfloat16`. No speculation.
- Atlas: binary `d92fc2488` (PR #533 tip), env `ATLAS_PREFILL_CODISPATCH=1
  ATLAS_FP8_ROWWISE=1`, flags: `--max-seq-len 2048 --max-batch-size 128
  --gpu-memory-utilization 0.85 --kv-cache-dtype bf16 --enable-prefix-caching true
  --ssm-cache-slots 8 --ssm-checkpoint-interval 32 --speculative --num-drafts 3
  --mtp-quantization bf16 --scheduling-policy fifo --disable-thinking
  --request-timeout 0 --ssm-h-dtype f16 --gdn-fused-norm --ssm-batched-recurrent
  --ssm-tail-midchunk false --mtp-gate force`. Spec width caps at 32 (C>32 decodes plain).
  C=1..16 rows are from the codispatch-only sweep; C=32 row is codispatch+rowwise
  (best measured); C=64/128 rows codispatch+rowwise.

## Scores (mean tok/s aggregate over 3 reps)

| C | Atlas | vLLM | ratio | rung |
|---:|---:|---:|---:|---|
| 1 | 22.96 | 11.04 | 2.08x | WON |
| 2 | 29.64 | 21.34 | 1.39x | WON |
| 4 | 51.66 | 41.20 | 1.25x | WON |
| 8 | 82.45 | 78.18 | 1.05x | WON |
| 16 | 149.78 | 137.11 | 1.09x | WON |
| 32 | 218.34 | 219.50 | 0.995x | OPEN (-0.5%; drain-tail graph reuse in validation, PR #536) |
| 64 | 338.38 | 312.26 | 1.08x | WON |
| 128 | 255.94 | 390.36 | 0.66x | OPEN (-34%; under diagnosis — KV pressure / graph churn / prefill ramp) |

Raw per-rung records with TTFT/TPOT percentiles and rep series: this directory.
vLLM rung spreads 0.02-0.32%; Atlas 1-5% (MTP accept-draw variance at temp 0).

## Known mechanics behind the open rungs

- C=32: deficit is the prefill ramp (Atlas ~620-745 tok/s prefill vs vLLM ~2.9k);
  Atlas DECODES 10.5% faster per token at this rung (TPOT p50 128.7 vs 143.5 ms).
  Spec dispatches on 100% of steps. Fix in flight: drain-tail CUDA-graph reuse
  (~+2%), then prefill throughput campaign (profiled, ranked targets on file).
- C=128: distress signatures (90k/131k tokens delivered, 38.7 s TTFT p50) —
  forensic analysis in progress.
