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

### THE APPLES-TO-APPLES REFERENCE (2026-08-17) — vLLM WITH MTP, fp8 KV

The earlier vLLM reference ran **speculative decoding OFF**, which understated it badly.
vLLM 0.27.1 registers `Qwen3_5MTP` and this checkpoint ships `mtp.*` weights, so vLLM can
and should run MTP here. Re-measured with every workload axis matched to Atlas — same
checkpoint/box/harness/prompts/ISL/OSL/temp/seed, ctx 2048 both, batch cap 128 both,
util 0.85 both, **fp8 KV both**, prefix caching on both, thinking off both, and
**MTP K=4 on both** (Atlas `--num-drafts 3`, vLLM `num_speculative_tokens: 3`):

| C | vLLM+MTP fp8 | (old no-spec ref) |
|---:|---:|---:|
| 1 | 19.72 | 11.04 |
| 2 | 38.79 | 21.34 |
| 4 | 71.61 | 41.20 |
| 8 | 124.48 | 78.18 |
| 16 | 197.03 | 137.11 |
| 32 | 283.48 | 219.50 |
| 64 | 361.39 | 312.26 |
| 128 | **358.57** | 390.36 |

Two structural facts: vLLM+MTP is 1.8-1.9x its own no-spec numbers at low C (so every
comparison against the no-spec reference is superseded), and **vLLM's C=128 is BELOW its
own C=64** — MTP verification costs it more than it gains at 128-wide, while Atlas's
speculation self-disables above 32 concurrent sequences and never pays that penalty.

Standing (Atlas C=128 at fp8 = 450.12; other Atlas rungs still bf16 KV pending round 4):

| C | Atlas | vLLM+MTP | ratio | rung |
|---:|---:|---:|---:|---|
| 1 | 21.74 | 19.72 | 1.10x | **WON** |
| 2 | 29.04 | 38.79 | 0.75x | open |
| 4 | 51.55 | 71.61 | 0.72x | open |
| 8 | 81.42 | 124.48 | 0.65x | open |
| 16 | 150.41 | 197.03 | 0.76x | open |
| 32 | 219.97 | 283.48 | 0.78x | open |
| 64 | 360.02 | 361.39 | 0.996x | open |
| 128 | **450.12** | 358.57 | **1.26x** | **WON** |

Measured root cause of the open rungs: Atlas's marginal cost per added concurrent sequence
is **4.28 ms/token/seq** vs vLLM's **1.94** (TPOT fits Atlas `58.9 + 4.28n`, collinear
across n=2,4,8; C=1 is off the line because `decode_a2.rs:65` routes n==1 to a different
single-sequence program). The hybrid carries ~102 MB of GDN recurrent state per sequence
per step; Atlas additionally paid 96 eager copy launches per sequence per step for SSM
rollback (PR #547 -> 2n) and stored h-state FP32 even under `--ssm-h-dtype f16`
(PR #548 -> `f16-pool`, halves the bytes). Round 4 measures both.

### Round 2 — full fix stack `ab97a7f24` (2026-08-17)

Stack = capacity PR #533 + graph-borrow #536 + varlen-prefill #538 + preempt-resume #540,
served with `ATLAS_PREFILL_CODISPATCH=1 ATLAS_FP8_ROWWISE=1` and `--prefill-varlen-batch`.

| C | Atlas | vLLM | ratio | rung |
|---:|---:|---:|---:|---|
| 1 | 21.74 | 11.04 | 1.97x | WON |
| 2 | 29.04 | 21.34 | 1.36x | WON |
| 4 | 51.55 | 41.20 | 1.25x | WON |
| 8 | 81.42 | 78.18 | 1.04x | WON |
| 16 | 150.41 | 137.11 | 1.10x | WON |
| 32 | 219.97 | 219.50 | 1.002x | **WON** (was 218.34 pre-stack) |
| 64 | 360.02 | 312.26 | 1.15x | WON (was 338.38) |
| 128 | 274.41 | 390.36 | 0.70x | OPEN — KV-capacity bound |

**7 of 8 rungs won.** C=128 mechanism is fully understood and no longer a correctness
problem: preempt-resume + depth-aware admission deliver all 131,072 tokens with ZERO
kills (the pre-stack build discarded 25% of decode work via 171 preempt-kills that
returned HTTP-200 empty bodies). The remaining deficit is capacity: the KV pool holds
102k tokens against a 157k-token demand, so only ~82 of 128 sequences run concurrently
and aggregate throughput follows batch width. Levers under test: fp8 KV (checkpoint's
declared kv_cache_quant_algo; needs both engines re-baselined), and completing the
fp16 SSM pool to cut the 36.7 GiB reserve. `--gpu-memory-utilization 0.90` was tried
and RETIRED: it froze the box (unified memory; 0.85 is the proven ceiling on GB10).

### Round 1 — pre-stack `d92fc2488` (2026-08-16, superseded)

| C | Atlas | vLLM | ratio |
|---:|---:|---:|---:|
| 1 | 22.96 | 11.04 | 2.08x |
| 2 | 30.61 | 21.34 | 1.43x |
| 4 | 53.72 | 41.20 | 1.30x |
| 8 | 83.10 | 78.18 | 1.06x |
| 16 | 150.90 | 137.11 | 1.10x |
| 32 | 218.34 | 219.50 | 0.995x |
| 64 | 338.38 | 312.26 | 1.08x |
| 128 | 255.94 | 390.36 | 0.66x |

C=1 and C=2 read lower in round 2 (-5%) but their rep spreads are 4.8%/6.6% versus
0.3-1.0% at the wider rungs, so the dip is not yet established as real; more reps
before any conclusion. Every other rung improved or held.

## Known mechanics behind the open rungs

- C=32: deficit is the prefill ramp (Atlas ~620-745 tok/s prefill vs vLLM ~2.9k);
  Atlas DECODES 10.5% faster per token at this rung (TPOT p50 128.7 vs 143.5 ms).
  Spec dispatches on 100% of steps. Fix in flight: drain-tail CUDA-graph reuse
  (~+2%), then prefill throughput campaign (profiled, ranked targets on file).
- C=128: distress signatures (90k/131k tokens delivered, 38.7 s TTFT p50) —
  forensic analysis in progress.
