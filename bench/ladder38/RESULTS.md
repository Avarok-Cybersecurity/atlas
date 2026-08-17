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

## MTP acceptance study (2026-08-17) — acceptance is NOT the gap

Instrumented `MTP accept` lines across every serve log on both boxes, bucketed by width:

| n | k_drafts | flushes | mean p1 | tok_step |
|---:|---:|---:|---:|---:|
| 1 | 3 | 68 | 0.80-0.90 | 2.75-3.26 |
| 4 | 3 | 28 | 0.84-0.88 | 2.75-3.03 |
| 8 | 3 | 55 | 0.78-0.87 | 2.57-2.99 |
| 16 | 1 | 51 | 0.770 | 1.770 |
| 16 | 2 | 31 | 0.863 | 2.582 |
| 32 | 1 | 843 | 0.64-0.68 | 1.64-1.68 |

**Per-draft acceptance (p1) is flat at 0.78-0.90 through n=16 — at or above the published
Qwen MTP band (0.7-0.85). Atlas's drafter is not the problem.** What collapses at n>=16 is
`tok_step`, because the K ladder (`speculative/ladder.rs:200`, `4:3,8:3,16:1,32:1`) hands
out ONE draft at those widths while vLLM keeps 3 at every width.

But the ladder cannot close the gap: break-even arithmetic bounds every admissible rung
change at ±10% on prose traffic, against a 29-31% deficit at C=16/32. Also `32:3` is not a
shape at all — 4 rows/seq x 32 = 128 > `VERIFY_ROW_BUDGET` 96 (`mtp_dcut.rs:55`), so it
serializes. Valid arms are `16:2`, `16:3`, `32:2` (96 rows exactly).

### Defects found while auditing the accept path (each with a proposed test)

- **B1 `--mtp-vocab 100000` makes every control token undraftable.** This checkpoint's added
  tokens are all in 248044..248076 (EOS 248046/248044, `</think>` 248069, `<tool_call>`
  248058), and the drafter's argmax is bounded at 100000 (`mtp_head/forward.rs:448-452`).
  Every such position is a guaranteed miss that truncates the rest of the span. Negligible on
  the prose ladder (~1 special per 1024 tokens); **4-6% of positions on BFCL/agentic**.
  Fix: `--mtp-vocab 0` (costs ~0.8 ms/propose — measure, don't assume).
- **B2 drafter carry is force-disabled on every default serve** — `mtp_carry.rs:98-103`
  requires `!mtp_multi_seq_mode()`, and that predicate is true whenever `mtp_max_seqs() > 1`
  (default 32), so carry is off even at C=1. Recorded worth: +0.079 p1 / +0.089 p2.
- **B5 zero-kept grammar truncation skips `trim_proposer_state`** (`mtp_step.rs:440-465`),
  leaving drafter KV rows for tokens the target never emitted — permanent desync.
- **B6 `--mtp-quantization bf16` does not cover the draft LM head** (`forward.rs:453-465`
  hard-wires NVFP4), a candidate for the n>=16 vs n<=8 p1 difference.
- **PR #549 (landed): accept-debug width buckets aliased.** `MAX_N` was 17 while the
  dispatch cap is 32, so every width 16..128 folded onto bucket 16 — the adaptive rung
  controller (BAND 9..=16) was steering on a mixture of n=16 and n>16 statistics.

### Where the gap actually is

Atlas's marginal cost per added concurrent sequence is **4.28 ms/token/seq vs vLLM's 1.94**.
That is not acceptance (p1 flat), not launch count (PR #547: 96n -> 2n launches moved C=8 by
+2.2%), and not state bytes (PR #548: h-state halved, reserve 36.6 -> 22.4 GB, same +2.2%).
Bandwidth arithmetic says 4.28 ms/seq at 273 GB/s implies ~1.17 GB moved per sequence per
step, versus ~72 MB of f16 h-state (x4 verify rows = ~288 MB). **The remaining ~4x is
unexplained by any traffic we have accounted for — the next step is an nsys profile of the
DECODE step at C=1 vs C=8, the decode analogue of the prefill profile that found the M=280
launch shape.**

## ROUND 4 (2026-08-17) — fp8 KV + PR #547 + PR #548, apples-to-apples

Stack `b508679e4`, Atlas served at **fp8 KV** (matching the reference at last) with
`--ssm-h-dtype f16-pool`, both marginal-cost fixes engaged (verified in the serve log:
"h pool SIZED at 2 bytes", no contiguous-block fallback, reserve 36.6 -> **22.4 GB**).

| C | round 4 | round 3 floor | Δ | vLLM+MTP | ratio | rung |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 22.44 | 21.74 | +3.2% | 19.72 | **1.14x** | **WON** |
| 2 | 30.32 | 29.04 | +4.4% | 38.79 | 0.78x | open |
| 4 | 52.35 | 51.55 | +1.6% | 71.61 | 0.73x | open |
| 8 | 83.22 | 81.42 | +2.2% | 124.48 | 0.67x | open |
| 16 | 154.30 | 150.41 | +2.6% | 197.03 | 0.78x | open |
| 32 | 225.37 | 219.97 | +2.5% | 283.48 | 0.79x | open |
| 64 | **373.90** | 360.02 | +3.9% | 361.39 | **1.035x** | **WON** |
| 128 | **442.83** | 450.12 | -1.6% | 358.57 | **1.235x** | **WON** |

**One regression to record honestly:** C=128 came in at 442.83 with `f16-pool` versus
450.12 without it (-1.6%) — the widen/narrow staging pair costs a little at the widest
rung even as it frees 14 GB. The rung is still won by 1.235x, but if C=128 ever tightens,
running that rung WITHOUT `f16-pool` is the cheaper config. Every other rung improved over
its own floor. Three rungs now won
apples-to-apples (C=1, C=64, C=128). The two fixes were worth +1.6-4.4% each rung — real,
but an order of magnitude short of the 30% needed at C=4/8, which is consistent with the
acceptance study's conclusion that the marginal cost lives somewhere we have not yet
profiled.

### K-ladder A/B (2026-08-17) — NEGATIVE RESULT, hypothesis closed

`ATLAS_MTP_K_LADDER="4:3,8:3,16:2,32:2"` (deeper drafts at the widths where the default
ladder hands out only one) measured at C=16: **153.92** versus 154.30 on the default
`16:1` — **-0.2%, i.e. nothing**, against a 28% deficit at that rung (vLLM+MTP 197.03).

This closes the draft-budget hypothesis. It matches the break-even arithmetic exactly:
on prose traffic the token-ratio gain of a second draft (~1.20) is cancelled by its cost
ratio (~1.17-1.26). The ladder is a +/-10% lever at best and this workload sits at its
break-even point. (`32:3` was never a candidate: 4 rows/seq x 32 = 128 > the 96-row
`VERIFY_ROW_BUDGET`, so it serializes.)

Ruled out for the mid-ladder gap, each by measurement rather than argument:
acceptance quality (p1 flat 0.78-0.90), draft budget (this A/B), launch count (PR #547),
state bytes (PR #548), KV dtype (round 4 at fp8). What remains is the per-sequence
marginal cost itself — 4.28 ms/token/seq vs vLLM's 1.94 — with no traffic accounting that
explains it. A decode-step nsys profile at C=1 vs C=8 is the next instrument.

## DECODE PROFILE (2026-08-17) — the missing 4x, found

nsys capture of steady-state decode at C=1 and C=8 on the round-4 binary. Artifacts:
`dgx2:/home/claude/prof_decode_c{1,8}.{nsys-rep,sqlite}`.

| | C=1 | C=8 |
|---|---:|---:|
| step wall | 94.0 ms | 209.7 ms |
| **GPU busy** | **96.3%** | **77.2%** |
| gaps > 50us | 556 | 7404 |

Marginal cost per added sequence, attributed:

| component | Δ/seq | share |
|---|---:|---:|
| main-model weight GEMM | 5.94 ms | 36% |
| **CUDA-graph instantiate/destroy** | **3.31 ms** | **20%** |
| GDN recurrent (wy4/3/2 + conv) | 2.42 ms | 15% |
| host sampling pipeline | 1.34 ms | 8% |
| MTP drafter GEMM | 1.13 ms | 7% |
| eager launch + D2D host calls | 0.70 ms | 4% |
| attention + elementwise | 0.60 ms | 4% |

### Three independent defects, two already measured on hardware

1. **GDN `in_proj_qkvz` decode reads a pre-dequantized FP8 weight copy instead of the
   NVFP4 one** (`layers/qwen3_ssm/trait_decode_batched.rs:342`; the NVFP4 copy is right
   there at :350, and `out_proj` ALREADY has the `num_tokens > 8 -> nvfp4` arm at
   :920-947). `fp8_fp8_gemm_ldmab` costs **42.58 ms/step = 26.3% of the whole step** at
   C=8, moves **+1.762 GB/step (+77.8%)** of extra weight traffic, and achieves only
   **94.6 GB/s against 203 GB/s** on this shape (its 128-row M tile is ~75% padding at
   M=32). ~90% of the C=1->C=8 GEMM-time regression is this one arm. The serve log
   asserts "decode keeps NVFP4" — the profile disproves it. Estimated C=4 +26%, C=8 +20%,
   C=16 +25%.
2. **Verify CUDA-graph key thrash.** `verify_e2.rs:58-71` keys on the interleaved
   (slot, depth) arrangement, and D-Cut re-ranks which slot gets which depth every step,
   so the key space at n=8 is **266 arrangements against a 32-entry cache** — 149 captures
   in 167 steps, **23.2 ms/step** of instantiate/destroy. Measured +6.9% at C=8 by
   disabling D-Cut (which also loses its pruning, so the thrash alone costs more).
   D-Cut's recorded +2.6% predates this key and is now net-negative.
3. **`presence_penalty=1.5` in the `non_thinking` preset disables four fast-sampling
   paths** (`fast_greedy.rs:59-70`). Measured **+7.8%** at C=8.
   ★ This was also a LIKE-FOR-LIKE VIOLATION: Atlas injects that penalty when a request
   omits it, while vLLM defaults to 0 — the two engines were doing different sampling work
   and emitting different text. The harness now sends `presence_penalty: 0.0` and
   `frequency_penalty: 0.0` explicitly to BOTH engines (`harness_w55_conc_ladder.py`),
   which makes the comparison honest and recovers the 7.8% legitimately.

Defects 2+3 measured together at C=8: **78.89 -> 91.96 tok/s (+16.6%)**. With defect 1's
estimate the rung lands near 110 against vLLM's 124.48.

Refuted by the same profile: serialized drafting (3 drafter forwards at batch dim n, not
3n). Surviving-but-secondary: h-state re-reads inside `gated_delta_rule_wy4_f16`
(2 reads + 4 writes per step; a resident K=2 twin exists, no K=4 twin).
