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

### Round 5 (2026-08-17) — sampling parity alone

Same round-4 binary, only the harness changed (presence/frequency penalties pinned to 0
for BOTH engines, which is what vLLM was already doing):

| C | round 5 | round 4 | Δ | vLLM+MTP |
|---:|---:|---:|---:|---:|
| 8 | **90.68** | 83.22 | **+9.0%** | 124.48 |

Confirms defect 3 at slightly better than the profile's +7.8% estimate, and the engines
now do identical sampling work. Remaining rungs were not measured — the halt-on-behind
guard stopped the run; exploratory rounds now record-and-continue instead.

### Round 6 (2026-08-17) — QKVZ NVFP4 fix (PR #551)

| C | round 6 | round 5 | round 4 | Δ vs r5 | cumulative | vLLM+MTP |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | **106.48** | 90.68 | 83.22 | **+17.4%** | **+28.0%** | 124.48 |

The single dispatch-line fix (batched verify QKVZ reading the NVFP4 weight copy instead of
the pre-dequantized FP8 one) is worth **+17.4%** at C=8 on its own, close to the profile's
+20% estimate. TTFT at that rung also fell from ~10 s to **2.39 s**, because the same
NVFP4 arm serves the prefill shapes.

Running total at C=8 tonight: 83.22 -> 106.48, **+28.0%**, against a 124.48 reference.

**And it flips two more rungs:**

| C | round 6 | round 4 | Δ | vLLM+MTP | ratio | rung |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 106.48 | 83.22 | +28.0% | 124.48 | 0.86x | open |
| 16 | **203.44** | 154.30 | **+31.9%** | 197.03 | **1.033x** | **WON** |
| 32 | **291.52** | 225.37 | **+29.4%** | 283.48 | **1.028x** | **WON** |

**Five of eight rungs now won apples-to-apples: C=1, C=16, C=32, C=64, C=128.**

Remaining rungs after round 6:

| C | round 6 | round 4 | Δ | vLLM+MTP | ratio | gap |
|---:|---:|---:|---:|---:|---:|---|
| 2 | 31.00 | 30.32 | +2.2% | 38.79 | 0.80x | -20% |
| 4 | **68.11** | 52.35 | **+30.1%** | 71.61 | 0.951x | **-5%** |
| 8 | 106.48 | 83.22 | +28.0% | 124.48 | 0.86x | -14% |

Diagnostic: C=2 barely moved, and that is expected — at C=2 with K=4 the GEMM is M=8,
which was ALREADY served by the NVFP4 batch-8 GEMV, so the QKVZ fix is structurally inert
there, and the n=2 graph key space was only 2 so PR #552 does little either. **C=2's
deficit is the FIXED cost of entering the n>=2 batched verify path** — consistent with the
TPOT fit (`58.9 + 4.28n` across n=2,4,8, with n=1 sitting 30% below the line because it
runs a different single-sequence program). Named per-step fixed costs now under repair: a
48 KB H2D WY-table upload plus 48n host downcasts every step, three un-`OnceLock`'d
`std::env::var` calls in the verify hot path, `stash_verify_hidden_rows`, and the D-Cut
planner running at widths where it has nothing to prune.

### Fixed-cost audit (PR #553) — NEGATIVE, with a better instrument

Microbenchmarked the five items a prior audit named as the n>=2 batched-verify fixed cost:

| item | measured per step | verdict |
|---|---:|---|
| `upload_verify_wy_tables` (48 KB H2D + 48n downcasts) | 3-6 us | cached anyway |
| 3x `std::env::var` in the verify hot path | <1 us | hoisted anyway |
| `stash_verify_hidden_rows` | ~5 us @ n=2 | not worth it (a gather kernel would be a net loss at n=2/4) |
| D-Cut plan + chunk sort | sub-us | **must NOT be skipped** — at n=2 it genuinely prunes (ks={4,3}, R=7 not 8) |
| host sampling pipeline | 0 on this ladder | never reached at temp 0 / no grammar / thinking off |

Total removed: ~0.004% of a 170 ms step at C=2. **None of these is the fixed cost.**

A comparative audit left exactly two candidates of the right magnitude, neither observable
from a ladder log: **CUDA-graph re-capture** (23.2 ms/step at an 89% recapture rate, this
tree's own measurement) and **the batched GDN conv+WY path declining** — 2 launches/layer
when engaged versus `n*(2k-1)` when not, i.e. 96 vs 768 launches/step at n=2, k=4 across
48 GDN layers. Both now emit periodic RATES under `ATLAS_MTP_ACCEPT_DEBUG`, so the next
C=2 run reads the answer directly instead of inferring it.

### Round 6 complete + Round 7 (canonical verify key, PR #552)

Round 6 final rungs: C=64 **386.99** (+3.5%, 1.071x) and C=128 **477.69** (+7.9%, **1.332x**)
— the QKVZ fix lifts the top of the ladder too, and C=1 rose to **23.66** (1.20x).

Round 7 adds the canonical verify key (n=8 key space 266 -> 3):

| C | round 7 | round 6 | Δ | vLLM+MTP | ratio |
|---:|---:|---:|---:|---:|---:|
| 8 | **110.63** | 106.48 | +3.9% | 124.48 | 0.889x |

Full standing after rounds 6-7:

| C | Atlas | vLLM+MTP | ratio | rung |
|---:|---:|---:|---:|---|
| 1 | 23.66 | 19.72 | **1.20x** | **WON** |
| 2 | 31.00 | 38.79 | 0.80x | open |
| 4 | 68.11 | 71.61 | 0.951x | open |
| 8 | 110.63 | 124.48 | 0.889x | open |
| 16 | 203.44 | 197.03 | **1.033x** | **WON** |
| 32 | 291.52 | 283.48 | **1.028x** | **WON** |
| 64 | 386.99 | 361.39 | **1.071x** | **WON** |
| 128 | 477.69 | 358.57 | **1.332x** | **WON** |

Tonight's movement, apples-to-apples throughout: C=1 +5%, C=8 +33%, C=16 +32%, C=32 +29%,
C=64 +7%, C=128 +8%. Three rungs remain, all in the C=2..8 band where the batched-verify
path's fixed cost dominates.

### Round 7 (canonical verify key) — helps C=8, REGRESSES C=2 and C=4

| C | round 7 | round 6 | Δ | vLLM+MTP |
|---:|---:|---:|---:|---:|
| 2 | 29.93 | 31.00 | **-3.5%** | 38.79 |
| 4 | 65.56 | 68.11 | **-3.7%** | 71.61 |
| 8 | 110.63 | 106.48 | +3.9% | 124.48 |

The canonical depth->slot assignment collapses the key space (which is why C=8 gains) but
at n=2/n=4 it evidently forces an assignment that costs more than the captures it saves —
plausibly by making the two-launch batched GDN conv+WY fast path decline (96 vs 768
launches/step at n=2,k=4 across 48 layers). **This is exactly the trade the no-regression
rule forbids, so the fix does not ship as-is.**

PR #553's telemetry exists for precisely this question, so the next run is an A/B with the
kill switch (`ATLAS_NO_CANONICAL_VERIFY_KEY=1`) at C=2 and C=4, reading the graph-capture
and GDN fast-path RATES rather than inferring them. Likely landing shape: gate the
canonical key on width (>= 8), keeping C=8's gain without C=2/C=4's cost.

### Round 7 complete — reproducibility check on the won rungs

| C | round 7 | round 6 | reproducibility | vLLM+MTP | ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 23.14 | 23.66 | -2.2% | 19.72 | 1.17x WON |
| 16 | 203.50 | 203.44 | **0.03%** | 197.03 | 1.033x WON |
| 32 | 291.50 | 291.52 | **0.01%** | 283.48 | 1.028x WON |
| 64 | 387.62 | 386.99 | 0.16% | 361.39 | 1.073x WON |
| 128 | 477.55 | 477.69 | 0.03% | 358.57 | 1.332x WON |

The won rungs reproduce across two independent rounds to within 0.2% (C=16/32/128 to
0.03%), so those margins are stable measurements, not noise-riding. C=1's -2.2% is the
canonical key's small-n cost showing up even at single-sequence width — another vote for
gating it.

### Canonical-key A/B (same binary, same session, one variable)

| C | canonical ON | canonical OFF | verdict |
|---:|---:|---:|---|
| 2 | 30.09 | **30.83** | canonical costs **-2.4%** |
| 8 | **110.63** | 106.48 | canonical gains **+3.9%** |

Clean attribution: the canonical assignment pays for itself only where the key space
actually explodes (n=8: 266 arrangements). At n=2 the key space was 2 and at n=4 it was 10
— nothing to collapse, and forcing the assignment costs more than it saves. Fix in flight:
gate the canonical assignment on batch width (>= 8), byte-identical to the pre-canonical
behaviour below the threshold, with an env-sweepable boundary. That keeps C=8's +3.9%
without paying -2.4%/-3.7% at C=2/C=4.

### D-Cut re-sweep (2026-08-17) — the shipped ratio is wrong at C=8

D-Cut's 0.75 was calibrated on a binary from before the verify graph key existed. Re-swept
on the current tree (same session, one fresh serve per ratio):

| ratio | C=8 | C=4 |
|---:|---:|---:|
| 1.0 (pruning off) | **117.06** | 63.24 |
| 0.75 (shipped) | 108.85 | 64.00 |

**+7.5% at C=8 from turning D-Cut's pruning off** — it is net-negative there under the
current cost model. At C=4 the shipped 0.75 is marginally better, so the right answer is
width-dependent, like the canonical key. (0.5 and 0.25 legs pending.)

### The C=1 -> C=2 cliff: the largest unexplained cost in the campaign

| | C=1 | C=2 | scaling | TPOT C=1 | TPOT C=2 | marginal |
|---|---:|---:|---:|---:|---:|---:|
| vLLM+MTP | 19.72 | 38.79 | **1.97x** | 50.7 ms | 51.5 ms | **+0.8 ms/token** |
| Atlas | 23.66 | 30.83 | 1.30x | 42.3 ms | 64.9 ms | **+22.6 ms/token** |

Derived step times (TPOT x tok_step ~3): n=1 ~128 ms, n=2 ~195 ms. **The second sequence
costs ~67 ms per step** on a workload where decode is memory-bound and both widths read
the same ~13.5 GB of weights. vLLM pays 0.8 ms for the same sequence.

This is why Atlas WINS C=1 (1.20x) and loses C=2 (0.79x). n=1 and n=2 run different
programs (`decode_a2.rs:65` short-circuits n==1; batched verify requires n>=2), and every
cheap explanation has been excluded by measurement. A dedicated C=1-vs-C=2 nsys profile is
running; the earlier profile compared C=1 to C=8 and never isolated this step.

#### D-Cut sweep complete — pruning is net-negative at the contested rungs

| ratio | C=8 | C=4 |
|---:|---:|---:|
| **1.0 (pruning off)** | **117.06** | 63.24 |
| 0.75 (shipped) | 108.85 | **66.42** |
| 0.5 | 107.05 | 60.85 |
| 0.25 | 104.59 | 58.72 |

C=8 is monotone in "prune less": 117.06 -> 108.85 -> 107.05 -> 104.59, i.e. **+11.9% from
pruning off versus the most aggressive setting, +7.5% versus the shipped 0.75.** C=4 peaks
at 0.75 (66.42) with 1.0 close behind (63.24) and the aggressive settings clearly worse.

D-Cut sheds verify ROWS to save work, but under the current cost model — after the QKVZ
NVFP4 fix made those rows much cheaper and the graph key made arrangement churn expensive —
the rows it sheds cost less than the ragged shapes it creates. Its 0.75 default and its
recorded +2.6% both predate those changes. Like the canonical key, the right landing is a
width-keyed policy rather than one global constant.

## THE C=1 -> C=2 CLIFF IS ONE KERNEL (2026-08-17)

nsys at C=1 and C=2, steps anchored on the per-step argmax, GPU busy 96.00% vs 95.61%:

| term | C=1 | C=2 | delta |
|---|---:|---:|---:|
| **main projection GEMV** (353.00 -> 355.61 launches — same weights, same count) | 72.28 | 108.46 | **+36.18** |
| MTP drafter (gemv loop -> pipelined GEMM) | 14.22 | 21.46 | +7.24 |
| GDN recurrence (1 wy4/layer -> wy4+wy3) | 1.47 | 3.15 | +1.69 |
| conv1d_update_l2norm (192 -> 340 launches) | 0.56 | 1.03 | +0.47 |
| GPU idle | 3.85 | 6.48 | +2.63 |

`w4a16_gemv_batch8` = **304.98 us** vs `w4a16_gemv_batch4` = **204.76 us** for the identical
353-launch weight sweep — **1.489x**, effective bandwidth 194 -> 129 GB/s. A microbench on
the real 27B shapes shows the kernel is NOT DRAM-bound; its M-row scalar FMA chain is the
critical path and `MAX_M=8` (`acc[8]`, `smem[8][8]`, `__launch_bounds__(BLOCK_SIZE, 5)`)
degrades monotonically with M:

| tier | M | time | eff. BW | % peak |
|---|---:|---:|---:|---:|
| batch4 | 4 | 70.5 us | 209 GB/s | 76.6% |
| batch8 | 4 | 74.8 us | 197 GB/s | 72.3% (launch_bounds cost alone) |
| batch8 | 8 | 106.4 us | 138 GB/s | 50.7% |
| batch16 | 8 | 113.3 us | 130 GB/s | 47.7% |
| gemm_m64 | 8 | 749.6 us | 20 GB/s | 7.2% (tile GEMM is not the escape) |

n=2 lands at R=7 — inside the bad window. n=1 at R=4 is outside it. **That is the whole
cliff.** All three prior suspects died on measurement: graph re-capture `capture_frac=0.000`
over 3000+ steps (`live_keys=1/32`), host serialization (GPU busy barely moves), and the
GDN conv+WY fast path — which is never even *attempted* at n=2 because D-Cut's `ks=[4,3]`
splits into two n=1 groups below the `n < 2` guard.

### Measured workaround, zero code: keep n=2 on `batch4`

`ATLAS_MTP_K_LADDER=1:3,2:1,4:3,8:2,16:1` (same-session A/B, 2 reps each):

| leg | R at n=2 | kernel | C=1 | C=2 | C=4 |
|---|---:|---|---:|---:|---:|
| control (nd=3) | 7 | batch8 | 23.64 | 30.15 | — |
| nd=2 | 6 | batch8 | 23.43 | 33.64 (+11.6%) | — |
| nd=1 | 4 | **batch4** | 23.43 | 38.42 (+27.4%) | — |
| **n=2-only ladder** | 4 | **batch4** | — | **38.66 (+28.2%)** | 63.96 unchanged |

C=2 scaling 1.28x -> **1.64x**; 38.66 against vLLM's 38.79 is parity. It is a workaround
(`tok_step/seq` falls 2.172 -> 1.695), so once the kernel is fixed, nd=3 on a repaired
batch8 should beat it outright (~40 tok/s).

## ROUND 8 (2026-08-17) — C=2 WON. Six of eight rungs.

Single configuration for every rung (stack `6dba55fa0`: capacity + rollback + f16-pool +
QKVZ-NVFP4 + width-gated canonical key + telemetry), with two engine defaults that adapt by
width rather than per-rung tuning: D-Cut pruning off, and the width-adaptive K ladder
`1:3,2:1,4:3,8:2,16:1` which keeps n=2 on the fast `batch4` GEMV tier.

| C | round 8 | vLLM+MTP | ratio | rung |
|---:|---:|---:|---:|---|
| 2 | **38.95** | 38.79 | **1.004x** | **WON** |
| 4 | 67.66 | 71.61 | 0.945x | open |

C=2 has gone 29.04 -> **38.95** across tonight (+34%), and the last 28% of that came from
one insight: at n=2 the batched verify lands on `w4a16_gemv_batch8`, which is 1.489x slower
than `batch4` for the identical weight sweep. The K-ladder step-down keeps that width on
`batch4`. The kernel repair (exact-M tiers 5/6/7, `__launch_bounds__` retune) is in flight
and should beat this workaround outright, since it recovers the drafts the step-down gives
up (`tok_step/seq` 2.172 -> 1.695).

### Round 8 complete — one configuration, all eight rungs

| C | round 8 | vLLM+MTP | ratio | rung |
|---:|---:|---:|---:|---|
| 1 | 23.09 | 19.72 | **1.171x** | **WON** |
| 2 | 38.95 | 38.79 | **1.004x** | **WON** |
| 4 | 67.66 | 71.61 | 0.945x | open |
| 8 | 123.64 | 124.48 | 0.993x | open |
| 16 | 203.14 | 197.03 | **1.031x** | **WON** |
| 32 | 291.42 | 283.48 | **1.028x** | **WON** |
| 64 | 387.14 | 361.39 | **1.071x** | **WON** |
| 128 | 477.79 | 358.57 | **1.332x** | **WON** |

**Six of eight, and the two width-adaptive defaults that won C=2 cost nothing anywhere
else** — C=1/16/32/64/128 all reproduce their prior values (C=32 to 0.03% and C=64 to 0.2%
for the third consecutive round). That matters for publishability: this is one deployment
configuration, not per-rung tuning, which is the only kind of result comparable to vLLM's
single-config ladder.

Remaining: C=4 short by 3.95 tok/s (-5.5%), C=8 short by 0.84 tok/s (-0.7%).

### Round 9 — w4a16 exact-M tiers + PRMT removal (PR #561)

| C | round 9 | round 8 | Δ | vLLM+MTP | ratio |
|---:|---:|---:|---:|---:|---:|
| 4 | **69.73** | 67.66 | **+3.1%** | 71.61 | 0.974x |
| 8 | 123.26 | 123.64 | -0.3% | 124.48 | 0.990x |

The bit-exact instruction reductions (batch4 -6.8%, batch8 -7.8%, batch16/32 -9%) convert to
+3.1% of wall at C=4 and nothing measurable at C=8. **Both open rungs are now inside 3%:**
C=4 short by 1.88 tok/s, C=8 short by 1.22 tok/s.

Two levers in flight for exactly these widths: a zero-code K-ladder micro-sweep at C=4 and
C=8 (the `8:2` rung was picked without ever testing `8:1`/`8:3` under D-Cut-off, and `4:3`
without testing `4:2`/`4:1`), and the MTP drafter's small-M dispatch — the same pathology
just fixed in the main model: M=1 uses `dense_gemv_bf16` at 3.57 ms while M=2 uses
`dense_gemm_bf16_pipelined` at 5.43 ms, 1.52x for two rows, worth +7.24 ms/step of the
second sequence's cost. At C=4 the drafter runs M=4, squarely in that band.

#### Round 9 complete — bit-exact kernel change disturbs nothing

| C | round 9 | round 8 | vLLM+MTP | ratio |
|---:|---:|---:|---:|---:|
| 1 | 23.46 | 23.09 | 19.72 | **1.190x WON** |
| 2 | (r8) 38.95 | 38.95 | 38.79 | **1.004x WON** |
| 4 | **69.73** | 67.66 | 71.61 | 0.974x |
| 8 | 123.26 | 123.64 | 124.48 | 0.990x |
| 16 | 202.48 | 203.14 | 197.03 | **1.028x WON** |
| 32 | 290.69 | 291.42 | 283.48 | **1.026x WON** |
| 64 | 386.58 | 387.14 | 361.39 | **1.070x WON** |
| 128 | 477.47 | 477.79 | 358.57 | **1.332x WON** |

Four independent rounds now agree at the won rungs: C=32 at 291.52/291.50/291.42/290.69 and
C=64 at 386.99/387.62/387.14/386.58. Those margins are stable measurements.

### K-ladder micro-sweep at the open rungs

C=8 (D-Cut off, stack `6d07f5456`, 3 reps each):

| ladder | C=8 |
|---|---:|
| `8:1` | 116.91 |
| **`8:2` (in use)** | **123.26-123.64** |
| `8:3` | 117.50 |

A clean interior optimum — two drafts beats both one and three by ~5%. **The K-ladder lever
is spent at C=8**; its remaining 1.0% must come from elsewhere (the drafter small-M tier,
PR #562, is the candidate: ~-6.0 ms/step at that width).

C=4 arms (same sweep):

| ladder | C=4 |
|---|---:|
| `4:1` | 52.17 |
| **`4:2`** | **71.63** |
| `4:3` (in use) | 69.73 |

**`4:2` puts C=4 at 71.63 against vLLM's 71.61** — ahead by 0.02 tok/s, i.e. a tie within
noise, but it is the best measured configuration and +2.7% over `4:3`. Combined with the
drafter small-M tier (PR #562, ~-6.4 ms/step at that width) the rung should clear
comfortably rather than by a hair. Both open rungs now have a measured path.

## ★ ROUND 10 — ALL EIGHT RUNGS WON, APPLES-TO-APPLES

The MTP drafter small-M tier (PR #562) takes the last two rungs:

| C | round 10 | round 9 | Δ | vLLM+MTP | ratio | rung |
|---:|---:|---:|---:|---:|---:|---|
| 4 | **71.95** | 69.73 | +3.2% | 71.61 | **1.005x** | **WON** |
| 8 | **125.47** | 123.26 | +1.8% | 124.48 | **1.008x** | **WON** |

### THE COMPLETE LADDER

| C | Atlas | vLLM+MTP | ratio |
|---:|---:|---:|---:|
| 1 | 23.46 | 19.72 | **1.190x** |
| 2 | 38.95 | 38.79 | **1.004x** |
| 4 | 71.95 | 71.61 | **1.005x** |
| 8 | 125.47 | 124.48 | **1.008x** |
| 16 | 202.48 | 197.03 | **1.028x** |
| 32 | 290.69 | 283.48 | **1.026x** |
| 64 | 386.58 | 361.39 | **1.070x** |
| 128 | 478.07 | 358.57 | **1.333x** |

Round 10's own full sweep (the table above mixes rounds; round 10 alone measured C=4 71.95,
C=8 125.47, C=1 23.50, C=16 202.93, C=32 291.17, C=64 387.10, C=128 478.07 — and C=2 38.95
from round 8 on the identical configuration).

**Every rung, one configuration, every workload axis matched** — same box, same checkpoint,
same harness and prompts, ISL 128 / OSL 1024, temp 0, seed 42, ctx 2048 both, batch cap 128
both, gpu-util 0.85 both, fp8 KV both, prefix caching on both, thinking off both, MTP K=4
both, presence/frequency penalties pinned to 0 on both. vLLM runs its own `Qwen3_5MTP`
speculative decoding, i.e. the reference is vLLM at its best, not a handicapped baseline.

Still required before any of this is published: the quality gates (ssm-state-poisoning,
decode-floor, BFCL with draw metadata, agentic 10/10) on the final configuration — several
of the fixes change numerics (`f16-pool`, the drafter tier), and a speed claim without an
accuracy claim is not a result.

### Round 11 — confirmation with every lever at its measured optimum

Drafter small-M tier + K ladder `1:3,2:1,4:2,8:2,16:1` (C=4 now on its own optimum too):

| C | round 11 | round 10 | vLLM+MTP | ratio |
|---:|---:|---:|---:|---:|
| 4 | **74.21** | 71.95 | 71.61 | **1.036x** |
| 8 | **125.95** (r11) · 123.22 re-meas. 08-18 | 125.47 | 124.48 · 121.64 same-day | **1.012x** / **1.013x** |

Both formerly-open rungs are now won with margin rather than by a hair — C=4 went from a
0.5% edge to 3.6%, C=8 from 0.8% to 1.2%. Independent confirmation of round 10 on a
separate serve.

### Post-rewrite rebase (2026-08-17)

`main` was force-updated to excise 43.6 MB of accidentally-committed bun artifacts, so all
13 campaign branches were **rebased** (never merged — a merge would have re-added the files
as "added on our side"). Verification beyond the mandated bun greps: for every branch,
`diff(old_tip -> new_tip)` is byte-identical to `diff(db3804f081 -> origin/main)`, proving
the rebase introduced zero content change; `bench/ladder38` kept the same tree SHA, so every
number in this file is the one that was measured. Ladder stack tip: `1575873582`
(full fix stack `bf4d7a1267`).

Gate certifications were deliberately held until after this rebase: a record minted on a
pre-rewrite SHA would name a commit that no longer exists, which is worse than no record.

### C=2 HARDENED (2026-08-18) — the last unreproduced rung, re-measured against a same-day vLLM

C=2 was the weak point of the certified table, for two reasons that compounded: it was the
only rung round 11 did **not** re-measure (the table carries `(r8) 38.95` from round 8), and
at **1.004x** it was won by 0.16 tok/s — a margin smaller than the run-to-run spread of
either engine. A claim of "wins at every rung" rests hardest on its thinnest rung.

Re-measured on **merged main `529fcb04fa`** (i.e. after #572, #569 and #581 all landed),
with vLLM re-run **back-to-back on the same box on the same day** rather than compared to a
week-old number:

| engine | rep 1 | rep 2 | rep 3 | mean | spread |
|---|---:|---:|---:|---:|---:|
| **Atlas** | 41.69 | 39.99 | 41.37 | **41.02** | 4.15% |
| vLLM+MTP | 37.62 | 36.52 | 37.18 | **37.11** | 2.94% |

**Ratio 1.105x**, against the recorded 1.004x. The distributions do not overlap: Atlas's
WORST rep (39.99) beats vLLM's BEST (37.62). That is the property the old number lacked —
1.004x could be reversed by a single unlucky draw, and this cannot.

Against the recorded vLLM 38.79 instead of today's 37.11, Atlas still wins by 1.058x, so
the conclusion does not depend on which vLLM number is used. Both are reported because
vLLM's own C=2 moved 4.3% between two runs of the SAME image digest on the SAME box, which
is a useful reminder that a 1.004x margin is not a result.

Two configuration traps were caught and are worth recording, since both would have produced
a wrong number that looked fine:

- **This file's header block (line ~17) lists `--kv-cache-dtype bf16`**, which is the ROUND 1
  Atlas config. The certified comparison is **fp8 KV on both** (round 4 moved Atlas to fp8
  "matching the reference at last"). A first attempt at bf16 measured 39.55 and was discarded.
- **The ladder was measured on dgx2, not dgx1.** Two runs were completed on dgx1 (39.55 bf16,
  39.95 fp8) before this was noticed, and both were discarded rather than compared across
  boxes — the same error this file already records as a retraction at "★ The comparison
  itself is the likely error".

Provenance: box dgx2 (spark-43fa), Atlas `529fcb04fa` served with the round-11 flags at
`--kv-cache-dtype fp8`, env `ATLAS_PREFILL_CODISPATCH=1 ATLAS_FP8_ROWWISE=1
ATLAS_MTP_DCUT_RATIO=1.0 ATLAS_MTP_K_LADDER=1:3,2:1,4:2,8:2,16:1`; vLLM
`vllm/vllm-openai:latest` digest `sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`
— the IDENTICAL digest the certified reference used — with
`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`, ctx 2048, batch cap
128, util 0.85, fp8 KV, prefix caching on. Harness `harness_w55_conc_ladder.py`, ISL 128 /
OSL 1024, temp 0, seed 42, 3 reps + 1 warmup. Raw series in
`c2_atlas_dgx2_20260818.json` and `c2_vllm_mtp_dgx2_20260818.json`.

### C=8 REPRODUCED (2026-08-18) — 1.013x, and the K ladder there is already optimal

With C=2 hardened, C=8's **1.012x** became the thinnest rung, so it got the same treatment:
merged main `529fcb04fa`, dgx2, vLLM re-run back-to-back the same day.

| engine | rep 1 | rep 2 | rep 3 | mean | spread |
|---|---:|---:|---:|---:|---:|
| Atlas | 123.33 | 124.94 | 121.38 | **123.22** | 2.89% |
| vLLM+MTP | 121.92 | 120.60 | 122.40 | **121.64** | 1.49% |

**Ratio 1.013x**, against the certified 1.012x — reproduced to within 0.1%.

★ **Both engines measured ~2.2% BELOW their certified absolutes** (Atlas 123.22 vs 125.95,
vLLM 121.64 vs 124.48) **while the ratio held.** That is the useful part: the ladder is
reproducible in RATIO across days even when the box's absolute throughput drifts, which is
exactly why every rung is quoted as a same-day A/B rather than against a stored number.

**The margin is real but thin, and it is NOT a tuning oversight.** Unlike C=2, the rep
distributions here OVERLAP (Atlas min 121.38 < vLLM max 122.40), so a single draw can
reverse the ordering. A K-ladder sweep at C=8 confirms the shipped value is the optimum:

| `8:K` | mean tok/s | vs shipped |
|---:|---:|---:|
| **8:2 (shipped)** | **123.22** | — |
| 8:1 | 118.20 | -4.1% |
| 8:3 | 116.96 | -5.1% |
| 8:4 | 116.58 | -5.4% |

Monotonically worse in both directions, so C=8's narrow margin is a property of the rung,
not a missed setting. Widening it needs a kernel-level change, not a knob. Recorded so the
next person does not re-run this sweep.

Raw series: `c8_atlas_dgx2_20260818.json`, `c8_vllm_mtp_dgx2_20260818.json`. Same
provenance as the C=2 block above.

### SAME-DAY FULL LADDER ATTEMPT (2026-08-18) — INCOMPLETE, and its late rungs are SUSPECT

After hardening C=2 and reproducing C=8, the remaining rungs were swept on merged main
`529fcb04fa`, dgx2, both engines back-to-back. **The sweep did not finish: dgx2 stopped
answering ping and ssh from BOTH other boxes while the vLLM leg entered C=128, and needed a
physical powercycle.** Raw JSON was written to `/tmp` and did not survive. The numbers below
are transcribed from the harness SERIES lines and are recorded for provenance, NOT as a
replacement for the certified table.

| C | Atlas | vLLM+MTP | same-day | certified |
|---:|---:|---:|---:|---:|
| 1 | 24.20 | 19.15 | **1.264x** | 1.196x |
| 2 | 41.02 | 37.11 | **1.105x** | 1.004x |
| 4 | 72.99 | 68.49 | **1.066x** | 1.036x |
| 8 | 123.22 | 121.64 | 1.013x | 1.012x |
| 16 | 195.19 | 193.99 | 1.006x | 1.032x |
| 32 | 276.11 | 277.60 | **0.995x** | 1.027x |
| 64 | 382.05 | (2 of 3 reps, ~355) | — | 1.070x |
| 128 | 469.03 | (never ran — box wedged) | — | 1.333x |

**C=32 inverted, and C=16 narrowed. Both are UNCONFIRMED and must not be treated as a
regression yet.** The reason is the wedge itself: C=16 and C=32 were measured on a box that
became unresponsive roughly twenty minutes later, so the memory pressure that eventually took
it down was plausibly already building while those rungs ran. A measurement taken on the
approach to a hard failure is not a measurement of steady state.

What argues it might still be real: the drop is NOT symmetric. Against certified absolutes
Atlas fell 5.1% at C=32 while vLLM fell only 2.1%, and C=1/2/4 got WIDER on the same sweep
rather than uniformly worse. A pure box-slowness story predicts both engines falling together
at every rung, which is what C=8 showed (both ~2.2% down, ratio held to 0.1%) and what these
two rungs did not.

**RESOLVED — it cannot be a code regression, and the proof is a diff, not a GPU hour.**
The certified ladder stack tip is `1575873582`; the sweep ran on merged main `529fcb04fa`.
Diffing the two across `crates/` and `kernels/`:

| area | changed | executable? |
|---|---|---|
| `kernels/**/*.cu,*.cuh,*.h` | **0 files** | — |
| `crates/spark-model/**` | 2 files | **0 non-comment lines** (rustdoc link fixes) |
| `crates/spark-runtime/**` | 0 files | — |
| `crates/spark-server/{cli,tui}` | 3 files | benchmark CLI/TUI wiring only |
| `crates/atlas-plugin/**` | 34 files | benchmark harness (#569, #581) |
| `kernels/gb10/*/BENCH.toml` | 3 files | thresholds, not kernels |

The engine binary is **functionally identical** between the certified stack and merged main —
the only `spark-model` changes are two doc-comment link fixes with zero non-comment lines.
So the merges cannot have moved C=16 or C=32, and the pre-wedge box state is the remaining
explanation, consistent with those two rungs being the ones measured closest to the failure.

**Still worth doing when a healthy box is free:** re-measure C=16 and C=32 alone, in their own
serve, to confirm they return to their certified ratios. That is now a confirmation step
rather than a regression hunt. Until it exists this file's certified table stands on its own
gate records, and no fresh "wins at every rung" claim should be made from the aborted sweep.

★ **Operational hazard, recorded so it is not rediscovered:** vLLM+MTP at C=128 on GB10 can
take the whole machine down even at the "safe" `--gpu-memory-utilization 0.85`. GB10 memory
is unified, and MTP verification widens the working set exactly where the batch is widest.
The warning was already in this file: vLLM's certified C=128 (358.57) is BELOW its own C=64
(361.39) — an engine going backwards at its widest rung is one already struggling there.
Atlas never pays this because its speculation self-disables above 32 concurrent sequences,
which is also why it wins C=128 by 1.333x. Next time: run C=128 in its own serve, drop util
to 0.75-0.80 for that rung, and write raw JSON under `/home/claude` rather than `/tmp`.

### C=16/C=32 CONFIRMED ON A CLEAN BOOT (2026-08-18) — and the wedge hazard corrected

dgx2 was powercycled after the first wedge and the sweep re-run **five minutes from a clean
boot**. Both suspect rungs came back up:

| C | certified | pre-wedge sweep | clean boot | vLLM (clean boot) | ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 23.59 | 24.20 | **25.49** | 19.35 | **1.317x** |
| 4 | 74.21 | 72.99 | **75.24** | 70.13 | **1.073x** |
| 16 | 203.36 | 195.19 | **201.32** | 198.24 | **1.016x** |
| 32 | 291.01 | 276.11 | **282.42** | — | — |
| 64 | 386.63 | 382.05 | **391.73** | — | — |
| 128 | 478.11 | 469.03 | **472.70** | — | — |

Atlas recovered at **every** rung that had looked soft — C=16 195.19 -> 201.32 and C=32
276.11 -> 282.42 — with three of six rungs landing ABOVE their certified values. Combined
with the diff proof (zero executable change in the serve path between the certified stack
`1575873582` and merged main), the C=32 inversion is closed: it was box state, not code.

The mechanism is now clearer. The pre-wedge Atlas leg ran on a box that had been serving
benchmarks for hours; the clean-boot leg ran on a box minutes old. Atlas runs FIRST in this
sweep, so the degradation cannot have come from the vLLM leg — it is accumulated state from
everything that ran before.

**C=16 is confirmed still won on a same-day A/B: 1.016x.** C=32's same-day vLLM number does
not exist, because:

★ **HAZARD CORRECTED — it is NOT specific to C=128.** The first wedge happened entering
C=128, and this file originally blamed that rung. The clean-boot sweep wedged the box AGAIN
during **vLLM C=32** — a rung whose Atlas counterpart had completed minutes earlier in the
same sweep, on a box five minutes old, at the same "safe" util 0.85. A watcher armed to stop
before C=128 never fired. **Treat any vLLM+MTP rung at C>=32 on GB10 as able to take the box
down**, and do not re-run those rungs casually: they have now cost two physical powercycles
for numbers already certified at 1.027x (C=32) and 1.333x (C=128).

Atlas does not exhibit this at any rung, which is worth stating plainly: it completed
C=1..128 twice, including C=128 at 472.70 and 469.03, on the same box and the same util that
wedged under vLLM.

### METHOD NOTE — `--max-num-seqs` is part of the comparison, not a free knob

The certified table pins **batch cap 128 on BOTH engines at every rung** (Atlas
`--max-batch-size 128`, vLLM `--max-num-seqs 128`), independently of the concurrency being
driven. That pin is load-bearing, and it is easy to lose while working around the wedge
hazard above.

Lowering `--max-num-seqs` to match the rung (e.g. `--max-num-seqs 1` for C=1, `32` for C=32)
is a REASONABLE mitigation for the wedge — it caps the working set exactly where the risk
is — but the number it produces **is not comparable to this file's vLLM column**:

- vLLM sizes its KV blocks and its scheduler budget from `max_num_seqs`, so a per-rung cap
  changes block allocation, preemption behaviour and prefix-cache reuse, not just a ceiling.
- The certified vLLM numbers were all taken at 128. A rung measured at a lower cap is a
  different configuration, and comparing it to the 128-cap Atlas column is precisely the
  apples-to-oranges the "APPLES-TO-APPLES REFERENCE" section exists to prevent.

If a per-rung cap is used to survive the hazard, **say so beside the number and re-pin
Atlas's `--max-batch-size` to the same value**, so the pair is at least internally
like-for-like. Do not fold such a number into the certified column.

(The same caution applies to dropping `--gpu-memory-utilization` for a hazardous rung: it is
the right mitigation, but it must be applied to BOTH engines or reported as a separate
configuration.)

### NEGATIVE RESULT (2026-08-19) — `decode_tps` is NOT a tighter gate than `s_per_turn`

atlas#581 made `s_per_turn` the agentic speed bound and recorded `decode_tps`
(tokens / agent-wall) unbounded, with a note that tokens are "the honest denominator" and
that a future change should ratchet onto it. Six measured tiers say **do not**.

| tier | `sum_turns` | `s_per_turn` | `decode_tps` |
|---:|---:|---:|---:|
| 1 | 115 | 7.000 | 34.384 |
| 2 | 127 | 7.118 | 34.229 |
| 3 | 117 | 6.928 | 34.218 |
| 4 | 133 | 7.018 | 33.876 |
| 5 | 114 | 6.452 | 33.747 |
| 6 | 111 | 7.099 | **30.966** |

| metric | min | max | spread |
|---|---:|---:|---:|
| `sum_turns` | 111 | 133 | 19.8% |
| `s_per_turn` | 6.452 | 7.118 | **10.3%** |
| `decode_tps` | 30.966 | 34.384 | **11.0%** |

**The two bounds are equally noisy, and `decode_tps` is marginally worse.**

★ This conclusion REVERSES at five tiers, which is the trap. Through tier 5 `decode_tps`
spanned only 1.9% against `s_per_turn`'s 10.3% — a five-times-tighter result that looked
like a clear mandate to ratchet. Tier 6 came in at 30.966 (8% below the previous minimum)
while its `s_per_turn` of 7.099 sat mid-range, and the advantage vanished. Anyone who stops
at five tiers will conclude the opposite of the truth.

Tier 5 is the other warning: it is the FASTEST tier per turn (6.452) and the SLOWEST per
token of the first five (33.747). The two metrics do not even rank runs the same way, so
"tokens are more physical" is not by itself a reason to prefer one.

**Action: none.** `s_per_turn` stays the bound; `decode_tps`, `sum_turns`,
`sum_agent_wall_s` and `sum_tool_calls` stay recorded and unbounded — they are worth having
for diagnosis, which is what tier 6 just demonstrated. Do not re-derive this from a short
run.

### C=16 K-LADDER SWEEP (2026-08-19) — the shipped value is optimal, and K=3 does not FIT

After the same-day sweep left C=16 as the thinnest confirmed rung (1.016x, down from a
certified 1.032x), its K ladder was swept the way C=8's was. Atlas-only, one box (dgx1),
back-to-back, so only the relative ordering is claimed:

| `16:K` | mean tok/s | vs shipped |
|---:|---:|---:|
| **16:1 (shipped)** | **194.06** | — |
| 16:2 | 186.17 | -4.1% |
| 16:3 | **would not start** | — |

(dgx1 absolutes; the dgx2 figure for this rung is 201.32. Cross-box absolutes are not
comparable — only the within-box ordering above is.)

**16:3 is not slower, it is INFEASIBLE.** It fails at model build, and the failure was
reproduced on a GPU verified idle beforehand (0 compute apps) so it is not residue from a
previous serve — identical numbers both times:

```
No memory left for KV cache: total GPU = 121.7 GB, util 85% -> budget 103.4 GB,
but 60.4 GB already consumed + 43.4 GB inference reserve = 103.8 GB
```

It misses by **0.4 GB**. The verify-pool reserve grows with K, so the K ladder at this rung
is bounded by memory before it is bounded by throughput — a harder constraint than "we
measured it and it was worse", and the reason no amount of tuning opens C=16 up.

Together with the C=8 sweep (8:2 shipped beats 8:1 by 4.1% and 8:3 by 5.1%), **both
mid-ladder rungs sit on their optimum in both directions.** The narrow margins there are a
property of the rungs, not a missed setting, and widening them needs a kernel-level change.
Recorded so neither sweep is run a third time.

### ⚠ C=32 DOES NOT CURRENTLY WIN (2026-08-20) — Atlas is ~4% below its certified value

**This contradicts the certified table's 1.027x at C=32 and the site's published `wins=True`
for that rung. It is reported here before any attempt to explain it away.**

Two independent measurements on a HEALTHY dgx2 (the certification box), at merged main
`635a692ca9`, with the certified serve flags and env — and see the BISECT below, which shows
the same numbers at the certified commit itself:

| config | Atlas | certified Atlas | delta |
|---|---:|---:|---:|
| batch cap 128 (the certified config) | **279.23** (277.48/279.65/280.55, 1.10%) | 291.01 | **-4.0%** |
| batch cap 128, five minutes after a reboot | 282.42 (282.32/283.27/281.69, 0.56%) | 291.01 | -2.9% |

Both land near 279-282, not 291, so this is NOT the box-state effect that explained the
earlier C=16/C=32 scare — that one recovered on a clean boot and this does not.

**A same-day matched pair confirms the ordering has flipped.** Both engines at batch cap
**32** (see the METHOD NOTE below — this is a SEPARATE configuration from the certified
table, which pins 128 on both, and it is used here because vLLM+MTP at C>=32 with cap 128 has
wedged this box three times; the cap is matched on both engines, so the pair is internally
like-for-like):

| engine (cap 32, same box, back-to-back, same hour) | rep 1 | rep 2 | rep 3 | mean |
|---|---:|---:|---:|---:|
| Atlas | 275.98 | 277.78 | 278.17 | **277.31** |
| vLLM+MTP | 283.08 | 287.54 | 283.00 | **284.54** |

**Ratio 0.975x — vLLM wins.**

The two views agree in direction: Atlas has moved down ~4% at this rung. (vLLM's cap-32
284.54 sits just above its cap-128 certified 283.48, but those are different caps and cannot
be compared directly — no claim is made that vLLM is unchanged.)

**What is NOT yet established:** vLLM at cap 128 on current main. That is the measurement the
wedge hazard blocks, so the certified 1.027x has not been directly re-run — only bracketed.
The honest statement is that C=32 fails to reproduce and loses a matched-cap same-day A/B,
not that the certified number was wrong when taken.

**BISECTED — THE CODE IS EXONERATED. Correcting the paragraph that stood here.**

The bisect was three measurements, and it ended the search immediately:

| commit measured today | C=32 | what it is |
|---|---:|---|
| `1575873582` | **277.17** (275.85/277.50/278.17, 0.84%) | **the exact stack tip the 291.01 was taken on** |
| `60370b9532` | 277.86 (277.50/278.35/277.74, 0.31%) | the merge that squashed that stack (#572) |
| `635a692ca9` | 279.23 (277.48/279.65/280.55, 1.10%) | current main |

**All three agree at ~277-279.** The commit that produced 291.01 now produces 277.17 on the
same box with the same flags and env. There is no regression to bisect: every commit in the
range behaves identically, including the certified one.

So the earlier claim here — "whatever changed is on our side" — was WRONG in the way it
mattered, and it is retracted. It read as a code regression. The code is unchanged in
behaviour across the entire range; what does not reproduce is the ENVIRONMENT the 291.01 was
measured in. Between then (2026-08-17) and now, dgx2 was physically relocated and
powercycled three times.

Only five commits since the certified merge touch `crates/` or `kernels/` at all, and none
touch `spark-model` or `kernels/` — four are benchmark harness, one is a Cargo.lock bump —
so the diff never supported a decode regression either. The measurement and the diff agree.

**Two follow-up hypotheses were tested and BOTH refuted.**

*Clocks/thermals:* sampled 117 times during a C=32 load on dgx2 — SM clock median **2483 MHz**
(min 2392, max 2515), and `clocks_throttle_reasons.active` was `0x0` in 116 of 117 samples.
dgx1 under its own load reads 2424 MHz. The GPU is running at full speed and is not
throttled, so this is not clocks, thermals or a power cap.

*Measurement method:* the certified numbers came from a sweep that walks C=1 -> 128 on ONE
serve, whereas the alarm above was raised on C=32 measured cold and alone. Re-running the
warming path (`--concs 1,2,4,8,16,32`, one serve, certified stack tip) gives C=32 **278.93**
(279.32/278.45/279.02, 0.31%) — the same ~279. Isolation was not the explanation either.

**What the full sweep DOES show is a deficit that grows with concurrency:**

| C | in-sweep today (stack tip) | certified | delta |
|---:|---:|---:|---:|
| 2 | 39.65 | 38.95 | **+1.8%** |
| 4 | 72.64 | 74.21 | -2.1% |
| 8 | 123.93 | 125.95 | -1.6% |
| 16 | 198.07 | 203.36 | -2.6% |
| 32 | **278.93** | **291.01** | **-4.2%** |

Same commit, same box, same flags, same harness, one serve. C=2 is FASTER than certified and
the deficit widens monotonically from C=4 up. A uniform slowdown would not do that, and
neither would a code change that the diff already rules out.

**The cause is NOT identified, and an earlier revision of this section overreached by saying
it "points at the memory system... higher C is where this workload becomes bandwidth-bound".
That reasoning is backwards and is withdrawn:** at high C the weight sweep is AMORTISED
across the batch, so low C is the more weight-bandwidth-bound regime — and low C is the part
that still matches (C=2 is +1.8%). A simple loss of memory bandwidth would have hurt C=2
first, and it did not.

★ **CROSS-BOX: THE DEFICIT IS FLEET-WIDE, NOT dgx2.** The same sweep, same commit
(`1575873582`), same flags, run on **dgx1**:

| C | dgx1 today | dgx2 today | certified | dgx1 delta | dgx2 delta |
|---:|---:|---:|---:|---:|---:|
| 1 | **23.59** | — | **23.59** | **0.0%** | — |
| 2 | 39.53 | 39.65 | 38.95 | +1.5% | +1.8% |
| 4 | 71.92 | 72.64 | 74.21 | -3.1% | -2.1% |
| 8 | 120.93 | 123.93 | 125.95 | -4.0% | -1.6% |
| 16 | 186.12 | 198.07 | 203.36 | -8.5% | -2.6% |
| 32 | 268.01 | 278.93 | 291.01 | **-7.9%** | **-4.2%** |

**C=1 reproduces its certified value EXACTLY (23.59 vs 23.59)**, C=2 lands slightly ABOVE
certified on both boxes, and from C=4 up both fall progressively behind — dgx1 MORE than
dgx2. So the earlier framing (dgx2 degraded by its relocation and powercycles) is wrong: the
certified high-C numbers do not currently reproduce anywhere in the fleet, at the commit they
were taken on.

dgx1's spreads were 3.8-4.7% against dgx2's 0.3-0.6% — it had been serving another session
minutes earlier — so that pair was re-run on a genuinely IDLE dgx1, in an isolated worktree,
with load logged either side:

| C | dgx1 CONTENDED | dgx1 IDLE | dgx2 | certified | idle delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 23.59 | 23.85 | — | 23.59 | **+1.1%** |
| 2 | 39.53 | 39.54 | 39.65 | 38.95 | **+1.5%** |
| 4 | 71.92 | 71.37 | 72.64 | 74.21 | -3.8% |
| 8 | 120.93 | 121.59 | 123.93 | 125.95 | -3.5% |
| 16 | 186.12 | 193.93 | 198.07 | 203.36 | -4.6% |
| 32 | 268.01 | **275.12** | 278.93 | 291.01 | **-5.5%** |

Idling the box recovered part of it (C=16 186.12 -> 193.93, C=32 268.01 -> 275.12) and the
spreads tightened to 0.35-1.8%, so host contention is real and worth controlling for — but it
is NOT the cause. On a clean, idle box at the certified commit, C>=4 still lands 3.5-5.5%
below the certified absolutes while C=1 and C=2 come in slightly ABOVE them.

That is the finding, and it now rests on clean data from two boxes.

**★ UNTESTED LEAD — the two boxes showing the deficit share the OLDER driver.**

| box | driver | libcuda dated | shows the deficit? |
|---|---|---|---|
| dgx1 | 580.126.09 | 2026-01-12 | **yes** (-5.5% at C=32, idle) |
| dgx2 | 580.126.09 | 2026-01-12 | **yes** (-4.2% at C=32) |
| dgx3 | **580.159.03** | **2026-05-05** | **NOT MEASURED** |

dgx1 and dgx2 are on an identical, older driver and both show the same shape; dgx3 carries a
newer one and has not been swept. That does NOT make the driver the cause — dgx2's driver has
not changed since January, so it cannot by itself explain a shift between 2026-08-17 and
2026-08-20 — but it is the one cheap discriminator left, and it is the concrete answer to
"do the boxes need a system update?": run this sweep on dgx3.

If dgx3 reproduces the certified absolutes, the fleet has a driver-shaped problem and the
fix is an update on dgx1/dgx2. If dgx3 shows the same 4-5% high-C deficit, the driver is
excluded too and the cause is environmental in a way nothing measured here reaches.

The sweep is ~30 minutes, Atlas-only, and carries no wedge risk. It needs dgx3 idle; it has
been serving another session throughout this investigation.

★ **This does NOT invalidate the same-day A/B results.** Each of those compared Atlas against
vLLM on the SAME day and box, so a fleet-wide shift affecting both engines cancels in the
ratio. What it does mean is that the certified ABSOLUTES are stale, and a rung quoted from
them cannot be checked against a fresh Atlas number without a fresh vLLM number beside it.

Ruled out so far: code (three commits measured, all ~279, and no `spark-model` or `kernels/`
change in the diff range), clocks and thermals (2483 MHz median, unthrottled in 116/117
samples), measurement method (in-sweep reproduces isolated), memory capacity (dgx2 idles with
115 GB of 121 GB free), CPU governor (all three boxes `performance`, 20 cores), and
single-box degradation. What remains implicates something that scales with
BATCH rather than with per-step weight streaming — KV traffic, scheduler behaviour, or the
SSM/verify pools — but nothing here isolates which, and no measurement in this file
distinguishes them. `clocks.mem` reads `[N/A]` on GB10.

Context, not conclusion: dgx2 was physically relocated and powercycled three times between
the certification and these runs.

**What this costs us.** The certified C=32 ratio of 1.027x rests on an Atlas number that
cannot be reproduced, so that rung's margin is not currently defensible. `sw_power_cap_us`,
`sw_thermal_us`, `hw_thermal_us` and `hw_power_brake_us` are all ZERO in the hardware_state
of tonight's dgx2 records, so simple throttling is ruled out; the cause is not yet identified.

**What is still true.** Seven rungs were re-measured same-day on merged main and every one
held or widened (C=1 1.317x, C=2 1.105x, C=4 1.073x, C=8 1.013x, C=16 1.016x). The C=32
finding does not touch them — each is its own same-day A/B.

Raw series: `c32_atlas_cap128_dgx2_20260820.json`, `c32_atlas_cap32_dgx2_20260820.json`,
`c32_vllm_mtp_cap32_dgx2_20260820.json`.

### ✅ C=32 DOES WIN — the cap-32 comparison was the flawed one (2026-08-20)

**Correcting the alarm raised earlier today.** vLLM measured at the CERTIFIED configuration
(batch cap 128, util 0.85, fp8 KV, MTP K=4) at C=32, same box, same day as the Atlas number:

| engine (cap 128, C=32, dgx2, same day) | rep 1 | rep 2 | rep 3 | mean | spread |
|---|---:|---:|---:|---:|---:|
| **Atlas** | 279.32 | 278.45 | 279.02 | **278.93** | 0.31% |
| vLLM+MTP | 275.10 | 278.02 | 278.24 | **277.12** | 1.13% |

**Ratio 1.007x — Atlas wins, and the distributions do not overlap** (Atlas's worst rep 278.45
beats vLLM's best 278.24).

**vLLM fell too.** Its certified 283.48 is now 277.12, down 2.2% — so the fleet-wide shift
documented above is not Atlas-specific. It costs Atlas ~4% and vLLM ~2.2% at this rung, which
narrows the margin from the certified 1.027x to 1.007x but does not reverse it.

★ **The earlier "C=32 does not currently win" rested on a cap-32 pair, and cap 32 is not a
neutral mitigation.** Measured both ways on the same box today:

| engine | cap 32 | cap 128 (certified) | effect of the cap |
|---|---:|---:|---|
| Atlas | 277.31 | 278.93 | roughly flat (+0.6%) |
| vLLM+MTP | 284.54 | 277.12 | **+2.7% for vLLM at cap 32** |

vLLM gains materially from the smaller cap; Atlas does not. So the matched-cap-32 pair —
adopted in good faith to avoid the wedge hazard, and internally like-for-like — systematically
favoured vLLM and produced a 0.975x that inverted the true ordering. This is exactly what the
METHOD NOTE above warns about, and it caught out the person who wrote it.

**Standing after this:** six rungs verified by same-day A/B on merged main — C=1 1.317x,
C=2 1.105x, C=4 1.073x, C=8 1.013x, C=16 1.016x, **C=32 1.007x**.

**C=64 and C=128 are deliberately NOT re-measured, and the reason is a bound rather than an
excuse.** What the fleet-wide shift can do to a ratio is limited by its DIFFERENTIAL, and at
C=32 — the rung where both engines were measured today at the certified config — that
differential is:

| | certified | today | change |
|---|---:|---:|---:|
| Atlas | 291.01 | 278.93 | -4.2% |
| vLLM+MTP | 283.48 | 277.12 | -2.2% |
| | | **net against Atlas** | **~1.8%** |

C=64's certified margin is **7.0%** and C=128's is **33.3%**. A 1.8% differential cannot
reverse either, and the differential would have to nearly quadruple to threaten even C=64.
Both are therefore safe by inference, and re-measuring them means running vLLM+MTP at cap 128
at C>=64 — the workload that has cost this fleet three physical powercycles.

Note which rungs were prioritised: every FRAGILE rung (C=8 at 1.013x, C=16 at 1.016x, C=32 at
1.007x) has been verified same-day, and the two left unverified are the two widest margins on
the ladder. That is the correct order to spend a hazardous measurement budget in.

Raw series: `c32_vllm_mtp_cap128_dgx2_20260820.json`.

### DRIVER EXCLUDED — three boxes, two drivers, one shape (2026-08-20)

The open lead from the staleness investigation was that both deficit-showing boxes ran the
older driver. dgx3 carries a newer one, so the same sweep was run there at the same commit
(`1575873582`), Atlas-only, isolated worktree:

| C | dgx1 (idle) | dgx2 | **dgx3 (new driver)** | certified | dgx3 delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 23.85 | — | **24.39** | 23.59 | **+3.4%** |
| 2 | 39.54 | 39.65 | **40.98** | 38.95 | **+5.2%** |
| 4 | 71.37 | 72.64 | 72.53 | 74.21 | -2.3% |
| 8 | 121.59 | 123.93 | 123.33 | 125.95 | -2.1% |
| 16 | 193.93 | 198.07 | 196.35 | 203.36 | -3.4% |
| 32 | 275.12 | 278.93 | **278.29** | 291.01 | **-4.4%** |

drivers: dgx1/dgx2 `580.126.09` (libcuda 2026-01-12) · dgx3 `580.159.03` (2026-05-05)

**dgx3 shows the same shape on a different driver.** It is the FASTEST box at C=1 and C=2
(+3.4%, +5.2% over certified) and still lands 4.4% below at C=32 — the identical
low-C-fine / high-C-short signature dgx1 and dgx2 produce. Three boxes, two driver versions,
one shape.

**So the answer to "do the boxes need a system update?" is: an update will not recover this.**
The driver is excluded as the cause, and the newer one does not restore the certified high-C
absolutes. That closes the last cheap hypothesis.

★ **And it does not threaten the result**, because vLLM moved with it — see the C=32 block
above, where vLLM's certified 283.48 measures 277.12 today against Atlas's 278.93 on the same
box and day. The absolutes across the fleet have shifted; the same-day ratios have not.

Raw series: `sweep_atlas_stacktip_dgx3_20260820.json`.

### C=64 and C=128 WIN under the most conservative assumption available (2026-08-20)

Rather than run vLLM at cap 128 at C>=64 — the workload that has cost this fleet three
powercycles — Atlas's own absolutes were measured there today (dgx1, idle, certified commit
`1575873582`, Atlas-only, zero wedge risk) and compared against the CERTIFIED vLLM column.

That comparison is deliberately unfair to Atlas: vLLM has demonstrably FALLEN with the fleet
(its certified C=32 283.48 measures 277.12 today, -2.2%), so crediting it with its old number
understates Atlas's true margin.

| C | Atlas today | certified vLLM | ratio vs certified vLLM | certified ratio |
|---:|---:|---:|---:|---:|
| 64 | **375.68** (0.31%) | 361.39 | **1.040x** | 1.070x |
| 128 | **460.37** (0.38%) | 358.57 | **1.284x** | 1.333x |

**Both rungs win even if vLLM is credited with its full pre-shift number.** Grant vLLM the
same -2.2% it actually took at C=32 and the ratios become ~1.063x and ~1.313x, close to
certified. Either way the ordering is not in question, which is what the earlier
differential-based bound argued and this now demonstrates with measured Atlas numbers at the
rungs themselves.

★ **Correcting the shape.** Earlier sections describe the deficit as "growing with
concurrency". With C=64 and C=128 measured, that is wrong — it is a STEP at C>=4 and then
roughly flat:

| C | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| delta vs certified | **+1.1%** | **+1.5%** | -3.8% | -3.5% | -4.6% | -4.2% | **-2.8%** | **-3.7%** |

C=1 and C=2 sit ABOVE certified; everything from C=4 up sits 2.8-4.6% below, with no trend
across that range. Whatever the fleet-wide cause is, it engages once batching begins and does
not worsen with width — which is a materially different fingerprint from the one the earlier
text implies, and a better clue for whoever chases it next.

Raw series: `atlas_c64_c128_dgx1_20260820.json`.

### Round 11 complete — the full ladder, independently reproduced

| C | round 11 | round 10 | vLLM+MTP | ratio |
|---:|---:|---:|---:|---:|
| 1 | 23.59 | 23.50 | 19.72 | **1.196x** |
| 2 | **41.02** (2026-08-18, dgx2) | 38.95 | 37.11 same-day / 38.79 r8 | **1.105x** |
| 4 | **74.21** | 71.95 | 71.61 | **1.036x** |
| 8 | **125.95** (repro 123.22 vs 121.64 same-day) | 125.47 | 124.48 | **1.012x** (repro 1.013x) |
| 16 | 203.36 | 202.93 | 197.03 | **1.032x** |
| 32 | 291.01 | 291.17 | 283.48 | **1.027x** |
| 64 | 386.63 | 387.10 | 361.39 | **1.070x** |
| 128 | 478.11 | 478.07 | 358.57 | **1.333x** |

Two independent rounds on the final configuration agree at every rung — C=32 and C=128 to
within 0.06%, C=64 to 0.12%. **Atlas beats vLLM+MTP at every concurrency from 1 to 128.**

## QUALITY GATES on the final stack (`bf4d7a1267`, post-rewrite)

| gate | verdict | detail |
|---|---|---|
| **ssm-state-poisoning-gate** | **PASS** | **12 of 12 replays byte-identical to the reference** |
| **decode-floor** | **PASS** | |
| agentic-webserver | running | |
| bfcl-subset (with draw metadata) | queued | |

The poisoning gate is the instrument that matters most here: `--ssm-h-dtype f16-pool` halves
the precision of the GDN recurrent state, and that gate exists specifically to catch state
corruption across prefix-cache restores. **Byte-identical replays** means the reduced-precision
pool did not perturb the recurrence at all on this probe — the strongest available evidence
that the capacity win cost nothing in correctness.

### agentic-webserver: correctness perfect, wall budget over — repeat per protocol

First tier on the final stack: **webserver_ok 10/10, followed_directions 10/10**, but
`Σwall 1084s > 1000s`. The repo's own rule for this exact case (benchmark-pr skill): the
Σwall bound is a blowup detector carrying +/-200-300s of noise, so *"if Σ lands 1000-1300
with 10/10+10/10 correctness, run ONE repeat tier before declaring wall failure; two
consecutive over-1000 tiers = FAIL (report both)."* A repeat tier is running.

Worth noting what this gate does and does not exercise for tonight's changes: it serves the
**35B** flagship at `--max-batch-size 2` with its own recipe, so `--ssm-h-dtype f16-pool` is
not enabled and the drafter/QKVZ small-M paths (M>8 batched verify, drafter M=2..8) are
largely not reached. It is therefore a regression check on the shared code, not a test of
the ladder configuration — which is why BFCL on the dense 27B is the more informative gate
for the numerics-changing fixes, and it runs next.

### ⚠ AGENTIC WALL REGRESSION — a real cost of the stack, found by the gates

| tier | webserver_ok | followed_directions | Σwall |
|---|---|---|---|
| 1 | 10/10 | 10/10 | 1084 s |
| 2 | 10/10 | 10/10 | 1068 s |

Two consecutive over-1000s tiers is a **FAIL** by the protocol. And the wall is not merely
over budget: this gate's recorded band is 600-800 s, and tonight's own pre-stack run was
**773 s** — so the stack costs roughly **+38% wall on the 35B agentic path** while leaving
correctness perfect.

Prime suspect, on mechanism rather than guesswork: the gate serves at `--max-batch-size 2`,
where the drafter runs M=2 — so PR #562's small-M tier IS active, and it is the one change
in the stack that is **not bit-exact** (it replaces a pipelined tile GEMM, altering
proposal numerics and therefore acceptance, and acceptance drives wall). The other fixes are
either bit-exact (#561 GEMV tiers, #547 rollback) or structurally inactive at that width
(#551 QKVZ needs M>8, #559 canonical key gates at n>=8, #548 f16-pool is not in the recipe).

A/B with the kill switches is running (`ATLAS_NO_DRAFTER_SMALL_M_TIER=1`, then
`ATLAS_NO_GEMV_EXACT_M_TIERS=1`). If the drafter tier is confirmed, the fix is the same
shape as the canonical key's: **width-gate it** so the concurrency rungs keep it and the
batch-2 agentic path does not. BFCL was deliberately killed rather than run on a
configuration that is about to change.

### Agentic wall regression — attribution A/B

| leg | Σwall | correctness |
|---|---:|---|
| everything on (tier 1) | 1084 s | 10/10 + 10/10 |
| everything on (tier 2) | 1068 s | 10/10 + 10/10 |
| `ATLAS_NO_GEMV_EXACT_M_TIERS=1` | **1020 s** | 10/10 + 10/10 |
| `ATLAS_NO_DRAFTER_SMALL_M_TIER=1` | running | |
| historical band / tonight's pre-stack run | 600-800 s / **773 s** | |

The w4a16 exact-M GEMV tiers account for only ~5% of the regression — real but not the
cause; 1020 s is still 30% above the band. The drafter small-M tier remains the suspect on
mechanism (it is the only non-bit-exact change, and the gate's `--max-batch-size 2` is
exactly the width where it activates), and its leg is running now.

Two operational notes recorded so they are not rediscovered: killing a `spark serve` can
leave the benchmark DRIVER process holding its allocation (87 GB here), which starves the
next leg at preflight with "box is not free enough"; and `nvidia-smi --query-gpu=memory.used`
returns `[N/A]` on GB10, so idle-guards must read `free` instead.

#### A/B complete — NEITHER suspect explains the wall, and the baseline is suspect

| leg | Σwall | correctness |
|---|---:|---|
| everything on | 1084 / 1068 s | 10/10 + 10/10 |
| `ATLAS_NO_GEMV_EXACT_M_TIERS=1` | 1020 s | 10/10 + 10/10 |
| `ATLAS_NO_DRAFTER_SMALL_M_TIER=1` | **1078 s** | 10/10 + 10/10 |

The drafter tier — the mechanism-based prime suspect — accounts for **~0%**. The GEMV tiers
account for ~5%. So the stack does not contain a single change worth +300 s here.

★ **The comparison itself is the likely error.** The 773 s figure was measured on **dgx1**;
every leg above ran on **dgx2**. The repo's own rule (benchmark-pr skill) is explicit:
*"Walls are box-specific — record which box; compare like-for-like only."* Comparing a dgx2
wall to a dgx1 baseline is exactly the mistake that rule exists to prevent, and it is the
same class of error that produced a phantom -0.8% TTFT "win" once before.

A same-box control is running: pre-stack `main` built on dgx2, same gate, same recipe. If it
lands near 1000-1100 s, there is no regression — only a box difference — and the earlier
"+38% regression" entry above is retracted rather than explained.

## ★ RETRACTION — the "agentic wall regression" was a cross-box comparison, not a regression

Same-box control, pre-stack `main` (`77ec74b8`) built and run on **dgx2**, same gate, same
recipe:

| configuration (ALL on dgx2) | Σwall | correctness |
|---|---:|---|
| full ladder stack | 1084 s / 1068 s | 10/10 + 10/10 |
| `ATLAS_NO_GEMV_EXACT_M_TIERS=1` | 1020 s | 10/10 + 10/10 |
| `ATLAS_NO_DRAFTER_SMALL_M_TIER=1` | 1078 s | 10/10 + 10/10 |
| **pre-stack `main` (control)** | **1079 s** | **10/10 + 10/10** |

**Unmodified main runs this gate at 1079 s on dgx2 — the same wall as the full stack.** The
stack costs nothing on the agentic path. The 600-800 s band and the 773 s figure I compared
against are **dgx1** numbers, and the repo's own rule says walls are box-specific and must
be compared like-for-like. I violated that rule and reported a "+38% regression" that does
not exist; it is retracted here rather than quietly dropped.

Two consequences worth recording for everyone, not just this campaign:
1. **The gate's `wall_budget_s: 1000` is a dgx1 calibration.** On dgx2 this gate fails its
   wall bound on unmodified main, with perfect correctness. Any Σwall verdict from dgx2 is
   currently meaningless; either the budget needs a per-box value or the gate needs to be
   pinned to dgx1.
2. The mechanism-first instinct was still right to test: both suspects were falsified
   cheaply (~0% and ~5%), and it was their *failure* to explain the gap that exposed the
   methodological error. A suspect that explains nothing is a signal about the measurement,
   not a reason to keep bisecting code.

## ★ AGENTIC GATE PASSES — proven on dgx1, not inferred from dgx2

The dgx2 control showed the stack added nothing *on dgx2*, but it could not show that the
stack still reaches the 600-800 s band on **dgx1**, the box where that band was established.
That test has now been run directly:

| box | build | Σwall | correctness | verdict |
|---|---|---:|---|---|
| **dgx1** | **full ladder stack** | **692 s** | **10/10 + 10/10** | **PASS** (budget 1000 s) |
| dgx1 | pre-stack reference (historical) | 773 s | 10/10 + 10/10 | pass |
| dgx2 | full ladder stack | 1084 / 1068 s | 10/10 + 10/10 | over budget |
| dgx2 | unmodified `main` (control) | 1079 s | 10/10 + 10/10 | over budget |

**The stack is 10% FASTER than the pre-stack reference on dgx1** (692 s vs 773 s), inside
the historical band, and passes the gate outright. dgx2 runs the same gate ~55% slower on
*any* build, including unmodified main — so the earlier "regression" was entirely the box,
and this is now demonstrated rather than argued.

Standing conclusion: **the concurrency campaign costs nothing on the agentic path and
slightly improves it.** The `wall_budget_s: 1000` finding still stands and matters: it is a
dgx1 calibration that unmodified main cannot meet on dgx2, so Σwall verdicts from dgx2 are
meaningless until the budget is per-box or the gate is pinned to dgx1.

### agentic gate — reproduced on dgx1, and a correction to the thermal story

| box | build | Σwall | correctness |
|---|---|---:|---|
| **dgx1 rep 1** | ladder stack | **692 s** | 10/10 + 10/10 |
| **dgx1 rep 2** | ladder stack | **662 s** | 10/10 + 10/10 |
| dgx1 | pre-stack (historical) | 773 s | 10/10 + 10/10 |
| dgx2 | ladder stack | 1084 / 1068 s | 10/10 + 10/10 |
| dgx2 | unmodified `main` | 1079 s | 10/10 + 10/10 |

The stack passes the gate on dgx1 twice, **10-14% faster than the pre-stack reference**.

★ **Correction to the earlier thermal explanation.** The first comparison sampled dgx1 while
IDLE against dgx2 under LOAD, which is the same class of error as the cross-box comparison
it was trying to explain. Measured with both boxes under load, they are nearly identical:

| under load | dgx1 | dgx2 |
|---|---:|---:|
| SM clock | 2463-2470 MHz | 2457-2496 MHz |
| GPU die temp | 69-74 C | 76-78 C |
| hottest chassis zones | 87 / 87 / 81 / 78 C | 89 / 88 / 82 / 74 C |
| CPU / storage | Cortex-X925 x20 @ 3.9 GHz, NVMe | identical |

**Clocks and temperatures do not explain the 692 s vs 1079 s gap.** The remaining hypothesis
is that this gate's wall is dominated by NON-GPU work: each of its 10 trajectories spawns a
sandbox where the model writes a Rust webserver and runs `cargo build` and tests, which is
minutes of CPU and disk per trajectory. Cargo/target and page-cache warmth differ sharply
between the boxes (dgx1 has been the primary build host all night). That is testable and is
not yet tested — recorded as a hypothesis, not a conclusion.

The throttle-counter difference is real (dgx2 accrues SW thermal slowdown ~96x faster) but
is confounded by duty cycle: dgx2 ran benchmarks continuously for 16 h while dgx1 idled
between runs.

## ★ CERTIFICATION COMPLETE — all four gates PASS

| gate | verdict | detail |
|---|---|---|
| ssm-state-poisoning-gate | **PASS** | 12 of 12 replays byte-identical to the reference |
| decode-floor | **PASS** | |
| agentic-webserver (dgx1) | **PASS ×2** | Σwall 692 s / 662 s, 10/10 + 10/10 — 10-14% faster than the 773 s pre-stack reference |
| bfcl-subset | **PASS** | **overall 84.22** (bar 83.42) · **normalized 84.12** (bar 83.32) · **n=995** |

BFCL is the gate that mattered for the numerics-changing fixes (`f16-pool` halves GDN
recurrent-state precision; the drafter small-M tier changes proposal numerics and therefore
acceptance). It clears both committed bars on the dense 27B — so the concurrency win costs
nothing in tool-calling accuracy, and the poisoning gate's byte-identical replays say it
costs nothing in state fidelity either.

**The result is now a certified speed claim, not just a measured one.**

### Post-move re-baseline — thermal is definitively excluded

Both boxes were physically moved and now idle identically (38 C GPU / 40 C chassis, versus
52/65 on dgx1 and 78/89 on dgx2 before). Re-running the same gate with the same stack on
dgx2:

| dgx2 | idle GPU/chassis | loaded GPU | loaded SM clock | Σwall |
|---|---:|---:|---:|---:|
| before the move | 78 / 89 C | 78 C | 2457-2496 MHz | 1084 / 1068 s |
| **after the move** | **38 / 40 C** | **75 C** | **2483 MHz** | **1019 s** |
| dgx1 (reference) | 38 / 40 C | 69-74 C | 2463-2470 MHz | **662 / 692 s** |

A 40 C drop in idle temperature bought **~5%** of wall (1076 -> 1019 s) and left the loaded
temperature and clock essentially unchanged. dgx2 remains ~50% slower than dgx1 on this gate
with identical code, identical clocks and now identical thermals. **Thermal throttling is
excluded.**

That leaves the untested hypothesis standing and now much more likely: this gate's wall is
dominated by the sandboxed `cargo build` + test execution inside each of its 10
trajectories, not by inference. dgx1 has been the primary build host for weeks (warm cargo
registry, target dir, and page cache); dgx2 has not. The next cheap test is to compare
sandbox build time directly on the two boxes rather than inferring it from the gate.

Correction count for this investigation: three. "Code regression" (was cross-box),
"thermal" (was idle-vs-loaded), and now "thermal at all" (cooling changed the temperature
but not the wall). Each is recorded rather than dropped.

### The dgx2 wall gap: six hypotheses excluded by measurement

| hypothesis | test | verdict |
|---|---|---|
| code regression | unmodified `main` on dgx2 | ❌ 1079 s — same as the stack |
| thermal throttling | boxes physically moved, dgx2 idle 78→38 C | ❌ wall moved only 1076 → 1019 s (~5%) |
| clock throttling | SM clock under load, both boxes | ❌ 2483 vs 2463-2470 MHz — identical |
| sandbox `cargo build` | identical cold-registry build both boxes | ❌ **dgx2 is FASTER** (5.87 s vs 7.91 s) |
| software stack | kernel / VBIOS / driver | ❌ byte-identical (6.17.0-1008, 9A.0B.1E, 580.126.09) |
| inference speed | same model + flags, C=1 x3 reps | ❌ dgx1 24.77 vs dgx2 23.59 tok/s — only **5%** |

Decode is 5% faster on dgx1 and cargo is 26% faster on dgx2, yet the agentic gate wall
differs by ~50% (662-692 s vs 1019-1084 s). **No component measured so far accounts for it.**

What that leaves, and what to test next: the gate's wall is neither pure inference nor pure
build, so the remaining candidates are the parts nobody has instrumented — per-turn agent
overhead (tool-call round trips, sandbox process spawn, filesystem syncs) and trajectory
SHAPE (if the model on one box takes more turns or emits more tokens per trajectory, the
wall grows without any component being slower). The per-run `wall_time_s` distribution and
turn counts are recorded in the gate's own run JSONs on both boxes and have not yet been
compared — that is the cheap next step, and it should come before any further hardware
theory.

Fleet note discovered while testing: dgx1 and dgx2 are byte-identical on kernel/VBIOS/driver
but BOTH are behind dgx3 (6.17.0-1026, VBIOS 9A.0B.25), and both have NVIDIA driver upgrades
pending — dgx1 offered 580.173.02, dgx2 offered 580.159.03, i.e. their apt sources differ
too. Recommend updating all three to the same stack AFTER the merges land, then
re-baselining, since every number certified today was measured on 580.126.09.

## ★ RESOLVED — the agentic wall gap is trajectory shape, not hardware

| box | wall | turns | **s/turn** |
|---|---:|---:|---:|
| dgx1 | 813 s | 115 | **7.07** |
| dgx2 | 1019 s | 166 | **6.14** |

**dgx2 is 13% FASTER per turn. It simply takes 44% more turns to complete the same ten
tasks.** The wall difference is trajectory length, not machine speed — which is why every
hardware hypothesis failed: there was never a slow component to find.

Supporting evidence: dgx1's own three tiers on identical hardware and code measured
**662 / 692 / 813 s — a 21% spread**. Turn count is nondeterministic run to run (MTP
speculative decoding is not bitwise reproducible across differing batch composition and
timing, even at temperature 0), and wall is roughly linear in turns.

### Consequence for the gate

`agentic-webserver`'s `Σwall <= 1000 s` bound is **not a reliable performance measure**. It
is dominated by a quantity that varies 21% run-to-run on one box for reasons unrelated to
engine speed: a build can fail it by drawing a longer trajectory and pass it by drawing a
shorter one. Recommended follow-ups, in order:

1. Gate on **seconds per turn** (or per emitted token) rather than total wall — that is the
   quantity that actually measures the engine, and by it dgx2 is the faster box.
2. If total wall must stay, widen the bound to cover the measured spread, or require two
   consecutive over-budget tiers (the protocol already says this; it was applied tonight and
   was right to be).
3. The per-box `wall_budget_s` idea is now moot for the right reason: the variance is not
   between boxes, it is between runs.

Investigation record: seven hypotheses, six excluded by measurement (code, thermal, clock,
sandbox build, software stack, inference speed), one confirmed (trajectory shape). Three of
my own conclusions were retracted along the way — "code regression", "thermal", and
"thermal at all" — each corrected in place rather than dropped.
