# W4A4 prefill on GB10 — measured DEAD on both speed and accuracy

**Date:** 2026-07-25 · **Box:** dgx1 (GB10) · **Model:** `centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf`

The last open lever from the decode-fold handoff. The thesis was that decode-W4A4 is dead only
because decode is weight-roofline-bound at M<=8, while **prefill is the opposite regime** —
compute-bound, M = seqlen >> 8, tiles full — so native sm_121a FP4 MMA should finally pay there.

**It does not pay, and it is not close.**

## No kernel had to be built

Both W4A4 prefill paths already exist behind opt-in env flags:

| path | flag | state before this test |
|---|---|---|
| dense-FFN prefill via `w4a4_gemm` | `ATLAS_FP4_PREFILL` | implemented, **never A/B'd** |
| QKV prefill via `w4a4_gemm_mfast` | `ATLAS_ATTN_W4A4` | already tried and **rejected** — hallucinated-multiple-choice signature on long prompts, and only ~2 ms |

So the "untested W4A4 prefill" in the handoff was one environment variable, not a build.

## The A/B

Same box, sequential legs (GB10 unified memory cannot host two serves at util 0.70), frozen
c2final env, nd=3, greedy temp 0 / seed 42. Only `ATLAS_FP4_PREFILL` differs. Each prompt is
issued cold behind a unique preamble so the prefix cache cannot hide the prefill cost.
`scripts/mlperf-edge/run_prefill_w4a4_ab.sh` + `prefill_w4a4_ab.py` + `prefill_w4a4_compare.py`.

The flag was confirmed live rather than assumed — the `[atlas] ATLAS_FP4_PREFILL=1: dense-FFN
prefill via w4a4_gemm (native FP4 MMA sm_121a, W4A4)` banner fires on the first prefill (not at
startup, which is why an early check reads zero).

### Speed — no gain at any prompt size

| prompt tokens | control | W4A4 | speedup |
|---|---|---|---|
| 1271 | 3525.5 ms | 3544.1 ms | **0.995x** |
| 4907 | 8818.4 ms | 8723.8 ms | 1.011x |
| 9755 | 16179.8 ms | 16107.3 ms | 1.005x |

All three are inside run-to-run noise. The marginal cost per 1k prompt tokens falls with size on
the control leg (2774 -> 1797 -> 1659 ms), confirming the probe really is in the compute-bound
regime the thesis wanted — FP4 activations simply buy nothing there.

**Why the thesis was wrong:** after the shared-memory bank-conflict fix (#356), the prefill FFN
GEMM is limited by smem/bandwidth, not by FP4 tensor-core throughput. Feeding it 4-bit
activations does not relieve the actual limiter.

### Accuracy — destroyed

Unlike an output-neutral change, W4A4 is lossy by construction (cos ~0.99), so a differing output
hash is expected and is NOT the gate. The gates are token match and top-20 logprob KL:

| metric | result | gate |
|---|---|---|
| token match | **70.6%** | >= 99% |
| mean KL | **7.64 nats** | < 1e-3 |
| p99 KL | 29.996 | — |
| median first divergence | position 29 of ~48 | — |

Worst individual prompt: **6.4% token match, diverging at position 3**. Several prompts hold at
100% match while others collapse entirely — the failure is input-dependent, not a uniform drift,
which is the harder kind to gate against.

## Verdict

**DO NOT FOLD, and do not re-open.** Zero speed upside and a catastrophic accuracy cost. Both
W4A4 prefill paths are now measured-dead: QKV on accuracy (pre-existing), FFN on both axes here.
Together with the decode result (activations are ~0.5% of traffic at M<=8), **W4A4 activation
quantization is closed on GB10 in every regime.**

The 4-bit *weights* remain fully exploited — that was never in question.
