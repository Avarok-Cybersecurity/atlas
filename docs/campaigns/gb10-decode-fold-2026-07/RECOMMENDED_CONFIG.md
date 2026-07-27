# Serve config — VERDICT: NO decode fold. bf16-KV main is the optimum.

The decode-gap campaign found **NO foldable decode win** — every lever measured-dead, INCLUDING fp8-KV.

## fp8-KV: DO-NOT-FOLD (corrected)
A synthetic decode-heavy microbench showed fp8-KV +4.8% TPOT, but it **did NOT survive the full e2e**
(prefill-heavy agentic turns, decode is a small slice of a ~26k-ctx turn). Proper same-binary A/B:

| config (fresh main 011bee65) | wall | TTFT p50 | TPOT p50 | TPS | IoU |
|---|---|---|---|---|---|
| **bf16-KV (OPTIMUM, keep)** | 4551.9s | 1264ms | **38.18ms** | **17.56** | 0.6285 |
| fp8-KV | 4534.9s | 1271ms | 39.08ms | 17.08 | 0.6223 |
| delta (fp8 vs bf16) | −0.4% (noise) | +0.6% | **+2.4% SLOWER** | **−2.7% WORSE** | −0.006 |

fp8-KV is neutral-on-wall, worse on TPOT/TPS, and drops IoU → **keep bf16-KV.** (My earlier "−9% wall"
was a comparison against a STALE 07-21 baseline/binary, not the same-binary control — corrected.)

## The e2e result stands (either config beats confirmed vLLM)
Atlas main (bf16-KV) e2e: wall 4551.9s / TPOT 38.18 / TPS 17.56 / BFCL ~87 / IoU 0.6285.
Confirmed vLLM: wall 5361s / tps 14.6 / BFCL 86.43 / IoU 0.6269.
→ **Atlas wins wall −15% / TPS +20% / BFCL / IoU-tie.** The RAW per-token TPOT (~38ms) is roofline-bound
(all kernel/weight levers measured-dead); not closable by a same-checkpoint fold. See FINDINGS.md.
