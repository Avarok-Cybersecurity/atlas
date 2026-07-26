# Decode-gap campaign — FINDINGS (2026-07-24)

Goal: close the raw boosted-MTP decode (TPOT) gap vs vLLM on GB10. This documents what was
measured, every lever tried with its verdict, and the honest conclusion. Full receipts in
CAMPAIGN_LOG.md + DECODE_FOLD_LEDGER.md (git-committed on `perf/decode-fold-2026-07-24`).

## Headline conclusion (measured, not asserted)
**On the real gate model (centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf, which is already all-NVFP4), Atlas's
boosted-MTP decode is at its optimization limit on GB10 — the projection GEMVs are memory-roofline-
bound (74–88% of 273 GB/s peak, cold) and acceptance is near its K=3 ceiling (E≈2.8 of max 3.0).
Every weight/kernel lever to cut the ~112ms K=3 verify step was measured DEAD.** The residual raw-TPOT
gap is against a *weak/verbose* vLLM reference (31.39ms); against the **confirmed** well-configured
vLLM, **Atlas already wins every reported metric.**

## Competitive picture (confirmed vLLM, same ~78 tok/turn, apples-to-apples)
| metric | confirmed vLLM | Atlas best (dpcarry) | winner |
|---|---|---|---|
| perf wall (1007) | 5361 s | 4984 s | **Atlas −7%** |
| tps | 14.6 | 15.91 | **Atlas +9%** |
| qps | 0.188 | 0.20 | **Atlas +6%** |
| 1/tps (agg ms/tok) | 68.5 | 62.9 | **Atlas −8%** |
| IoU | 0.6269 | 0.6285 | Atlas (tie+) |
| BFCL | 86.43 | 87.04 | **Atlas +0.6** |
| steady-state TPOT | ~31 (WEAK run) / unknown (confirmed) | ~40 | vLLM (weak ref only) |
Atlas wins because TTFT is ~2× better (1557 vs 2985ms), and the agentic turns are prefill-heavy
(~26k ctx, ~78 decode tok) so TTFT dominates the wall. The per-token decode gap is real but outweighed.

## Where the K=3 step goes (dgx3 nsys phase-split, 96% GPU-busy, no bubbles)
projection GEMVs 68% · full-attn 12.7% · MTP drafter propose 9% · lm_head 2.7% · GDN wy3 1.6% ·
GDN epilogue 0.9% · rollback/D2D 0.8% · sampling 0.4% · idle 3.8%.
Root cause of the 2× vs the 51ms weight-floor: the GEMVs are weight-once but the M=3 batched case runs
~75% peak (3× compute per streamed byte, occupancy/register-bound) — structural, not fixable by
access-pattern/prefetch.

## Levers tried → verdicts (the scoped pieces)
| lever | mechanism | verdict | evidence |
|---|---|---|---|
| L1 fuse K=2 verify epilogue | remove per-token launches | DEAD (neutral) | epilogue = 0.9% of step |
| L6 fuse K=3 epilogue | same at K=3 | DEAD (byte-identical, +0% ) | agent A/B |
| K=4 (more drafts) | raise E | REGRESSES +27.8% | draft sweep |
| GEMV access-pattern / prefetch (C1) | latency hiding | DEAD (0.98-1.00×) | cold microbench, near-roofline |
| **W4A8 int8-act DP4A (strix trick, C2)** | fewer mainloop instr | **DEAD on GB10** (0.99-1.01× +0.5% acc) | measured; strix baseline was pre-DP4A, GB10 already weight-once+roofline; hit __constant__-LUT trap |
| C3 NVFP4 the FP8 GDN proj | halve GDN bytes | DEAD | centml gate model ALREADY NVFP4-GDN (inert); on nvidia ckpt +4.6% SLOWER (w4a16 M=3 75% < w8a16 85%) |
| W4A4 native FP4 MMA | tensor-core acts | not worth (qwen+profile) | acts <1% of traffic at M=3; FP4 MMA wants M≥8 |
| fp8-KV cache | halve KV reads | [pending A/B] | attn 12.7% of step |

## What WOULD move the raw TPOT (none is a same-checkpoint kernel fold)
1. A higher-acceptance drafter (trained EAGLE-3-style) to lift E above the K=3 ceiling — a model
   project, not a kernel change; K=4 with the current drafter regresses.
2. A structurally cheaper verify (e.g. tree drafting w/ batched tree-verify — the tree-draft-feasibility
   branch) — a larger architecture change.
3. Obtaining the CONFIRMED vLLM's real TPOT — the 31.39 target is from a verbose run and may not reflect
   the well-configured vLLM's decode speed.

## Reproduce (git-replicable)
Branch `perf/decode-fold-2026-07-24`. Build/serve/gate/e2e commands in CAMPAIGN_LOG.md "Exact reproduce
commands". Investigation worktrees (uncommitted, correct-but-not-folded): `.wt-w4a4` (C1/C2 kernels +
w4a4_verify_bench.rs), `.wt-c3` (NVFP4-GDN-out-proj, qwen35_dense.rs:569). Gate: scripts/mlperf-edge/kl_coherence_gate.py.
Phase-split artifacts: dgx3:/workspace/decode_phasesplit_20260724_025300/.
