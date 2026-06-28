# SBR M1 (tail-pin eviction) — e2e results on dgx2 (Qwen3-Next-80B-NVFP4)

## Win (stranding scenario: deep ~24k conv, warmed, idle under 30-conv pressure bursts, resumed)

`--enable-prefix-caching --ssm-cache-slots 16`, `ATLAS_SBR_TAIL_PIN` 0 vs 1, top-K=4.

| resume cycle | Baseline TTFT / replay-tok | Tail-pin TTFT / replay-tok |
|---|---|---|
| cyc0 | 9.31s / 7456 | 3.32s / 2417 |
| cyc1 | 9.48s / —    | 0.44s / 984  |
| cyc2 | 9.58s / 7568 | 9.29s / 7633 (gap) |
| cyc3 | 9.74s / 7760 | 0.45s / 11   |
| **mean** | **9.53s** | **3.37s** |

- **2.8× mean TTFT, up to 21× (0.45s vs 9.74s) on the best cycle.**
- Replay distance collapses from ~7600 tokens → **11–984** when the anchor survives.
- **Exact by construction**: tail-pin changes only WHICH checkpoint is restored; the
  replay from it is the unchanged bit-exact WY4 path (`gdn_exact_replay`), so reconstructed
  state at the match point is identical regardless of anchor depth. (Formal argmax/KL
  parity check pending for the paper.)
- Baseline is steadily stranded (~9.5s, replay ~7600) — the 1s→21s pathology, here at
  ~24k depth manifesting as ~9.5s.

## Root cause confirmed (snap-lookup token_counts)
Baseline at match 23968 keeps `[…,16512]` (deep region evicted) → replay 7456. The fix
keeps the resumable session's **top-K deepest**; pinning the single deepest alone failed
because the leaf **overshoots** the match point (`24061 > 23968`, lookup excludes it). K≥2
keeps a usable ≤-match anchor.

## Robustness FIXED with top-K=8 (now the default)

| resume cycle | Baseline | K=4 | **K=8 (default)** |
|---|---|---|---|
| cyc0 | 9.31s | 3.32s | 3.35s |
| cyc1 | 9.48s | 0.44s | 0.45s |
| cyc2 | 9.58s | 9.29s | **0.46s** |
| cyc3 | 9.74s | 0.45s | 0.45s |
| **mean** | **9.53s** | 3.37s | **1.18s** |

K=8 eliminates the cyc2 spike (replay now 11–19 tok on warm cycles). **8× mean
speedup, up to 21×, FLAT and exact.** The warm-cycle 0.45s **matches llama.cpp's
continuous-sequence resume** — i.e. SBR attains the "never recompute" latency
WITHOUT keeping the sequence live (slots still evicted/shared). Default set to K=8
(`ATLAS_SBR_TAIL_PIN_K`). cyc0 (first post-pressure resume, 3.35s) is the only
non-flat point — the ideal anchor wasn't yet in top-K when the first burst hit;
still 2.8× better than baseline.

## Final shipping form (2026-06-27): top-K=8, DEFAULT OFF

Multi-conversation robustness sweep settled the policy honestly:

| regime | baseline | tail-pin ON | verdict |
|---|---|---|---|
| single deep conv (strand) | 9.53s | **1.18s (8×)** | decisive win |
| balanced 8-conv / 24-slot round-robin | 5.89s | 7.68–7.86s | **~30% REGRESSION** |

Across all four policy variants tried (single-deepest, top-K, budget-aware,
session-count gate, two-tier), enabling tail-pin reliably **regresses balanced
many-conversation round-robin ~30%** — the recency·hit forecast is already
near-optimal there and pinning fights it. Attempts to detect the regime from the
index's local view (budget/n, in-index session count, two-tier resumable-set)
all failed because that view doesn't reflect the true active-conversation count.
The two-tier variant was also empirically *worse* on strand (cyc0 9.04s) and is
reverted.

**Decision: ship the top-K=8 policy OFF by default** (`ATLAS_SBR_TAIL_PIN` unset
= pure baseline forecast = provably do-no-harm everywhere), enabled via
`ATLAS_SBR_TAIL_PIN=1` for deep single/few-conversation agentic workloads (the
user's 1s→21s symptom), where it delivers the 8× win. Exact in both modes.
Policy selection split into `evict_lru_inner(pin, k)` for unit-testability.

## Open follow-ups (lower priority)
- Regime auto-detection from a TRUE active-conversation signal (scheduler-side,
  not index-local) to enable safe default-on — the 4 index-local attempts failed.
- Formal argmax/KL parity table for the paper (exactness is structural; measure it).
- M3 2-D sheaf reconciliation: prototyped → honest negative (see M3_FINDINGS.md).

## Config / reproduction
Binary: SBR `feat/sheaf-based-replaying` built on dgx1, bind-mounted into
`atlas-gb10:ornith-perf` on dgx2. Scripts: `research/sbr/sbr_strand.py`,
`run_strand_ab.sh`. See [[project_atlas_marconi_warmhit_bench_gotchas]] for the
must-pass gates (esp. `--enable-prefix-caching`).
