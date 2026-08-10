# Where the GB10 MLPerf-edge wall actually goes

**Date:** 2026-07-25 · **Source:** `results/chainK_golden_e2e_20260724_131209/events.jsonl`
(the shipped K=4 golden run, perf phase, 1007 samples, wall 4104.0 s)

Everything here is measured from the **harness's own event stream**, not the serve log. That matters:
the serve log's TTFT flatters the real number, and the harness's view is what the submission reports.

**Method note that cost me one wrong table:** `events.jsonl` contains BOTH the perf and accuracy
phases. Scope to the window between `session.start_performance_tracking` and
`session.stop_performance_tracking` or the sums are nonsense (unfiltered I got 2.6M s of "TTFT").
Samples inside the perf window = 1007; the raw file has 2002 `sample.issued`.

## The decomposition

| slice | seconds | % of wall |
|---|---|---|
| decode (first token -> complete) | 2447.6 | 59.6% |
| **fixed per-turn TTFT** (879 ms x 1007) | **867** | **21.1%** |
| marginal prefill (1.753 ms/new token) | 771 | 18.8% |
| client-side gap (wall − union of request spans) | **0.1** | **0.0%** |

It reconciles: mean decode 2431 ms = 78.2 output tokens x 32.62 ms TPOT.

**The client adds nothing.** The union of request spans is 4103.9 s of a 4104.0 s wall — requests are
back to back. This **corrects the older "wall is CLIENT-side +456 s/1007" note**: that does not hold
for this run, and no wall is recoverable by making the harness faster.

## TTFT is a clean two-parameter law

Fit over the 987 warm turns:

```
TTFT = 879 ms + 1.753 ms per new token          (marginal prefill = 570 tok/s)
```

- `corr(TTFT, prompt_chars) = +0.077` — **conversation depth is essentially free.** p50 is flat from
  1141 ms at <10k chars to 1509 ms at 80-90k chars.
- `corr(TTFT, delta_chars) = +0.585`, `corr(TTFT, turn_index) = +0.010`.
- No 512-token step structure — prefill is smoothly linear in the delta, so chunk quantisation is not
  a factor.

Real delta distribution (what the warm prefill actually sees): p5 61 / p25 106 / **p50 210** / p75 396
/ p90 698 / p99 2210 tokens, mean 331.

## The same law predicts COLD turn-1 within 3%

Applied unchanged to the 20 cold turn-1 requests (median prompt 1696 tokens): predicted 3852 ms vs
observed 3973 ms, **median ratio 1.03**, range 0.91-1.13. Two consequences, both important:

1. **Prefix caching + SSM snapshots are already behaving optimally.** A warm turn costs exactly what
   a cold prefill of only its delta would cost. There is no hidden replay and no hidden recompute to
   go find — a warm-path audit would be wasted effort.
2. **The 879 ms constant is present on cold requests too**, where there is no snapshot to restore and
   no radix walk to speak of. It is therefore not a caching artifact.

## What the 879 ms is NOT

Each ruled out from source or from the data, not by assertion:

- **Not SSM replay distance.** `prefill_b/save_checkpoint.rs:40` saves tail checkpoints at the last
  TWO block boundaries below the prompt end, so replay is <= 32 tokens.
- **Not an MTP drafter rebuild.** `drafter_context.rs` documents carry as already incremental:
  ~21.5 ms mean append, versus 1136 ms for a from-scratch rebuild.
- **Not client dead time.** 0.1 s across the whole run (above).
- **Not the prefix cache.** Present cold.
- **Not GEMM tile padding.** At the real delta distribution, M128 -> M64 would cut padded rows only
  7.4% = 57 s = **1.4% of wall**. Bounded and not worth chasing.

Remaining candidates are transport/HTTP, request admission/scheduling, per-request graph capture, and
streaming first-flush. **vLLM's min TTFT on this same harness and box is 616 ms** (best Atlas run 650,
golden 782), which bounds how much of the constant can be Atlas-specific — plausibly ~260 ms/turn
(~6.4% of wall), but that split is **not yet proven** and is the highest-value open measurement.
Probe: `scratchpad/ttft_floor_probe.py` (crosses prompt length x delta so the two effects separate).

## Why this reprioritises the campaign

Median output is only **45 tokens** (avg 78.2, total 78,786). So for the median request,
latency 2847 ms = TTFT 1298 ms + 45 x 32.6 ms — and **the 879 ms floor alone is ~32% of median sample
latency**. With outputs this short, per-request overhead competes directly with decode, which the
campaign had been treating as the only remaining axis.

Decode itself is at its measured limit: TPOT 32.62 ms against well-tuned vLLM's 31.39 ms, K=4 is the
verified optimum of the K-ladder, and the `n in 4..=8` FFN arm is correctly wired through
`w4a16_gemv_batch8` (checked — this was the failure mode that made every historical "K=4 regresses"
result wrong, so it was worth re-verifying at K=5..8).

## The workload runs one conversation at a time

20 conversations, 27-61 turns each, **not interleaved**: distinct other conversations between
consecutive turns of the same conversation is p50 0, mean 0.0, **max 0**.

- No cross-session snapshot eviction exists in this benchmark. Any interleaved multi-session probe
  measures a pressure the workload never experiences — see `SSM_SLOTS_AB.md` for the probe that
  over-predicted a tail win by ~50x on exactly this mistake.
- KV at 331,664 tokens is ~13x the largest single conversation (~24k tokens), so KV is not the
  binding constraint and trading it away is cheap here.
- Slot pressure is INTRA-conversation: ~3 slots/turn means the 6 longest conversations need 165-183
  slots against the 128 pool. Real, but worth only a −133 ms -> +86 ms residual drift across a
  conversation.
