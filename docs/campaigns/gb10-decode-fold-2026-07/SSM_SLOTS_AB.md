# `--ssm-cache-slots` on GB10 — 192 is a measured NULL, and the probe that motivated it was invalid

**Date:** 2026-07-25 · **Box:** dgx1 (GB10) · **Model:** `centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf`
**Verdict: keep `--ssm-cache-slots 128`.**

## What a slot costs

Per Marconi snapshot slot, for each of the 48 GDN layers:

| tensor | shape | dtype | bytes/layer |
|---|---|---|---|
| `h_state` | 48 heads x 128 vdim x 128 kdim | **FP32** | 3.00 MB |
| `conv_state` | 10240 x 4 | **FP32** | 0.156 MB |

3.156 MB x 48 = **151.5 MB/slot**, which matches the allocator log exactly
(`Marconi 192 slots (29088 MB)`). **`h_state` is 95% of a slot.** Slots come straight out of the KV
budget: 128 slots = 19.4 GB pool -> 20.2 GB KV (331,664 tokens); 192 slots = 29.1 GB -> 10.8 GB KV
(177,152 tokens). 256 slots is unrunnable (~0.8 GB KV, less than one max-length sequence).

## The probe said 3.8x. The e2e said 0.8%.

`scripts/mlperf-edge/ssm_slots_probe.py` interleaves S=24 conversations round-robin so that
snapshots get evicted between turns of the same session. It reported warm-TTFT:

| slots | p50 | p90 | p99 | max |
|---|---|---|---|---|
| 32 | 2724 | 2897 | 2903 | 2906 |
| 128 | 750 | 2850 | 2895 | 2898 |
| 192 | **655** | **759** | **762** | **762** |

Read literally that says 128 — what we ship — sits at the worst point on the curve: enough to fix
the median but with a p90/p99/max statistically identical to 32 slots, and only 192 collapses the
tail. So we ran the full both-phase e2e, changing only the slot count:

| metric | 128 (tip) | 192 | delta |
|---|---|---|---|
| wall | 4134.01 s | **4145.41 s** | **+11.4 s** |
| tps | 19.25 | 19.12 | −0.13 |
| TTFT p50 / p90 / p99 | 1301.6 / 2544.9 / 5415.6 ms | 1292.9 / 2524.0 / 5328.2 ms | −0.7% / **−0.8%** / **−1.6%** |
| TTFT max | 26801 ms | 26448 ms | −1.3% |
| TPOT p50 | 32.62 ms | 32.54 ms | −0.08 |
| IoU | 0.6279 | 0.6258 | −0.0021 |
| BFCL | 87.24 | 87.04 | −0.20 |

Both phases 1007/1007, 0 failed. Every delta is inside the measured noise floor (two IDENTICAL runs
differ by IoU 0.0048 / BFCL 0.20). **The probe over-predicted the tail win by roughly 50x.**

## Why the probe was wrong — check the workload's issue order

From the golden run's `events.jsonl`: the MLPerf-edge agentic workload has 20 conversations of 27-61
turns, and **it does not interleave them.** Each conversation runs to completion before the next
begins. Distinct *other* conversations between consecutive turns of the same conversation:
**p50 0, mean 0.0, max 0.**

Cross-session snapshot eviction — the only pressure the probe creates — therefore never happens.
The probe was measuring a regime the benchmark does not have.

This is a stronger failure than the caveat originally attached to it ("biased toward more slots
because its sessions are short"). The sessions being short was a second-order bias; the first-order
defect was that the pressure axis itself was absent.

**Rule going forward: before building a probe, check the issue order of the target workload.**

## What *is* true about slots, and why it still doesn't pay

Slot demand here is INTRA-conversation, not cross-session: each turn saves ~3 slots (2 tail
checkpoints from `prefill_b/save_checkpoint.rs` + 1 leaf), so the six longest conversations need
165-183 slots against the 128 pool and genuinely do evict their own checkpoints.

That effect is real but small. Removing the fitted TTFT law (`879 ms + 1.753 ms/new_token`, see
`WALL_DECOMPOSITION.md`) leaves a residual that drifts only from **−133 ms** at turns 0-4 to
**+86 ms** at turns 45-49. Covering it with 192 slots bought nothing measurable at the wall.

The KV-squeeze worry dies on the same data: with one conversation live and a max conversation of
~24k tokens, even the reduced 177,152-token KV is ~7x headroom. Neither side of the tradeoff bites,
which is exactly why the result is a flat null rather than a win or a regression.

## Open follow-up

Because `h_state` is 95% of a slot and is FP32 purely as *storage* (compute stays FP32), narrowing
the snapshot dtype would halve the pool and halve save/restore D2D traffic. That is being measured
separately — but note this result removes the motivation of "buy more slots with it", since more
slots are worth nothing here. Its remaining value is the freed memory and the faster restore.
