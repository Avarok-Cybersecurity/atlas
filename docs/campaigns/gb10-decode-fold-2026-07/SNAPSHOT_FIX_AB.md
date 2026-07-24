# Snapshot-lookup fix on GB10 — measured NEUTRAL, folded anyway

**Date:** 2026-07-24 · **Box:** dgx1 (GB10) · **Model:** `centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf`

The commit `fix(ssm): stop warm-turn snapshot misses that make TTFT climb on short prompts`
was carried over from the Strix branch, where it was measured as a large warm-TTFT win
(turn-4 TTFT 6973 → 2991 ms, and on the desktop probe warm p90 6.1 s → 2.7 s). It was
expected here to close the one latency axis a tuned vLLM still wins — TTFT p99.

**It does not, because the pathology does not reproduce on GB10 under the golden config.**

## The A/B

Same box, sequential (GB10 unified memory cannot host two serves at `--gpu-memory-utilization
0.70`), frozen c2final env, `--num-drafts 3`, greedy temp 0 / seed 42. One growing 16-turn
conversation with a >1024-token system prompt so every turn hashes to the same session.
`scripts/mlperf-edge/ab_ttft_probe.sh` + `warm_probe.py`.

Legs differ only in the binary: control = `3e609cee` (K=4 golden), candidate = control + this fix.

| leg | TTFT p50 | TTFT mean | TTFT p90 | TTFT max | TPOT p50 | output sha |
|---|---|---|---|---|---|---|
| control | 906.0 ms | 936.4 ms | 1049.6 ms | 1148.2 ms | 43.59 ms | `b5d6a89aab5528ea` |
| + snapshot fix | 906.5 ms | 935.6 ms | 1049.3 ms | 1144.5 ms | 43.62 ms | `b5d6a89aab5528ea` |
| delta | +0.06% | −0.09% | −0.03% | −0.31% | +0.07% | identical |

Raw per-turn data: `ab_snapshotfix_ctl.json`, `ab_snapshotfix_conglom.json`.

## Why it is neutral here

The control leg has **no warm-TTFT climb to fix**. Across turns 2–15 its warm TTFT is flat at
850–1150 ms while context grows from 11.7k to 25.7k characters — the curve is flat and mildly
*decreasing*, not climbing. The Strix branch lacked the mitigations the GB10 golden config
already ships:

- `ATLAS_SSM_TAIL_PROTECT=1` + `ATLAS_SSM_TAIL_LEASE_TTL=128`
- `--ssm-cache-slots 128`, `--ssm-checkpoint-interval 32`
- the main-line recency-only snapshot eviction (#287) and the tail lease (#345)

With those in place the anchor this fix preserves is already being preserved by other means.

## What this does NOT tell us about the e2e TTFT p99

This probe cannot reach the regime the e2e outliers live in, and that limit should be stated
rather than papered over:

1. **One session.** The probe runs a single conversation, so it never creates contention for the
   128 SSM cache slots. The e2e runs 1007 samples across many sessions — eviction pressure is a
   multi-session phenomenon this probe structurally cannot produce.
2. **Depth.** Max context reached is ~25.7k chars (~6.4k tokens). The golden run's TTFT p99 is
   5354 ms with a single 26.6 s outlier; the Strix campaign localised its residual tail to SSM
   snapshot *eviction* at ~12.5k context, deeper than this probe goes.

So the e2e TTFT tail is most likely eviction-driven, not lookup-driven, and this fix targets
lookup. Closing the p99 axis is still open work.

## Verdict

**FOLD — neutral, not a win.** It is output-neutral (identical greedy output hash over 16 turns,
~3.4k generated tokens), costs nothing measurable, and is a genuine correctness fix for configs
that do not carry the tail-protect mitigations. It is carried on this branch for that reason and
**no TTFT improvement is claimed from it on GB10**.
