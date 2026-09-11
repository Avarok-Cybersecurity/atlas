# Why concurrent prefills ran one at a time (#927, H100)

Phase-1 attribution for the batched (varlen) prefill lever. **Headline: the
scheduler was already co-admitting all sixteen streams; the serialisation is one
crate down, and the documented fix could not engage because four sites disagreed
about one boolean.**

## Anchors (measured, 1xH100, Qwen/Qwen3.8-27B-FP8, round 11 cells A and E)

nsys over two back-to-back 16-way bursts of 1193-token prompts,
`--max-prefill-tokens 8192` (`prefill_budget=8192, max_batch_tokens=8196`).
`rms_norm_residual` launches once per layer per forward, `GrdX` = that forward's
token count; all 576 detected forwards have exactly 64. **64 prefill chunks for
32 requests, strictly `1168, 25, 1168, 25, …`.**

| forward `GrdX` | count | busy ms | ms/forward | **µs/token** |
|---|---|---|---|---|
| 1168 (prompt head) | 32 | 7064.4 | 220.76 | 189.0 |
| 25 (prompt tail) | 32 | 936.2 | 29.26 | **1170.2** |

A 16-row decode step costs 1234.5 µs/token — the tail chunk is per-token
indistinguishable from decode. Window: PREFILL 8.0 s (39.8%), DECODE 10.1 s
(50.2%), IDLE 2.0 s (10.0%). C=16 TTFT p50 2.28 s / p99 4.22 s (vLLM 1.05 / 1.17,
same box). Budget at 2048 or 24576 moved the aggregate ±0.4% — **the budget was
never binding**, which is what sent this to the trace.

## 1. Why the default path prefills one prompt per step

Not the scheduler. `continue_in_progress_prefills` takes
`can_batch_prefill_only` (N≥2 prefilling, no active decode, not EP) and hands ALL
sixteen streams to `Model::prefill_batch_chunk` in one call. The serialisation is
inside that call: `prefill_batch_chunk_dispatch` asks `kernel_batched_eligible`
first, and that predicate rejects any batch with `chunk_start == 0` unless
`allow_chunk_zero` — false with neither `ATLAS_PREFILL_CODISPATCH` nor
`--prefill-varlen-batch` set. A burst of fresh prompts is all chunk 0, so it falls
through to the per-stream loop: sixteen sequential forwards at M=1193 instead of
one at M≈7000, every projection GEMM at a sixth of the arithmetic intensity it
could have. The `seq_len_start > 0` precondition exists because the
batched attention kernels are **paged-only** and chunk 0 historically took the
non-paged `prefill_attention_with_cache_skip` arm. VARLEN v1 lifted that (chunk-0
streams get a forced paged upload in `batch_kernel.rs`), so the precondition was
already stale — for three of its four readers.

## 2. Why `--prefill-varlen-batch` returned empty 200s instead

Four sites decide chunk-zero co-admission. Three read `codispatch || varlen`:
`check_kernel_batched_eligible`, the paged-upload force, and
`prefill_attention_paged_attn_batched`. The fourth,
`Qwen3AttentionLayer::prefill_inner`, read `codispatch` **alone**. Admission said
yes; the layer said no — at layer 0, with Phase A already done (KV blocks
allocated, prefix reservations taken, hidden staged for every member).

```
ERROR ...run_batched_prefill: Batched prefill error (wave of 6 streams, 6
prefilling): prefill_inner: batched mode requires seq_len_start > 0 (paged
path); got seq_len_start=0. Caller must fall back to per-stream for this chunk.
```

The error asked the caller to fall back. The caller did not: it pushed
`(i, None)` for every member and `promote_completed_prefills` freed each sequence
while **dropping its `ResponseSink`**. The SSE body was already committed with a
200, so sixteen clients got a stream ending after the role delta — no content, no
`finish_reason`, no usage frame, `err=0`. 3/3 reps; a recheck gave one clean rep
at 438 tok/s (2% ahead of the default) and one dropping 3 of 16. Filed separately
as Avarok-Cybersecurity/atlas#1000 — silent truncation is a bug on its own.

## 3. What chunk 0 needed

Nothing structural. KV writes go through `slot_stacked` per token; RoPE reads
`positions_stacked`, so positions start at 0 with no special case; the paged
batched kernel takes `q_offset = 0` with per-stream `cu_seqlens` / `kv_lens` and
masks causally relative to it; GDN is per-request under `cu_seqlens`, each over
its own `h_state`, whose fresh value is the zeroed one `alloc_state` returns. The
fix is the predicate, not the plumbing: one function,
`ops::prefill_batched_chunk_zero_allowed`, read by all four sites, with a
source-scanning test that fails if any re-derives the disjunction.

## 4. The 25-token tails

`prefill_chunk_dispatch` splits a prompt's FINAL chunk once at
`((total-1)/bs)*bs - bs` = 1168 for a 1193-token prompt at `bs=16`, to land an SSM
tail checkpoint a later turn's block-floored prefix match can use. It is
unconditional on hybrid-SSM models with the prefix cache active and must stay so:
a shape differing cold vs warm flips a temperature-0 argmax (Puzzle-75B: "17
barrels" cold, "15 barrels" warm). The batched path does not split, so a whole prompt handed to it would take a
different shape than the same prompt takes per-stream. `Model::prefill_tail_cut`
reports the cut and the scheduler pre-splits there under VARLEN: geometry stays
identical per sequence, and the tails BATCH — sixteen tails of one burst share
`chunk_start = 1168`, so the planner puts all of them in one forward instead of
sixteen standalone passes. Ragged lengths give different cuts and so separate
tail waves; pairing a tail with another prompt's head is not available, because a
batch must share `chunk_start` (it sets `effective_seq_len_start`) and
`is_last_chunk` (finalize_last and save_checkpoint cannot dispatch together).
**Admission arithmetic:** `wave_cap = min(--max-prefill-tokens,
max_batch_tokens) = 8192`. Sixteen whole 1193-token prompts pack 6 per wave
(6×1193 = 7158; a seventh is 8351) → 3 dispatches, not 16; pre-split heads pack 7
(7×1168 = 8176). Both pinned in `prefill_waves::tests`. The lever stays opt-in
(`--prefill-varlen-batch`, legacy `ATLAS_PREFILL_VARLEN=1`, default OFF) until
the H100 A/B; the boot route line prints the resolved chunk-zero decision beside
it, and `ATLAS_NO_TAIL_SPLIT=1` stays the A/B for the split itself.

## 5. Why the lever then degenerated its output (#1002, H100 round 13)

Phase-2 attribution. The lever now engages as §1-§4 describe — `16 streams ->
3 wave(s), M per wave [50, 8176, 8176] (cap 8192)`, `Q12 kernel-batched prefill
dispatched (fused large-M) n=7 total_tokens=8176`, every response carrying
tokens and a `finish_reason`. And cell V logged **24 content-loop-watchdog
fires, 2 fuzzy-repetition stops and 5 SimHash stops against ZERO on cells A, D,
T1 and T2** — same binary, same prompts, temp 0, seed 42 — with 6/16 probe
responses cut at 49 tokens.

**Not the varlen kernel path. The scheduler hands two live sequences one SSM
pool slot.** A sequence claims its slot at admission, so a stream parked in
`prefilling` owns one; `compact_survivors_into_range` derived its free targets
from the DECODING set alone. Varlen is the first configuration that parks
slot-owning streams there (`want_varlen_defer`: 96 deferrals on cell V, zero on
A/D/T1), so the per-tick compaction migrated an active survivor onto a
prefilling stream's slot — visible in the log 30 ms apart as
`slots=Some([0, 7])` then `slots=Some([0, 1])`. `compact_sequence` calls
`ssm_pool.claim_specific` and discards its false return, so the collision is
silent, and two sequences then share one GDN `h_state`/`conv_state`.

It outlives the burst: both owners release that index, `release_slot`'s only
guard was a `debug_assert` (a no-op in the shipped `--release` binary), and the
duplicated free-list entry makes `claim_slot` issue it to two fresh sequences
for the life of the process. That is why the 16-way probe four minutes later —
**with varlen not engaging at all**, every prefill logging `Prefilled (single
chunk)` — still decoded at `slots=[0, 0, 0, 1, 1, 2, 2, 3, ...]` and lost 6/16
responses. Duplicate REAL slots appear in 16 of cell V's 24 batched-decode
captures and in none of any other cell's in this campaign.

**The `50`-token wave is not a truncated chunk 0.** `[50, 8176, 8176]` is two
25-token TAILS (from streams that arrived one tick early, their heads dispatched
as the `2336` forward) plus fourteen 1168-token heads: `2 x 25 + 14 x 1168 =
16402`. The budget can never shrink a chunk 0 — `plan_stream_chunk` budgets
against `max_prefill_tokens` and the WAVE cap belongs to the planner, which
opens a new wave rather than trimming a member. Pinned in `prefill_waves_tests`
alongside a tick-by-tick replay of the burst's arrival pattern.

**The long shape's +83% TTFT was deferral with no batching.** 4593 pre-splits
to `4576 + 17` and `2 x 4576 = 9152 > 8192`, so the planner emitted 14 waves for
16 streams; waves run back-to-back inside one tick, so no stream is promoted
until all of them have run and every TTFT collapses onto the p99 (7 524.7 ->
13 756.1 ms while p99 barely moved). `varlen_defer_pays` now declines the
deferral when the two smallest chunk-0s cannot share a wave.

**And a watchdog stop is no longer only a `"length"`.** The wire
`finish_reason` still reports `"length"` for every non-timeout guard — that
mapping is a measured contract, and minting a new enum value hard-fails typed
clients — but the guard's name now rides beside it in the `stop_reason`
extension field, on the streaming chunk and the blocking choice alike (the
round-13 probe ran `stream=false`).

**Still open after this:** the batching itself does not pay. nsys on cell V
measured `M=8176` (7 prompts fused) at **190.4 us/token against 188.8 us/token
at M=1168** — 0.8% WORSE — with total prefill busy moving 8 000.5 -> 7 925.2 ms
(-0.9%) for the same 38 176 tokens, and total GDN prefill identical at 54% of
prefill either way. Round 11's arithmetic-intensity hypothesis (§1, "every
projection GEMM at a sixth of the arithmetic intensity it could have") is
refuted by measurement: the M=1168 GEMMs are already at full efficiency on an
H100. Varlen repackages the GDN work; it does not reduce it. The lever stays
default OFF, and the C=16 gap to vLLM is not a prefill-batching deficit.

## 6. The slot fix held; the SCHEDULING of the waves was the loss (#1002, H100 round 15)

Round 15 rebuilt at `8a6f50b61` with #1002's slot-aliasing fix in. **Every
structural check passes**: cell V15 logged **0 `SSM pool slot SHARED`, 0
`release_slot … already free`, 0 content-loop / fuzzy / SimHash stops, 16/16
responses at the full 256 tokens**, and every `Captured CUDA graph … slots=`
list carries distinct real slots (index 16 is the padding sentinel). Against
round 13's cell V — 24 content-loop fires, 2 fuzzy stops, 5 SimHash stops, 6/16
probe responses cut at 49 tokens — that is a clean reversal. On the LONG shape
`deferral SKIPPED` fires and the numbers land on the no-lever cell's exactly
(398.34 vs 401.28 tok/s, 4 115.2 vs 4 118.8 ms TTFT): round 13's +83% TTFT
regression is gone, replaced by a null.

**And the lever was still a loss on the SHORT shape, for a scheduling reason,
not a kernel one.** Whenever the 16-stream varlen batching engaged:

| | A15 (no lever) | **V15 (engaged)** | Δ |
|---|---:|---:|---:|
| `1024x256` C=16 aggregate | 513.86 | **427.53** | **−16.8%** |
| TTFT p50 | 1 359.4 ms | **4 141.2 ms** | **3.05×** |
| TTFT p99 | 2 504.5 ms | 4 142.2 ms | +65% |
| **TPOT** | 25.90 ms | **21.28 ms** | **−17.8%** |
| e2e p50 | 7.97 s | 9.57 s | +20% |

**TPOT moved the right way by 17.8%** — batching the prefill up front genuinely
removes the prefill/decode interference round 11 traced — so the batching works.
What did not work was that **p50 = p99**: `16 streams -> 3 wave(s)` ran
back-to-back inside ONE tick. Promotion is a phase, not a callback: a stream's
first token reaches its client in `promote_completed_prefills`, after
`continue_in_progress_prefills` returns, and decode runs later still in the
tick's own decode step. So a stream that finished in wave 1 waited for waves 2
and 3 before anyone heard from it, and every TTFT in the burst landed on the
slowest.

**The fix is ONE WAVE PER TICK** (`prefill_waves::waves_this_tick`), not a
second guard condition. Wave 1 runs, its finished streams are promoted at the
end of that tick, decode interleaves from the next one, and the streams that did
not fit re-plan next tick — where they batch among themselves, because the
planner is re-run from the live geometry every tick. On the round-15 shape:
sixteen 1193-token prompts plan 7 / 7 / 2, seven prefill on tick 1, and on tick
2 those seven stand first in FIFO order and their tails complete as one
175-token forward before any further head wave is issued.

The alternative — promoting inside the wave loop — buys nothing here.
`promote_completed_prefills` removes from `prefilling`, which invalidates the
wave indices mid-loop, and decode still would not run until the prefill phase
returned; the promoted stream's token would leave at the same wall time. The
tick boundary is where the scheduler already interleaves, so that is where the
cut belongs.

FAIRNESS is unchanged in kind. The planner is first-fit in FIFO order, so wave 1
always contains stream 0: the head of the queue advances exactly one chunk per
tick, which is the guarantee the single-stream `prefilling.first_mut()` path
gives — and it now carries everyone who fits with it. With the flag OFF the
planner returns one wave holding every stream and this cap returns it whole, so
that path stays byte-identical. Pinned in `prefill_waves_tick_tests`.

**Why engagement varied burst to burst, and what is done about it.** Round 15
saw two of four otherwise identical short-shape bursts engage (427 tok/s) and
two not (514 tok/s — the no-lever numbers), a 17.52% rep spread that is not
measurement noise. The planner is not the source: `plan_prefill_waves` is a pure
function of the geometries. The ADMISSION is: it requires `active.is_empty()`
and two chunk-0s co-admitted in the SAME tick, and a burst whose first request
has already been promoted to decode by the time the rest arrive fails the first
test. That is arrival timing against the scheduler's tick period — **inherent,
not pinnable** — so `varlen_admission` now returns the one input that decided
each verdict and `phase_start_prefills` logs it per burst:

```
Varlen prefill admission: defer=false reason="decode already active this tick (arrival timing)" new_reqs=9 prefilling=0 active=7 smallest_chunk0=[1193, 1193] cap=8192
```

A run's engagement is now readable from its serve log instead of inferred from
its throughput. The `deferral SKIPPED` line is unchanged and still fires on the
long shape.

**The lever stays default OFF.** §5's finding stands: `M=8176` (7 prompts fused)
measured **190.4 µs/token against 188.8 µs/token at M=1168** — 0.8% WORSE —
so varlen repackages the GDN work rather than reducing it. This change removes
the TTFT collapse that made the lever a 16.8% aggregate loss on top of that; it
does not make batching pay, and the C=16 gap to vLLM is still not a
prefill-batching deficit.
