# QSA prefill-select determinism fix (second temp-0 nondeterminism source)

Date: 2026-09-02. Worktree: `/home/ms/atlas/.claude/worktrees/exl3-qsadet`
(branch `wip/exl3-qsadet`, base `e76b349fe`). Uncommitted, for review.
All runs on dgx-00, checkpoint `/tank/exl3-ckpt/qwen38-flash-next-2.05bpw`,
CTX 8192, SEQS 1, bf16 KV, `--deterministic-moe-prefill` default (on).

## Defect, verified (not just trusted)

`qsa_topk_rows` (`kernels/gb10/qwen3.8-flash-next/nvfp4/qsa_indexer.cu`), the
"emit everything strictly greater" pass: output slots are handed out by
`atomicAdd(&s_emitted, 1u)` across 256 threads, i.e. in race order. Same SET,
different ORDER every run.

Verified live two ways before changing anything:

1. **Consumer read**: `qsa_prefill_attn` consumes `lists` directly (no
   `qsa_expand_sel` re-sort — that is decode-only) and runs an online softmax
   whose `m`/`l`/`acc` running-rescale accumulates in list order
   (qsa_indexer.cu lines ~492-501). Different order = different fp32 rounding
   sequence = bit-different attention context. The old header comment "output
   order ... is irrelevant to the consumer" was true mathematically, false
   bit-wise.
2. **Kernel-level repro**: a new GPU test ran the pre-fix kernel 8x on
   identical device-resident scores (production shape: 64 rows, topk 512,
   ratio 4, relu-floored tie-heavy scores). Run 1 differed from run 0
   bit-wise, with the emitted order visibly warp-shuffled
   (`test_prefix.log`). The host `prefill_select` path and
   `qsa_score`/`qprep`/`prefill_attn` needed no exoneration beyond this: the
   racy emit alone reproduces, and fixing it alone clears the e2e probes.

## What shipped

**Canonicalise `out[]` to ascending block id at the end of `qsa_topk_rows`**:
a bitonic sort in shared memory (`QSA_TOPK_SORT_MAX 512`, next-pow-2 padded
with INT_MAX) — byte-for-byte the same pattern `qsa_expand_sel` already uses
to make DECODE's order match the host's `sort_unstable()`. The emit pass and
tie-fill pass are untouched, so the SET and tie-break semantics ("top-k by
score, ties broken by lower index") are exactly as before; only the order
becomes a pure function of the set. Ascending is also decode's canonical
order, so both consumers now see the same convention.

Supporting changes:
- `crates/spark-model/src/layers/qsa.rs`: `QsaIndexer::new` now ensures
  `block_topk <= 512` (the shared-memory cap of BOTH device sorts; decode's
  `qsa_expand_sel` always relied on it silently — production is exactly 512).
- `crates/spark-model/src/layers/qsa_tests_topk.rs` (new, 134 LoC) +
  wiring in `qsa_tests.rs`: GPU test `qsa_prefill_topk_bit_deterministic` —
  8 runs on identical scores (relu_floor / all_tied / distinct), asserts
  (a) bit-equality across runs, (b) strictly-ascending `out[]`,
  (c) set == host reference (sort by (-score, index), take k).
  **Fails on the pre-fix kernel** ((a) and (b)); passes fixed.

No kill switch: the change is semantics-preserving on the selected set, costs
nothing measurable (below), and a switch would only re-enable a race.

**Rejected alternative**: making the strictly-greater emit itself walk in
index order with block-wide ballot ranking (sharing the tie pass's walk).
Deterministic by construction and saves the sort, but it is the intricate
option in exactly the code being fixed (two-level ballot prefix with a tie
quota crossing warp boundaries), ends at the same ascending order, and the
sort's cost is unmeasurable. The sort is the cheapest correct fix and is
verifiable at a glance.

## Validation (all measured, logs in this directory)

Bit-level echo probes (8 identical echo+logprobs requests, prompt logprobs
compared bit-for-bit; `echoprobe.py` / `echolong.py`). "distinct n/8": 1/8 =
bit-identical across all 8.

| prompt tokens | pre-fix (no prefix cache) | fixed (no prefix cache) | fixed (prefix cache ON) |
|---|---|---|---|
| 93   | 1/8 | 1/8 | — |
| 1938 | 1/8 | 1/8 | — |
| 2103 | **8/8** | **1/8** | 1/8 |
| 2213 | 8/8 | 1/8 | — |
| 2323 | 8/8 | 1/8 | — |
| 5513 | **8/8** | **1/8** | 1/8 |

(The brief's "~4963" leg: this filler tokenizes to 5513 at FR=100 on this
serve; same neighborhood, above-bound, long multi-chunk prefill.)

Selected-set preservation (fix reorders, never re-selects):
- New GPU test leg (c): fixed kernel's set == host reference on tie-heavy
  relu-floored scores, all-tied scores, and distinct scores (64 rows each).
- `qsa_prefill_select_sets_match_reference` (real checkpoint weights, golden
  mask fixtures): PASS on the fixed kernel.
- `qsa_decode_select_parity` example: 200 cases device == host — decode,
  which shares `qsa_topk_rows`, is byte-identical to the host path after the
  change (the added sort feeds `qsa_expand_sel` an already-ascending list;
  its own sort is a no-op on it).

Speed (TTFT, streamed first token, 6 reps, median ms — same box, same boot):

| prompt tokens | pre-fix | fixed | delta |
|---|---|---|---|
| 2323 | 4256 (best 4238) | 4264 (best 4258) | +0.2% |
| 5513 | 11374 (best 11348) | 11386 (best 11372) | +0.1% |

Within run-to-run spread; #820's device top-k win is untouched (the sort adds
2 KB smem + ~45 barrier stages per 512-entry row vs 5 full passes over up to
~25K scores).

Gates (all exit 0, `gate_*.log`):
- `cargo test --release -p spark-model --lib`: 730 passed / 0 failed.
- `cargo test --release -p spark-runtime --lib`: 0 failed.
- `cargo clippy --release -p spark-runtime -p spark-model -p spark-server
  --all-targets`: clean.
- `exl3_native_parity` example: PASS every leg.
- `qsa_decode_select_parity` example: PASS, 200 cases.
- GPU QSA fixture tests (`qsa -- --ignored`): 6 passed / 0 failed.
- New `qsa_prefill_topk_bit_deterministic`: FAILS pre-fix, PASSES fixed
  (re-verified after the rustfmt touch-up).

Serve hygiene: no serve left running; watchdogs never fired; no unexplained
SIGTERMs this session.

## What remains nondeterministic

Nothing observed at temp-0 prefill in this configuration: 1/8 at every
probed length 93..5513, prefix cache on and off. Not re-probed here: C=4
mixed concurrency above the bound (below-bound C=4 was already 1/8 after the
MoE fix), and MTP/spec decode (known separate behavior, default off).

## Files changed (uncommitted in the worktree)

- `kernels/gb10/qwen3.8-flash-next/nvfp4/qsa_indexer.cu` (+51/-3): comment
  rewrite + `QSA_TOPK_SORT_MAX` + canonicalising sort.
- `crates/spark-model/src/layers/qsa.rs` (+6): block_topk cap ensure.
- `crates/spark-model/src/layers/qsa_tests.rs` (+5): test module wiring.
- `crates/spark-model/src/layers/qsa_tests_topk.rs` (new): determinism test.

Binaries for A/B kept here: `spark_prefix` (e76b349fe), `spark_fixed`.
