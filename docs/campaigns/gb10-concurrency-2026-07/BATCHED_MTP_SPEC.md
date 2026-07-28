# Batched speculative decoding — implementation spec

**Status: green-lit, NOT started.** This is the only remaining lever sized to close the C=8/16 gap.
Everything below is grounded in measurements taken 2026-07-27/28 or in the vendored vLLM checkout.

## Why this and nothing else

Measured position after five shipped kernel wins (+15% at C=16):

| C | Atlas | vLLM | ratio | MTP |
|---|---|---|---|---|
| 1 | 25.45 | 14.2 | **1.792x** | **ON** |
| 2 | 26.20 | 27.8 | 0.942x | off |
| 4 | 48.65 | 53.3 | 0.913x | off |
| 8 | 74.65 | 98.8 | 0.756x | off |
| 16 | 130.75 | 168.9 | 0.774x | off |

**We beat vLLM by 1.79x at exactly the one concurrency where MTP runs, and lose at every
concurrency where it does not.** That is the whole gap.

The kernel route is exhausted, measured three ways: 87% of the step is four kernels at 86-100% of
achievable bandwidth; the entire remaining byte-identical basket is ~1.5 ms of a 110 ms step; and
MMQ cp.async is dead on the same occupancy mechanism that made a 4-stage pipeline a **-5%**
regression. Five wins totalled +15%. C=8/16 need **+29-32%**.

## Why it is not a constant change (measured)

`ATLAS_MTP_MAX_SEQS` is a GUARD, not a knob:
- cap=2 at C=2: **25.45** vs cap=1 (MTP off) **26.35** => enabling it LOSES 3.4%. Default is now 1.
- cap=4 at C=4: **25.8** vs 48.5 => **HALVES** throughput. Output coherent and identical, so this is
  SERIALIZATION, not corruption.

The verify runs per-sequence, so its cost scales with concurrency while the drafting benefit does
not. At C=1 the extra K tokens ride nearly free on a bandwidth-bound step; at C=2 the second
sequence's verify is additive and overtakes the gain.

## Target design (vLLM V1 shape; blueprint in `scratchpad/vllm-src/`)

1. **Verification IS the normal decode forward.** Draft tokens are appended to each request's
   scheduled tokens; the ragged batch goes through the ordinary varlen path. No separate verify
   pass. See `gpu_model_runner.py:2743` `_calc_spec_decode_metadata` (handles mixed draft lengths,
   e.g. `[3, 0, 2, 0, 1]`); uniform-decode graph captured at query length 1+k
   (`gpu_model_runner.py:817`).
2. **Batched rejection sampling** — one Triton program per request over `[batch, k]`, emitting
   variable-length accepted prefixes plus bonus tokens (`vllm/v1/sample/rejection_sampler.py`). We
   run temp 0.0, so the greedy path suffices.
3. **Per-request rewind is scheduler arithmetic** — `num_computed_tokens -= num_rejected`;
   rejected KV is overwritten next step (`scheduler.py:1547-1571`). No trees needed; vLLM uses
   linear chains.
4. **GDN state rollback is an INDEX LOOKUP, not recomputation.** `gdn_attn.py` allocates
   **num_spec+1 recurrent-state slots per sequence** (`spec_state_indices_tensor [batch, num_spec+1]`)
   and the FLA kernel writes a checkpoint per draft position INLINE (`INPLACE_FINAL_STATE`,
   `fla/ops/fused_recurrent.py:104-166`); the next step loads slot `num_accepted[i]-1`.
   ★ This is the piece that makes the whole thing viable — it eliminates exactly the per-token FLA
   overhead that killed the earlier TRT-LLM ngram v21 attempt (73% rejection, 1.5x SLOWER than
   baseline). The checkpoints are a byproduct of one fused kernel, not extra passes.
5. **Dynamic K-vs-batch ladder** — ship WITH the above, never after. vLLM's documented EAGLE3
   ladder is K=5 at batch 1-16, K=4 at 17-32, K=3 at 33-64. Fixed large K on a saturated engine is
   what produced vLLM's historical 1.4-1.8x SLOWDOWNS (SmartSpec, arXiv 2406.14066).

## Local code sites

| what | where |
|---|---|
| scheduler gate | `crates/spark-server/src/scheduler/mod.rs:551` (`active.len() <= mtp_max_seqs()`) |
| cap constant | `scheduler/mod.rs:189-197` (`mtp_max_seqs`, now defaults 1) |
| spec step driver | `crates/spark-server/src/scheduler/spec_step.rs:94, 266` (`decode_verify`, `decode_verify_graphed`) |
| verify dispatch (per-K) | `crates/spark-model/src/model/trait_impl/verify_a.rs:31`, `verify_b.rs:39`, `verify_c.rs:47`, `verify_c2.rs:36` (K=4), `verify_d.rs:40` (K=gamma) |
| **the missing piece** | a BATCHED `decode_verify` taking `n` sequences x (K+1) rows. The model trait has only single-sequence forms. |
| MTP single-seq assumptions | `crates/spark-model/src/model/mtp_carry.rs:37`; one MTP slot at `model/types.rs:187` |
| prefill-continuation gate | `scheduler/phase_continue_prefills.rs:101` (also `active.len() == 1`) |
| multi-seq decode (the path to generalize) | `crates/spark-model/src/model/trait_impl/decode_a2.rs` — already handles n sequences x 1 token with `padded_n` rows, `meta.positions`, `meta.slot` |

**The natural implementation** is to make the verify reuse the multi-seq decode path with
`padded_n = n*(K+1)` rows rather than building a new kernel path: that is precisely vLLM's
"verification is just the decode forward" property, and our multi-seq path already does ragged
per-sequence positions and slots.

## Expected outcome

Measured elsewhere: EAGLE-3/SGLang H100 **B=32: 1.30x TPOT, 1.70x aggregate**; EAGLE-3 paper 1.38x
at B=64; MagicDec 1.18-1.91x at B=32 with **speedup GROWING with batch in the memory-bound regime**
— GB10 qualifies (FFN at 87% of achievable bandwidth; weights are read once per step regardless of
batch, so k+1 tokens per weight-read is nearly free).

Applied to 130.75 at C=16 => **170-222 tok/s**, i.e. clearing vLLM's 168.9 and the campaign goal.

★ It is UNKNOWN whether the vLLM 168.9 reference itself ran with speculation. If it did not, this
does not merely close the gap — it is how Atlas passes it.

## Gates

- Coherence + tool-call smoke at C=1,2,8,16 (spec paths change emitted tokens by construction —
  `spec_not_output_neutral`; do NOT expect byte-identity).
- Acceptance-rate telemetry per K and per batch size before tuning the ladder.
- C-sweep vs the kill switch, >=3 reps, ranges must not overlap.
- Accuracy (BFCL/IoU) only AFTER parity — the standing embargo.

---

# PROGRESS

## DONE — step 1: batched verify conv (commit `5b4c40cb`)
`gdn_verify_fused_conv_kn_batched` in `kernels/gb10/common/gdn_verify_fused_conv_kn.cu`
(verified: NO 27B shadow, common/ is live), plus:
- wrapper `ops::gdn_verify_fused_conv_kn_batched` (`layers/ops/ssm_mamba.rs`), grid
  `(ceil(d_inner/256), n_seq, 1)`, four extra per-sequence stride args
- handle `gdn_verify_fused_conv_kn_batched_k` (`qwen3_ssm/init.rs:~236`, field in `mod.rs:~162`)
Additive and UNCALLED — HEAD behaviour unchanged (kernel audit clean, C=16 132.5).
Bit-identical to n separate launches: per-sequence conv windows are independent, so the per-token
sequential loop is untouched; only base addresses move.

## NEXT — step 2: batch the recurrent scan
Consumer of the conv is `qwen3_ssm/trait_decode_batched_conv_gdn_wyn.rs:89` (`fused_conv` gate).
The recurrent/WY side needs the SAME `gridDim.y = n_seq` treatment plus per-sequence state strides.
★ Check `ATLAS_GDN_FUSED_CONV17` (`:92`) — the fused path is gated by it; do not assume it is on.
★ Memory records GDN multi-token VERIFY is SUPERLINEAR in K on strix (85/249/623 ms at K=1/2/3).
   Batching across sequences does NOT fix superlinearity in K — it fixes the n-fold WEIGHT re-read.
   Size the two separately.

## THEN — steps 3-5
3. Batched `decode_verify` on the model trait. **The multi-seq decode path
   (`model/trait_impl/decode_a2.rs`) is the natural host** — it already carries per-row
   `meta.positions` / `meta.slot` and `padded_n` rows, which IS vLLM's "verification is just the
   decode forward" property. Feed it `n*(K+1)` rows: FFN/projections/lm_head are
   position-independent and batch as-is; paged attention already takes per-row block tables and
   seq lens, so give each of a sequence's K+1 rows that sequence's block table with
   `seq_len = base + j`.
4. Batched rejection (greedy suffices at temp 0.0) + per-request rewind
   (`num_computed_tokens -= num_rejected`; rejected KV is overwritten next step).
5. Re-raise `mtp_max_seqs` (currently 1) and add the K-vs-batch ladder. **Do not raise the cap
   before 1-4 land** — measured, it HALVES C=4.

## Gate before believing any of it
`ATLAS_MTP_MAX_SEQS=4` at C=4 must go from 25.8 (today's serialized number) to >48.5 (today's
MTP-off number) before the lever is real.
