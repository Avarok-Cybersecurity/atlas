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

## DONE — steps 3+4 v1: batched verify + batched propose (fixer round 1, 2026-07-28)

**GATE: PASS (marginal).** C=4 cap=4 = **48.8 / 48.5 / 49.6** (3 scored reps, mean 49.0) vs the
bar 48.5 — up from 25.8 serialized (+90%) and from 43.9 with only the verify batched. C=1 cap=4
25.5-25.6, C=1 default-cap unchanged; coherence + tool-call smokes PASS; MTP sustained (16 K4
summaries), zero errors. Batched MTP is now at parity-to-slightly-ahead of MTP-off at C=4; the
remaining upside (per-seq GDN conv/WY loop = spec step 2, K-vs-batch ladder = step 5) is what
takes it decisively past.

Three fixes landed on top of the eb85ce41 scheduler/model wiring:
1. **Batched cross-sequence PROPOSE** (the measured gap: 12 per-seq drafter forwards x ~5 ms =
   ~62 ms of a ~180 ms step — the BF16 drafter re-reads ~850 MB per forward). One M=n forward
   per draft position: big projections on `dense_gemm_bf16_pipelined` (microbenched 2.7x the
   4x-GEMV loop at M=4), LM head on `w4a16_gemv_batch4`, small per-row ops loop per seq with
   `forward_one`'s exact kernels. `mtp_head/forward_batch.rs`; scheduler defers the verdict
   propose (`K4Hidden::DeferPropose`) and batches it in Phase 4 of `verify_k4_batch_step.rs`.
   Kill switch `ATLAS_NO_MTP_BATCH_PROPOSE`. Measured alone: 46.3-46.8 vs 42.9-43.6 control.
2. **out_proj M>8 dispatch fix** (m_dispatch class): the R=16 verify rows fell into the
   pre-dequanted-FP8 PREFILL arm (2x the weight bytes of NVFP4) — 379 us vs 182 us per call,
   x48 layers = ~9.5 ms/step at C=4. New arm routes M>8 to the NVFP4 transposed-twin tile GEMM
   via `deep_k_gemm` (same kernel the multi-seq decode out_proj uses). Kill switch
   `ATLAS_NO_VERIFY_OUTPROJ_TGEMM`.
3. **Batched argmax** in the R-row verify and the batched propose (1 launch vs R serial one-CTA
   scans; ~2 ms/step).

★ TRAP hit during diagnosis: a stale copy of the profiling script (from a serve whose build
FAILED on un-drained page cache) was still polling :8888; when the next serve came up it fired
its own C=4 drive → 8 active > cap 4 → MTP correctly disabled by `active.len() <= mtp_max_seqs()`
→ a whole nsys profile of the WRONG regime (36.5 tok/s, zero K4 lines) that looked exactly like
"my change killed MTP". Drop caches (`vm.drop_caches=3`) before every serve of this config, and
`pgrep` for stale drivers before ANY benchmark serve.

## NEXT (in order, each measured)
1. Step 2 proper: batch the GDN conv/WY per-seq loop across sequences (landed
   `gdn_verify_fused_conv_kn_batched` + a recurrent/WY sibling): ~5 ms/step of small kernels +
   launch gaps at C=4.
2. K-vs-batch ladder / D-Cut (task #35) before re-raising the cap past 4.
3. Slot-vector-keyed CUDA graph for the batched verify (eager gaps ~22 ms/step at C=4).

## IMPLEMENTED, UNMEASURED — items 1 + 3 above (2026-07-28, build green, no GPU validation yet)

Both landed in one lever set (they share the verify_e forward and the same
slot-vector stability argument):

1. **Slot-vector-keyed CUDA graphs for the batched K=4 verify** (`verify_e.rs` +
   `verify_e2.rs`, cache `verify_batched_graphs` on the model). Captured span =
   layer loop + final norm + lm_head + argmax (the ~4-5k eager launches);
   pre-graph each step = embed, KV-block ensure, metadata/bt H2D, WY-table H2D
   — all into FIXED addresses (decode_a2 pattern); post-graph = the argmax D2H.
   Key = each sequence's ssm-pool slot in batch order + a wy-tables sentinel;
   cache capped at 32 (overflow runs eager). Kill switch
   `ATLAS_NO_MTP_VERIFY_GRAPHS` (PRESENCE). ATLAS_K4_DIAG still forces eager.
2. **Cross-sequence batched GDN conv+WY** (`trait_decode_batched_conv_gdn_multi.rs`):
   the shipped-but-uncalled `gdn_verify_fused_conv_kn_batched` now has its call
   site (one launch, gridDim.y = n, snapshots inline — kills the n×(4 conv +
   3 D2D) loop), and the WY side batches to ONE `gdn_decode_wy4` launch at
   batch_size = n via the existing `state_is_table` pointer-table form.
   Tables ([h|Hi0|Hi1|Hi2] × 4 u64 entries per GDN layer, ~6 KB) are staged by
   the model into a fixed buffer (`verify_wy_tables`) refreshed pre-graph
   every step. Preconditions (consecutive-slot conv layout, intermediates
   0..3 intra-slot contiguous) are checked on the ACTUAL pointers per layer;
   any failure falls back to the per-seq loop (engage/decline logged once +
   counted — grep serve logs for "batched-verify GDN conv+WY"). Kill switch
   `ATLAS_NO_VERIFY_GDN_BATCH` (PRESENCE). Contract delta: the batched conv
   also writes the dead t=K-1 snapshot (pool has num_intermediates = K slots;
   verified before launch).

Left EAGER by design: Phase-1 host picks (policy decision, not graph work),
Phase-2 stash (verdict-dependent row), Phase-3 verdict commits (dynamic index,
secondary stream), Phase-4 batched propose (3 per-position host syncs), the
argmax D2H. NOT ported: table-form wy2/wy3 (K<4 batched verify still refuses
batch>1 — irrelevant for the K=4 path).

Validation still owed (validator): serve + `ATLAS_MTP_MAX_SEQS` sweep vs the
two kill switches, coherence + tool-call smoke, PROOF the batched conv kernel
resolves (the "ENGAGED" log line / nsys instance count), and the accept-rate
telemetry unchanged.
