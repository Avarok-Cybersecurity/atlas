# GPU sampler for the wide MTP verify — design (decode lever 5)

Date 2026-09-05. Branch `perf/mtp-gpu-sampler-scaffold` on top of `wip/exl3-research`
(`ffdffa467`). Box dgx-00 (one GB10). Checkpoint `/tank/exl3-ckpt/qwen38-flash-next-4.05bpw`
(vocab 248320, hidden 2560, 48 layers, 512 experts top-10). **Every performance statement in this
document is a hypothesis until the A/B in `ab_mtp_gpu_greedy.sh` has run on the GPU.**

## 0. What is being removed, and what is shipped here

`verify_mtp_wide::finish` (crates/spark-server/src/scheduler/verify_mtp_wide.rs) is the tail of
every K=3/K=4 MTP verify step on the mHC path. The nsys step anatomy in `EXL3_DECODE_PERF.md`
("Launch count" section) shows it as **one 1.55 ms host gap per step** between the verify
lm_head and draft 1 (~1.8% of an ~86 ms step at C=1; it serialises per sequence at concurrency
because the scheduler thread does it inline). The gap is entirely host work on logits that already
live on the device:

| # | host step today (per step, K rows) | cost driver |
|---|---|---|
| 1 | `copy_logits_to_host` of `[K, 248320]` BF16 (1.49 MB at K=3, 1.99 MB at K=4) | D2H + a stream sync |
| 2 | per row: BF16 -> f32 expansion of 248320 entries into `scratch.seq_f32` | 248K scalar ops, host memory bandwidth |
| 3 | per row: `ATLAS_DUMP_LOGITS_PATH` raw dump (env-gated, off) | file I/O when armed |
| 4 | per row: `penalty_params_for(FinalDecode, temp, seed, logit_bias)` | tiny (a Vec clone) |
| 5 | per row: `process_position_logits` = force-temp-zero bypass -> 8 masking stages -> B1 margin scan -> `apply_penalties_and_bias` | up to three full-vocab scans (F2 softmax when armed, B1 top-2, penalties over history) |
| 6 | per row: `sample_with_params_history` = penalties again with NEUTRAL params (no-op) -> at temp<=0 `greedy_pick_last_wins` (one vocab scan) ; at temp>0 top-n-sigma (two scans) -> temperature -> top-k quickselect -> softmax -> min-p -> top-p -> one `StdRng` draw | 1-4 vocab scans |
| 7 | per row: `extract_logprobs_from_f32` when `top_logprobs` is set | log-softmax + top-k scan |
| 8 | `emit_token` between rows | host state machine — stays on the host by design |

Rows are sequentially dependent: row t+1's penalty history, grammar matcher, think/tool state and
seed offset all come from row t's EMISSION. The device already produces the per-row argmax
(`decode_verify_graphed_k3/k4` -> `argmax_bf16` + 4-byte D2H per row, `verify_hc.rs` tail) and
`finish` discarded it.

**Shipped in this commit (part b):** the GREEDY fast path only, opt-in `ATLAS_MTP_GPU_GREEDY`
(presence), per-row eligibility with per-row host fallback, cross-check mode
`ATLAS_MTP_GPU_GREEDY_CHECK`, unit tests. Section 2 specifies it; section 3 is the sampled path
design (not implemented); section 4 lists integration points and tests for both.

## 1. Host step -> device equivalent, step by step

Existing device kernels (kernels/gb10/common): `argmax_bf16` (single row, one CTA, 1024 threads,
strided per-thread first-strict-max then a lower-tid-wins tree), `argmax_bf16_batch` (one block per
row, identical body), `argmax_bf16_batch_lp` (same index semantics plus `log softmax[argmax]` by
online softmax in the same pass), `argmax_fp32`, `embed_from_argmax` / `batched_embed*`, and the
MoE routing top-k family (`moe_topk*.cu`, `moe_gate_topk.cu`, top-k over 512 router logits, NOT a
vocab-scale kernel). There is **no** vocab-scale softmax, top-k, top-p, penalty-scatter or
bitmask-apply kernel in the tree today. The Rust wrappers live in
crates/spark-model/src/layers/ops/sampling.rs; the `Model` trait exposes `argmax_on_device`,
`argmax_batch`, `logits_buffer_ptr`, `copy_logits_to_host`, `logits_ptr_is_fp32`.

| host step | device equivalent | kernel status | equivalence |
|---|---|---|---|
| 1 D2H `[K, V]` | none — logits stay in `buffers.logits()` | — | exact (nothing to compare) |
| 2 BF16->f32 | `__bfloat162float` inside every kernel below | exists (all argmax kernels) | exact: the host `bf16_to_f32` is the same bit-shift widening |
| 3 raw dump | none — a host consumer; fast paths refuse the step | — | n/a (StepGate) |
| 4 `penalty_params_for` | stays host (it reads request policy only) | — | exact |
| 5a force-temp-zero bypass | `argmax_bf16(_batch)` on the raw row | exists | exact when the maximum is unique; **tie-break differs** (host `>` first-wins here; kernel per-thread first then lower-tid) |
| 5b F2 confidence (top-1 prob >= 0.95) | `argmax_bf16_batch_lp`: confident iff `out_logprob >= ln 0.95` | exists | same predicate, different f32 summation order (online vs host two-pass) — a threshold decision can flip at the margin; the state mutation (`consecutive_confident`, `force_end_thinking`) stays on the host |
| 5c MidWord `</think>` mask, PostClose `</think>`/`<think>` mask, ToolDuringThink `<tool_call>` -inf / -12 | (i) host post-check: if the device argmax is one of the <=3 maskable ids the row is host-decided (shipped); (ii) later: a `mask_ids` kernel writing -inf / -12 at <=3 indices before the argmax/sampler (a few bytes) | (ii) new, trivial | exact for (i) by construction: masks that only lower non-argmax ids cannot move the argmax; exact for (ii) with the same f32 ops |
| 5d ForcedThinkEnd injector, PinToToolCall | no logits needed: the pick IS `</think>` / `<tool_call>`; the host decides and emits without any kernel (only `top_logprobs` needs the row) | — | exact |
| 5e ForcedToken (grammar admits one token) | no logits needed (matcher decides) | — | exact |
| 5f Grammar bitmask (`gs.fill_bitmask` -> `apply_bitmask_to_logits`) | H2D of the xgrammar bitmask (`V/32` u32 = 7.8 KB) + `apply_token_bitmask` kernel (-inf where bit clear) — the kernel vLLM/xgrammar ships as `apply_token_bitmask_inplace_cuda` | new | exact (a mask) |
| 5g B1 margin observer (top-1/top-2 gap inside parameter bodies) | a top-2 variant of `argmax_bf16_batch` (carry `(max, idx, second)` through the tree) | new, small | diagnostic only; not on the emission path |
| 5h `apply_penalties_and_bias` (rep divide/multiply, presence/frequency subtract, LZ, DRY, bias add) | host builds the sparse `(id, delta_kind)` list — history ids (already host-resident), LZ/DRY pattern ids (host n-gram search stays host), bias ids — then ONE scatter kernel applies them in the host order: rep -> presence/frequency -> LZ -> DRY -> bias | new | exact if the per-element op sequence is identical (each id receives a handful of f32 ops in a fixed order; no reductions) |
| 6a greedy `greedy_pick_last_wins` | `argmax_bf16_batch` (shipped) or a `_lastwins` twin (per-thread `>=`, tree prefers the HIGHER index) | twin is new | **not exact on ties today** (see 2.3); the twin makes it exact, NaN aside |
| 6b top-n-sigma (mean/var over V) | two block reductions (sum, sum of squares or two-pass) | new | reduction order differs from the host's sequential f32 sums -> threshold can differ at the margin. All four card presets pin `top_n_sigma = 0.0`, so the stage is off on this model |
| 6c temperature divide | in-kernel | new | exact (same division) |
| 6d top-k (host: quickselect + sort of k) | radix select over the f32 bit pattern (histogram passes over V; k=20 in every preset) -> gather survivors -> in-block sort of <=k | new | exact SET of survivors when the k-th value is unique; ties at the k-th value are broken by `select_nth_unstable_by` on the host (unspecified order) so parity there is not a contract even on the host |
| 6e softmax over survivors, min-p, top-p | in-block over <=k survivors (`exp`, prefix sum, cutoff) | new | `expf` vs Rust `f32::exp` differ in the last ulp; the `>=`/cumsum cutoffs can flip at a boundary |
| 6f draw: `StdRng::seed_from_u64(seed).gen::<f32>()` (rand 0.8, ChaCha12), or `thread_rng` when unseeded | host computes the K per-row uniforms (seed = `seq.seed + output_tokens.len() + row`) and passes them as kernel arguments — no device RNG | — | exact draw value; the pick differs only when the CDF boundary moves by the rounding above |
| 7 logprobs | `logprob_of` on device = log-softmax + top-k (reuses 6d/6e machinery) or keep host (needs the row) | new | log-softmax sum order differs at the ulp level |
| 8 `emit_token` | host | — | exact — the row loop stays on the host; only the PICK moves |

**Row chaining.** For the greedy path the only chain is "which row are we on" (rows are
independent given neutral penalties), so the K already-computed argmaxes are sufficient. For the
sampled path with `presence_penalty = 1.5` (the non-thinking preset) row t's pick must be
penalised in row t+1 if it is new to the history: either K chained launches with the pick in
device memory and a device-side scoped-history buffer per slot (append in-kernel), or one
cooperative kernel that walks the K rows in order. Either way the ONLY D2H is `K` u32 at the end,
which is what `decode_verify_graphed_k*` already pays.

## 2. The greedy fast path (implemented)

### 2.1 Claim

At `temperature <= 0` with `penalty_params_for(FinalDecode)` classifying `PenaltyGate::Neutral`,
no grammar, no `top_logprobs`, no host logits consumer, and none of {F2, ForcedThinkEnd,
PinToToolCall} armed for the row, the host emission for a row is

    greedy_pick_last_wins(pipeline(row)) == greedy_pick_last_wins(row)

because every stage that still runs (MidWord, PostClose, ToolDuringThink) can only LOWER the
logit of one of three ids (`ctx.think_end_token`, `seq.think_start_token`,
`ctx.tool_call_start_token`); lowering a non-maximum entry cannot change the argmax; and
`apply_penalties_and_bias` with neutral params is a mathematical no-op (each branch is gated on its
parameter being non-neutral). If the device argmax IS one of the three ids the row is not
decided by the arm (host fallback), so the claim is only ever invoked on rows where the masks were
provably inert.

`Neutral` is the right gate, not `ReduceOnly`: the host path here samples with
`greedy_pick_last_wins` over the PENALISED row, and a reduce-only penalty on a non-argmax token
cannot move the argmax (fast_greedy.rs proof) — that extension is valid but needs the per-row
immunity test (membership in the scoped history + one 2-byte D2H for the `> 0` guard) and is
deferred (2.5).

### 2.2 When it is bit-equivalent, and when it is not

Bit-equivalent to the host pipeline (same emitted token) when ALL hold for the row:

* `ATLAS_MTP_GPU_GREEDY` present, no `ATLAS_LOGIT_DUMP`, `ATLAS_DUMP_LOGITS_PATH`,
  `ATLAS_ADADEC_DIAGNOSTIC`; the verify logits are BF16 (`!logits_ptr_is_fp32`);
* `seq.temperature <= 0.0` (the sampler's own greedy test) or `ATLAS_FORCE_TEMP_ZERO`;
* `seq.top_logprobs.is_none()`;
* `seq.grammar_state.is_none()` (ForcedToken + bitmask stages inert; also no forced picks);
* `classify_penalties(penalty_params_for(seq, FinalDecode, 0.0, None, seq.logit_bias.clone()))
  == Neutral`: `repetition_penalty == 1.0`, `presence_penalty == 0.0`, `frequency_penalty ==
  0.0`, `lz_penalty == 0.0`, `dry_multiplier == 0.0`, and the built `logit_bias` EMPTY — which
  covers the request bias, the tools `<tool_call>` opener nudge (+3.0 / -5 / -10 from
  `sampling_setup`) and the A4 `</think>` floor (-8.0 while `inside_thinking && thinking_tokens <
  min_reasoning_floor`);
* not `f2_active` (`inside_thinking && !force_end_thinking && thinking_tokens >= 400 &&
  watchdog.confidence_early_stop && !disable_watchdogs`);
* not `think_end_inject_armed` (`inside_thinking && (force_end_thinking || defer_hard_override)`);
* not `pin_tool_armed` (`think_just_ended && require_tool_call && !tool_call_opened &&
  !inside_thinking`);
* the device pick is not `ctx.think_end_token`, `seq.think_start_token` or
  `ctx.tool_call_start_token`;
* **the row's maximum is unique** (see 2.3).

Under `ATLAS_FORCE_TEMP_ZERO` the host returns the raw first-wins argmax with no pipeline and no
penalties, so only `top_logprobs`, the host consumers and the tie caveat apply.

Not equivalent — the arm does NOT fire (row goes to the host) when: penalties or bias present
(this includes the whole non-thinking / tools preset at `presence_penalty = 1.5`, and every
tools-active request while the opener nudge is armed); a grammar is live; logprobs requested;
any forced/stateful stage armed; the pick is a maskable id; a host logits consumer is armed;
FP32 logits.

Not equivalent — and the arm DOES fire: an exact BF16 tie at the maximum (2.3). Everything else
that differs between the two paths is diagnostics only: the B1 margin gauge
(`stats.b1_low_margin`) is not fed for device rows, the `ATLAS_MTP_TIMING` phases `Dequant` /
`PipelineProc` / `Argmax` are not recorded for them, and `scratch.seq_f32` is untouched.

### 2.3 Tie-breaking

Three argmaxes with three tie rules are in play: `greedy_pick_last_wins` (LAST index; the host
sampler's rule, pinned by `verify_k2_step/sampling/tests::greedy_verify_uses_serial_tie_breaking`),
`argmax_first_wins` (FIRST index; `verify_pipeline_helper` and the force-temp-zero bypass), and
the kernel (thread `tid` owns indices `tid, tid+1024, ...`, keeps its FIRST strict max, and the
tree prefers the LOWER tid — neither first nor last globally). They agree iff the maximum is
unique. BF16 has 8 significant bits, so two logits within one bf16 quantum at the top (e.g.
within 0.125 of each other at magnitude 16-32) are an EXACT tie; low-margin positions are exactly
where these occur, and `fast_masked.rs` records a temp-0 MTP drift measured 2026-07-11 from this
very difference. The cross-check arm counts these (`mtp_gpu_greedy_check_ties`) separately from
predicate defects (`mtp_gpu_greedy_check_defects`) by reading the two bf16 values from the copied
row: equal values = tie.

To make the arm bit-identical: add `argmax_bf16_batch_lastwins` (per-thread `>=` keeps the last,
tree prefers the HIGHER index on `==`), wire `decode_verify_graphed_k3/k4` to it under the same
lever, and keep the plain kernel for every other caller (the `_lp` kernel's comment explains why a
new kernel beats a new argument). NaN rows remain a corner: the host `max_by` fallback and the
kernel's `>` scan differ; `argmax_first_wins` documents the same edge for the other host path.

### 2.4 Mechanism (what the code does)

* `verify_k3_step` / `verify_k4_step` pass the `[u32; K]` argmax ids the forward returned into
  `finish(.., device_argmax)`.
* `gpu_greedy::StepGate::new(ctx, fp32, raw_dump).admits()` decides once per step.
* `emit_rows` walks rows 0..=K-1 exactly as before (set `seq_len` to the committed prefix, pick,
  `emit_token`, restore `seq_len`, stop on finish or first rejected draft). Before EVERY row it
  calls `gpu_greedy::device_pick(ids, row, seq, ctx)`, which re-reads the live sequence
  (`RowGate::from_seq`) — required because `emit_token` on row t mutates `inside_thinking`,
  `force_end_thinking` (budget), `thinking_tokens` (the F2 threshold), `think_just_ended`,
  `tool_call_opened`, `output_tokens`. An eligible row emits the device id with no logits; an
  ineligible row triggers the `[K, V]` D2H LATE (the buffer is untouched until the next forward)
  and runs `process_seq_logits` for that row and every later ineligible row. The copy happens at
  most once per step and not at all when every emitted row is eligible.
* If the late copy fails after rows were emitted, the prefix stands, the sequence finishes, and
  the normal verdict/rewind path runs with `na = picks.len() - 1`; if it fails before the first
  row the pre-existing failure path runs (broadcast 0, finish).
* Counters on `SpecStats`: `mtp_gpu_greedy_rows`, `mtp_gpu_greedy_host_rows`,
  `mtp_gpu_greedy_check_ties`, `mtp_gpu_greedy_check_defects`. One `info!` line on first
  activation (`stats.once("log:mtp_gpu_greedy")`).
* Kill switch: the arm is OPT-IN by presence of `ATLAS_MTP_GPU_GREEDY` (house convention: `=0`
  still arms it; absence is off). `ATLAS_MTP_GPU_GREEDY_CHECK` (presence) is the cross-check:
  both pickers run, the HOST pick is emitted, disagreements are classified and logged
  (`MTP GPU-greedy check: exact bf16 tie` at info, `... DEFECT #n` at warn — both visible
  under the standard `RUST_LOG=info` serve, which is what the A/B script greps).

### 2.5 Expected effect (hypotheses)

* Removes the 1.49-1.99 MB D2H + stream sync and K host vocab passes from the step tail:
  ~1.5 ms/step at C=1 => ~+1.8% tok/s on the MTP profile (`ab_mtp_gpu_greedy.sh` measures it).
  At C=4 the gap is paid per sequence on the scheduler thread, so the relative gain should be
  larger per step; unmeasured.
* The tie rate on this checkpoint is unknown; the CHECK arm measures it on the 300-token code
  prompt. If it is non-zero the `_lastwins` kernel twin (2.3) is the follow-up before the arm
  can graduate.
* Coverage on real traffic: thinking-preset requests at temp 0 with no tools qualify; the
  non-thinking/tools presets (presence 1.5, opener bias) do NOT — extending to them is the
  penalty-scatter kernel (section 3) or the `ReduceOnly` immunity extension (one 2-byte D2H per
  row, `fast_greedy::logit_is_positive`), in that order of value.

## 3. The sampled path (design only)

Targets the two card presets: thinking `temp 1.0 / top_p 0.95 / top_k 20 / min_p 0 /
top_n_sigma 0` (no penalties) and non-thinking / tools `temp 0.7 / top_p 0.8 / top_k 20 /
presence 1.5`. Rows: K = 3 or 4; the verify logits are `[K, V]` BF16 at `logits_buffer_ptr()`.

### 3.1 Kernels (all new, kernels/gb10/common/verify_sample.cu)

1. `logits_scatter_penalty(row, ids[], kinds[], n, rep, pres, freq, bias[])` — one thread per
   sparse entry, applies the host op order per id: rep (`> 0 ? /= rep : *= rep`), then
   `-= freq*count + pres`, then LZ/DRY deltas (host-computed values), then `+= bias`. Inputs are
   built on the host from the scoped history (`penalty_history_scope`) and the built
   `logit_bias`, uploaded as one small H2D (<= a few KB) per step. Exact reproduction of
   `apply_penalties_and_bias`.
2. `apply_token_bitmask(row, mask[V/32])` — grammar rows (-inf where the bit is clear). Same
   as xgrammar's CUDA kernel.
3. `mask_ids(row, ids[<=3], values[<=3])` — the think/tool masks (-inf or -12.0), so the
   masked-greedy rows in 5c(ii) can also stay on device.
4. `sample_row_topk(row, temp, top_k, top_p, min_p, top_n_sigma, u, out_tok, out_lp?)` — one
   block (1024 threads) per row: (a) optional top-n-sigma mean/var reductions; (b) radix select
   over the f32 bit patterns of `logit / temp` to find the k-th value (k <= 20 in every preset;
   for k=0 skip); (c) gather survivors (<= k, plus ties at the k-th value handled as the host's
   quickselect does: unspecified, so take the first k encountered); (d) in-block sort
   descending; (e) `exp(v - max)`; (f) min-p (`>= min_p * p_max`), then top-p exactly as the
   host loop (`sample_impl.rs` step 6): `cumsum += p_i / sum` in descending order and truncate
   AFTER the first `i` with `cumsum >= top_p`; (g) CDF draw with the host-supplied uniform `u`
   (`threshold = u * sum_survivors`, first `i` whose running `cumsum >= threshold`).
   Optionally write `log softmax[pick]` and the top-`top_logprobs` for the logprobs consumer.
5. Row chaining for presence/frequency: K launches of (1 -> 3 -> 2 -> 4) on the same stream with
   the pick written to device memory and a per-slot device history append kernel (or one
   cooperative kernel that loops K rows). The K uniforms are computed on the host from
   `seq.seed.map(|s| s.wrapping_add(output_tokens.len() + row))` with `StdRng::seed_from_u64`
   (rand 0.8 / ChaCha12, `gen::<f32>()`), so a seeded request draws the identical `u` the host
   would; unseeded requests use `thread_rng` on the host today and would use `rand::random` on
   the host here as well — still host-generated.

Only ONE D2H at the end: `K` u32 picks (+ logprobs if requested). The device path must refuse
the step (fall back to the host loop) when: `ATLAS_LOGIT_DUMP` / `ATLAS_DUMP_LOGITS_PATH` /
`ATLAS_ADADEC_DIAGNOSTIC` armed, FP32 logits, `top_n_sigma > 0` until (a) is validated, a
forced-token/forced-think-end/pin row (host decides those with no logits anyway — the loop can
simply emit them and skip the kernel for that row), F2 armed (needs `out_lp`: use
`argmax_bf16_batch_lp` semantics inside 4 and keep the state mutation on the host).

### 3.2 Equivalence contract for the sampled path

Not bit-identical to the host sampler and not required to be: `expf`, the parallel prefix sums
and the reduction order move a CDF boundary by an ulp, so a pick flips only when `u` lands within
that ulp of a boundary. The contract is the SAME distribution: validated by (i) a fixed-`u`
kernel-vs-host test over random bf16 rows counting boundary-adjacent disagreements only, (ii) the
existing agentic gates under `ATLAS_AGENTIC_SAMPLING=model-card` (webserver_ok, followed_directions,
0 mangling markers), (iii) `echoprobe.py`-style determinism runs with a fixed seed (same seed =>
same tokens on the same binary). Ties at the k-th survivor are already unspecified on the host.

### 3.3 Cost model (hypothesis)

Per row: radix select = 2-3 passes over 0.5 MB of bf16 (L2-resident after the argmax) ~ 5-10 us
of one block, gather + sort + softmax over 20 survivors negligible; K rows serial ~ 30-40 us plus
launch gaps (~2.6 us each on this box). Versus today's 1.55 ms host gap. The win is the same
~1.5 ms/step as the greedy arm, but for the presets that carry real traffic.

## 4. Integration points and tests

### 4.1 Where it plugs in

| site | today | greedy arm (shipped) | sampled arm |
|---|---|---|---|
| `scheduler/verify_mtp_wide.rs::finish` | host copy + `process_seq_logits` per row | `emit_rows` with `device_argmax`, per-row `gpu_greedy::device_pick`, late host copy | `emit_rows` gains a third source: device sample results (`Vec<(u32, Option<TokenLogprobs>)>`) produced by one `model.verify_sample_rows(..)` call before the loop; rows that the loop decides host-side (forced/pin) skip it |
| `scheduler/verify_k3_step.rs`, `verify_k4_step.rs` | dropped the `[u32; K]` argmax | pass `result_vec` | same |
| `scheduler/levers.rs`, `logit_processors/mod.rs::SamplingLevers` | — | `mtp_gpu_greedy`, `mtp_gpu_greedy_check` (presence) | `mtp_gpu_sample` (presence), `mtp_gpu_sample_check` |
| `scheduler/spec_stats.rs` | — | four counters | `mtp_gpu_sample_rows`, `_host_rows`, `_check_boundary_flips` |
| `spark_model::traits::Model` | `argmax_batch`, `copy_logits_to_host` | unchanged | `verify_sample_rows(logits_ptr, k, &RowSampleParams, stream) -> Result<Vec<u32>>`; `argmax_batch_lastwins` |
| `spark-model/src/layers/ops/sampling.rs` | argmax wrappers | unchanged | launch wrappers for 3.1 kernels |
| `kernels/gb10/common/argmax_bf16.cu` | first/lower-tid tie rule | unchanged | `argmax_bf16_batch_lastwins` twin (2.3) |
| `kernels/gb10/common/verify_sample.cu` | — | — | new file, section 3.1 |
| `scheduler/verify_k2_step/sampling.rs::Rows`, `verify_k4_batch_step.rs`, `verify_pipeline_helper.rs` (chat/grammar fast paths) | the other host-copy sites | untouched | candidates for the same `RowSource` once the kernels exist (the batched K=4 site is the C>=2 win) |

### 4.2 Tests

Shipped (`cargo test -p spark-server --bin spark verify_mtp_wide`, 26 pass):
* `gpu_greedy_tests::predicate_admits_only_the_plain_greedy_row` — every single deviation closes
  the arm; `force_temp_zero_bypasses_everything_but_logprobs`.
* `row_gate_reads_penalties_bias_and_the_a4_floor_from_the_sequence` — presence, repetition,
  request bias, the A4 floor (asserted against the SSOT builder, floor width is per-model),
  temperature, logprobs, the three structural ids, the forced/pin arms.
* `step_gate_needs_the_lever_and_no_host_logits_consumer`.
* `device_rows_skip_the_host_copy_and_match_the_host_picks` (K=3 and K=4; the host closure is
  never called; picks and `output_tokens` equal the host path's).
* `ineligible_sequence_ignores_the_device_ids_entirely` (wrong device ids cannot leak).
* `a_structural_pick_falls_back_for_that_row_only` (late copy once; 2 device rows, 1 host row).
* `a_failed_late_copy_keeps_the_emitted_prefix_and_finishes`, `a_failed_first_copy_emits_nothing`.
* `cross_check_emits_the_host_pick_and_classifies_disagreements` (tie vs defect counters),
  `cross_check_ignores_agreement_and_out_of_range_ids`, `device_pick_is_none_past_the_ids`.
* `levers::tests::the_gpu_greedy_arm_is_presence_gated_in_the_resolver` (`=0` arms it).
* The pre-existing `verify_mtp_wide::tests` (host path) run unchanged through `sample_and_emit`.

To add with the kernels (GPU-gated, `cargo test -p spark-model`):
* `argmax_bf16_batch_lastwins` vs `greedy_pick_last_wins` on random bf16 rows with planted
  ties across the 1024-thread stride (indices `i` and `i + 1024*m`), NaN rows excluded.
* `logits_scatter_penalty` vs `apply_penalties_and_bias` bit-for-bit on random rows and
  histories (rep 1.05/1.1, presence 1.5, frequency 0.2, bias entries, window 0 and 256).
* `sample_row_topk` vs `sample_with_params_seeded` at fixed `u` over random rows for the two
  presets; assert equal picks except where the host CDF boundary is within 4 ulp of `u * sum`
  (report the boundary-flip rate).
* `apply_token_bitmask` vs `GrammarState::apply_bitmask_to_logits`.

E2E (operator, GPU): `ab_mtp_gpu_greedy.sh` (this directory) — preset boot, flag the only
variable, `measure_decode.py` x3 at 300 tokens, a 200-token temp-0 sample per arm for byte
equality, the CHECK arm's tie/defect counts from the server log; then the agentic gate
(`benchmark run agentic-webserver` under `ATLAS_AGENTIC_SAMPLING=model-card` with prefix caching
on, as `agentic/run_agentic.sh` does) before any graduation.
