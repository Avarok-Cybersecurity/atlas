# qwen4_exp concurrent-decode flatline — root-cause diagnosis

Investigation of the C=1..8 aggregate-throughput flatline (12.6 -> 16.2 t/s, TPOT ~linear in C)
on the EXL3 2.05bpw qwen4_exp checkpoint, single node dgx-00, CTX 8192, SEQS 8.
Worktree `/home/ms/atlas/.claude/worktrees/exl3-research` @ 616178062. All experiments used a
throwaway patch that is REVERTED (tree clean); the patch is saved as
`hc_batched_moe_experiment.patch` in this directory, logs/benches alongside it
(`boot1.log`, `boot_prof.log`, `bootB.log`, `bootC.log`, `bench_*.txt`, `probe*.txt`).

## 1. Verdict

**Decode IS batched at the scheduler and dispatch level. The flatline is caused inside the
per-layer mHC (hyper-connection) decode bodies: both layer kinds run the MoE FFN as a
PER-SEQUENCE loop, so the single most expensive component of the step scales linearly with C.**
Proven by a 121-line counter-experiment that routes the same batch through the already-existing
batched EXL3 MoE arm: C=8 aggregate throughput doubles (16.5 -> 32.6 t/s) and the per-step time
drops 404 -> 159 ms, with coherent output and C=1 unchanged. A second, independent defect makes
concurrency collapse entirely past 2051 context tokens (QSA-active batches abandon the batched
path for a per-seq staging loop with a D2H sync per sequence per step) — measured: C=4 at
ISL 4096 falls to 2.0 tok/s/seq (aggregate 4.0) on the SAME build that does 22.8 at ISL 1024.

This is a known-but-unquantified gap, self-documented at boot:
`crates/spark-server/src/main_modules/serve_load.rs:887` — "mHC highway model: concurrency N
via the per-seq highway decode loop (batched highway kernels are the perf follow-up)" (#753
item B). The follow-up was half-built: GDN projections and attention are batched; the MoE loop
(the dominant cost at 2 bpw) was left per-row.

### Evidence chain (what proved it)

Decode is batched — rules out "sequences decoded one at a time":
* `crates/spark-server/src/scheduler/decode_step.rs:82` -> `decode_batch_with_preemption` ->
  `decode_batch_dispatch` (`crates/spark-model/src/model/trait_impl/decode_a2.rs:42`) — ONE
  dispatch per step for the whole batch.
* With `ATLAS_DECODE_BATCH_LOG=1` every joint step logs
  `ATLAS_DECODE_BATCH: n=8 slots=[0..7] contiguous_0..n=true` (boot1.log, 209 n=8 steps).
* `SSM batched projections ACTIVE — QKVZ/out_proj read once per step` — the batched GDN
  projection arm engages (`qwen3_ssm/trait_decode_multi_seq/ssm_batched.rs`).

The per-row MoE loops — the mechanism (file:line):
* SSM/GDN layers (36 of 48): `qwen3_ssm/trait_decode_multi_seq/hc.rs:195-214` — MoE fused at
  n=2 (`forward_k2`), but any other n runs `for i in 0..n { self.ffn.forward(row_i) }`: n full
  single-token MoE forwards (router GEMM + top-10 EXL3 cooperative mgemm pipeline + shared
  expert + blend, per row). The padded ladder (`traits/model.rs:90` = [2,4,8,12,16..]) pads
  n=3 to 4, so the `forward_k3` arm is dead and every C>=3 batch takes the loop.
* Attention layers (12 of 48): `qwen3_attention/trait_impl/multi_seq/mod.rs:391-412` — the hc
  body has ONLY the per-token sequential FFN loop, for every n>=2. The engine's whole batched
  FFN ladder (`multi_seq/ffn.rs`: k2/k3, k4..8 batched GEMV, grouped read-once, token-major)
  is bypassed by hc models.
* The batched arm already exists and is used by every non-hc entry:
  `moe/forward_exl3.rs::forward_exl3_decode(input, n, ..)` — batched router + three cooperative
  mgemms over S=n*top_k slots, slot capacity 5120 (`moe/tables.rs:164`); n=8 is S=80.

Graphs are structurally vetoed — every launch above is paid eagerly every step:
* `decode_a2.rs:278-292` (and the n=1 twin `decode_a.rs:250-263`): `layer_veto` — PLE on GDN
  layers (`qwen3_ssm/trait_layer.rs:46`), QSA on attention layers
  (`qwen3_attention/trait_impl.rs:111`) — plus the `lm_head_exl3` cooperative-launch veto.
  Zero `Captured CUDA graph` lines in any run. PLE/QSA are architectural, so the fully
  materialized control can't graph either: eager is permanent for qwen4_exp as built.

Counter-experiment (the proof, and the counter-proof for the alternatives):
* Patch (saved, reverted): behind `ATLAS_HC_BATCHED_MOE=1`, replace both hc per-row MoE loops
  with one `forward_exl3_decode(normed, n)` + one n-row `hc_post_site`. 3 files, no kernel
  changes, no scheduler changes, no host-sync changes.
* Result (distinct-prompt harness, ISL 1024 / OSL 256): see table below. C=8 step time
  403.9 -> 158.8 ms; aggregate 16.5 -> 32.6 t/s. C=1 identical (12.6 vs 12.8 sTPS — patch
  inert at n=1). Coherence probe: 4 distinct concurrent questions, 4/4 correct answers
  (probeB.txt), identical again with arm C (probeC.txt).
* Because ONLY GPU-side MoE batching changed, this also kills the host-round-trip hypothesis:
  PLE reads host ids (no D2H, `ple/layer.rs:270-276`), QSA below its bound returns before its
  top-k D2H (`qsa.rs:368-372`), sampling is one batched D2H + rayon host pass
  (`decode_logits_step.rs:224`). None were touched, yet the curve moved 2.5x.

Alternatives tested and ruled out:
* Prefix-cache dedupe distorting the numbers: `bench_concurrency.py` DOES send identical
  prompts per slot (`make_prompt` is deterministic), but a distinct-prompt harness reproduces
  the same curve — the flatline is real, not a cache artifact.
* EXL3 dense arms: gate-off control identical (given in the brief); the native-MoE arm is on
  in both configs — the defect is calling it n times, not the arm itself.
* EXL3 `launch_state` host mutex: single scheduler thread submits all decode work; and the
  A/B changes nothing about the mutex yet fixes the curve.
* GDN recurrence per-seq loop: `ATLAS_SSM_BATCHED_RECURRENT=1` (existing engine arm, default
  off, `gdn_flags.rs:95`) measured NEUTRAL on top of the MoE fix (C=8 30.4 vs 32.6 t/s — run
  noise; identical 158.8 ms step median). Partly because it declines when SSM pool slots
  fragment ("SSM batched recurrent DECLINED (n=4): pool slots are not contiguous", bootC.log)
  and partly because the block is small (~28% of ~0.6ms/layer per its own docs).

## 2. Corrected baseline (distinct prompts) and the A/B

Measurement notes: client TPOT under-reports at high C (SSE chunk coalescing merges arrivals
while the token count comes from server usage); the trustworthy numbers are the server-side
step time (`ATLAS_DECODE_BATCH` log timestamp deltas, p50 over 200+ steps) and sTPS, which
agree with each other (1/sTPS matches step p50 within ~5% during pure-decode phases).

ISL 1024 / OSL 256, one request per slot, unique prompt per slot:

| C | control Tput | control sTPS | control step p50 | batched-MoE Tput | sTPS | step p50 |
|---|---|---|---|---|---|---|
| 1 | 11.2 | 12.8 | ~78 ms (1/sTPS) | 11.3 | 12.6 | ~79 ms |
| 2 | 14.9 | 9.5 | 109.5 ms | 15.5 | 9.7 | 104.6 ms |
| 4 | 15.1 | 4.5 | 217.3 ms | 23.9 (+58%) | 7.6 | 121.9 ms (−44%) |
| 8 | 16.5 | 2.4 | 403.9 ms | 32.6 (+98%) | 5.2 | 158.8 ms (−61%) |

* Control fits: step ≈ 11 ms shared + ~49 ms per sequence (linear in C — the reported curve).
  49 ms/seq ≈ 48 MoE sublayers × ~1 ms per single-row MoE forward.
* Patched fits: step ≈ ~85 ms flat + ~9.2 ms per sequence. C=2 gains little by design (SSM
  layers already had the fused K2 arm; only the 12 attention layers' loop changed).
* The original same-prompt sweep numbers were not materially distorted; conclusions carry over.

Long-context cliff (same patched build, arm C boot): ISL 4096, C=4, OSL 64:
aggregate 4.0 t/s, sTPS 2.0/seq — and the `ATLAS_DECODE_BATCH` counter did not advance at all
during the run (523 lines before = 523 after): every step took the per-seq staging loop in
`decode_a2.rs:145-191` (per-seq `decode()` + full-vocab-row `copy_d2h_on_stream` sync per seq
+ graph suppression), selected by the `hc_perseq` gate at `decode_a2.rs:91-100` because
seq_len >= index_topk + index_compress_ratio − 1 = 2048+4−1 = 2051 (indexer_budget 2048,
ratio 4 in the checkpoint's text_config). Above that context, the batched path — and with it
the whole MoE fix — is abandoned for any batch containing one such sequence.

## 3. Per-step budget

Control build, `ATLAS_MS_PROFILE=1` (per-layer host timers, stream sync per layer; overhead
<2% — profiled total 398 ms vs 404 ms unprofiled at n=8; n=1 not measurable this way — the
n==1 path bypasses `decode_batch_compute_main`):

| n | step total | SSM layers (36) | attn layers (12) | lm_head (batched EXL3) |
|---|---|---|---|---|
| 2 | 108.1 ms | 76.1 ms (2.11/L) | 30.5 ms (2.54/L) | 1.6 ms |
| 8 | 398.2 ms | 301.4 ms (8.37/L) | 95.1 ms (7.93/L) | 1.7 ms |

* Per-layer increment n=2 -> n=8 is ~1.0 ms/row/layer on SSM layers and ~0.9 on attention —
  the cost of one per-row MoE forward; ~48 × n of these dominate the step. lm_head is flat in
  n — direct proof that batching works where it is wired.
* Scheduler + sampling + metadata + embed overhead: step_total − (ssm+attn+head) ≈ 4-6 ms.
* Post-patch residual slope ~9.2 ms/seq/step is distributed: per-row PLE injection (36 L),
  QSA ingest GEMV (12 L), per-seq GDN recurrence kernels (36 L), per-seq attention KV work,
  n-row host sampling, embeds, plus the batched MoE arm's slot-linear compute (S = 10n).
  Splitting that finely needs per-phase timers or nsys (not done — see §5).
* GPU-side saturation cross-check (user observation during the patched C=8 run): SM util
  ~88-90% at ~22 W — consistent with an eager step of thousands of small, bandwidth-bound
  M≈1..8 kernels with launch gaps, not a compute-saturated GPU. Batching raises M and removes
  launches; graphs are off the table while EXL3 cooperative kernels + PLE/QSA vetoes stand.

## 4. Fix plan, ranked

1. **Wire the hc decode paths to the batched MoE arm (quick win, measured +98% at C=8).**
   What: replace the per-row loops with one n-row MoE call + one n-row `hc_post_site`:
   - `crates/spark-model/src/layers/qwen3_ssm/trait_decode_multi_seq/hc.rs:195-214`
   - `crates/spark-model/src/layers/qwen3_attention/trait_impl/multi_seq/mod.rs:391-412`
   For production, call `forward_token_major_decode(input, n)` (moe/forward_token_major.rs:19)
   rather than the exl3-only helper in my patch: it already delegates EXL3 -> forward_exl3_decode,
   NVFP4 -> token-major kernels, LoRA/other -> forward_batched, so every quant gets a defined
   path; keep the per-row loop only as the final fallback. Delete the now-redundant n==2/3
   special cases (n=3 is already dead due to the pad ladder).
   Expected: C=8 16.5 -> ~32 t/s aggregate, TPOT p50 ~404 -> ~160 ms (measured on EXL3 2.05bpw);
   the NVFP4 canonical config should see the same shape of win but MUST be re-measured (its
   per-row arm and batched arm have different relative costs; 512-expert grouped-vs-pairwise at
   small n was never settled — `ATLAS_MOE_GROUPED_ROUTED_DECODE` exists for exactly that A/B).
   Risk: numerics on the routed blend (same kernels as the shipping k2/k3 and non-hc paths;
   my 4-question coherence probe passed, but run the standard 3/3 agentic gate + a BFCL smoke
   before defaulting). Validate: distinct-prompt sweep + step-time medians from
   `ATLAS_DECODE_BATCH_LOG=1`, plus the agentic gate.
   **This is the single change I would make first.**

2. **Fix the >2051-token concurrency collapse (real work, big win for agentic/long-context).**
   The `hc_perseq`/`qsa_active` gate (`decode_a2.rs:91-100,145-191`) throws every batch with
   one QSA-active sequence onto per-seq `decode()` + a per-seq D2H sync. Measured: 2.0 tok/s/seq
   at C=4/ISL 4096 vs 7.4 at ISL 1024 (patched build). Fix = per-sequence QSA selection support
   in the batched ms attention path (`multi_seq/attn.rs` — per-row `decode_select` already runs
   there for ingest; the missing piece is consuming a per-row `Some(selection)` in
   `ms_phase_paged_decode` instead of refusing). Until then, even fix #1 only helps short
   contexts. Expected: restores the (post-fix-#1) curve at long context, i.e. ~3-4x at C=4
   ISL>=4K. Risk: attention correctness under selection — needs the QSA parity tests.
   Interim mitigation: none good — capping context per batch reintroduces the same cliff.

3. **Batch the remaining per-row loops in the hc bodies (medium, ~grinds the 9.2 ms/seq slope).**
   PLE row loop -> one n-row injection (`hc.rs:82-114`; PLE's forward already takes
   num_tokens>1 — needs a per-row state/carry variant); QSA ingest n rows -> one batched qk
   GEMV (`multi_seq/mod.rs:243-265`); GDN recurrence: `ATLAS_SSM_BATCHED_RECURRENT=1` measured
   neutral, so first make its contiguity precondition hold (slots fragment as sequences finish
   — either compact slots on free or lift the strict [0..n) base+stride requirement in
   `ssm_batched_recurrent.rs:86-146`) before re-measuring. Expected: maybe another 15-25% at
   C=8 (residual slope 9.2 -> ~5 ms/seq); each piece is small, so measure per piece.

4. **Do NOT bother (tested/settled):** `ATLAS_SSM_BATCHED_RECURRENT=1` as-is (neutral);
   client-TPOT-based tuning (measurement artifact — fix the harness first, as done here);
   chasing decode graphs for this model while EXL3 native + PLE/QSA vetoes stand
   (`decode_a2.rs:278-292` — three independent structural vetoes).

5. **Bigger levers beyond batching** (context for prioritization): with fix #1+#3 the step is
   still ~85 ms of shared eager work at 2 bpw (weight-read-bound trellis GEMMs at tiny M).
   The lever that raises arithmetic intensity across the board is speculative decode (the MTP
   draft head already works per qwen4exp-mtp-scope), which multiplies rows per weight read in
   verify — orthogonal to and compounding with this fix.

## 5. Not settled / what it would take

* **Fine split of the residual 9.2 ms/seq slope** (PLE vs QSA vs recurrence vs attention vs
  sampling): needs env-gated per-phase timers inside the two hc bodies or an nsys pass
  (nsys not attempted on this box during the window). ~1 rebuild + 2 short runs.
* **NVFP4-config replication**: the defect is code-path-identical for the canonical NVFP4
  serving config (the hc loops don't branch on quant), but the win size there is unmeasured;
  one boot of the NVFP4 checkpoint with the production fix behind a flag settles it.
* **Whether the Holo/35B "flatline C=4->C=8" is the same mechanism: it is NOT.** Qwen3.5-35B
  is non-hc; its batched decode uses the ffn.rs ladder + default-on multi-seq CUDA graphs
  (`decode_a2.rs:36-39`), none of which qwen4_exp can reach. Same symptom, different disease
  (35B's is expert-read bandwidth + kernel-chain overhead per the existing holo notes). Code
  evidence only — I did not boot a 35B (one boot would confirm its batch log + graph lines).
* **Why arm B C=2 improved only ~5%**: expected (SSM layers already fused at n=2; only 12
  attention-layer rows changed) — consistent, not investigated further.
* sTPS vs step-p50 at C=8 differ ~15% (5.2 -> 192 ms vs 158.8 ms measured): sTPS averages in
  drain-phase steps at smaller n and inter-step scheduler gaps; treated as bounded uncertainty.

## Hygiene

* Worktree left clean (`git status` empty); `target/release/spark` rebuilt from the pristine
  tree after the revert. Experiment patch preserved at
  `/home/ms/.claude/jobs/5a7bd33d/tmp/cinvest/hc_batched_moe_experiment.patch`.
* All serves killed; memory verified >100 GB available before each boot; MemAvailable watchdog
  ran throughout (boot script built-in).
