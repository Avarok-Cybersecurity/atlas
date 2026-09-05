// SPDX-License-Identifier: AGPL-3.0-only

//! K-row speculative verify for models carrying an mHC highway.
//!
//! # Why this exists
//!
//! `decode_verify_dispatch` (verify_a.rs) verifies K tokens by running the
//! attention layers per-token through `decode()` and the SSM layers through
//! `decode_batched()`. Under an mHC highway that second call REFUSES:
//! `refuse_batched_under_hc` fires because the batched paths keep their own
//! residual bookkeeping and the highway replaces it, so running them would add
//! every block output to the residual twice. The scheduler turns that error into
//! `a.finished = true` — a SILENTLY TRUNCATED RESPONSE, not a fallback. So
//! speculation has been unavailable on this model class, for ANY proposer: the
//! same refusal sits under DFlash's batched verify (`verify_e.rs` routes the GDN
//! conv+WY body through `decode_verify_multi`).
//!
//! # The shape
//!
//! The only working multi-row mHC path is `prefill_inner_hc`, and BOTH layer
//! types have one (`qwen3_ssm/trait_prefill_hc.rs`,
//! `qwen3_attention/trait_impl/prefill_inner.rs:531`). `prefill()` dispatches to
//! it whenever `self.hc.is_some()`. So a K-row verify is expressible as a
//! MINI-PREFILL of the K candidate tokens at positions
//! `[seq_len, seq_len + K)`.
//!
//! Running EVERY layer through prefill was the FIRST cut, on the belief that
//! mixing per-token attention decode with a K-row SSM body could not work: the
//! highway buffer is laid out `[T, hc, H]`, and a 1-row attention path was
//! assumed to disagree with a K-row SSM path about what row a stream belongs
//! to. That is not true, and the belief cost correctness. `hc_row_offset` is
//! already a ROW INDEX into that same buffer -- `prefill_inner_hc` addresses
//! the highway at `hc_row_offset * hc_mult * H * 4`
//! (`qwen3_attention/trait_impl/prefill_inner.rs:565`), and the K-row GDN body
//! at `trait_decode_batched_hc.rs:103` uses the identical expression. Teaching
//! `decode_inner_hc` the same arithmetic (it had hard-coded row 0) makes a
//! one-row decode body at `hc_row_offset = t` land on exactly the row a K-row
//! body would have written. So the two layouts ARE the same layout, indexed
//! the same way.
//!
//! That is what the default path now does: the 12 attention layers run as K
//! sequential one-row `decode()` bodies (rows 0..K), the 36 GDN layers run as
//! ONE K-row `decode_batched()` pass. Both write the same `[T, hc, H]`
//! highway. It matters because verify row 0 re-processes a token a serial
//! decode already committed, and prefill attention (chunked/flash paged
//! kernel, GEMM projections, grouped MoE, QSA `prefill_ingest`) is a
//! different reduction order from decode attention (paged-decode GEMV, GEMV
//! projections, decode MoE, QSA `decode_select`) -- equivalent in exact
//! arithmetic, not in bf16.
//!
//! Kill switch: `ATLAS_QWEN4EXP_MTP_HC_ATTN_DECODE=0` restores the K-row
//! `prefill()` body for the attention layers.
//!
//! # MEASURED END TO END (2026-08-28) — RUNS, BUT WRONG AND SLOW
//!
//! With the proposer armed (`--speculative --num-drafts 1` +
//! `ATLAS_QWEN4EXP_MTP_VERIFY=1`), 4K ctx, greedy, vs a same-config baseline:
//! ```text
//!   baseline      19.8 tok/s  (50.5 ms/token)  correct output
//!   speculative,
//!     before      ~4.9 tok/s  (205 ms/token)   degenerate output
//!     after        8.3 tok/s  (120 ms/token)   degenerate output
//!   errors: 0
//! ```
//! The chain is COMPLETE — draft, verify, rollback and both carries run without
//! a single error, which no earlier revision managed. Two problems remain. The
//! COST one is now largely understood and 1.7x better (item 2); the CORRECTNESS
//! one is still open and its leading hypothesis has been disproved (item 1).
//!
//! 1. CORRECTNESS - STILL OPEN, and the leading hypothesis was TESTED AND
//!    DISPROVED. Four arms, same prompts, greedy, 4K ctx:
//!    ```text
//!      spec off (baseline)              "Red, blue, and green."   coherent
//!      spec on, rollback off            "Red light")..."          diverges ~tok 2-3
//!      spec on, rollback on             "Redaccion, 1."           diverges ~tok 2
//!      spec on, rollback on, old MoE    "Redaccion, ..."          diverges ~tok 2
//!      rollback errors: 0   panics: 0
//!    ```
//!    Read these carefully, because two plausible culprits are ELIMINATED:
//!
//!    * The missing rollback was the leading suspect - `rollback_verify_hc` was
//!      written but NOTHING CALLED IT, so a rejected draft left the aux carries
//!      un-restored. It is now wired (`Model::rollback_verify_rows`, called from
//!      the scheduler's K=2 reject branch) and it changes NOTHING: armed and
//!      unarmed diverge at the same point. It ships OFF
//!      (`ATLAS_QWEN4EXP_MTP_ROLLBACK=1` to arm) as unproven, not as harmful.
//!    * The small-M FFN substitution below is likewise exonerated - forcing the
//!      OLD grouped-MoE verify reproduces the identical corruption.
//!
//!    Note also that the "first ~12 tokens match the baseline" behaviour an
//!    earlier revision recorded DOES NOT REPRODUCE under this harness; every
//!    speculative arm diverges within 2-3 tokens. Treat the 12-token figure as
//!    prompt-specific and do not reason from it.
//!
//!    Divergence that early, with leaked raw special-token ids in the output
//!    (`| 100257`, `<|fim_prefix|>`), is a wrong-LOGITS signature rather than a
//!    slow state leak - the verify appears to return bad rows from nearly the
//!    first step, which no rewind can repair. Next suspects, in order: (a) the
//!    K-row logits the mini-prefill hands back - row indexing/aliasing into the
//!    logits buffer, the defect class this repo has hit repeatedly; (b) what
//!    `apply_aux_states` restores, PLE's rolling conv/history window especially,
//!    since unlike QSA's contiguous marks it cannot be rebuilt by truncation;
//!    (c) the scheduler's accepted-row bookkeeping vs what this verify advances.
//!    A row-by-row A/B of verify logits against a serial decode of the same
//!    tokens would settle (a) immediately and is the cheapest next experiment.
//!
//! 2. COST. Measured, gamma=1:
//!    ```text
//!      decode step           50.5 ms
//!      draft forward          2.6 ms   (shadow-on 53.1 vs baseline 50.5)
//!      verify (before)      ~395 ms  -> 205 ms/token end to end
//!      verify (after)                   120 ms/token end to end
//!    ```
//!    ★ THE DRAFT IS ESSENTIALLY FREE - 5% of a decode. The economics are
//!    entirely about verify. Break-even at ~91% accept needs
//!    `draft + verify < 95 ms`.
//!
//!    AN EARLIER REVISION OF THIS BLOCK CALLED THAT STRUCTURALLY BLOCKED, on
//!    the theory that the GDN prefill floor made a 2-row verify cost what a
//!    large chunk costs. PROFILING DISPROVED IT. Per-layer, per-verify-row:
//!    ```text
//!                  before    after
//!      moe        2700 us    191 us   (14x)
//!      gdn_block   860 us    862 us   (unchanged)
//!    ```
//!    The dominant term was never the GDN. It was the MoE: `forward_prefill`
//!    routes through the grouped GEMM, which streams every one of the 512
//!    experts' weights regardless of row count, so ONE row paid nearly what a
//!    28-row chunk paid (T=16 6.7-9.6 ms, T=28 8.5-12.3 ms -- 1.75x the rows
//!    for 1.2x the time). Substituting the single-token/K=2/K=3 MoE kernels at
//!    small row counts (`ATLAS_QWEN4EXP_HC_SMALL_M_FFN`, default on) cut it 14x.
//!
//!    NOTE the K=1 arm is the one that matters: `decode_verify_hc` splits a
//!    verify into row-0-then-drafts, so at gamma=1 BOTH calls arrive as a
//!    single row and the k2/k3 arms never fire.
//!
//!    WHERE IT STANDS: 120 ms/token vs a 50 ms decode -- speculation still does
//!    not pay, but it is now ~2.4x rather than ~4x, and the remaining cost has
//!    moved to the GDN: 36 layers x 862 us x 2 rows ~= 62 ms.
//!
//!    NEXT LEVER, and it is the same shape as the fix above: at T=1 a "prefill"
//!    row under the highway is just a decode step, so the hc decode body
//!    (`qwen3_ssm/trait_decode_hc.rs`) should serve it instead of the chunk
//!    scan. That is a 1-row substitution -- it does NOT require the batched
//!    multi-row GDN feature (#753 item B) that the earlier conclusion pinned
//!    this on. A batched K-row step remains the better endpoint, since two
//!    serial decodes (~101 ms) still exceed the ~92 ms budget on their own.
//!
//! Speculation therefore stays behind BOTH `--speculative` and
//! `ATLAS_QWEN4EXP_MTP_VERIFY=1`, and neither is a default.
//!
//! # THE THREE CARRIES, AND WHAT LANDS EACH (2026-09-03)
//!
//! A K-row mini-prefill advances THREE pieces of per-sequence state one row at
//! a time. A partial accept keeps `num_accepted` of those rows and discards the
//! rest, so all three must be walked back to the SAME boundary — "state after
//! row `num_accepted - 1`". They need three different mechanisms, which is why
//! one blanket "restore the snapshot" never worked:
//!
//! | carry | mechanism | published by | landed by |
//! |---|---|---|---|
//! | SSM `h_state`/`conv_state` | per-row publish into the pool intermediates | `publish_verify_row_state` | `commit_accepted_prefix` |
//! | PLE conv + n-gram history | per-row SNAPSHOT | `collect_verify_aux_states` | `restore_verify_aux_at` |
//! | QSA `ingested`/`pooled` | ABSOLUTE mark rewind, no blob | — | `align_verify_aux_states` |
//!
//! * **QSA is a mark rewind.** `ingested`/`pooled` are contiguous marks and
//!   both device buffers are written forward from them, so moving the marks
//!   back is sufficient; stale bytes past the mark are overwritten by the next
//!   ingest. It is also the one that MUST NOT be snapshotted: the blob carries
//!   `ingested * head_dim * 2` bytes of raw keys PER ATTENTION LAYER, which at
//!   context is megabytes through the host on every speculative step.
//!   `Layer::aux_rewind_is_exact` is what routes it here.
//! * **PLE needs a SNAPSHOT.** `PleSeqState::conv` is a rolling FP32 device
//!   convolution state and `history` is a fixed-length window whose oldest
//!   entries have already rolled off, so neither can be reconstructed by
//!   truncation. It is one layer and a ~100 KB blob, so per-row is cheap.
//!
//! ## ★ The index is the whole fix
//!
//! The earlier cut took ONE snapshot, after row 0, and restored it
//! unconditionally from the K=2 reject branch. Two things were wrong with that,
//! and the second is the one that explains the severity gradient:
//!
//! 1. It was DEFAULT-OFF (`ATLAS_QWEN4EXP_MTP_ROLLBACK=1` to arm), so on the
//!    default path a rejected row left both carries permanently advanced.
//! 2. Row 0 is the right snapshot only for a ONE-ROW commit. `verify_k3_step`
//!    computes `num_accepted <= 2` against `k = 3`, so EVERY K=3 step is a
//!    partial accept, and its two-row commit needed row 1. Restoring row 0
//!    there left the carries one row BEHIND the SSM state the very same commit
//!    had just rewound — the two halves of one commit landing on different
//!    tokens.
//!
//! So the stash is now per-row over `hc_publish_rows(k)` — the SAME range the
//! SSM intermediates are published over — and the restore index is
//! `commit_rewind_index(num_accepted)`, the SAME function
//! `commit_accepted_prefix` uses. `verify_aux_restore_row` is the named pairing;
//! `aux_restore_row_tracks_the_ssm_rewind_index` is the CPU test that fails if
//! the two ever drift.
//!
//! The absolute base matters too. `verify_hc_rows` advances `seq.seq_len` by K
//! and the scheduler's reject branches rewind it at different points, so
//! `VerifyAuxRows::base_pos` is captured BEFORE the pass and the QSA alignment
//! target is `base_pos + num_accepted`, never a delta off a moving `seq_len`.
//! `restore_verify_aux_at` asserts the two agree.
//!
//! DEFAULT ON, kill switch `ATLAS_QWEN4EXP_MTP_AUX_COMMIT=0`. Callers must
//! still `checkpoint_ssm_states` before the verify, exactly as the non-hc path
//! requires.
//!
//! ## Call sites
//!
//! Every scheduler branch that calls `commit_accepted_prefix(n, k)` for an mHC
//! model calls `commit_verify_aux(n, k)` beside it with the same arguments:
//! `verify_k2_step` (both branches), `verify_k3_step` (all three) and
//! `verify_k4_verdict` (both). One shared helper,
//! `commit_verify_aux_or_finish`, so the K=2/K=3/K=4 copies cannot drift.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::types::{TransformerModel, VerifyAuxRows};
use super::async_chkpt::commit_rewind_index;
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::ops;
use crate::traits::SequenceState;


impl TransformerModel {
    /// True when K-row verify must take the mHC path.
    pub(super) fn verify_needs_hc_path(&self) -> bool {
        self.config.hc_mult > 0
    }

    /// Verify `tokens` by mini-prefill, SPLIT so a rejected draft can be rolled
    /// back exactly.
    ///
    /// Row 0 is the already-sampled real token and is always kept; rows 1.. are
    /// the drafts. The PLE carry is snapshotted after EVERY row a partial
    /// accept can land on, not just after row 0, so `commit_verify_aux` can
    /// restore the row the commit actually kept. A single row-0 snapshot is
    /// correct only for a one-row commit; every wider partial accept restored
    /// it a row SHORT of the sequence, which is a silent desync, not an error.
    pub(super) fn decode_verify_hc(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let k = tokens.len();
        if k == 0 {
            return Ok(Vec::new());
        }
        let stream_d = self.gpu.default_stream();

        // ── ROW-AT-A-TIME so the per-row SSM intermediates exist ──
        //
        // MEASURED ROOT CAUSE (2026-09-03). `commit_accepted_prefix` rewinds
        // every GDN layer's live `h_state`/`conv_state` from
        // `ssm_pool.h_intermediate(l, slot, num_accepted - 1)` — buffers that
        // are written ONLY by the fused batched verify kernels in
        // `qwen3_ssm/trait_decode_batched_conv_gdn*.rs`. This mHC verify runs
        // every layer through `prefill()` and writes NONE of them, so a
        // partial accept copied NEVER-WRITTEN pool memory into 36 layers of
        // live recurrent state.
        //
        // The arithmetic says which widths are hit: `verify_k3_step` computes
        // `num_accepted` from TWO drafts, so it is at most 2 against `k = 3`
        // — EVERY K=3 step took the partial-accept branch and corrupted the
        // state. K=2 only corrupted on a REJECT (`(1, 2)`); its accept branch
        // passes `(2, 2)`, which `commit_accepted_prefix_dispatch`
        // short-circuits. Hence the observed split: gamma=1 mostly coherent
        // with occasional dropped tokens, gamma=2 degenerate from step one.
        //
        // The evidence that this, and not the K-row pass, is the defect:
        //   K3 verify: tokens=[1156,369,29350] -> v=[369,9859,391]
        //              drafts=[369,29350] accepted=1
        // against a serial decode of `1156, 369, 9859, 364, ...`. Rows 0 and 1
        // MATCH serial exactly; row 2's input was the REJECTED draft 29350,
        // not 9859, so its 391 is a correct logit row for a different token.
        // The verify was right; the commit that followed it was not.
        //
        // Fix: run one row per pass and publish the live state into
        // `h_state_intermediates[t]` / `conv_state_intermediates[t]` after row
        // `t`, which is exactly the contract `commit_accepted_prefix_dispatch`
        // reads (index `num_accepted - 1` = "state after token
        // num_accepted-1"). `ATLAS_QWEN4EXP_MTP_HC_COMMIT=0` restores the old
        // fused 1 + (K-1) split for A/B — it re-enables the corruption, so it
        // is a diagnostic switch, not a supported mode.
        // ── ONE K-ROW PASS (ATLAS_QWEN4EXP_MTP_HC_BATCHED=1) ──
        //
        // With the batched conv+GDN kernels serving the GDN layers, the per-row
        // intermediates are written by the kernel, so the K single-row passes
        // that e53b78427 needed collapse back into one.
        //
        // ALL THREE CARRIES ARE PER-ROW IN THIS PASS, which matters more than
        // the pass count:
        //   * SSM `h_state`/`conv_state` — the conv+GDN kernels write
        //     `h_state_intermediates[t]` / `conv_state_intermediates[t]`
        //     natively for `t in 0..K-1`, exactly the range
        //     `commit_rewind_index` reads.
        //   * PLE's rolling conv + history — `decode_batched_inner_hc` runs
        //     `forward_row` ONE ROW AT A TIME and snapshots the carry at every
        //     boundary a commit can land on (`push_verify_row`).
        //   * QSA's `ingested`/`pooled` — contiguous marks, so no snapshot is
        //     needed: `align_aux` rewinds them to `base + num_accepted`.
        // `commit_verify_aux_rows` lands the last two, called from
        // `commit_accepted_prefix` immediately after the SSM copies.
        //
        // `ATLAS_QWEN4EXP_MTP_ROLLBACK=1` is REFUSED alongside this arm. That
        // path restores a PRE-verify aux blob, which here would undo the
        // committed row 0 on top of a commit that already landed correctly.
        // It is default-off and documented unproven; this arm supersedes it.
        if crate::layers::qwen3_ssm::trait_decode_batched_hc::hc_batched_verify_enabled() {
            anyhow::ensure!(
                !rollback_armed(),
                "ATLAS_QWEN4EXP_MTP_HC_BATCHED=1 and ATLAS_QWEN4EXP_MTP_ROLLBACK=1 \
                 are incompatible: the batched arm commits the PLE and QSA carries \
                 per row through commit_accepted_prefix, and rollback would then \
                 restore a PRE-verify blob over it, undoing the committed row. \
                 Arm one or the other."
            );
            // The ABSOLUTE base a partial accept is measured from. Recorded
            // before the pass, because `verify_hc_rows` advances `seq.seq_len`
            // by K and the scheduler's reject branches rewind it at different
            // points — deriving the base from a moving `seq_len` is how the
            // carries end up one row off.
            *self
                .pending_verify_span
                .lock()
                .map_err(|_| anyhow::anyhow!("verify span stash poisoned"))? =
                Some((seq.seq_len, k));
            return self.verify_hc_rows(tokens, seq, stream);
        }

        let base_pos = seq.seq_len;
        if !hc_verify_publishes_intermediates() {
            let mut out = self.verify_hc_rows(&tokens[..1], seq, stream)?;
            if k == 1 {
                return Ok(out);
            }
            // The pre-fix stash, reproduced deliberately: ONE snapshot after
            // row 0, handed to every restore index. That is the single-snapshot
            // behaviour this file's fix replaced, so the diagnostic arm
            // reproduces the corruption rather than erroring on a missing row.
            let stash = self.collect_verify_aux_states(seq, stream_d)?;
            self.stash_verify_aux(VerifyAuxRows {
                base_pos,
                k,
                rows: vec![stash; hc_publish_rows(k).len().max(1)],
            })?;
            out.extend(self.verify_hc_rows(&tokens[1..], seq, stream)?);
            return Ok(out);
        }

        let mut out = Vec::with_capacity(k);
        let mut aux_rows: Vec<Vec<(u32, Vec<u8>)>> = Vec::with_capacity(hc_publish_rows(k).len());
        for t in 0..k {
            out.extend(self.verify_hc_rows(&tokens[t..t + 1], seq, stream)?);
            // State after token `t`. Only indices [0, k-2] are reachable by
            // `commit_accepted_prefix` (`num_accepted <= k-1` on every partial
            // accept), so the last row's snapshot is skipped: it is the live
            // state already, and `num_accepted == k` short-circuits.
            //
            // ★ ONE RANGE GOVERNS BOTH HALVES OF THE ROLLBACK. The SSM
            // intermediates and the auxiliary carries are rewound by the same
            // index (`commit_rewind_index(num_accepted)`) from the same
            // commit, so they must be published for the same rows or the two
            // land on different tokens. Publishing them in one branch is what
            // keeps that true by construction.
            if hc_publish_rows(k).contains(&t) {
                self.publish_verify_row_state(seq, t, stream_d)?;
                // Carries with rows [0..=t] applied and no later draft. Only
                // the non-mark-rewindable half is serialized (PLE's conv +
                // n-gram history); QSA is realigned by absolute position in
                // `restore_verify_aux_at`, which costs nothing and avoids a
                // per-layer raw-key round trip on every speculative step.
                aux_rows.push(self.collect_verify_aux_states(seq, stream_d)?);
            }
        }
        if k > 1 {
            self.stash_verify_aux(VerifyAuxRows {
                base_pos,
                k,
                rows: aux_rows,
            })?;
        }
        Ok(out)
    }

    fn stash_verify_aux(&self, stash: VerifyAuxRows) -> Result<()> {
        *self
            .pending_verify_aux
            .lock()
            .map_err(|_| anyhow::anyhow!("verify aux stash poisoned"))? = Some(stash);
        Ok(())
    }

    /// Copy this sequence's live GDN `h_state`/`conv_state` into the verify
    /// intermediate slot for row `t`, so `commit_accepted_prefix` has a real
    /// snapshot to rewind to.
    ///
    /// The batched verify kernels populate these slots as a side effect of
    /// their own scan; the mHC prefill body has no such side effect, so this
    /// publishes them explicitly. The widths are the SSOT ones
    /// `commit_accepted_prefix_dispatch` copies BACK —
    /// `ssm_pool.h_stored_bytes` for h (it tracks `--ssm-h-dtype`) and the
    /// conv blob computed from config — so the two must not drift apart.
    fn publish_verify_row_state(
        &self,
        seq: &mut SequenceState,
        t: usize,
        stream: u64,
    ) -> Result<()> {
        use atlas_core::config::LayerType;

        use crate::layer::SsmLayerState;

        let nv = self.config.linear_num_value_heads;
        let vd = self.config.linear_value_head_dim;
        let nk = self.config.linear_num_key_heads;
        let kd = self.config.linear_key_head_dim;
        let h_bytes = self.ssm_pool.h_stored_bytes;
        let conv_bytes = (nk * kd * 2 + nv * vd) * self.config.linear_conv_kernel_dim * 4;

        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let ssm = layer_state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;
            // No MTP pools (`has_mtp == false`) means no speculative verify
            // reaches `commit_accepted_prefix` either — nothing to publish.
            if ssm.h_state_intermediates.is_empty() && ssm.conv_state_intermediates.is_empty() {
                return Ok(());
            }
            let h_dst = *ssm.h_state_intermediates.get(t).ok_or_else(|| {
                anyhow::anyhow!(
                    "mHC verify: row {t} has no h intermediate slot ({} sized) — the \
                     verify width exceeds what ssm_reserve sized, and a partial accept \
                     would rewind onto never-written pool memory",
                    ssm.h_state_intermediates.len()
                )
            })?;
            let c_dst = *ssm.conv_state_intermediates.get(t).ok_or_else(|| {
                anyhow::anyhow!(
                    "mHC verify: row {t} has no conv intermediate slot ({} sized)",
                    ssm.conv_state_intermediates.len()
                )
            })?;
            self.gpu
                .copy_d2d_async(ssm.h_state, h_dst, h_bytes, stream)?;
            self.gpu
                .copy_d2d_async(ssm.conv_state, c_dst, conv_bytes, stream)?;
        }
        Ok(())
    }

    /// Land the auxiliary carries on a commit of `num_accepted` out of `k`
    /// verify rows, leaving `seq_len`/`tokens` to the caller.
    ///
    /// PLE is restored from the per-row snapshot at
    /// `verify_aux_restore_row(num_accepted, k)`; QSA is realigned to the
    /// absolute position `base_pos + num_accepted`. The KV written for the
    /// discarded positions is left alone deliberately: it is past `seq_len` and
    /// the next step overwrites it.
    ///
    /// The scheduler's branch already owns the token/`seq_len` rewind
    /// (`seq_len -= rejected` + `commit_accepted_prefix`); what it cannot do is
    /// walk back the carries a mini-prefill advanced. Splitting the aux half out
    /// lets the scheduler call exactly the missing piece instead of rewinding
    /// `seq_len` a second time.
    pub(super) fn restore_verify_aux_at(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
    ) -> Result<()> {
        let stream = self.gpu.default_stream();
        let stash = self
            .pending_verify_aux
            .lock()
            .map_err(|_| anyhow::anyhow!("verify aux stash poisoned"))?
            .take();
        let Some(stash) = stash else {
            anyhow::bail!(
                "commit_verify_aux({num_accepted}/{k}) with no stashed aux snapshot — \
                 decode_verify_hc must run first, or the carries cannot be rewound"
            );
        };
        anyhow::ensure!(
            stash.k == k,
            "commit_verify_aux quotes k={k} but the stash was taken for k={} — the \
             scheduler and the verify disagree about the draft width, and restoring \
             across that mismatch would land the carries on the wrong token",
            stash.k
        );
        anyhow::ensure!(
            num_accepted >= 1 && num_accepted <= k,
            "commit_verify_aux: num_accepted={num_accepted} outside 1..={k}"
        );
        // The scheduler owns the token/`seq_len` rewind and runs it BEFORE the
        // commit; this pins that the two agree about where the sequence landed.
        // If they ever disagree the QSA alignment below would move the marks to
        // a position the sequence is not at, which the next decode reports as
        // "decode at pos N but M tokens ingested" — a confusing symptom a long
        // way from its cause.
        anyhow::ensure!(
            stash.base_pos + num_accepted == seq.seq_len,
            "commit_verify_aux({num_accepted}/{k}): verify started at {} so the \
             sequence should be at {} after committing {num_accepted} rows, but \
             seq_len is {} — the scheduler's rewind and this commit disagree",
            stash.base_pos,
            stash.base_pos + num_accepted,
            seq.seq_len
        );

        // ── The snapshot half: PLE's conv + n-gram history ──
        // Skipped on a FULL accept: no row was discarded, so the live carry is
        // already the committed one (and `hc_publish_rows` never snapshots the
        // last row for exactly that reason).
        if let Some(idx) = verify_aux_restore_row(num_accepted, k) {
            let blobs = stash.rows.get(idx).ok_or_else(|| {
                anyhow::anyhow!(
                    "mHC verify: commit of {num_accepted}/{k} rows needs aux snapshot \
                     {idx}, but the verify stashed only {} — the publish range and \
                     `commit_rewind_index` have drifted apart",
                    stash.rows.len()
                )
            })?;
            self.apply_aux_states(seq, blobs, stream)?;
        }

        // ── The mark half: QSA `ingested`/`pooled` ──
        // ALWAYS, including full accept, and by ABSOLUTE position. The verify
        // advanced the marks by `k`; the sequence kept `num_accepted`. Aligning
        // to `base_pos + num_accepted` is a no-op when they already agree, so
        // this is safe to run on every branch and leaves nothing for the next
        // step's `align_aux` to discover — which matters because a Marconi
        // checkpoint can be taken between the two, and would otherwise
        // serialize marks that are ahead of the sequence.
        self.align_verify_aux_states(seq, stash.base_pos + num_accepted, stream)
    }

    /// Land ALL THREE per-row carries on a partial accept of `num_accepted`
    /// rows out of the K-row batched mHC verify.
    ///
    /// The SSM carry is not here — `commit_accepted_prefix` already rewinds
    /// `h_state`/`conv_state` from the pool intermediates the conv+GDN kernels
    /// wrote. What this adds is the other two, which that function has never
    /// touched:
    ///
    /// * PLE's rolling conv + history window, from the per-row snapshot
    ///   `decode_batched_inner_hc` recorded (`commit_verify_row`); and
    /// * QSA's `ingested`/`pooled` marks, rewound to the ABSOLUTE position
    ///   `base + num_accepted` (`align_aux`) — contiguous marks need no blob.
    ///
    /// Leaving either advanced over the rejected rows is the measured
    /// degeneration: PLE then hashes a window shifted by the discarded
    /// drafts, and QSA trips its own `pos == ingested` assertion or indexes
    /// past the committed prefix.
    ///
    /// No-op unless the BATCHED arm ran: the per-row reference path advances
    /// its carries one row per pass and snapshots between rows 0 and 1
    /// instead.
    pub(super) fn commit_verify_aux_rows(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        stream: u64,
    ) -> Result<()> {
        let span = self
            .pending_verify_span
            .lock()
            .map_err(|_| anyhow::anyhow!("verify span stash poisoned"))?
            .take();
        let Some((base, k)) = span else {
            return Ok(());
        };
        // A full accept keeps the live carries — they are already exactly
        // `base + k`. Zero accepted has no row-0 snapshot to land on and is
        // rejected upstream by `commit_accepted_prefix`.
        if num_accepted == 0 || num_accepted >= k {
            return Ok(());
        }
        let row = commit_rewind_index(num_accepted);
        let to_pos = base + num_accepted;
        for (i, layer) in self.layers.iter().enumerate() {
            let st = seq.layer_states[i].as_mut();
            layer.commit_verify_row(st, row, self.gpu.as_ref(), stream)?;
            layer.align_aux(st, to_pos, self.gpu.as_ref(), stream)?;
        }
        Ok(())
    }


    /// One K-row mini-prefill. Advances sequence state by K rows.
    fn verify_hc_rows(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        let k = tokens.len();
        if k == 0 {
            return Ok(Vec::new());
        }
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let bf16 = 2usize;

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();
        let mut kv_cache = self.kv_cache.lock();

        // ── KV blocks for every position this verify will write ──
        let bs = kv_cache.block_size();
        let last_pos = seq.seq_len + k - 1;
        let blocks_needed = (last_pos / bs) + 1;
        while seq.block_table.len() < blocks_needed {
            let blk = kv_cache.alloc_block()?;
            seq.block_table.push(blk);
        }

        // ── Align the QSA carry with the rows about to be replayed ──
        // The graphed K=2 verify re-processes the CURRENT position: row 0 is
        // token_0, which the bootstrap decode already emitted — and already
        // ingested into the indexer. Replaying it without rewinding trips
        // `QSA: prefill chunk starts at 367 but 368 tokens are ingested`.
        // ALIGN to seq_len — absolute, not a fixed rewind. The overlap is not
        // constant: rewinding by 1 unconditionally produced the mirror-image
        // failure ("starts at 366 but 365 ingested"). Aligning never advances
        // the mark, so a carry that is already correct is untouched.
        for (i, layer) in self.layers.iter().enumerate() {
            layer.align_aux(
                seq.layer_states[i].as_mut(),
                seq.seq_len,
                self.gpu.as_ref(),
                stream,
            )?;
        }

        // ── Embed the K candidates into hidden[K, H] ──
        // FP32 stride: `hidden_states` is the FP32 residual-stream buffer on
        // this path, matching verify_a.
        for (t, &token) in tokens.iter().enumerate() {
            self.embed(token, hidden.offset(t * h * 2), stream)?;
        }

        // ── Prefill-shaped metadata for K rows at [seq_len, seq_len+K) ──
        // Reuses the prefill packer rather than hand-rolling: it owns the
        // MRoPE stream layout and bounds the write against the scratch region.
        let meta_base = self.buffers.scratch().offset(32768);
        let meta_region = self.buffers.scratch_bytes().saturating_sub(32768);
        let all_tokens: Vec<u32> = seq
            .tokens
            .iter()
            .copied()
            .chain(tokens.iter().copied())
            .collect();
        let chunk_start = seq.tokens.len();
        let meta = self.prefill_b_upload_meta_at(
            &all_tokens,
            seq,
            chunk_start,
            k,
            seq.seq_len,
            k,
            seq.seq_len,
            &kv_cache,
            meta_base,
            meta_region,
            stream,
        )?;

        // Paged metadata (block table delta + seq_len) — the same helper the
        // chunked-prefill path uses. `needs_paged` is always true here: verify
        // only ever runs at seq_len_start > 0.
        if meta.needs_paged {
            // GROW the paged metadata. It was allocated for the ORIGINAL
            // prefill and verify extends past it — measured: "chunked prefill
            // metadata capacity 4 < required 7 blocks". `ensure_...` BAILS on a
            // short capacity rather than growing, so drop the old one first and
            // let it allocate at the size this verify needs. The old device
            // buffers are freed explicitly: `DevicePtr` has no Drop.
            let bs_meta = kv_cache.block_size();
            let need_blocks = all_tokens.len().saturating_sub(1) / bs_meta + 1;
            let too_small = seq
                .chunked_prefill_meta
                .as_ref()
                .is_some_and(|m| m.block_capacity < need_blocks);
            if too_small && let Some(old) = seq.chunked_prefill_meta.take() {
                let _ = self.gpu.free(old.block_table);
                let _ = self.gpu.free(old.seq_len);
            }
            self.ensure_chunked_prefill_meta(seq, all_tokens.len(), bs_meta)?;
            self.prefill_b_upload_paged(
                seq,
                all_tokens.len(),
                seq.seq_len,
                k,
                meta_base,
                meta.slot_offset,
                &kv_cache,
                stream,
            )?;
        }
        let (block_table_dev, seq_len_dev) = if meta.needs_paged {
            let pm = seq
                .chunked_prefill_meta
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("verify_hc: paged meta missing after upload"))?;
            (pm.block_table, pm.seq_len)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };
        let seq_slot = self.upload_seq_slot_uniform(
            seq.adapter_slot,
            k,
            self.buffers.lora_seq_slot(),
            stream,
        )?;

        // Field-for-field as prefill_c builds it. Pointing these at `meta_base`
        // wholesale (an earlier cut of this file) makes attention read the
        // position stream as its slot/seq_len/block table — silently wrong.
        let attn_metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(meta.slot_offset),
            seq_len: seq_len_dev,
            block_table: block_table_dev,
            max_blocks_per_seq: seq.block_table.len() as u32,
            num_seqs: 1,
            seq_slot,
            moe_row_adapter: DevicePtr::NULL,
        };

        // ── Per-row `seq_len` for the decode-shaped attention replay ──
        //
        // `prefill_b_upload_paged` writes ONE `i32` (the chunk end,
        // `seq_len + K`) because a prefill's paged attention is causal-masked
        // over the whole chunk. A DECODE step instead reads
        // `seq_lens[0]` as "how many keys are visible", so row `t` needs its
        // own value `seq_len + t + 1`. Parked immediately past the i64 slot
        // table inside the SAME metadata region, which nothing else writes.
        let attn_rows = verify_attn_decode_enabled();
        let row_seq_lens = meta_base.offset(verify_row_seq_len_offset(meta.slot_offset, k));
        if attn_rows {
            let need = verify_row_seq_len_offset(meta.slot_offset, k) + k * VERIFY_SEQ_LEN_STRIDE;
            anyhow::ensure!(
                need <= meta_region,
                "verify_hc: metadata region {meta_region} B cannot hold the per-row \
                 seq_len array ({need} B needed)"
            );
            let vals: Vec<i32> = (0..k)
                .map(|t| verify_row_seq_len_value(seq.seq_len, t))
                .collect();
            // SAFETY: exactly `k * 4` bytes over the live, fully initialised
            // `vals` Vec built on the lines above.
            let bytes = unsafe {
                std::slice::from_raw_parts(vals.as_ptr() as *const u8, k * VERIFY_SEQ_LEN_STRIDE)
            };
            self.gpu.copy_h2d(bytes, row_seq_lens)?;
        }

        let ctx = ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(attn_metadata),
            profile: false,
            comm: self.comm_ref(),
            // Host-built metadata: capture is illegal here.
            graph_capture: false,
            // The verify must agree TOKEN-FOR-TOKEN with serial decode: row 0
            // re-processes a row the decode already committed. `prefill()`
            // would otherwise pick the FLA chunked scan while `decode()`
            // carries H forward one token at a time — equivalent in exact
            // arithmetic, not in bf16 (measured on this model: chunked 110428
            // / 2097152 words differ, relL2 1.212e-3; sequential 0). That ~1%
            // per layer compounds over 48 layers and flips greedy argmaxes
            // wherever the top-2 margin is under ~0.9 logit units.
            // `ATLAS_QWEN4EXP_MTP_VERIFY_FLA=1` restores the chunked scan.
            gdn_exact_replay: !verify_uses_fla_scan(),
            token_ids: None,
            // PLE reads HOST ids for the rows it is about to process.
            host_token_ids: Some(tokens),
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(),
        };

        // ── Every layer over the same K rows ──
        //
        // The 12 full-attention layers always take the K-row mHC prefill path
        // — that is also what advances the QSA `ingested`/`pooled` marks, the
        // third of the three per-row carries.
        //
        // The 36 GDN layers take one of two bodies:
        //   * DEFAULT — `prefill()` -> `prefill_inner_hc` -> `prefill_block`,
        //     the CHUNK SCAN. It writes no `h_state_intermediates`, so the
        //     caller must run one row per pass and publish them by hand
        //     (`publish_verify_row_state`).
        //   * `ATLAS_QWEN4EXP_MTP_HC_BATCHED=1` — `decode_batched()` ->
        //     `decode_batched_inner_hc` -> `decode_batched_block`, the fused
        //     conv+GDN verify kernels. They advance the recurrence over all K
        //     rows in ONE pass and write the per-row intermediates natively,
        //     which is the whole point: one pass instead of K, and the state
        //     published by the kernel that owns it.
        //
        // Both bodies are K-row and both sit at `hc_row_offset = 0`, so the
        // `[T, hc, H]` highway layout is uniform either way.
        let batched_gdn = crate::layers::qwen3_ssm::trait_decode_batched_hc::
            hc_batched_verify_enabled();
        // LIVENESS, once per process. The switch is an env read; this line is
        // the proof the batched body actually ran, which a flag is not.
        if batched_gdn {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                tracing::info!(
                    "mHC verify: K-row BATCHED GDN armed (ATLAS_QWEN4EXP_MTP_HC_BATCHED=1),                      first pass k={k} over {} layers",
                    self.layers.len()
                );
            });
        }
        if attn_rows {
            static SAID_ATTN: std::sync::Once = std::sync::Once::new();
            SAID_ATTN.call_once(|| {
                tracing::info!(
                    "mHC verify: attention layers replayed as K sequential one-row \
                     DECODE bodies (kill switch ATLAS_QWEN4EXP_MTP_HC_ATTN_DECODE=0)"
                );
            });
        }
        // EXPERIMENTAL, opt-in (`ATLAS_QWEN4EXP_MTP_HC_SSM_DECODE=1`): send the
        // GDN layers down the same one-row decode body. Legal ONLY at k == 1 --
        // the per-row reference arm's pass width -- because a plain `decode()`
        // writes no `h_state_intermediates`, and at k > 1 the commit rewind
        // would read never-written pool memory. At k == 1 `hc_publish_rows(1)`
        // is empty and the CALLER publishes the row state after the pass
        // (`publish_verify_row_state`), so the contract still holds.
        let ssm_rows = attn_rows && k == 1 && verify_ssm_decode_enabled();
        anyhow::ensure!(
            !(verify_ssm_decode_enabled() && k > 1),
            "ATLAS_QWEN4EXP_MTP_HC_SSM_DECODE=1 needs the per-row verify arm \
             (one row per pass); this pass is k={k}. Unset \
             ATLAS_QWEN4EXP_MTP_HC_BATCHED."
        );
        let base_seq_len = seq.seq_len;
        for (i, layer) in self.layers.iter().enumerate() {
            // ── Attention: K sequential one-row decode bodies ──
            //
            // Row 0 re-processes a token a serial decode already committed, so
            // its logits MUST equal that decode's. `prefill()` cannot give
            // that: prefill attention is the chunked/flash paged kernel over K
            // queries with a causal mask, decode attention is the paged-decode
            // GEMV against the KV cache at M=1 -- different reduction order,
            // and on this checkpoint different enough to flip greedy argmaxes.
            // The same split applies to the projections (GEMM at M=K vs GEMV
            // at M=1), to the MoE (grouped/fused-small-M vs the decode
            // expert path) and to QSA (`prefill_ingest` vs `decode_select`).
            //
            // LAYOUT RECONCILIATION -- the reason the module doc above said
            // this could not be done. The highway is `[T, hc_mult, H]` FP32
            // and `prefill_inner_hc` addresses it at
            // `hc_row_offset * hc_mult * H * 4` (prefill_inner.rs:565).
            // `decode_inner_hc` used to hard-code row 0. It now applies the
            // IDENTICAL arithmetic (decode_inner.rs:467), so a one-row decode
            // body at `hc_row_offset = t` reads and writes exactly the row the
            // K-row GDN body would have. `hidden`/`residual` are offset by the
            // same row in BF16 stride, and `hc_post`/`hc_comb`/`norm_output`
            // are per-pass scratch that a one-row body uses at its base. So
            // the 1-row attention path and the K-row SSM path agree about
            // which row a stream belongs to.
            //
            // The metadata is re-pointed per row rather than rebuilt: the
            // K-row pack already holds `[K]` u32 positions and `[K]` i64
            // slots, so row `t` is a pointer bump of `t*4` / `t*8`. Only the
            // device `seq_len` differs in KIND between the two shapes, and it
            // is uploaded above.
            if attn_rows && (ssm_rows || !layer.is_ssm_layer()) {
                for t in 0..k {
                    let row_meta = AttnMetadataDev {
                        positions: attn_metadata.positions.offset(t * VERIFY_POS_STRIDE),
                        positions_h: attn_metadata.positions_h.offset(t * VERIFY_POS_STRIDE),
                        positions_w: attn_metadata.positions_w.offset(t * VERIFY_POS_STRIDE),
                        slot: attn_metadata.slot.offset(t * VERIFY_SLOT_STRIDE),
                        seq_len: row_seq_lens.offset(t * VERIFY_SEQ_LEN_STRIDE),
                        block_table: attn_metadata.block_table,
                        max_blocks_per_seq: attn_metadata.max_blocks_per_seq,
                        num_seqs: 1,
                        seq_slot: attn_metadata.seq_slot,
                        moe_row_adapter: attn_metadata.moe_row_adapter,
                    };
                    let row_ctx = ForwardContext {
                        buffers: &self.buffers,
                        hc_row_offset: t,
                        gpu: self.gpu.as_ref(),
                        config: &self.config,
                        dispatch: &self.dispatch,
                        derived: &self.derived,
                        levers: &self.levers,
                        stats: &self.stats,
                        attn_metadata: Some(row_meta),
                        profile: false,
                        comm: self.comm_ref(),
                        graph_capture: false,
                        // Attention layers never read it; carried so the two
                        // contexts cannot drift.
                        gdn_exact_replay: !verify_uses_fla_scan(),
                        token_ids: None,
                        host_token_ids: Some(&tokens[t..t + 1]),
                        routed_lora_layers: None,
                        midchunk_capture: None,
                        moe_lora_route: self.decode_moe_route(),
                    };
                    layer.decode(
                        hidden.offset(t * h * 2),
                        residual.offset(t * h * 2),
                        seq.layer_states[i].as_mut(),
                        &mut kv_cache,
                        // PRE-APPEND length, the decode convention: the token
                        // being processed sits at absolute position
                        // `base_seq_len + t`.
                        verify_row_decode_seq_len(base_seq_len, t),
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        &row_ctx,
                        stream,
                    )?;
                }
                self.hidden_probe_layer("verify_hc", i, 0, hidden, stream);
                continue;
            }
            if batched_gdn && layer.is_ssm_layer() {
                layer.decode_batched(
                    hidden,
                    residual,
                    k,
                    seq.layer_states[i].as_mut(),
                    &mut kv_cache,
                    seq.seq_len,
                    &mut seq.block_table,
                    &mut seq.disk_block_ids,
                    &mut seq.disk_last_offloaded_per_layer,
                    &ctx,
                    stream,
                )?;
                continue;
            }
            layer.prefill(
                hidden,
                residual,
                k,
                seq.layer_states[i].as_mut(),
                &mut kv_cache,
                seq.seq_len,
                &mut seq.block_table,
                &mut seq.disk_block_ids,
                &mut seq.disk_last_offloaded_per_layer,
                seq.seq_len,
                &ctx,
                stream,
            )?;
            self.hidden_probe_layer("verify_hc", i, 0, hidden, stream);
        }
        drop(kv_cache);

        // ── K-row head: same tail as the non-hc verify ──
        let normed = self.buffers.norm_output();
        let eps = self.config.rms_norm_eps as f32;
        // ATLAS_LOGIT_PROBE=1: the VERIFY side of the hidden-state A/B against
        // `decode_forward_body`. Same point in the pipeline (pre-final-norm),
        // same row stride, so an equal fingerprint blames the head and an
        // unequal one blames the layer bodies.
        for t in 0..k {
            self.hidden_probe("verify_hc", t, hidden.offset(t * h * 2), stream);
        }
        self.final_norm_apply(hidden, normed, k as u32, h as u32, eps, stream)?;
        self.lm_head_batched(normed, k as u32, self.buffers.logits(), stream)?;

        let mut out = Vec::with_capacity(k);
        for t in 0..k {
            let logits_t = self.buffers.logits().offset(t * vocab * bf16);
            // ATLAS_LOGIT_PROBE=1: the verify side of the row-by-row A/B
            // against a serial decode of the same prefix. `lm_head_batched`
            // always writes BF16 here (the FP32-logits buffer is the
            // single-token decode path only).
            self.logit_probe("verify_hc", t, logits_t, false, stream);
            let out_ptr = self.buffers.scratch().offset(t * 4);
            ops::argmax_bf16(
                self.gpu.as_ref(),
                self.argmax_kernel,
                logits_t,
                out_ptr,
                vocab as u32,
                stream,
            )?;
            let mut b = [0u8; 4];
            self.gpu.copy_d2h(out_ptr, &mut b)?;
            out.push(u32::from_le_bytes(b));
        }

        seq.tokens.extend_from_slice(tokens);
        seq.seq_len += k;
        Ok(out)
    }
}

/// `ATLAS_QWEN4EXP_MTP_HC_COMMIT=0` reverts to the pre-fix fused split, which
/// leaves the SSM verify intermediates unwritten. Diagnostic only.
pub(crate) fn hc_verify_publishes_intermediates() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MTP_HC_COMMIT").as_deref() != Ok("0"))
}

/// `ATLAS_QWEN4EXP_MTP_ROLLBACK=1` — the same switch `rollback_verify_rows`
/// reads (`trait_impl/mod.rs`), duplicated here so the batched arm can refuse
/// the incompatible combination at the point of use rather than desyncing.
fn rollback_armed() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MTP_ROLLBACK").as_deref() == Ok("1"))
}

/// The verify rows an mHC verify of width `k` must publish an SSM
/// intermediate for.
///
/// The last row is excluded on purpose: `commit_accepted_prefix` reads
/// `commit_rewind_index(num_accepted)` and short-circuits at
/// `num_accepted == k`, so the highest index it can ever read is `k - 2`.
/// `hc_publish_covers_every_commit_rewind` pins that agreement.
/// Opt back into the FLA chunked GDN scan inside the mHC verify.
///
/// Default OFF: the verify has to reproduce `decode()` bit-for-bit or greedy
/// speculation is not lossless, and the chunked scan does not. This exists so
/// the cost of the sequential path can be A/B'd, not because the chunked one is
/// ever correct here.
pub(super) fn verify_uses_fla_scan() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MTP_VERIFY_FLA").as_deref() == Ok("1"))
}

/// Element strides of the three per-row attention metadata streams, as
/// `prefill_b_upload_meta_at` packs them and as `layer::AttnMetadataDev`
/// documents them. They are NOT the same width, which is the whole reason
/// these are named constants: `positions` is `[N] u32`, `slots` is `[N] i64`
/// (`fill_slots_from_block_table` / `reshape_and_cache` both take i64), and
/// `seq_lens` is `[N] i32`. Bumping the slot pointer by 4 instead of 8 aims
/// the KV write at the wrong cache slot for every row past 0 -- silently,
/// because the value there is still a plausible slot index.
pub(super) const VERIFY_POS_STRIDE: usize = 4;
pub(super) const VERIFY_SLOT_STRIDE: usize = 8;
pub(super) const VERIFY_SEQ_LEN_STRIDE: usize = 4;

/// Byte offset, inside the verify's metadata block at `meta_base`, of the
/// `[k]` i32 per-row `seq_len` array.
///
/// The pack owns `[0, slot_offset)` for the position streams and
/// `[slot_offset, slot_offset + k*8)` for the i64 slot table
/// (`prefill_b_upload_paged` fills exactly `k` entries there). The seq_len
/// array goes immediately past it, 4-byte aligned.
pub(super) fn verify_row_seq_len_offset(slot_offset: usize, k: usize) -> usize {
    (slot_offset + k * VERIFY_SLOT_STRIDE).next_multiple_of(VERIFY_SEQ_LEN_STRIDE)
}

/// The DEVICE `seq_len` value row `t` of a verify based at `base_seq_len`
/// must present to the paged-decode attention: the number of VISIBLE KEYS,
/// which includes the row's own token because `write_kv_cache` runs first.
pub(super) fn verify_row_seq_len_value(base_seq_len: usize, t: usize) -> i32 {
    (base_seq_len + t + 1) as i32
}

/// The HOST `seq_len` argument for row `t`'s `Layer::decode`. That parameter
/// is the PRE-APPEND length -- serial decode of the token at absolute
/// position `p` passes `p` (see `attention_forward.rs`, the QSA
/// `decode_select` call site) -- so it is one less than the device value.
pub(super) fn verify_row_decode_seq_len(base_seq_len: usize, t: usize) -> usize {
    base_seq_len + t
}

/// Default-ON. `ATLAS_QWEN4EXP_MTP_HC_ATTN_DECODE=0` puts the attention
/// layers back on the K-row `prefill()` body.
pub(super) fn verify_attn_decode_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_QWEN4EXP_MTP_HC_ATTN_DECODE").as_deref() != Ok("0")
    })
}

/// EXPERIMENTAL, default-OFF. `ATLAS_QWEN4EXP_MTP_HC_SSM_DECODE=1` puts the
/// GDN layers on the same one-row `decode()` body the attention layers take,
/// making the WHOLE verify decode-shaped. Only valid at k == 1 (see the call
/// site); the check there refuses anything wider rather than corrupting the
/// commit rewind.
pub(super) fn verify_ssm_decode_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MTP_HC_SSM_DECODE").as_deref() == Ok("1"))
}

pub(super) fn hc_publish_rows(k: usize) -> std::ops::Range<usize> {
    0..k.saturating_sub(1)
}

/// `ATLAS_QWEN4EXP_MTP_AUX_COMMIT=0` disables the auxiliary-carry commit.
/// Diagnostic only — with it off, every rejected verify row leaves PLE's
/// rolling conv/history and QSA's marks one row ahead of the sequence.
///
/// It replaces `ATLAS_QWEN4EXP_MTP_ROLLBACK=1`, which was the same rollback in
/// arm-to-use polarity AND wrong: reachable only from the K=2 reject branch,
/// and hard-wired to snapshot row 0.
pub(super) fn hc_verify_commits_aux() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MTP_AUX_COMMIT").as_deref() != Ok("0"))
}

/// The aux snapshot row a commit of `num_accepted` out of `k` verify rows must
/// restore, or `None` when the live carries are already correct.
///
/// This is the auxiliary-carry twin of `commit_rewind_index`, and it delegates
/// to it rather than restating the arithmetic: the PLE carry and the SSM state
/// are rewound by the SAME commit and must land on the SAME token, so if the
/// two indices could drift the model would run one carry a row out from the
/// other. `hc_publish_rows` is the range both are published over.
///
/// `None` at `num_accepted == k` mirrors `commit_accepted_prefix`'s
/// short-circuit: nothing was discarded, so nothing is restored.
pub(super) fn verify_aux_restore_row(num_accepted: usize, k: usize) -> Option<usize> {
    if num_accepted == 0 || num_accepted >= k {
        return None;
    }
    Some(commit_rewind_index(num_accepted))
}

#[cfg(test)]
mod verify_attn_row_tests {
    use super::{
        VERIFY_POS_STRIDE, VERIFY_SEQ_LEN_STRIDE, VERIFY_SLOT_STRIDE, verify_row_decode_seq_len,
        verify_row_seq_len_offset, verify_row_seq_len_value,
    };

    /// THE contract the decode-shaped attention replay rests on: row 0 of a
    /// K-row verify re-processes a token a serial decode already committed,
    /// so it must be handed EXACTLY the arguments that decode was handed.
    /// Serial decode of the token at absolute position `p` passes host
    /// `seq_len = p` (pre-append) and a device `seq_len = p + 1` (keys
    /// visible after `write_kv_cache`). Off by one in either direction and
    /// row 0 attends over the wrong prefix.
    #[test]
    fn row_zero_matches_a_serial_decode_of_the_committed_token() {
        for base in [1usize, 35, 367, 4096] {
            assert_eq!(verify_row_decode_seq_len(base, 0), base);
            assert_eq!(verify_row_seq_len_value(base, 0), base as i32 + 1);
        }
    }

    /// Each further row is one token later, and the device value stays
    /// exactly one ahead of the host one.
    #[test]
    fn each_row_advances_by_exactly_one_token() {
        let base = 367usize;
        for k in 1..=8usize {
            for t in 0..k {
                assert_eq!(verify_row_decode_seq_len(base, t), base + t, "t={t}");
                assert_eq!(
                    verify_row_seq_len_value(base, t),
                    verify_row_decode_seq_len(base, t) as i32 + 1,
                    "t={t}: device seq_len must count the row's own key"
                );
            }
        }
    }

    /// The three metadata streams are packed at DIFFERENT element widths.
    /// This is the arithmetic the per-row pointer bumps use; the failure it
    /// pins is the natural (and wrong) assumption that a row is `t * 4` in
    /// all three.
    #[test]
    fn the_per_row_metadata_strides_are_not_uniform() {
        assert_eq!(VERIFY_POS_STRIDE, 4, "positions are [N] u32");
        assert_eq!(VERIFY_SLOT_STRIDE, 8, "slots are [N] i64");
        assert_eq!(VERIFY_SEQ_LEN_STRIDE, 4, "seq_lens are [N] i32");
        assert_ne!(
            VERIFY_SLOT_STRIDE, VERIFY_POS_STRIDE,
            "a row's slot is not at the same byte offset as its position"
        );
    }

    /// The per-row seq_len array must start past the LAST slot entry the
    /// pack writes -- `fill_slots_from_block_table` fills `k` i64 slots at
    /// `slot_offset` -- and must be 4-byte aligned for an i32 store.
    #[test]
    fn the_row_seq_len_array_clears_the_slot_table() {
        for slot_offset in [8usize, 16, 24, 4104] {
            for k in 1..=8usize {
                let at = verify_row_seq_len_offset(slot_offset, k);
                assert!(
                    at >= slot_offset + k * VERIFY_SLOT_STRIDE,
                    "slot_offset={slot_offset} k={k}: seq_len array at {at} overlaps \
                     the {k}-entry i64 slot table"
                );
                assert_eq!(at % VERIFY_SEQ_LEN_STRIDE, 0, "unaligned i32 store");
            }
        }
    }

    /// Distinct rows address distinct seq_len words -- a shared word would
    /// give every row the same visible-key count, which is precisely the
    /// prefill shape this replaces.
    #[test]
    fn every_row_gets_its_own_seq_len_word() {
        let base = verify_row_seq_len_offset(16, 4);
        let mut seen = std::collections::BTreeSet::new();
        for t in 0..4usize {
            assert!(seen.insert(base + t * VERIFY_SEQ_LEN_STRIDE));
        }
        assert_eq!(seen.len(), 4);
    }
}

#[cfg(test)]
mod hc_intermediate_contract_tests {
    use super::super::async_chkpt::commit_rewind_index;
    use super::hc_publish_rows;

    /// THE regression this pins: every intermediate slot
    /// `commit_accepted_prefix` can read on a partial accept must have been
    /// published by the verify that preceded it. When the mHC verify ran as a
    /// fused `1 + (K-1)` pair it published NOTHING, so every K=3 step rewound
    /// 36 GDN layers onto never-written pool memory.
    #[test]
    fn hc_publish_covers_every_commit_rewind() {
        for k in 2..=8usize {
            let published = hc_publish_rows(k);
            // `num_accepted == k` short-circuits before any rewind; 0 bails.
            for num_accepted in 1..k {
                let idx = commit_rewind_index(num_accepted);
                assert!(
                    published.contains(&idx),
                    "k={k}: commit_accepted_prefix({num_accepted}) reads intermediate \
                     {idx}, which the verify never publishes ({published:?})"
                );
            }
        }
    }

    /// The PLE carry's snapshot boundaries must be the SAME set as the SSM
    /// carry's. They are recorded in different crates' worth of code — the
    /// SSM's by the conv+GDN kernels via `hc_publish_rows`, PLE's by
    /// `decode_batched_inner_hc`'s row loop via `hc_verify_snapshot_rows` —
    /// and a partial accept reads BOTH at the same index. One range shorter
    /// than the other is a silent desync of exactly the kind this pins.
    #[test]
    fn hc_ple_snapshot_range_matches_the_ssm_one() {
        use crate::layers::qwen3_ssm::trait_decode_batched_hc::hc_verify_snapshot_rows;
        for k in 1..=8usize {
            assert_eq!(
                hc_verify_snapshot_rows(k),
                hc_publish_rows(k),
                "k={k}: the PLE and SSM verify carries disagree about which row \
                 boundaries a commit can land on"
            );
            for num_accepted in 1..k {
                assert!(
                    hc_verify_snapshot_rows(k).contains(&commit_rewind_index(num_accepted)),
                    "k={k}: commit of {num_accepted} rows has no PLE snapshot"
                );
            }
        }
    }

    /// And nothing beyond: publishing the LAST row would cost a copy per GDN
    /// layer that no rewind can reach (`num_accepted == k` short-circuits).
    #[test]
    fn hc_publish_stops_at_the_last_reachable_row() {
        assert_eq!(hc_publish_rows(1), 0..0);
        assert_eq!(hc_publish_rows(2), 0..1);
        assert_eq!(hc_publish_rows(3), 0..2);
        assert_eq!(hc_publish_rows(4), 0..3);
    }

    /// THE regression this pins, aux half: every aux snapshot a commit can
    /// restore must be one the verify actually stashed, and it must be the row
    /// the SSM rewind lands on.
    ///
    /// This FAILS against the pre-fix code, which stashed exactly ONE snapshot
    /// (after row 0) and restored it unconditionally: at k=3 with two rows
    /// committed the correct row is 1, so the carries came back a row short of
    /// the SSM state the same commit had just rewound.
    #[test]
    fn aux_restore_row_tracks_the_ssm_rewind_index() {
        for k in 2..=8usize {
            let published = hc_publish_rows(k);
            for num_accepted in 1..k {
                let idx = super::verify_aux_restore_row(num_accepted, k).unwrap_or_else(|| {
                    panic!("k={k}: partial accept of {num_accepted} restores no aux row")
                });
                assert_eq!(
                    idx,
                    commit_rewind_index(num_accepted),
                    "k={k}, accepted={num_accepted}: the aux carry and the SSM state \
                     would be rewound to different tokens"
                );
                assert!(
                    published.contains(&idx),
                    "k={k}: commit of {num_accepted} rows restores aux snapshot {idx}, \
                     which the verify never stashed ({published:?})"
                );
            }
        }
    }

    /// The single-snapshot behaviour this replaced, stated as the thing that is
    /// now false. Row 0 is correct ONLY for a one-row commit.
    #[test]
    fn aux_restore_row_is_not_always_row_zero() {
        assert_eq!(super::verify_aux_restore_row(1, 2), Some(0));
        assert_eq!(super::verify_aux_restore_row(1, 3), Some(0));
        assert_eq!(super::verify_aux_restore_row(2, 3), Some(1));
        assert_eq!(super::verify_aux_restore_row(2, 4), Some(1));
        assert_eq!(super::verify_aux_restore_row(3, 4), Some(2));
    }

    /// A full accept discarded nothing, so it restores nothing — the same
    /// short-circuit `commit_accepted_prefix` takes at `num_accepted == k`.
    #[test]
    fn aux_restore_row_is_none_on_a_full_accept() {
        for k in 1..=8usize {
            assert_eq!(super::verify_aux_restore_row(k, k), None, "k={k}");
        }
    }

    /// The stash the verify builds must be exactly as long as the restore can
    /// index. `decode_verify_hc` pushes one entry per `hc_publish_rows` row, so
    /// the highest valid index is `len - 1`.
    #[test]
    fn every_restorable_row_is_inside_the_stash() {
        for k in 2..=8usize {
            let stashed = hc_publish_rows(k).len();
            for num_accepted in 1..=k {
                if let Some(idx) = super::verify_aux_restore_row(num_accepted, k) {
                    assert!(
                        idx < stashed,
                        "k={k}: restore row {idx} but the verify stashes {stashed}"
                    );
                }
            }
        }
    }
}
