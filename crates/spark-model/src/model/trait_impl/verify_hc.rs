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
//! Kill switch: `AVAROK_QWEN4EXP_MTP_HC_ATTN_DECODE=0` restores the K-row
//! `prefill()` body for the attention layers.
//!
//! # MEASURED END TO END (2026-08-28) — RUNS, BUT WRONG AND SLOW
//!
//! With the proposer armed (`--speculative --num-drafts 1` +
//! `AVAROK_QWEN4EXP_MTP_VERIFY=1`), 4K ctx, greedy, vs a same-config baseline:
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
//!      (`AVAROK_QWEN4EXP_MTP_ROLLBACK=1` to arm) as unproven, not as harmful.
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
//!    small row counts (`AVAROK_QWEN4EXP_HC_SMALL_M_FFN`, default on) cut it 14x.
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
//! `AVAROK_QWEN4EXP_MTP_VERIFY=1`, and neither is a default.
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
//! 1. It was DEFAULT-OFF (`AVAROK_QWEN4EXP_MTP_ROLLBACK=1` to arm), so on the
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
//! DEFAULT ON, kill switch `AVAROK_QWEN4EXP_MTP_AUX_COMMIT=0`. Callers must
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
        // num_accepted-1"). `AVAROK_QWEN4EXP_MTP_HC_COMMIT=0` restores the old
        // fused 1 + (K-1) split for A/B — it re-enables the corruption, so it
        // is a diagnostic switch, not a supported mode.
        // ── ONE K-ROW PASS (AVAROK_QWEN4EXP_MTP_HC_BATCHED=1) ──
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
        // `AVAROK_QWEN4EXP_MTP_ROLLBACK=1` is REFUSED alongside this arm. That
        // path restores a PRE-verify aux blob, which here would undo the
        // committed row 0 on top of a commit that already landed correctly.
        // It is default-off and documented unproven; this arm supersedes it.
        if crate::layers::qwen3_ssm::trait_decode_batched_hc::hc_batched_verify_enabled() {
            anyhow::ensure!(
                !rollback_armed(),
                "AVAROK_QWEN4EXP_MTP_HC_BATCHED=1 and AVAROK_QWEN4EXP_MTP_ROLLBACK=1 \
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
            // Keyed by SLOT: the batched multi-sequence verify has N of these
            // in flight at once, and a single slot would have one sequence
            // consume another's base — a silently wrong rewind, which surfaces
            // as an EMPTY completion rather than an error.
            self.pending_verify_span
                .lock()
                .map_err(|_| anyhow::anyhow!("verify span stash poisoned"))?
                .insert(seq.slot_idx, (seq.seq_len, k));
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
            self.stash_verify_aux(
                seq.slot_idx,
                VerifyAuxRows {
                    base_pos,
                    k,
                    rows: vec![stash; hc_publish_rows(k).len().max(1)],
                },
            )?;
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
            self.stash_verify_aux(
                seq.slot_idx,
                VerifyAuxRows {
                    base_pos,
                    k,
                    rows: aux_rows,
                },
            )?;
        }
        Ok(out)
    }
}

#[path = "verify_hc_publish.rs"]
mod verify_hc_publish;

#[path = "verify_hc_rows.rs"]
mod verify_hc_rows;

#[path = "verify_hc_flags.rs"]
mod verify_hc_flags;
pub(super) use verify_hc_flags::*;
#[cfg(test)]
#[path = "verify_hc_attn_row_tests.rs"]
mod verify_attn_row_tests;

#[cfg(test)]
#[path = "verify_hc_intermediate_tests.rs"]
mod hc_intermediate_contract_tests;
