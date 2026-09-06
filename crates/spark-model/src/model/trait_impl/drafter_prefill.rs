// SPDX-License-Identifier: AGPL-3.0-only

//! Whole-prompt drafter context: the capture write and its EAGER consume.
//!
//! # The defect this file closes (measured recon, 2026-07-28)
//!
//! `mtp_prefill_hidden` is ONE shared slot stamped with a generation
//! (`mtp_prefill_capture_gen`). The consume used to live at the first
//! `propose` of a sequence (`ensure_drafter_context`), which is one or more
//! scheduler ticks after that sequence's prefill. At C >= 2 the sequences
//! prefill back-to-back, so by the time sequence `k` proposes, sequence
//! `k+1`'s prefill has already restarted the capture — the ownership stamp
//! then correctly refuses to build drafter KV from another sequence's
//! hiddens ("blind beats poisoned"), and every sequence but the LAST-prefilled
//! one runs a drafter that never saw its prompt.
//!
//! Worse, since `e793d1c5` ("fix the can_mix spec gate") a `--speculative`
//! serve takes the MIXED path for requests 3..n
//! (`run_standard.rs`: `spec_step_this_tick = active.len() == 1`), and
//! `mixed_forward_dispatch` never captured at all. At C=8 that is ~6 of 8
//! sequences drafting blind.
//!
//! # The fix
//!
//! Two halves, both behind `ATLAS_NO_MTP_EAGER_DRAFTER` (PRESENCE):
//!
//! 1. `try_mtp_prefill_capture_from` — the capture body, parameterised by the
//!    SOURCE pointer, so the mixed path can hand it the prefill rows (which
//!    live at `hidden + padded_n * h`, NOT at the head of the buffer: copying
//!    from the head there would capture the DECODE rows, i.e. poison rather
//!    than blindness).
//! 2. `try_eager_drafter_prefill` — consume the capture at the END of the
//!    sequence's own prefill, while it provably still owns the generation
//!    stamp. Called from the `Model` trait wrappers in `mod.rs` (the single
//!    funnel all four prefill entry points pass through) AFTER the dispatch
//!    returns, so `seq.tokens` / `seq.prompt_len` are already updated.
//!
//! Nothing about the guard set changes: `ensure_drafter_context` still
//! enforces `captured >= prompt_len`, the ownership stamp, and first-propose.
//! Moving it earlier only means the guards can now PASS for more than one
//! sequence per serve. The propose-site call stays (it is the WARM-turn carry
//! entry point and a no-op once the drafter owns rows).
//!
//! Cost: `prefill_drafter` is ~0.095 ms/row and now runs inside TTFT instead
//! of inside the first decode step. It fires only when the capture covers the
//! whole prompt — i.e. exactly the COLD-turn case that already paid it one
//! tick later. A warm turn (prefix reuse / Marconi restore) leaves the capture
//! short, so this adds nothing to the ~1136 ms warm rebuild trap documented in
//! `mtp_carry`.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::layer::ForwardContext;
use crate::traits::SequenceState;

/// `ATLAS_NO_MTP_EAGER_DRAFTER` (PRESENCE): restore the propose-site-only
/// consume, i.e. the pre-fix behaviour where only the last-prefilled sequence
/// of a concurrent group can prefill its drafter.
pub fn eager_drafter_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("ATLAS_NO_MTP_EAGER_DRAFTER").is_ok())
}

impl TransformerModel {
    /// ATLAS_MTP_DRAFTER_PREFILL: copy this prefill chunk's final-layer
    /// hiddens (`[proc_count, h]` BF16, contiguous at the head of the hidden
    /// buffer) into the whole-prompt capture at row `chunk_start`.
    ///
    /// Contiguity-tracked: `chunk_start == 0` (re)starts the capture; a chunk
    /// extending the current range appends; anything else (prefix-cache
    /// reuse, Marconi warm restore — rows whose hiddens were never computed)
    /// leaves the tracked length short, which safely disables the drafter
    /// prefill for that sequence via the coverage check at the consume site.
    pub(super) fn try_mtp_prefill_capture(
        &self,
        seq: &mut SequenceState,
        chunk_start: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        self.try_mtp_prefill_capture_from(
            seq,
            chunk_start,
            proc_count,
            self.buffers.hidden_states(),
            stream,
        )
    }

    /// [`Self::try_mtp_prefill_capture`] with an explicit SOURCE pointer.
    ///
    /// The mixed forward lays out `[decode rows | prefill rows]`, so its
    /// prefill hiddens start at `hidden + padded_n * h * 2` — passing the
    /// buffer head there would capture decode rows.
    pub(super) fn try_mtp_prefill_capture_from(
        &self,
        seq: &mut SequenceState,
        chunk_start: usize,
        proc_count: usize,
        src: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if self.mtp_prefill_hidden.is_null() || proc_count == 0 {
            return Ok(());
        }
        use std::sync::atomic::Ordering;
        if chunk_start + proc_count > self.mtp_prefill_capacity {
            return Ok(());
        }
        let len = self.mtp_prefill_capture_len.load(Ordering::Relaxed);
        // Ownership: the capture buffer is a SINGLE shared slot. Chunk 0
        // claims it under a fresh generation; an append is valid only while
        // this sequence still owns the current generation — at C>=2 another
        // sequence's chunk 0 may have restarted the capture in between, and
        // appending onto foreign rows would pair mixed hiddens under one
        // contiguous length. Mismatch leaves the length stale-short, which
        // safely disables the drafter prefill (coverage check at consume).
        let contiguous_from_zero = if chunk_start == 0 {
            let generation = self.mtp_prefill_capture_gen.fetch_add(1, Ordering::Relaxed) + 1;
            seq.mtp_capture_gen = generation;
            Some(proc_count)
        } else if chunk_start == len
            && seq.mtp_capture_gen != 0
            && seq.mtp_capture_gen == self.mtp_prefill_capture_gen.load(Ordering::Relaxed)
        {
            Some(len + proc_count)
        } else {
            None
        };
        // ATLAS_MTP_CARRY_DRAFTER: a warm turn's chunk starts at the reused-
        // prefix boundary, which the contiguous-from-zero tracker above must
        // reject (its consumer prefills the drafter from row 0). The carry
        // path consumes the SAME buffer position-indexed, so it wants the
        // write regardless of where the chunk starts — the rows are still
        // `hidden_i` at absolute row `i`. Note the SOURCE is this chunk's
        // rows, only the DESTINATION is absolute.
        let carry_on = crate::model::mtp_carry::mtp_carry_drafter_enabled(&self.levers);
        if contiguous_from_zero.is_none() && !carry_on {
            return Ok(());
        }
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        self.gpu.copy_d2d_async(
            src,
            self.mtp_prefill_hidden.offset(chunk_start * h * bf16),
            proc_count * h * bf16,
            stream,
        )?;
        if let Some(new_len) = contiguous_from_zero {
            self.mtp_prefill_capture_len
                .store(new_len, Ordering::Relaxed);
        }
        if carry_on {
            let mut r = self.mtp_store_range.lock();
            *r = crate::model::mtp_carry::merge_interval(*r, chunk_start, proc_count);
        }
        Ok(())
    }

    /// Consume the whole-prompt capture at the END of this sequence's prefill,
    /// while it provably still owns the capture generation.
    ///
    /// `is_last` is the caller's last-chunk flag: a mid-chunk call would only
    /// build partial drafter rows and then permanently block the full prefill
    /// (`prefill_drafter` fast-returns once the drafter owns rows).
    ///
    /// Runs on the SAME `stream` the prefill used, so it is ordered after the
    /// capture copy it reads. Cross-stream visibility to the later propose is
    /// the pre-existing contract (the propose-site consume already read this
    /// buffer from `default_stream`).
    ///
    /// Never fails a prefill: a drafter with fewer rows costs acceptance, not
    /// correctness, because the target verifies every draft.
    pub(super) fn try_eager_drafter_prefill(
        &self,
        seq: &mut SequenceState,
        is_last: bool,
        stream: u64,
    ) {
        if !is_last || eager_drafter_disabled() || self.mtp_prefill_hidden.is_null() {
            return;
        }
        let Some(proposer) = self.proposer.clone() else {
            return;
        };
        if seq.proposer_state.is_none() {
            return;
        }
        let ctx = ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            // Route-aware v0: base (Skip) proceeds free; an active adapter is
            // rejected before the fold on these multi-seq/speculative paths
            // (reject_decode_lora), so Fold is inert here.
            moe_lora_route: self.decode_moe_route(),
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            // None: drafter routing is not the target model's (see
            // `MoeSite::MtpHead`).
            expert_telemetry: None,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
        };
        self.ensure_drafter_context(proposer.as_ref(), seq, &ctx, stream);
        if crate::speculative::mtp_accept_debug() {
            let rows =
                proposer.drafter_rows(seq.proposer_state.as_mut().expect("checked above").as_mut());
            let captured = self
                .mtp_prefill_capture_len
                .load(std::sync::atomic::Ordering::Relaxed);
            tracing::info!(
                "MTP drafter coverage: prompt_len={} captured={captured} drafter_rows={rows}",
                seq.prompt_len,
            );
        }
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::info!(
                "MTP eager drafter prefill ENGAGED: the whole-prompt capture is consumed at \
                 end-of-prefill, so every concurrent sequence (not only the last-prefilled) \
                 can build drafter KV over its own prompt"
            );
        }
    }
}
