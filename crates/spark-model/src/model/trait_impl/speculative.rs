// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn generate_speculative_dispatch(
        &self,
        prompt_tokens: &[u32],
        params: &spark_runtime::sampler::SamplingParams,
        num_drafts: usize,
    ) -> Result<crate::engine::GenerateResult> {
        // Self-speculative mode: draft via layer-skipping (no MTP weights needed)
        if self.self_speculative {
            let mut seq = self.alloc_sequence()?;
            let stream = self.gpu.default_stream();
            let result = self.generate_self_speculative_inner(
                prompt_tokens,
                params,
                num_drafts,
                &mut seq,
                stream,
            );
            self.free_sequence(&mut seq)?;
            return result;
        }

        let proposer = match &self.proposer {
            Some(p) => p.clone(),
            None => {
                // Fallback to regular generation
                return crate::engine::generate(self, prompt_tokens, params);
            }
        };

        let mut seq = self.alloc_sequence()?;
        let stream = self.gpu.default_stream();

        let result = self.generate_speculative_inner(
            prompt_tokens,
            params,
            num_drafts,
            &proposer,
            &mut seq,
            stream,
        );

        self.free_sequence(&mut seq)?;

        result
    }

    pub(super) fn has_proposer_dispatch(&self) -> bool {
        self.proposer.is_some() || self.self_speculative
    }

    pub(super) fn has_self_speculative_dispatch(&self) -> bool {
        self.self_speculative
    }

    pub(super) fn decode_draft_dispatch(
        &self,
        token: u32,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<DevicePtr> {
        TransformerModel::decode_draft(self, token, seq, stream)
    }

    /// ATLAS_MTP_DRAFTER_PREFILL: copy this prefill chunk's final-layer
    /// hiddens (`[proc_count, h]` BF16, contiguous at the head of the hidden
    /// buffer) into the whole-prompt capture at row `chunk_start`.
    ///
    /// Contiguity-tracked: `chunk_start == 0` (re)starts the capture; a chunk
    /// extending the current range appends; anything else (prefix-cache
    /// reuse, Marconi warm restore — rows whose hiddens were never computed)
    /// leaves the tracked length short, which safely disables the drafter
    /// prefill for that sequence via the coverage check at the propose site.
    pub(super) fn try_mtp_prefill_capture(
        &self,
        chunk_start: usize,
        proc_count: usize,
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
        let contiguous_from_zero = if chunk_start == 0 {
            Some(proc_count)
        } else if chunk_start == len {
            Some(len + proc_count)
        } else {
            None
        };
        // ATLAS_MTP_CARRY_DRAFTER: a warm turn's chunk starts at the reused-
        // prefix boundary, which the contiguous-from-zero tracker above must
        // reject (its consumer prefills the drafter from row 0). The carry
        // path consumes the SAME buffer position-indexed, so it wants the
        // write regardless of where the chunk starts — the rows are still
        // `hidden_i` at absolute row `i`. Note the SOURCE is the head of the
        // hidden buffer (this chunk's rows), only the DESTINATION is absolute.
        let carry_on = crate::model::mtp_carry::mtp_carry_drafter_enabled();
        if contiguous_from_zero.is_none() && !carry_on {
            return Ok(());
        }
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        self.gpu.copy_d2d_async(
            self.buffers.hidden_states(),
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

    /// Give the drafter its prompt context on the FIRST propose of a sequence.
    ///
    /// COLD turn: the whole-prompt capture covers the prompt, so run the
    /// classic `prefill_drafter`. WARM turn: the capture never covers it (a
    /// reused prefix computes nothing), which is the context-blindness defect
    /// — adopt the previous turn's drafter KV and append only the new span.
    ///
    /// Extracted from `run_mtp_propose_inner` so `impl_b3.rs` does not grow
    /// past its size budget.
    pub(in crate::model) fn ensure_drafter_context(
        &self,
        proposer: &dyn DraftProposer,
        seq: &mut SequenceState,
        ctx: &ForwardContext,
        stream: u64,
    ) {
        // Disjoint field borrows: the proposer state is mutated while the
        // token slice is read. Destructuring is what makes that legal, and it
        // avoids cloning a 12k-token vector on every propose.
        let SequenceState {
            tokens: seq_tokens,
            prompt_len,
            proposer_state,
            ..
        } = seq;
        let Some(prop_state) = proposer_state.as_mut() else {
            return;
        };
        let prompt_len = *prompt_len;
        // ATLAS_MTP_DRAFTER_PREFILL: on the FIRST propose of a sequence,
        // batch-prefill the drafter's KV over the prompt (fresh-state check
        // and quant support live inside prefill_drafter; it fast-returns 0 on
        // every later call). Requires the capture to cover the full prompt —
        // a COLD turn satisfies that; a WARM turn never does, which is the
        // context-blindness defect ATLAS_MTP_CARRY_DRAFTER closes below.
        if !self.mtp_prefill_hidden.is_null() {
            let p = prompt_len;
            let captured = self
                .mtp_prefill_capture_len
                .load(std::sync::atomic::Ordering::Relaxed);
            let cold_prefill_ok = p >= 2 && captured >= p && seq_tokens.len() >= p;
            let carry_on = crate::model::mtp_carry::mtp_carry_drafter_enabled();
            // Both branches below are FIRST-PROPOSE only: `prefill_drafter`
            // enforces that itself (`mtp_state.seq_len != row_base` fast-return),
            // and the carry must not re-run once the drafter owns rows.
            let first_propose = proposer.drafter_rows(prop_state.as_mut()) == 0;
            if cold_prefill_ok {
                // A cold turn builds its own rows, so any carried entry is
                // dead. It MUST be released here: the drafter KV pool holds
                // exactly `max_seq_len / block_size + 1` blocks — one
                // sequence's worth — so a carried entry left alive would
                // starve this prefill's `alloc_block` calls.
                if carry_on && let Some(old) = self.mtp_carry.lock().take() {
                    proposer.free_drafter_kv(&old.block_table);
                }
                if let Err(e) = proposer.prefill_drafter(
                    &seq_tokens[..p],
                    self.mtp_prefill_hidden,
                    prop_state.as_mut(),
                    ctx,
                    stream,
                ) {
                    tracing::warn!("MTP drafter prefill failed (continuing without): {e:#}");
                }
            } else if carry_on && first_propose && p >= 2 {
                // WARM turn: adopt the previous turn's drafter KV and append
                // only this turn's newly-computed span. See `try_carry_drafter`.
                let outcome = self.try_carry_drafter(
                    proposer,
                    seq_tokens,
                    p,
                    prop_state.as_mut(),
                    ctx,
                    stream,
                );
                if crate::model::mtp_carry::mtp_carry_debug() {
                    tracing::info!(
                        "MTP_CARRY adopt: prompt_len={p} store={:?} -> {outcome}",
                        *self.mtp_store_range.lock(),
                    );
                }
            }
        }
    }

    /// ATLAS_MTP_CARRY_DRAFTER: give the drafter this turn's prompt context on
    /// the FIRST propose of a sequence, by adopting the previous turn's
    /// drafter KV and appending only the span this turn actually computed.
    ///
    /// Why not just re-run `prefill_drafter`: measured 1136 ms over 11,947
    /// rows on GB10 (2026-07-21) against a 1134 ms warm TTFT — a full rebuild
    /// spends more TTFT than the ~10% acceptance gain returns on the scored
    /// workload. The append here is proportional to the NEW tokens.
    ///
    /// Conventions (see `mtp_carry` module docs): pair key `k` is
    /// `(embed(t_{k+1}), hidden_k)` with RoPE `k + 1`; `mtp_prefill_hidden`
    /// row `i` is `hidden_i`. Rows are compacted, so a skipped key leaves no
    /// hole — only a missing row, which is the steady state of this row space
    /// anyway.
    ///
    /// Returns the outcome for logging. Never fails the propose: every branch
    /// degrades to "drafter has fewer rows", which costs acceptance, not
    /// correctness, because the target verifies every draft.
    pub(in crate::model) fn try_carry_drafter(
        &self,
        proposer: &dyn DraftProposer,
        seq_tokens: &[u32],
        prompt_len: usize,
        prop_state: &mut dyn crate::speculative::ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> crate::model::mtp_carry::CarryOutcome {
        use crate::model::mtp_carry::{CarryOutcome, hidden_row_offset, plan_append};
        let prompt = &seq_tokens[..prompt_len.min(seq_tokens.len())];
        let Some(entry) = self.mtp_carry.lock().take() else {
            return CarryOutcome::NoCarry;
        };
        let Some((rows, last_key)) = entry.usable_by(prompt) else {
            let common = entry.common_prefix_len(prompt);
            proposer.free_drafter_kv(&entry.block_table);
            return CarryOutcome::PrefixMismatch {
                common,
                entry_rows: entry.rows,
            };
        };
        // `install_drafter_kv` takes ownership on success only; keep a copy of
        // the ids so a refused install frees them instead of leaking.
        let block_ids = entry.block_table.clone();
        if !proposer.install_drafter_kv(prop_state, entry.block_table, rows, Some(last_key)) {
            // Fresh-state precondition violated (the drafter already has rows).
            // Nothing owns these blocks now, so release them here.
            proposer.free_drafter_kv(&block_ids);
            return CarryOutcome::NoCarry;
        }
        let (lo, hi) = *self.mtp_store_range.lock();
        let Some(plan) = plan_append(last_key, prompt.len(), lo, hi) else {
            return CarryOutcome::NoHiddens;
        };
        // `drafter_rows_impl` reads `tokens[r + 1]` and `hiddens` row `r` for
        // row r, and RoPE `pos_base + r`. Row r must be pair key
        // `first_key + r`, i.e. `(embed(t_{first_key+r+1}), hidden_{first_key+r})`
        // at RoPE `first_key + r + 1`.
        let tokens = &prompt[plan.first_key..];
        let hiddens = hidden_row_offset(
            self.mtp_prefill_hidden,
            plan.first_key,
            self.config.hidden_size,
        );
        match proposer.catchup_drafter(
            tokens,
            hiddens,
            rows,
            plan.first_key + 1,
            prop_state,
            ctx,
            stream,
        ) {
            Ok(appended) => CarryOutcome::Adopted {
                rows,
                appended,
                first_key: plan.first_key,
            },
            Err(e) => {
                tracing::warn!("MTP carry append failed (drafter keeps carried rows): {e:#}");
                CarryOutcome::Adopted {
                    rows,
                    appended: 0,
                    first_key: plan.first_key,
                }
            }
        }
    }

    pub(super) fn save_hidden_for_mtp_dispatch(
        &self,
        token_idx: usize,
        _stream: u64,
    ) -> Result<()> {
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        // Residual stream is always BF16, so the saved hidden is BF16.
        let fp32 = 2usize;
        // Save the RAW hidden state (before final_norm), not norm_output.
        // The MTP head applies its own pre_fc_norm_hidden — passing norm_output
        // would double-normalize and degrade prediction accuracy.
        let src = self.buffers.hidden_states().offset(token_idx * h * fp32);
        self.gpu
            .copy_d2d_async(src, self.mtp_hidden_save, h * fp32, stream)?;
        self.last_mtp_hidden_idx
            .store(token_idx, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// ATLAS_MTP_CATCHUP: ring-capture the final hidden of a serially
    /// decoded token (position `pos`), keeping the ring's position range
    /// contiguous (a gap resets the range to just this row).
    pub(super) fn save_hidden_for_catchup_dispatch(
        &self,
        token_idx: usize,
        pos: usize,
    ) -> Result<()> {
        if self.mtp_catchup_ring.is_null() {
            return Ok(());
        }
        let ring_rows = super::super::types::MTP_CATCHUP_RING_ROWS;
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let src = self.buffers.hidden_states().offset(token_idx * h * bf16);
        let dst = self.mtp_catchup_ring.offset((pos % ring_rows) * h * bf16);
        self.gpu.copy_d2d_async(src, dst, h * bf16, stream)?;
        if crate::speculative::mtp_refeed_debug() {
            self.gpu.synchronize(stream)?;
            let fp_src = crate::speculative::hidden_fingerprint(self.gpu.as_ref(), src, h);
            let fp_dst = crate::speculative::hidden_fingerprint(self.gpu.as_ref(), dst, h);
            tracing::info!(
                "REFEED_DBG ring_write label={pos} row={token_idx} slot={} \
                 fp_src={fp_src:016x} fp_dst={fp_dst:016x} match={}",
                pos % ring_rows,
                fp_src == fp_dst,
            );
        }
        let mut meta = self.mtp_catchup_meta.lock();
        let (start, count) = *meta;
        *meta = if count > 0 && pos == start + count {
            // Contiguous append; cap the range at ring capacity by advancing
            // the start once the ring wraps (oldest row overwritten).
            if count == ring_rows {
                (start + 1, ring_rows)
            } else {
                (start, count + 1)
            }
        } else {
            (pos, 1)
        };
        Ok(())
    }

    pub(super) fn run_mtp_propose_dispatch(
        &self,
        token: u32,
        position: usize,
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Option<u32>> {
        let drafts = self.run_mtp_propose_multi(token, position, 1, seq, 0, None)?;
        Ok(drafts.into_iter().next())
    }

    pub(super) fn run_mtp_propose_multi_dispatch(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        _stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        // MTP loads ALL experts on every rank — no EP all_reduce needed.
        // Rank 1 does not participate in MTP propose.
        self.run_mtp_propose_inner(token, position, num_drafts, seq, grammar_bitmask)
    }

    pub(super) fn read_deferred_draft_token_dispatch(&self) -> Result<u32> {
        let proposer = match &self.proposer {
            Some(p) => p.as_ref(),
            None => return Ok(0),
        };
        proposer.read_deferred_draft_token(self.gpu.as_ref())
    }

    pub(super) fn trim_proposer_state_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        _stream: u64,
    ) -> Result<()> {
        let proposer = match &self.proposer {
            Some(p) => p.as_ref(),
            None => return Ok(()),
        };
        let stream = self.gpu.default_stream();
        if let Some(ref mut state) = seq.proposer_state {
            proposer.after_verify(num_accepted, state.as_mut(), stream)?;
        }
        Ok(())
    }
}
