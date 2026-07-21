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

    /// ATLAS_MTP_DRAFTER_PREFILL: copy this prefill chunk's final-layer hiddens
    /// (`[proc_count, h]` BF16, contiguous at the head of the hidden buffer)
    /// into the whole-prompt capture at row `chunk_start`.
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
        // ATLAS_MTP_CARRY_DRAFTER: a warm turn's chunk starts at the reused-prefix
        // boundary, which the contiguous-from-zero tracker must reject. The carry
        // path consumes the buffer position-indexed, so it wants the write anyway.
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
            self.mtp_prefill_capture_len.store(new_len, Ordering::Relaxed);
        }
        if carry_on {
            let mut r = self.mtp_store_range.lock();
            *r = crate::model::mtp_carry::merge_interval(*r, chunk_start, proc_count);
        }
        Ok(())
    }

    /// Give the drafter its prompt context on the FIRST propose of a sequence.
    /// COLD turn: classic `prefill_drafter`. WARM turn: adopt the previous
    /// turn's drafter KV and append only the new span.
    pub(in crate::model) fn ensure_drafter_context(
        &self,
        proposer: &dyn DraftProposer,
        seq: &mut SequenceState,
        ctx: &ForwardContext,
        stream: u64,
    ) {
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
        if !self.mtp_prefill_hidden.is_null() {
            let p = prompt_len;
            let captured = self
                .mtp_prefill_capture_len
                .load(std::sync::atomic::Ordering::Relaxed);
            let cold_prefill_ok = p >= 2 && captured >= p && seq_tokens.len() >= p;
            let carry_on = crate::model::mtp_carry::mtp_carry_drafter_enabled();
            let first_propose = proposer.drafter_rows(prop_state.as_mut()) == 0;
            if cold_prefill_ok {
                // A cold turn builds its own rows, so any carried entry is dead
                // and MUST be released (the drafter KV pool holds one seq worth).
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

    /// ATLAS_MTP_CARRY_DRAFTER: adopt the previous turn's drafter KV and append
    /// only the span this turn actually computed. Never fails the propose.
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
        let block_ids = entry.block_table.clone();
        if !proposer.install_drafter_kv(prop_state, entry.block_table, rows, Some(last_key)) {
            proposer.free_drafter_kv(&block_ids);
            return CarryOutcome::NoCarry;
        }
        let (lo, hi) = *self.mtp_store_range.lock();
        let Some(plan) = plan_append(last_key, prompt.len(), lo, hi) else {
            return CarryOutcome::NoHiddens;
        };
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
