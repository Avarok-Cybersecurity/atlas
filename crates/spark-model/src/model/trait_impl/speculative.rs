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
        let capture_gen = seq.mtp_capture_gen;
        // Read before the destructure below borrows `seq` field-wise. This is
        // the identity the carry slot is gated on; see `CarriedDrafter`.
        let session_hash = seq.session_hash;
        // And the ticket for the shared hidden rows; see `StoreRange`.
        let store_gen = seq.mtp_store_gen;
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
            // Ownership check: the shared capture must still be THIS
            // sequence's (stamp == current generation). At C>=2 the seqs
            // prefill back-to-back before any propose, so without this the
            // first propose of every seq but the LAST-prefilled would build
            // drafter KV from its own tokens paired with a DIFFERENT
            // sequence's hiddens. Blind (skip) beats poisoned.
            let owns_capture = capture_gen != 0
                && capture_gen
                    == self
                        .mtp_prefill_capture_gen
                        .load(std::sync::atomic::Ordering::Relaxed);
            let cold_prefill_ok = p >= 2 && captured >= p && seq_tokens.len() >= p && owns_capture;
            let carry_on = crate::model::mtp_carry::mtp_carry_drafter_enabled(&self.levers);
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
                    session_hash,
                    store_gen,
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
        session_hash: u64,
        store_gen: u64,
        prop_state: &mut dyn crate::speculative::ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> crate::model::mtp_carry::CarryOutcome {
        use crate::model::mtp_carry::{CarryOutcome, hidden_row_offset, plan_append};
        let prompt = &seq_tokens[..prompt_len.min(seq_tokens.len())];
        let Some(entry) = self.mtp_carry.lock().take() else {
            return CarryOutcome::NoCarry;
        };
        let Some((rows, last_key)) = entry.usable_by(prompt, session_hash) else {
            // `usable_by` is the single authority on admission; this only
            // LABELS its refusal, by asking the same predicate which of the
            // two rules said no. A foreign-session refusal reported as a
            // prefix mismatch is how the channel stayed invisible.
            let outcome = if entry.session_matches(session_hash) {
                CarryOutcome::PrefixMismatch {
                    common: entry.common_prefix_len(prompt),
                    entry_rows: entry.rows,
                }
            } else {
                CarryOutcome::ForeignSession {
                    entry_session: entry.session_hash,
                    prompt_session: session_hash,
                }
            };
            proposer.free_drafter_kv(&entry.block_table);
            return outcome;
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
        // Only rows THIS sequence wrote are visible; another owner's interval
        // reads as empty, which `plan_append` then refuses.
        //
        // Deliberately placed AFTER `install_drafter_kv`: the carried rows are
        // already proven valid for this session, and refusing to APPEND is no
        // reason to throw them away. It also means the blocks must NOT be freed
        // here — `install_drafter_kv` has taken ownership, and the proposer
        // state releases or re-deposits them.
        let stored = *self.mtp_store_range.lock();
        let (lo, hi) = stored.visible_to(store_gen);
        let Some(plan) = plan_append(last_key, prompt.len(), lo, hi) else {
            return if stored.owner != store_gen && stored.owner != 0 {
                CarryOutcome::ForeignHiddens {
                    owner: stored.owner,
                    expected: store_gen,
                }
            } else {
                CarryOutcome::NoHiddens
            };
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

    /// Batched-verify Phase 2: copy the raw-hidden row `rows[i]` (the
    /// accepted position of sequence i in the just-run batched verify
    /// forward) into stash slot i, BEFORE any propose clobbers the shared
    /// `hidden_states` buffer (every drafter `forward_one` writes into it —
    /// mtp_multi.rs). Same RAW-hidden (pre-final-norm) contract as
    /// `save_hidden_for_mtp_dispatch`.
    pub(super) fn stash_verify_hidden_rows_dispatch(
        &self,
        rows: &[usize],
        _stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            !self.verify_hidden_stash.is_null(),
            "stash_verify_hidden_rows: verify_hidden_stash not allocated (no MTP proposer)"
        );
        anyhow::ensure!(
            rows.len() <= crate::layer::VERIFY_WY_TABLE_SEQS,
            "stash_verify_hidden_rows: {} rows exceeds the {}-slot stash",
            rows.len(),
            crate::layer::VERIFY_WY_TABLE_SEQS
        );
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize; // residual stream is BF16
        for (i, &row) in rows.iter().enumerate() {
            let src = self.buffers.hidden_states().offset(row * h * bf16);
            let dst = self.verify_hidden_stash.offset(i * h * bf16);
            self.gpu.copy_d2d_async(src, dst, h * bf16, stream)?;
        }
        // Keep the ABSOLUTE rows beside the stash: the qwen4_exp proposer
        // reads the accepted target's highway row (`impl_b3.rs`), and the slot
        // index is not that row. Without this the restore below cannot publish
        // a correct `last_mtp_hidden_idx` and every sequence drafts from
        // whichever row the global happens to hold.
        *self
            .verify_stash_rows
            .lock()
            .map_err(|_| anyhow::anyhow!("verify stash rows poisoned"))? = rows.to_vec();
        Ok(())
    }

    /// Batched-verify Phase 3: stash slot `idx` → `mtp_hidden_save` (the MTP
    /// head's input). Stashed-row variant of `save_hidden_for_mtp_dispatch`
    /// for verdicts applied AFTER a propose has overwritten the live rows.
    pub(super) fn save_hidden_for_mtp_from_stash_dispatch(
        &self,
        idx: usize,
        _stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            !self.verify_hidden_stash.is_null(),
            "save_hidden_for_mtp_from_stash: verify_hidden_stash not allocated"
        );
        anyhow::ensure!(
            idx < crate::layer::VERIFY_WY_TABLE_SEQS,
            "save_hidden_for_mtp_from_stash: idx {idx} >= {}",
            crate::layer::VERIFY_WY_TABLE_SEQS
        );
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let src = self.verify_hidden_stash.offset(idx * h * bf16);
        self.gpu
            .copy_d2d_async(src, self.mtp_hidden_save, h * bf16, stream)?;
        // Publish the row this restore corresponds to, exactly as
        // `save_hidden_for_mtp_dispatch` does for the single-sequence path.
        // The qwen4_exp proposer reads the target HIGHWAY row at this index;
        // leaving it stale makes every sequence in a batched step draft from
        // another sequence's row (measured: draft match 0.24 vs 0.86).
        if let Some(&row) = self
            .verify_stash_rows
            .lock()
            .map_err(|_| anyhow::anyhow!("verify stash rows poisoned"))?
            .get(idx)
        {
            self.last_mtp_hidden_idx
                .store(row, std::sync::atomic::Ordering::Relaxed);
        }
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
        // 🔴 Whether rank 1 participates is a PROPERTY OF THE PROPOSER, not of MTP.
        //
        // The Qwen and DeepSeek-V4 MTP modules load every expert on every rank, so their
        // propose is complete on rank 0 alone and a comm would double the output via SUM —
        // that is what the old unconditional comment ("MTP loads ALL experts on every rank")
        // described, and it stays the default.
        //
        // GLM-5.3's MTP block is EP-sharded (144 of 288 experts) with a row-parallel DSA
        // `o_proj`, so rank-0-only means drafting from half of both. Under
        // `ATLAS_MTP_EP_PROPOSE=1` the head tells the worker to run the same propose FIRST,
        // then both ranks issue the same collectives in the same order.
        //
        // 🪤 Order matters: the command and its three scalars must be on the wire BEFORE
        // this rank enters the drafter forward, or the worker is still blocked in
        // `ep_recv_seq_and_cmd` when rank 0 hits its first all-reduce. That is `t58`.
        if self.multi_rank_protocol_active()
            && self.proposer.as_ref().is_some_and(|p| p.needs_comm())
        {
            self.ep_broadcast_cmd_for_seq(
                seq.slot_idx as u32,
                crate::speculative::EP_CMD_MTP_PROPOSE,
            )?;
            self.ep_broadcast_u32(token)?;
            self.ep_broadcast_u32(position as u32)?;
            self.ep_broadcast_u32(num_drafts as u32)?;
            // 🔴 THE FOURTH WORD IS THE WHOLE POINT OF AN ALL-REDUCE HERE.
            //
            // `run_mtp_propose_inner` reads its `target_hidden` from `mtp_hidden_save`, and
            // the worker's command arms never write it — only the head calls
            // `save_hidden_for_mtp`. Without this the worker drafts from a STALE hidden, so
            // the two ranks reduce partials computed from DIFFERENT input vectors and the sum
            // is not the block's output at all. Measured: p1 0.747 -> 0.530, worse than the
            // rank-0-only half-sum it was meant to fix.
            //
            // The index is the one the head's own `save_hidden_for_mtp` just used (verify row
            // 1 on a K=2 accept, row 0 on a reject), latched by
            // `save_hidden_for_mtp_dispatch`. Both ranks ran the same verify forward, so row
            // `idx` of `hidden_states()` holds the same vector on both.
            //
            // 🪤 Paths that save from the verify STASH (`save_hidden_for_mtp_from_stash`, the
            // batched multi-seq verify) do not latch this index — they are multi-seq mode,
            // which this single-sequence propose protocol does not serve.
            self.ep_broadcast_u32(
                self.last_mtp_hidden_idx
                    .load(std::sync::atomic::Ordering::Relaxed) as u32,
            )?;
        }
        self.run_mtp_propose_inner(token, position, num_drafts, seq, grammar_bitmask)
    }

    /// Batched cross-sequence propose (batched K=4 verify path). Target
    /// hiddens are read DIRECTLY from the verify stash rows (`stash_idx[i]`),
    /// so the single-slot `mtp_hidden_save` is never involved. The catchup /
    /// refeed / carry blocks of `run_mtp_propose_inner` are force-disabled in
    /// multi-seq MTP mode (the only mode that reaches this path), so skipping
    /// them here matches the per-seq behavior at cap > 1 exactly.
    pub(super) fn run_mtp_propose_batched_dispatch(
        &self,
        tokens: &[u32],
        positions: &[usize],
        stash_idx: &[usize],
        num_drafts: usize,
        seqs: &mut [&mut SequenceState],
        out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let proposer = match &self.proposer {
            Some(p) => p.as_ref(),
            None => return Ok(None),
        };
        // The confidence clamp is a per-seq propose feature; keep semantics
        // by falling back whenever it is armed.
        if self.levers.draft_conf_tau > 0.0 {
            return Ok(None);
        }
        if self.verify_hidden_stash.is_null() {
            return Ok(None);
        }
        let stream = self.gpu.default_stream();
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
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            decode_step: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
        };
        // First-propose drafter context (cold-turn prefill); fast no-op on
        // every later call — same as the per-seq path.
        for seq in seqs.iter_mut() {
            self.ensure_drafter_context(proposer, seq, &ctx, stream);
        }
        let h = self.config.hidden_size;
        let hiddens: Vec<spark_runtime::gpu::DevicePtr> = if self.config.model_type == "qwen4_exp"
        {
            // This proposer's combiner needs the TARGET HIGHWAY row, not the
            // collapsed hidden. `verify_stash_rows[slot]` is the absolute
            // verify row the stash was filled from — the same row the
            // per-sequence path publishes via `last_mtp_hidden_idx`, which
            // is the highway-row fix's whole point. Without a recorded row
            // there is nothing correct to draft from: decline to the
            // per-sequence path rather than guess.
            let rows = self
                .verify_stash_rows
                .lock()
                .map_err(|_| anyhow::anyhow!("verify stash rows poisoned"))?;
            let hc = self.config.hc_mult.max(1);
            let base = self.buffers.hc_streams();
            let mut out = Vec::with_capacity(stash_idx.len());
            for &i in stash_idx {
                let Some(&row) = rows.get(i) else {
                    return Ok(None);
                };
                out.push(base.offset(row * hc * h * 4));
            }
            out
        } else {
            stash_idx
                .iter()
                .map(|&i| self.verify_hidden_stash.offset(i * h * 2))
                .collect()
        };
        let mut states: Vec<&mut dyn crate::speculative::ProposerState> = Vec::new();
        for seq in seqs.iter_mut() {
            match seq.proposer_state.as_mut() {
                Some(s) => states.push(s.as_mut()),
                None => return Ok(None),
            }
        }
        proposer.propose_batch(
            tokens,
            &hiddens,
            positions,
            num_drafts,
            &mut states,
            &ctx,
            stream,
            out_conf,
        )
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
