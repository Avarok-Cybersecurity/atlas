// SPDX-License-Identifier: AGPL-3.0-only

//! `DraftProposer` implementation for [`MtpHead`] (split from mtp_head.rs for the 500-LoC cap).
use super::*;

impl DraftProposer for MtpHead {
    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(MtpProposerState {
            block_table: Vec::new(),
            seq_len: 0,
            last_num_drafted: 0,
            last_pair_key: None,
        }))
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let mtp_state = state
            .as_any_mut()
            .downcast_mut::<MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid MTP proposer state"))?;

        // Reset chain confidence for this propose (forward_one mins into it).
        self.last_conf_bits
            .store(1.0f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
        let mut drafts = Vec::with_capacity(num_drafts);
        let mut current_token = last_token;
        let mut current_hidden = target_hidden;

        for i in 0..num_drafts {
            // Only the LAST draft gets GPU-side embedding (it's the one
            // used in the next verify step).
            let embed_target = if i == num_drafts - 1 {
                draft_embed_target
            } else {
                None
            };
            // Grammar-masked drafting (num_drafts==1 path only for now).
            // For num_drafts > 1 we would need to speculatively advance the
            // matcher between drafts and roll back before returning; the
            // current scheduler only uses num_drafts==1, so we pass the same
            // mask for every i and warn loudly if K>1 + grammar combine.
            if grammar_bitmask.is_some() && i > 0 {
                tracing::warn!(
                    "MTP grammar-masked drafting called with num_drafts>1 (i={i}); \
                     mask held fixed across draft positions — acceptance may drop."
                );
            }
            let mask_for_draft = grammar_bitmask;
            let draft = self.forward_one(
                current_token,
                current_hidden,
                position + i,
                mtp_state,
                ctx,
                stream,
                embed_target,
                mask_for_draft,
            )?;
            tracing::debug!(
                "MTP propose[{i}]: token={current_token} pos={} mtp_seq_len={} → draft={draft}",
                position + i,
                mtp_state.seq_len,
            );
            drafts.push(draft);
            current_token = draft;
            // For subsequent drafts, use the MTP head's own hidden state
            current_hidden = ctx.buffers.hidden_states();
        }

        mtp_state.last_num_drafted = drafts.len();
        Ok(drafts)
    }

    fn prefill_drafter(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        self.prefill_drafter_impl(prompt_tokens, hiddens, state, ctx, stream)
    }

    fn drafter_rows(&self, state: &mut dyn ProposerState) -> usize {
        state
            .as_any_mut()
            .downcast_mut::<MtpProposerState>()
            .map(|s| s.seq_len)
            .unwrap_or(0)
    }

    fn last_pair_key(&self, state: &mut dyn ProposerState) -> Option<usize> {
        state
            .as_any_mut()
            .downcast_mut::<MtpProposerState>()
            .and_then(|s| s.last_pair_key)
    }

    fn take_drafter_kv(
        &self,
        state: &mut dyn ProposerState,
    ) -> Option<(Vec<u32>, usize, Option<usize>)> {
        let st = state.as_any_mut().downcast_mut::<MtpProposerState>()?;
        if st.block_table.is_empty() || st.seq_len == 0 {
            return None;
        }
        let blocks = std::mem::take(&mut st.block_table);
        let rows = st.seq_len;
        let key = st.last_pair_key;
        // Leave the state exactly as `alloc_state` would: no blocks, no rows,
        // no pair key. `free_state` then has nothing to release.
        st.seq_len = 0;
        st.last_pair_key = None;
        st.last_num_drafted = 0;
        Some((blocks, rows, key))
    }

    fn install_drafter_kv(
        &self,
        state: &mut dyn ProposerState,
        blocks: Vec<u32>,
        rows: usize,
        last_pair_key: Option<usize>,
    ) -> bool {
        let Some(st) = state.as_any_mut().downcast_mut::<MtpProposerState>() else {
            return false;
        };
        // Only ever into a fresh state — otherwise the old blocks would leak.
        if !st.block_table.is_empty() || st.seq_len != 0 {
            return false;
        }
        st.block_table = blocks;
        st.seq_len = rows;
        st.last_pair_key = last_pair_key;
        true
    }

    fn free_drafter_kv(&self, blocks: &[u32]) {
        if !blocks.is_empty() {
            self.kv_cache.lock().free_blocks(blocks);
        }
    }

    fn catchup_drafter(
        &self,
        tokens: &[u32],
        hiddens: DevicePtr,
        row_base: usize,
        pos_base: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        self.drafter_rows_impl(tokens, hiddens, row_base, pos_base, state, ctx, stream)
    }

    fn read_deferred_draft_token(&self, gpu: &dyn GpuBackend) -> Result<u32> {
        self.read_deferred_draft_token(gpu)
    }

    fn last_confidence(&self) -> Option<f32> {
        if crate::speculative::draft_conf_tau() <= 0.0 {
            return None;
        }
        Some(f32::from_bits(
            self.last_conf_bits
                .load(std::sync::atomic::Ordering::Relaxed),
        ))
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let mtp_state = state
            .as_any_mut()
            .downcast_mut::<MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid MTP proposer state"))?;

        // Trim rejected drafts from MTP KV cache.
        // num_drafted was recorded in the last propose() call.
        // We trim `num_drafted - num_accepted` entries.
        // e.g. K=2: drafted 1, accepted 0 → trim 1. accepted 1 → trim 0.
        // e.g. K=3: drafted 2, accepted 0 → trim 2. accepted 1 → trim 1. accepted 2 → trim 0.
        let num_drafted = mtp_state.last_num_drafted.max(1);
        let num_to_trim = mtp_rows_to_trim(
            num_drafted,
            num_accepted,
            crate::speculative::mtp_refeed_accepted_enabled(),
        );
        let old_sl = mtp_state.seq_len;
        if num_to_trim > 0 {
            mtp_state.seq_len = mtp_state.seq_len.saturating_sub(num_to_trim);
            // Trimmed rows have consecutive pair keys; the newest surviving
            // key moves back by the same count.
            if let Some(k) = mtp_state.last_pair_key {
                mtp_state.last_pair_key = Some(k.saturating_sub(num_to_trim));
            }
        }
        tracing::debug!(
            "MTP after_verify: accepted={num_accepted} drafted={num_drafted} trim={num_to_trim} mtp_seq_len: {old_sl} → {}",
            mtp_state.seq_len,
        );
        Ok(())
    }

    fn free_state(&self, _gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let mtp_state = state
            .as_any_mut()
            .downcast_mut::<MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid MTP proposer state"))?;
        if !mtp_state.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&mtp_state.block_table);
            mtp_state.block_table.clear();
        }
        mtp_state.seq_len = 0;
        Ok(())
    }
}
