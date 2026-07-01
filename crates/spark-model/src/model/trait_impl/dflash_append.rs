// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash EAGLE ctx-slot append helpers backing `Model::dflash_accept_append`,
//! `dflash_eagle_accept_append`, and `dflash_eagle_kgamma_append`. Extracted
//! from `mod.rs` to keep the trait-impl file under the 500-LoC cap.

use anyhow::Result;

use super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    pub(super) fn dflash_accept_append_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        let base = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        let proposer = match &self.proposer {
            Some(p) => p.as_ref(),
            None => return Ok(()),
        };
        let n_layers = self.dflash_capture_layers.len();
        if n_layers == 0 {
            return Ok(());
        }
        let ctx_slot_bytes = n_layers * self.config.hidden_size * 2;
        let save_1 = base.offset(ctx_slot_bytes); // second half = row 1 (draft hidden)
        let prop_state = seq
            .proposer_state
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("no proposer state"))?;
        let stream = self.gpu.default_stream();
        // The accepted draft was emitted at position seq_len-1 (seq_len was already
        // incremented past it). actual_pos records the true sequence position so
        // forward_block assigns the correct RoPE rotation to this ctx K-vector.
        let actual_pos = seq.seq_len.saturating_sub(1) as i32;
        proposer.append_ctx_slot(
            save_1,
            actual_pos,
            prop_state.as_mut(),
            self.gpu.as_ref(),
            stream,
        )
    }

    pub(super) fn dflash_eagle_accept_append_dispatch(
        &self,
        seq: &mut SequenceState,
    ) -> Result<()> {
        let base = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        let proposer = match &self.proposer {
            Some(p) => p.as_ref(),
            None => return Ok(()),
        };
        let n_layers = self.dflash_capture_layers.len();
        if n_layers == 0 {
            return Ok(());
        }
        let ctx_slot_bytes = n_layers * self.config.hidden_size * 2;
        let save_0 = base; // 1st half = row 0 (last_token@N hidden)
        let save_1 = base.offset(ctx_slot_bytes); // 2nd half = row 1 (draft@N+1 hidden)
        // Positions computed before the proposer_state borrow. On K=2 accept
        // seq_len = N+2, so row 0 → N (= seq_len-2), row 1 → N+1 (= seq_len-1).
        let pos_row0 = (seq.seq_len as i32) - 2;
        let pos_row1 = (seq.seq_len as i32) - 1;
        let prop_state = seq
            .proposer_state
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("no proposer state"))?;
        let stream = self.gpu.default_stream();
        // EAGLE order: row 0 (older) THEN row 1 (freshest) — so forward_block's
        // freshest ctx slot is row 1 = the hidden that generated the bonus.
        proposer.append_ctx_slot(
            save_0,
            pos_row0.max(0),
            prop_state.as_mut(),
            self.gpu.as_ref(),
            stream,
        )?;
        proposer.append_ctx_slot(
            save_1,
            pos_row1.max(0),
            prop_state.as_mut(),
            self.gpu.as_ref(),
            stream,
        )?;
        // One-shot: tell the upcoming propose() to skip its own decode-append.
        if let Some(d) = prop_state
            .as_any_mut()
            .downcast_mut::<crate::layers::DflashProposerState>()
        {
            d.skip_next_decode_append = true;
        }
        Ok(())
    }

    pub(super) fn dflash_eagle_kgamma_append_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        base_pos: usize,
    ) -> Result<()> {
        let base = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        let proposer = match &self.proposer {
            Some(p) => p.as_ref(),
            None => return Ok(()),
        };
        let n_layers = self.dflash_capture_layers.len();
        if n_layers == 0 {
            return Ok(());
        }
        let ctx_slot_bytes = n_layers * self.config.hidden_size * 2;
        let prop_state = seq
            .proposer_state
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("no proposer state"))?;
        let stream = self.gpu.default_stream();
        // Append rows 0..=num_accepted at positions base_pos..=base_pos+num_accepted.
        // Row t = dflash_hidden_save.offset(t * ctx_slot_bytes) (row-major K-row
        // buffer). Row num_accepted is appended LAST → freshest ctx slot = the
        // hidden that generated the bonus (EAGLE). On REJECT (num_accepted=0) this
        // appends only row 0 @ base_pos = the generator of v0 — already EAGLE-correct.
        for t in 0..=num_accepted {
            let row = base.offset(t * ctx_slot_bytes);
            let pos = (base_pos + t) as i32;
            proposer.append_ctx_slot(row, pos, prop_state.as_mut(), self.gpu.as_ref(), stream)?;
        }
        // One-shot: tell the upcoming propose() to skip its own decode-append
        // (row 0 already appended above).
        if let Some(d) = prop_state
            .as_any_mut()
            .downcast_mut::<crate::layers::DflashProposerState>()
        {
            d.skip_next_decode_append = true;
        }
        Ok(())
    }
}
