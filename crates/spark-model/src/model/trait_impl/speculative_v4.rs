// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;

use super::super::types::TransformerModel;
use crate::layer::ForwardContext;
use crate::traits::SequenceState;

impl TransformerModel {
    /// Consume a target prefill chunk's live V4 highway into the MTP cache.
    pub(super) fn try_v4_mtp_prompt_prefill(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        first_position: usize,
        proc_count: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<usize> {
        if std::env::var("ATLAS_V4_MTP_PROMPT_PREFILL").ok().as_deref() != Some("1")
            || self.config.model_type != "deepseek_v4"
            || proc_count == 0
        {
            return Ok(0);
        }
        let Some(proposer) = self.proposer.as_ref() else {
            return Ok(0);
        };
        let Some(prop_state) = seq.proposer_state.as_mut() else {
            return Ok(0);
        };

        // The final prompt hidden/highway feeds the first immediate proposal.
        if is_last_chunk {
            self.save_hidden_for_mtp_dispatch(proc_count - 1, stream)?;
        }

        let available = tokens
            .len()
            .saturating_sub(first_position.saturating_add(1));
        let rows = proc_count.min(available);
        let current_rows = proposer.drafter_rows(prop_state.as_mut());
        let result = if rows == 0 || current_rows != first_position {
            if rows > 0 {
                tracing::debug!(
                    "V4 MTP prompt prefill skipped non-contiguous chunk: \
                     state_rows={current_rows} first_position={first_position} rows={rows}"
                );
            }
            Ok(0)
        } else {
            let next_tokens = &tokens[first_position + 1..first_position + 1 + rows];
            let ctx = ForwardContext {
                buffers: &self.buffers,
                gpu: self.gpu.as_ref(),
                config: &self.config,
                attn_metadata: None,
                profile: false,
                comm: None,
                graph_capture: false,
                gdn_exact_replay: false,
                token_ids: Some(self.buffers.token_ids()),
                routed_lora_layers: None,
                midchunk_capture: None,
            };
            proposer.prefill_v4_stream_rows(
                next_tokens,
                self.buffers.hc_streams(),
                first_position,
                prop_state.as_mut(),
                &ctx,
                stream,
            )
        };

        // The target LM head still owns first-token logits. Restore only its
        // final hidden row; the first proposal restores the saved V4 highway.
        if is_last_chunk {
            let h = self.config.hidden_size;
            self.gpu.copy_d2d_async(
                self.mtp_hidden_save,
                self.buffers
                    .hidden_states()
                    .offset((proc_count - 1) * h * 2),
                h * 2,
                stream,
            )?;
        }
        result
    }
}
