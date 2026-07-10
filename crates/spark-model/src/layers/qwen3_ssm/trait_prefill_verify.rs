// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash K=γ verify entry into the batched phase-1 pipeline.

use super::*;
impl Qwen3SsmLayer {
    /// DFlash K=γ verify: run the full batched phase-1 pipeline once over
    /// `num_tokens`, writing per-token conv_state intermediates inline from
    /// the contiguous intermediate pool (conv_state_intermediates[t] =
    /// base + t * conv_dim * d_conv floats). Ok(false) → caller falls back
    /// to the per-token phase-1 loop.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_phase1_verify_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_write_start: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if self.conv1d_prefill_inter_k.0 == 0 {
            return Ok(false);
        }
        let (inter_base, inter_stride) = {
            let ssm_state = state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;
            if ssm_state.conv_state_intermediates.len() < num_tokens.saturating_sub(1) {
                return Ok(false);
            }
            let conv_dim = ctx.config.linear_num_key_heads * ctx.config.linear_key_head_dim * 2
                + ctx.config.linear_num_value_heads * ctx.config.linear_value_head_dim;
            let d_conv = ctx.config.linear_conv_kernel_dim;
            (
                ssm_state.conv_state_intermediates[0],
                (conv_dim * d_conv) as u32,
            )
        };
        self.prefill_phase1_inner(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len_start,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_write_start,
            gdn_bufs,
            0, // token_offset: single batched call covers all k tokens
            ctx,
            stream,
            Some((inter_base, inter_stride)),
        )?;
        Ok(true)
    }
}
