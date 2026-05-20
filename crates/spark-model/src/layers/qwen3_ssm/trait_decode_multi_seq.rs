// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode_multi_seq.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    /// Multi-sequence decode: dispatches the proven single-sequence path per
    /// sequence.
    ///
    /// A previous batched path shared `conv_out`, `gdn_out`, and
    /// `moe_output` regions across sequences, which the per-token MoE call
    /// then rewrote before the next sequence read them — surfacing as
    /// Chinese/multilingual gibberish (#6). Per-sequence dispatch avoids
    /// the aliasing entirely and is effectively free at decode time:
    /// SSM decode is memory-bandwidth-bound and GEMV weights stay in L2
    /// across iterations.
    ///
    /// Per-seq stride uses the actual residual element size (BF16 by
    /// default on GB10 LPDDR5X; FP32 when [`use_fp32_residual`] is on).
    pub(super) fn decode_multi_seq_inner<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let residual_elem = if ctx.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };

        let mut stub_disk = Vec::<u32>::new();
        let mut stub_last_offloaded = Vec::<u32>::new();
        for i in 0..num_seqs {
            let hidden_i = hidden.offset(i * h * residual_elem);
            let residual_i = residual.offset(i * h * residual_elem);
            self.decode(
                hidden_i,
                residual_i,
                states[i],
                kv_cache,
                seq_lens[i],
                &mut block_tables[i].clone(),
                &mut stub_disk,
                &mut stub_last_offloaded,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }
}
