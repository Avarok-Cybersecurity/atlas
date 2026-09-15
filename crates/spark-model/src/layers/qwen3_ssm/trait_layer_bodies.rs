// SPDX-License-Identifier: AGPL-3.0-only

//! The bodies of the fatter `TransformerLayer` methods on
//! `Qwen3SsmLayer`. A trait impl cannot span two files, so the bodies move
//! here as inherent methods and the trait methods in `trait_layer.rs` call
//! them -- which keeps that file under the 500-line cap.

use super::*;

impl Qwen3SsmLayer {
    pub(super) fn decode_prestage_body(
        &self,
        token: u32,
        state: &mut dyn LayerState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if let Some(ple) = self.ple.as_ref() {
            let st = ple_seq_state(ple, state, gpu)?;
            ple.prestage(st, &[token], gpu, stream)?;
        }
        Ok(())
    }

    pub(super) fn snapshot_aux_into_body(
        &self,
        state: &dyn LayerState,
        gpu: &dyn GpuBackend,
        stream: u64,
        dst: &mut [u8],
    ) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        let ple = self
            .ple
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("snapshot_aux_into: no PLE layer"))?;
        let ssm = state
            .as_any()
            .downcast_ref::<crate::layer::SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("PLE host layer state is not SsmLayerState"))?;
        let st = ssm
            .ple
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("snapshot_aux_into: no PLE sequence state"))?;
        ple.snapshot_aux_into(st, gpu, stream, dst)
    }

    pub(super) fn snapshot_aux_body(
        &self,
        state: &dyn LayerState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<Vec<u8>>> {
        let Some(ple) = self.ple.as_ref() else {
            return Ok(None);
        };
        let ssm = state
            .as_any()
            .downcast_ref::<crate::layer::SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("PLE host layer state is not SsmLayerState"))?;
        match ssm.ple.as_ref() {
            Some(st) => Ok(Some(ple.snapshot_aux(st, gpu, stream)?)),
            // Sequence never ran this layer (snapshot before first pass):
            // nothing to carry, and restore-side declines aux-less slots.
            None => Ok(None),
        }
    }

    pub(super) fn restore_aux_body(
        &self,
        state: &mut dyn LayerState,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let ple = self
            .ple
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("restore_aux: no PLE on this layer"))?;
        let st = ple_seq_state(ple, state, gpu)?;
        ple.restore_aux(st, blob, gpu, stream)
    }

    pub(super) fn commit_verify_row_body(
        &self,
        state: &mut dyn LayerState,
        row: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let Some(ple) = self.ple.as_ref() else {
            return Ok(());
        };
        let Some(ssm) = state
            .as_any_mut()
            .downcast_mut::<crate::layer::SsmLayerState>()
        else {
            return Ok(());
        };
        let Some(st) = ssm.ple.as_mut() else {
            return Ok(());
        };
        ple.rewind_verify_row(st, row, gpu, stream)
    }

    pub(super) fn replay_verify_rows_body(
        &self,
        state: &mut dyn LayerState,
        num_accepted: usize,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if num_accepted == 0 {
            return Ok(());
        }
        let Some(ssm) = state
            .as_any_mut()
            .downcast_mut::<crate::layer::SsmLayerState>()
        else {
            return Ok(());
        };
        if ssm.replay_inputs.is_empty() {
            return Ok(());
        }
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let nv = ctx.config.linear_num_value_heads;
        let bf16 = 2usize;
        let fp32 = 4usize;
        // Row 0 of the shared decode scratch: the verify forward is complete
        // and the next one has not started (the scheduler applies verdicts
        // between forwards), and this runs on the default stream, so staging
        // each replayed token through row 0 cannot race the forward.
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let gates_buf = ctx.buffers.ssm_gates();
        for t in 0..num_accepted {
            let Some(&row) = ssm.replay_inputs.get(t) else {
                anyhow::bail!(
                    "replay rollback: {num_accepted} rows accepted but only {} cached — \
                     the verify window and the replay ring disagree",
                    ssm.replay_inputs.len()
                );
            };
            ctx.gpu
                .copy_d2d_async(row, deinterleaved, qkvz_size * bf16, stream)?;
            ctx.gpu.copy_d2d_async(
                row.offset(qkvz_size * bf16),
                gates_buf,
                nv * 2 * fp32,
                stream,
            )?;
            let args = self.conv_gdn_args_single(ctx, 1, deinterleaved, gates_buf, stream);
            self.decode_batched_conv_gdn(ssm, ctx, &args)?;
        }
        Ok(())
    }

    pub(super) fn decode_batched_body(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // K-row batched GDN verify under the highway (#753 item B, the
        // single-sequence axis). `decode_batched_inner_hc` replaces the
        // residual bracket with hc_pre/hc_post around the SAME residual-free
        // block this path uses, so the highway is not double-counted.
        //
        // ARMED BY ENV, not by `hc.is_some()`. The refusal below still guards
        // every OTHER caller: `decode_verify_dispatch` (verify_a.rs) mixes
        // per-token attention `decode()` with a K-row SSM `decode_batched()`,
        // and those two disagree about which highway row a stream belongs to
        // (the buffer is `[T, hc, H]`). Only `verify_hc.rs`, which runs a
        // uniform K on every layer, may take this path.
        if self.hc.is_some() && super::super::trait_decode_batched_hc::hc_batched_verify_enabled() {
            return self.decode_batched_inner_hc(hidden, num_tokens, state, ctx, stream);
        }
        // v1 is C=1 only under an mHC highway: these paths keep their own
        // residual bookkeeping, which the highway replaces. Refusing is the
        // point — a batched GDN step running on an unmixed stream produces
        // plausible, wrong activations. Avarok #753.
        self.refuse_batched_under_hc("decode_batched")?;
        self.decode_batched_inner(
            hidden,
            residual,
            num_tokens,
            super::super::trait_decode_batched::GdnStates::Single(state),
            ctx,
            stream,
        )
    }

    pub(super) fn prefill_body(
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
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Under an mHC highway the residual bookkeeping is completely
        // different — the highway IS the residual — so this is a second entry
        // path, not a flag on the first. See `trait_prefill_hc.rs`.
        if self.hc.is_some() {
            return self.prefill_inner_hc(hidden, num_tokens, state, seq_len_start, ctx, stream);
        }
        self.prefill_inner(
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
            ctx,
            stream,
        )
    }

    pub(super) fn prefill_phase3_body(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        token_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_phase3_inner(
            hidden,
            residual,
            num_tokens,
            gdn_bufs,
            token_offset,
            ctx,
            stream,
        )
    }
}
