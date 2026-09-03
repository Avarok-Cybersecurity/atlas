// SPDX-License-Identifier: AGPL-3.0-only

//! `TransformerLayer` impl for [`Qwen3SsmLayer`] — the trait surface that
//! forwards into the `trait_*` sibling modules holding the actual phases.
//! Split out of `mod.rs` to keep it under the 500-LoC cap.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::Qwen3SsmLayer;
use crate::layer::{ForwardContext, GdnPrefillBuffers, LayerState, TransformerLayer};

impl TransformerLayer for Qwen3SsmLayer {
    fn gdn_fold_accepted(
        &self,
        gpu: &dyn GpuBackend,
        h_table: DevicePtr,
        na_tab: DevicePtr,
        k_rows: usize,
        n: usize,
        stream: u64,
    ) -> Result<bool> {
        // Flag-driven, not host-armed: the batched verify replays as a CUDA
        // graph, so the only reliable record of WHICH kernel ran is the device
        // word the woa twin writes inside the graph. The fold kernel reads it
        // and either folds the stash (woa ran) or performs the parent's
        // partial-accept restore from the Hi tables (parent ran). Either way
        // h is committed here, so the caller skips its h restore.
        let _ = self
            .woa_armed
            .swap(false, std::sync::atomic::Ordering::AcqRel);
        if !super::gdn_flags::gdn_woa_enabled()
            || self.gdn_wy4_fold_k.0 == 0
            || self.gdn_wy4_clear_k.0 == 0
            || self.woa_flag.is_null()
            || h_table.is_null()
        {
            return Ok(false);
        }
        let [nk, nv, kd, vd] = self.woa_dims;
        let hi_tables = h_table.offset(crate::layer::VERIFY_WY_TABLE_STRIDE_BYTES);
        crate::layers::ops::gdn_wy4_fold(
            gpu,
            self.gdn_wy4_fold_k,
            h_table,
            self.woa_stash,
            na_tab,
            hi_tables,
            crate::layer::VERIFY_WY_TABLE_SEQS as u32,
            self.woa_flag,
            k_rows as u32,
            n as u32,
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            self.woa_stash_seq_floats as u32,
            stream,
        )?;
        crate::layers::ops::gdn_wy4_flag_clear(gpu, self.gdn_wy4_clear_k, self.woa_flag, stream)?;
        Ok(true)
    }

    /// Downcast hook so the LoRA install walk can reach this layer's MoE FFN
    /// (Feature-1: routed-expert/router deltas exist on GDN layers too).
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    /// PLE's host half (hash + NVMe fault-in + slot upload), hoisted before
    /// graph replay/capture. No-op on the 47 layers without a PLE site.
    fn decode_prestage(
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

    fn has_aux_state(&self) -> bool {
        self.ple.is_some()
    }

    /// PLE's per-seq host hash on the hc multi-seq decode path is
    /// capture-illegal (pageable reads); the single-decode path prestages
    /// around it, the batched path does not — veto batched graphs.
    fn decode_graph_unsupported(&self) -> bool {
        self.ple.is_some()
    }

    fn snapshot_aux(
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

    fn restore_aux(
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

    fn decode_prestage_rearm(&self, state: &mut dyn LayerState) {
        if let Some(ple) = self.ple.as_ref()
            && let Some(ssm) = state
                .as_any_mut()
                .downcast_mut::<crate::layer::SsmLayerState>()
            && let Some(st) = ssm.ple.as_mut()
        {
            ple.rearm(st);
        }
    }

    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.hc.is_some() {
            return self.decode_inner_hc(hidden, state, ctx, stream);
        }
        self.decode_inner(
            hidden,
            residual,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    fn decode_batched(
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
        // v1 is C=1 only under an mHC highway: these paths keep their own
        // residual bookkeeping, which the highway replaces. Refusing is the
        // point — a batched GDN step running on an unmixed stream produces
        // plausible, wrong activations. Avarok #753.
        self.refuse_batched_under_hc("decode_batched")?;
        self.decode_batched_inner(
            hidden,
            residual,
            num_tokens,
            super::trait_decode_batched::GdnStates::Single(state),
            ctx,
            stream,
        )
    }

    fn decode_verify_multi<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        n_seqs: usize,
        ks: &[usize],
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        wy_tables: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.refuse_batched_under_hc("decode_verify_multi")?;
        anyhow::ensure!(
            states.len() == n_seqs && ks.len() == n_seqs,
            "decode_verify_multi: states/ks/n mismatch"
        );
        let num_tokens: usize = ks.iter().sum();
        self.decode_batched_inner(
            hidden,
            residual,
            num_tokens,
            super::trait_decode_batched::GdnStates::Multi {
                states,
                ks,
                wy_tables,
            },
            ctx,
            stream,
        )
    }

    fn decode_multi_seq<'a, 'b: 'a>(
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
        if self.hc.is_some() {
            // #753 item B milestone 2: the highway replaces the residual the
            // non-hc path folds into its fused norm kernels; run the
            // hc-bracketed variant instead of refusing.
            return self.decode_multi_seq_inner_hc(hidden, num_seqs, states, seq_lens, ctx, stream);
        }
        self.decode_multi_seq_inner(
            hidden,
            residual,
            num_seqs,
            states,
            kv_cache,
            seq_lens,
            block_tables,
            ctx,
            stream,
        )
    }

    fn prefill(
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

    fn is_ssm_layer(&self) -> bool {
        self.is_ssm_layer_inner()
    }

    fn prefill_phase1(
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
        token_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
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
            token_offset,
            ctx,
            stream,
        )
    }

    fn prefill_phase1_proj_batched(
        &self,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        total_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_phase1_proj_batched_inner(
            hidden_stacked,
            residual_stacked,
            total_tokens,
            gdn_bufs,
            ctx,
            stream,
        )
    }

    fn prefill_phase1_conv1d_one(
        &self,
        state: &mut dyn LayerState,
        token_offset: usize,
        len: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_phase1_conv1d_one_inner(state, token_offset, len, gdn_bufs, ctx, stream)
    }

    fn prefill_phase1_l2_batched(
        &self,
        total_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_phase1_l2_batched_inner(total_tokens, gdn_bufs, ctx, stream)
    }

    fn prefill_gdn_full(
        &self,
        state: &mut dyn LayerState,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_gdn_full_inner(state, gdn_bufs, ctx, stream)
    }

    fn prefill_gdn_full_batched(
        &self,
        h_state_ptrs: DevicePtr,
        gdn_bufs: &GdnPrefillBuffers,
        batch_size: u32,
        chunk_len: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_gdn_full_batched_inner(
            h_state_ptrs,
            gdn_bufs,
            batch_size,
            chunk_len,
            ctx,
            stream,
        )
    }

    fn prefill_gdn_full_batched_fla_varlen(
        &self,
        h_state_ptrs: DevicePtr,
        gdn_bufs: &GdnPrefillBuffers,
        batch_size: u32,
        cu_seqlens: DevicePtr,
        max_num_chunks: u32,
        total_nt: usize,
        max_seqlen: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        self.prefill_gdn_full_batched_fla_varlen_inner(
            h_state_ptrs,
            gdn_bufs,
            batch_size,
            cu_seqlens,
            max_num_chunks,
            total_nt,
            max_seqlen,
            ctx,
            stream,
        )
    }

    fn prefill_phase3(
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

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        self.alloc_state_inner(gpu)
    }
}

/// The PLE per-seq carry from a sequence's [`SsmLayerState`], lazily created
/// on first use. Errors if the state is not an `SsmLayerState`.
fn ple_seq_state<'a>(
    ple: &crate::layers::ple::PleLayer,
    state: &'a mut dyn LayerState,
    gpu: &dyn GpuBackend,
) -> Result<&'a mut crate::layers::ple::PleSeqState> {
    let ssm = state
        .as_any_mut()
        .downcast_mut::<crate::layer::SsmLayerState>()
        .ok_or_else(|| anyhow::anyhow!("PLE host layer state is not SsmLayerState"))?;
    if ssm.ple.is_none() {
        ssm.ple = Some(ple.new_seq_state(gpu)?);
    }
    Ok(ssm.ple.as_mut().expect("just created"))
}
