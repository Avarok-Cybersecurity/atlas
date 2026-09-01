// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::Qwen3AttentionLayer;
use crate::layer::{
    BatchedAttnMetadata, EmptyLayerState, ForwardContext, LayerState, TransformerLayer,
};
use crate::layers::FfnComponent;

mod decode_inner;
mod multi_seq;
mod prefill_inner;

/// Debug: read back BF16 GPU tensor and compute L2 norm + first 4 values.
pub(super) fn diag_norm(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    n_elements: usize,
    stream: u64,
    label: &str,
) {
    let _ = gpu.synchronize(stream);
    let mut buf = vec![0u16; n_elements];
    // SAFETY: `buf` is `vec![0u16; n_elements]` on the line above, so
    // `buf.len() == n_elements` and `n_elements * 2 == buf.len() *
    // size_of::<u16>()` — the span is exactly the Vec's buffer, all of it
    // zero-initialised. `bytes` is the sole reference derived from `buf` while it
    // is live: it is dead after the `copy_d2h` below, before `buf.iter()` runs.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_elements * 2) };
    if gpu.copy_d2h(ptr, bytes).is_err() {
        return;
    }
    let vals: Vec<f32> = buf
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let norm: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
    let max_abs: f32 = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let f4 = if vals.len() >= 4 {
        format!(
            "[{:.4},{:.4},{:.4},{:.4}]",
            vals[0], vals[1], vals[2], vals[3]
        )
    } else {
        format!("{:?}", &vals[..vals.len().min(4)])
    };
    tracing::info!("DIAG {label}: norm={norm:.4} max={max_abs:.4} first4={f4} n={n_elements}");
}

/// Debug: read back FP32 GPU tensor and compute L2 norm + first 4 values.
/// Used by the DeepSeek-V4 multi-seq decode diagnostic path (post/comb-attn
/// holographic tensors are FP32). V4-only — no non-V4 caller.
pub fn diag_norm_f32(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    n_elements: usize,
    stream: u64,
    label: &str,
) {
    let _ = gpu.synchronize(stream);
    let mut buf = vec![0f32; n_elements];
    // SAFETY: `buf` is `vec![0f32; n_elements]` on the line above, so
    // `buf.len() == n_elements` and `n_elements * 4 == buf.len() *
    // size_of::<f32>()` — the span is exactly the Vec's buffer, all of it
    // zero-initialised. `bytes` is the sole reference derived from `buf` while it
    // is live: it is dead after the `copy_d2h` below, before `buf.iter()` runs.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_elements * 4) };
    if gpu.copy_d2h(ptr, bytes).is_err() {
        return;
    }
    let norm: f32 = buf.iter().map(|v| v * v).sum::<f32>().sqrt();
    let max_abs: f32 = buf.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let f4 = if buf.len() >= 4 {
        format!("[{:.4},{:.4},{:.4},{:.4}]", buf[0], buf[1], buf[2], buf[3])
    } else {
        format!("{:?}", &buf[..buf.len().min(4)])
    };
    tracing::info!(
        "DIAG {label}: norm={norm:.4} max={max_abs:.4} first4={f4} n={n_elements} (FP32)"
    );
}

// The `OnceLock<bool>` static that lived here is now
// `layers::ops::ModelLevers::gemma4_diag`, resolved when the model is built
// and carried on `ForwardContext`.

impl TransformerLayer for Qwen3AttentionLayer {
    fn uses_local_mla_prefill(&self) -> bool {
        self.mla.is_some()
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn fp8_calibration_frozen(&self) -> Option<bool> {
        self.fp8_calibration
            .as_ref()
            .map(|cal| !cal.is_calibrating())
    }

    /// QSA selection does a host top-k per step — never capturable, and a
    /// graph captured on the dense path would replay wrong attention once
    /// selection activates. Native EXL3 MoE experts (main FFN or the LongCat
    /// shortcut MoE) launch cooperatively — same veto.
    fn decode_graph_unsupported(&self) -> bool {
        self.qsa.is_some() || self.exl3_graph_veto()
    }

    fn exl3_graph_veto(&self) -> bool {
        self.ffn.exl3_native_moe() || self.moe_ffn.as_ref().is_some_and(|f| f.exl3_native_moe())
    }

    fn has_aux_state(&self) -> bool {
        self.qsa.is_some()
    }

    fn snapshot_aux(
        &self,
        state: &dyn LayerState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<Vec<u8>>> {
        let Some(qsa) = self.qsa.as_ref() else {
            return Ok(None);
        };
        let attn = state
            .as_any()
            .downcast_ref::<crate::layer::AttnLayerState>()
            .ok_or_else(|| anyhow::anyhow!("QSA host layer state is not AttnLayerState"))?;
        match attn.qsa.as_ref() {
            Some(st) => Ok(Some(qsa.snapshot_aux(st, gpu, stream)?)),
            // Sequence never reached this layer's ingest: nothing to carry.
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
        let qsa = self
            .qsa
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("restore_aux: no QSA on this layer"))?;
        let attn = state
            .as_any_mut()
            .downcast_mut::<crate::layer::AttnLayerState>()
            .ok_or_else(|| anyhow::anyhow!("QSA host layer state is not AttnLayerState"))?;
        if attn.qsa.is_none() {
            attn.qsa = Some(qsa.new_seq_state(gpu)?);
        }
        qsa.restore_aux(attn.qsa.as_mut().expect("just created"), blob, gpu, stream)
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

    #[allow(clippy::too_many_arguments)]
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
            None, // batched_meta — single-stream
            ctx,
            stream,
        )
    }

    /// Q12 Path B: batched-mode attention prefill via `prefill_inner` with
    /// `batched_meta = Some`. The model-level `prefill_attn_batched_layer`
    /// calls this method. Per-stream block_table is unused under batched
    /// mode (block_table_ptrs from batched_meta carries them); we still
    /// pass an empty Vec to satisfy the signature.
    fn prefill_inner_batched_q12(
        &self,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        num_tokens: usize,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        batched_meta: &BatchedAttnMetadata,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mut empty_state = EmptyLayerState;
        let mut empty_block_table: Vec<u32> = Vec::new();
        let mut empty_disk_block_ids: Vec<u32> = Vec::new();
        let mut empty_disk_last: Vec<u32> = Vec::new();
        self.prefill_inner(
            hidden_stacked,
            residual_stacked,
            num_tokens,
            &mut empty_state,
            kv_cache,
            seq_len_start,
            &mut empty_block_table,
            &mut empty_disk_block_ids,
            &mut empty_disk_last,
            0,
            Some(batched_meta),
            ctx,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
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

    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(crate::layer::AttnLayerState::default()))
    }

    /// Free the QSA indexer carry this sequence lazily attached.
    ///
    /// `alloc_state` hands back an EMPTY `AttnLayerState`; the buffers appear
    /// later, on first use, via `qsa_seq_state`. So the thing to release is
    /// not what `alloc_state` returned — it is whatever the sequence grew.
    /// `take()` makes this idempotent and leaves the state in the same shape
    /// `alloc_state` produced.
    ///
    /// A layer with no QSA indexer (plain attention) never populates the
    /// field, so the `take()` yields `None` and this costs nothing.
    fn release_state(&self, state: &mut dyn LayerState, gpu: &dyn GpuBackend) -> Result<()> {
        let Some(attn) = state
            .as_any_mut()
            .downcast_mut::<crate::layer::AttnLayerState>()
        else {
            return Ok(());
        };
        let Some(mut st) = attn.qsa.take() else {
            return Ok(());
        };
        let Some(qsa) = self.qsa.as_ref() else {
            // Buffers exist but the indexer is gone: nothing can size or
            // free them correctly, and guessing would be worse than saying so.
            anyhow::bail!("release_state: QSA seq state present but layer has no QSA indexer");
        };
        qsa.release_seq_state(&mut st, gpu)
    }

    fn transpose_moe_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_for_prefill(gpu, config)?;
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.transpose_for_prefill(gpu, config)?;
        }
        Ok(())
    }

    fn transpose_moe_gate_up_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_gate_up_for_prefill(gpu, config)?;
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.transpose_gate_up_for_prefill(gpu, config)?;
        }
        Ok(())
    }

    fn set_moe_down_transpose_scratch(
        &mut self,
        scratch_packed: DevicePtr,
        scratch_scale: DevicePtr,
        packed_ptrs_t: DevicePtr,
        scale_ptrs_t: DevicePtr,
    ) {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.set_down_transpose_scratch(
                scratch_packed,
                scratch_scale,
                packed_ptrs_t,
                scale_ptrs_t,
            );
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.set_down_transpose_scratch(
                scratch_packed,
                scratch_scale,
                packed_ptrs_t,
                scale_ptrs_t,
            );
        }
    }

    fn transpose_moe_for_prefill_unified(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_for_prefill_unified(gpu, config)?;
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.transpose_for_prefill_unified(gpu, config)?;
        }
        Ok(())
    }

    fn transpose_moe_for_prefill_hybrid(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_for_prefill_hybrid(gpu, config)?;
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.transpose_for_prefill_hybrid(gpu, config)?;
        }
        Ok(())
    }
}
