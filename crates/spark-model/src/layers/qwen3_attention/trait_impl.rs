// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::Qwen3AttentionLayer;
use crate::layer::{EmptyLayerState, ForwardContext, LayerState, TransformerLayer};
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
pub fn diag_norm_f32(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    n_elements: usize,
    stream: u64,
    label: &str,
) {
    let _ = gpu.synchronize(stream);
    let mut buf = vec![0f32; n_elements];
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

/// Gemma-4 diagnostic gate. Set ATLAS_DIAG_GEMMA4=1 to enable per-layer
/// hidden-state norm dumps in the decode path. Heavy (one d2h copy per
pub(super) fn gemma4_diag_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_DIAG_GEMMA4").ok().as_deref(),
            Some("1") | Some("true")
        )
    })
}

impl TransformerLayer for Qwen3AttentionLayer {
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
        Ok(Box::new(EmptyLayerState))
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

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn test_alloc_state_returns_empty() {
        let gpu = MockGpuBackend::new();
        assert!(gpu.kernel("norm", "rms_norm").is_ok());
        assert!(gpu.kernel("rope", "rope_forward").is_ok());
        assert!(
            gpu.kernel("paged_decode_fp8", "paged_decode_attn_fp8")
                .is_ok()
        );
    }
}
