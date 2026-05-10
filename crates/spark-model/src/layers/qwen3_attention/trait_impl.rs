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

    /// Q12 Phase 3: batched attention prefill (same-chunk-len, N≥2 streams).
    ///
    /// **Current status:** scaffolded — checks the same-chunk-len constraint
    /// and the availability of a batched paged-prefill kernel; on either
    /// failing, falls through to per-stream sequential calls. The
    /// kernel-batched body using `ops::prefill_attention_paged_fp8_batched`
    /// / `_nvfp4_batched` / `_batched` is documented inline as a TODO and
    /// pending the kernel-validation session.
    ///
    /// Implementation plan (kernel session):
    ///   1. Per-stream sequential `q_proj` / `k_proj` / `v_proj` GEMMs +
    ///      RoPE + KV-cache write — each writes to stacked Q at offset
    ///      `b * q_len * num_q_heads * head_dim`. Q/K/V are stored
    ///      contiguously across batched streams.
    ///   2. Build `block_table_ptrs[n]` device array from each stream's
    ///      `seq.chunked_prefill_meta.block_table` device pointer.
    ///   3. Once: batched paged-prefill via
    ///      `ops::prefill_attention_paged_fp8_batched{,_64}` (or BF16 /
    ///      NVFP4 sibling depending on KV dtype). Grid becomes
    ///      `(num_q_heads, q_chunks, batch_size)` — kernel reads Q/O at
    ///      per-batch offsets and dereferences `block_table_ptrs[b]` to
    ///      get each stream's paged KV view.
    ///   4. Per-stream sequential `o_proj` + residual add — each reads
    ///      from stacked O at `b * q_len * num_q_heads * head_dim`.
    ///
    /// Open issues to address in the kernel session:
    ///   - `reshape_and_cache` (KV-cache write) is per-stream today. Wrap
    ///     in a per-stream loop or add a batched variant. Cheap operation
    ///     so per-stream loop is acceptable.
    ///   - q_offset / kv_len must match across batched streams (kernel
    ///     constraint). Scheduler-side `can_batch_prefill_only` ensures
    ///     same chunk_len; q_offset = prior tokens before this chunk = also
    ///     same when prior prefix matches. Cross-check at dispatch boundary.
    #[allow(clippy::too_many_arguments)]
    fn prefill_batched(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        cu_seqlens: &[usize],
        states: &mut [&mut dyn LayerState],
        kv_cache: &mut PagedKvCache,
        seq_lens_start: &[usize],
        block_tables: &mut [&mut Vec<u32>],
        disk_block_ids: &mut [&mut Vec<u32>],
        disk_last_offloaded_per_layer: &mut [&mut Vec<u32>],
        kv_write_starts: &[usize],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let n = cu_seqlens.len().saturating_sub(1);
        // Eligibility gate. Same-chunk-len check identical to SSM override.
        // Additional check: q_offset (seq_lens_start) must match (since kernel
        // takes one q_offset for the whole batch).
        let same_chunk_len = n >= 2
            && {
                let chunk0 = cu_seqlens[1] - cu_seqlens[0];
                (1..n).all(|i| (cu_seqlens[i + 1] - cu_seqlens[i]) == chunk0)
            };
        let same_q_offset = n >= 2 && seq_lens_start.iter().all(|&s| s == seq_lens_start[0]);
        let batched_kernel_ready = self.prefill_attn_paged_fp8_batched_k.0 != 0
            || self.prefill_attn_paged_batched_k.0 != 0
            || self.prefill_attn_paged_nvfp4_batched_k.0 != 0;
        let _ = (same_chunk_len, same_q_offset, batched_kernel_ready);

        // TODO(kernel-session): kernel-batched paged-prefill path.
        //
        // Pseudocode using ops shipped in commit a96fc67:
        //
        //   let chunk_len = cu_seqlens[1] - cu_seqlens[0];
        //   // (1) per-stream q_proj/k_proj/v_proj + RoPE + reshape_and_cache,
        //   //     writing Q/K/V at stacked offset b * chunk_len * H
        //   for b in 0..n { run_qkv_proj_at_offset(b * chunk_len) }
        //   // (2) stage block_table_ptrs[n] device array
        //   let bt_ptrs_dev = stage_block_table_ptrs(states, block_tables, ctx, stream)?;
        //   // (3) batched paged-prefill (choose op based on KV dtype + q_len)
        //   ops::prefill_attention_paged_fp8_batched(
        //       ctx.gpu, self.prefill_attn_paged_fp8_batched_k,
        //       q_ptr, k_cache, v_cache, o_ptr, bt_ptrs_dev,
        //       n as u32, chunk_len as u32, kv_len as u32, q_offset as u32,
        //       num_q_heads, num_kv_heads, head_dim, cache_block_size,
        //       sliding_window, inv_sqrt_d, k_scale, v_scale, cache_stride,
        //       stream,
        //   )?;
        //   // (4) per-stream o_proj + residual
        //   for b in 0..n { run_o_proj_at_offset(b * chunk_len) }
        //
        // Until validated, fall through to per-stream sequential below.
        let h_d = ctx.config.hidden_size;
        let bf16 = 2usize;
        let mut sit = states.iter_mut();
        let mut bit = block_tables.iter_mut();
        let mut dib = disk_block_ids.iter_mut();
        let mut dlb = disk_last_offloaded_per_layer.iter_mut();
        for i in 0..n {
            let off = cu_seqlens[i];
            let chunk_len = cu_seqlens[i + 1] - off;
            if chunk_len == 0 {
                continue;
            }
            let h_i = hidden.offset(off * h_d * bf16);
            let r_i = residual.offset(off * h_d * bf16);
            self.prefill(
                h_i,
                r_i,
                chunk_len,
                *sit.next().unwrap(),
                kv_cache,
                seq_lens_start[i],
                *bit.next().unwrap(),
                *dib.next().unwrap(),
                *dlb.next().unwrap(),
                kv_write_starts[i],
                ctx,
                stream,
            )?;
        }
        Ok(())
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
