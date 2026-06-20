// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3-Next SSM (Gated Delta Net) layer implementing TransformerLayer.
//!
//! Corrected pipeline matching the HuggingFace reference implementation:
//!   1. QKVZ projection (interleaved output)
//!   2. Deinterleave QKVZ → sequential [Q | K | V | Z]
//!   3. BA projection (interleaved output)
//!   4. Compute GDN gates: gate = exp(-A * softplus(alpha + dt_bias)), beta = sigmoid(b)
//!   5. Conv1d update on [Q | K | V] concatenated (d_inner=8192)
//!   6. Split conv output → Q', K', V'
//!   7. GDN decode (Q', K', V', gate, beta) — kernel handles GQA internally
//!   8. Gated RMS norm (GDN output, Z gate)
//!   9. Output projection [value_dim → hidden_size]
//!  10. MoE FFN

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{
    ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::FfnComponent;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Fp8Weight, QuantizedWeight, SsmWeights};

/// Qwen3-Next SSM/GDN layer (36 of 48 layers).
///
/// Supports two QKVZ projection modes:
/// - **Interleaved** (80B): `w4a16_gemv_qkvz` or GEMV + `deinterleave_qkvz`
/// - **Sequential** (3.5-35B): plain GEMV → `[Q|K|V|Z]` already in order
#[allow(dead_code)]
pub struct Qwen3SsmLayer {
    input_norm: DenseWeight,
    ssm: SsmWeights,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
    // NVFP4-quantized QKVZ weight (quarters bandwidth vs BF16)
    qkvz_nvfp4: Option<QuantizedWeight>,
    // Transposed [K/2, N] copy for coalesced w4a16_gemm reads (prefill)
    qkvz_nvfp4_t: Option<QuantizedWeight>,
    // Transposed out_proj for prefill GEMM
    out_proj_nvfp4_t: Option<QuantizedWeight>,
    // BF16 out_proj for models where SSM weights are not pre-quantized
    pub out_proj_dense: Option<DenseWeight>,
    // FP8 E4M3 checkpoint weights for native FP8 serving (w8a16_gemv LUT kernel)
    qkvz_fp8w: Option<Fp8Weight>,
    out_proj_fp8w: Option<Fp8Weight>,
    /// When true, QKVZ projection output is already sequential [Q|K|V|Z].
    /// Skips the deinterleave kernel (used by Qwen3.5 where QKV+Z are
    /// concatenated at load time rather than interleaved per-group).
    sequential_qkvz: bool,
    // Kernels — decode path (single-token GEMV)
    rms_norm_residual_k: KernelHandle,
    gated_rms_norm_k: KernelHandle,
    gated_rms_norm_f32_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    w8a16_gemv_k: KernelHandle,
    w4a16_gemv_qkvz_k: KernelHandle,
    deinterleave_k: KernelHandle,
    conv1d_k: KernelHandle,
    conv1d_l2norm_k: KernelHandle,
    conv1d_l2norm_f32_k: KernelHandle,
    gdn_k: KernelHandle,
    gdn_f32_k: KernelHandle,
    ba_gates_k: KernelHandle,
    residual_add_k: KernelHandle,
    l2_norm_k: KernelHandle,
    residual_add_rms_norm_k: KernelHandle,
    /// Dual-output (bf16 + f32) MoE-input norm for ATLAS_FP32_ROUTING. Zero if absent.
    residual_add_rms_norm_gatef32_k: KernelHandle,
    gated_rms_norm_prefill_k: KernelHandle,
    // Kernels — batched verification path (multi-token GEMM)
    w4a16_gemm_k: KernelHandle,
    w4a16_gemm_t_k: KernelHandle, // Transposed B layout [K/2, N] — K_STEP_T=32
    w4a16_gemm_t_k64_k: KernelHandle, // K64 variant: K_STEP_T=64, halves outer loop
    w4a16_gemm_t_m128_k: KernelHandle, // M128 variant: 2 M-chunks per CTA, halves B re-reads
    w4a16_gemv_batch2_k: KernelHandle,
    dense_gemm_k: KernelHandle,
    dense_gemm_pipelined_k: KernelHandle,
    gdn_prefill_k: KernelHandle,
    gdn_prefill_split_k: KernelHandle,
    gdn_prefill_split4_k: KernelHandle,
    gdn_prefill_persistent_k: KernelHandle,
    gdn_prefill_persistent_wy4_k: KernelHandle,
    /// FLA multi-kernel chunked prefill (baked default for 128-dim GDN): recompute_wu →
    /// chunk_delta_h_ksplit (k-split occupancy) → chunk_fwd_o. 1.75x vs wy4 @16k,
    /// token-equal (cos=1.0 vs scalar). Three handles; all must be non-null.
    gdn_prefill_fla_recompute_wu_k: KernelHandle,
    gdn_prefill_fla_chunk_delta_h_k: KernelHandle,
    gdn_prefill_fla_chunk_fwd_o_k: KernelHandle,
    /// WY32 chunked prefill: processes 32 tokens per WY iteration with H in
    /// shared memory. ~30x faster than per-token for 14k+ sequences.
    gdn_prefill_wy32_k: KernelHandle,
    // ── Q12 Phase 2b: same-chunk-len batched GDN prefill kernels ──
    // Each takes `float* const* h_state_ptrs` plus stacked QKV/gate/beta/output.
    // Used by `Qwen3SsmLayer::prefill_batched` when N≥2 streams have matching
    // chunk_len. Null on targets that don't carry the corresponding kernel.
    gdn_prefill_wy32_batched_k: KernelHandle,
    gdn_prefill_persistent_batched_k: KernelHandle,
    gdn_prefill_persistent_wy4_batched_k: KernelHandle,
    gdn_prefill_split4_batched_k: KernelHandle,
    compute_gdn_gates_k: KernelHandle,
    ba_gates_prefill_k: KernelHandle,
    // Kernels — prefill (multi-token sequential)
    conv1d_prefill_k: KernelHandle,
    // Kernels — fused chunk2 path (2-token verification)
    gdn_chunk2_k: KernelHandle,
    conv1d_chunk2_k: KernelHandle,
    // Kernels — fused chunk3 path (3-token verification)
    gdn_chunk3_k: KernelHandle,
    w4a16_gemv_batch3_k: KernelHandle,
    // Kernels — WY-chunkwise path (2-pass verification)
    gdn_wy2_k: KernelHandle,
    gdn_wy3_k: KernelHandle,
    gdn_wy4_k: KernelHandle,
    /// STAGE 1 fused K=2 MTP-verify epilogue: conv1d+L2norm ×2 and
    /// gated-RMS-norm ×2 each folded into a single launch. Dispatched only
    /// when the `ATLAS_GDN_FUSED_VERIFY` env flag is set (default OFF); the
    /// per-token path runs unchanged otherwise. Bit-identical (cos == 1.0).
    gdn_verify_fused_conv_k2_k: KernelHandle,
    gdn_verify_fused_norm_k2_k: KernelHandle,
    /// WY-Chunkwise K=17 GDN verify (DFlash γ+1). Only present in
    /// qwen3.6-35b-a3b's PTX module set; NULL handle for other targets,
    /// in which case decode_batched(K=17) falls through to the sequential
    /// per-token path.
    gdn_wy17_k: KernelHandle,
    // State allocation sizes (pre-computed from config)
    h_state_bytes: usize,
    conv_state_bytes: usize,
    // Pre-dequanted FP8 weights for zero-overhead prefill GEMMs
    qkvz_fp8: Option<DevicePtr>,
    out_proj_fp8: Option<DevicePtr>,
    fp8_gemm_k: KernelHandle,
    fp8_gemm_t_m128_k: KernelHandle, // M128: halves B re-reads for out_proj at ISL > 128
    // Block-scaled W8A16 prefill kernels (preferred over single-scale
    // fp8_gemm_n128 when block-scaled FP8 weights are available — matches
    // vLLM's per-128-block scale precision instead of single-scale).
    w8a16_gemm_k: KernelHandle,
    // Pipelined (cp.async) rewrite of w8a16_gemm: bit-identical, ~4.6× faster.
    // KernelHandle(0) when not linked into the image. Gated ON only when
    // ATLAS_W8A16_PIPELINED=1 (default OFF — production dispatch unchanged).
    w8a16_gemm_pipelined_k: KernelHandle,
    w8a16_gemm_t_k: KernelHandle,
    // W8A8 + FP32 epilogue (vLLM-equivalent) prefill kernels.
    // `per_token_group_quant_fp8` produces FP8 activations + per-token-per-128
    // FP32 scale; `fp8_gemm_t_blockscaled` consumes both with FP8 MMA and
    // applies a_scale × b_scale in the FP32 epilogue. Gated behind
    // `ATLAS_FP8_W8A8=1` for staged rollout.
    per_token_group_quant_fp8_k: KernelHandle,
    fp8_gemm_t_blockscaled_k: KernelHandle,
}

// ── Sub-files (split for ≤500 LoC) ────────────────────────────────────────
mod debug;
mod init;
mod ssm_forward;
mod trait_decode;
mod trait_decode_batched;
mod trait_decode_batched_conv_gdn;
mod trait_decode_multi_seq;
mod trait_prefill;
mod trait_prefill_gdn;
mod trait_prefill_helper;
mod trait_prefill_phase1;
mod trait_prefill_phase3;
mod trait_prefill_proj;
mod trait_prefill_recur;

// ── TransformerLayer impl (delegates to per-file inherent _inner methods) ──
impl TransformerLayer for Qwen3SsmLayer {
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

    fn decode_batched(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_batched_inner(
            hidden,
            residual,
            num_tokens,
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

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::config::ModelConfig;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn test_ssm_state_allocation_sizes() {
        let config = ModelConfig::qwen3_next_80b_nvfp4();
        let nv = config.linear_num_value_heads; // 32
        let vd = config.linear_value_head_dim; // 128
        let nk = config.linear_num_key_heads; // 16
        let kd = config.linear_key_head_dim; // 128
        let d_conv = config.linear_conv_kernel_dim; // 4

        let h_bytes = nv * vd * kd * 4;
        assert_eq!(h_bytes, 32 * 128 * 128 * 4); // 2 MB

        // conv_dim = 2*key_dim + value_dim = 2*2048 + 4096 = 8192
        let conv_dim = nk * kd * 2 + nv * vd;
        let conv_bytes = conv_dim * d_conv * 4;
        assert_eq!(conv_bytes, 8192 * 4 * 4); // 128 KB

        // Verify allocations
        let gpu = MockGpuBackend::new();
        let h_state = gpu.alloc(h_bytes).unwrap();
        let conv_state = gpu.alloc(conv_bytes).unwrap();
        assert!(!h_state.is_null());
        assert!(!conv_state.is_null());
    }
}
