// SPDX-License-Identifier: AGPL-3.0-only

//! Pre-allocated GPU buffer arena for intermediate tensors.
//!
//! All buffer sizes derive from [`ModelConfig`] (SSOT). The arena is
//! allocated once during initialization and reused across decode steps.

use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::Result;
use atlas_core::config::ModelConfig;

mod accessors;
mod sizes;
mod sizes_q12;
pub use sizes::BufferSizes;
pub use sizes_q12::{Q12_SIZING_STREAMS, q12_batched_scratch_bytes};

/// Pre-allocated GPU buffers for a single forward pass.
///
/// Each buffer is sized for `max_batch_tokens` tokens through the model.
/// Buffers are reused across steps — no per-step allocation.
///
/// Expert output buffers are sized for max(k_max, max_batch_tokens) to
/// support both speculative decode (K=3) and batched MoE prefill. At N=512,
/// this adds ~31 MB (vs the old grouped-GEMM approach that needed 260 MB
/// and caused a 15% decode regression). The GEMV-based prefill kernels
/// only touch k_max slots during decode, so the extra pages don't affect
/// decode bandwidth on unified memory.
pub struct BufferArena {
    /// Hidden states: [M, hidden_size] in BF16.
    hidden_states: DevicePtr,
    /// Residual stream: [M, hidden_size] in BF16.
    residual: DevicePtr,
    /// Post-norm output: [M, hidden_size] in BF16.
    norm_output: DevicePtr,
    /// QKV projection output for full attention: [M, (Hq + 2*Hkv) * D] in BF16.
    qkv_output: DevicePtr,
    /// Attention output: [M, Hq * D] in BF16.
    attn_output: DevicePtr,
    /// MoE gate logits: [M, num_experts] in BF16.
    gate_logits: DevicePtr,
    /// MoE gate logits: [M, num_experts] in FP32 (ATLAS_FP32_GATE path).
    gate_logits_f32: DevicePtr,
    /// MoE-input norm output: [M, hidden_size] in FP32 (ATLAS_FP32_ROUTING).
    moe_router_in_f32: DevicePtr,
    /// MoE output: [M, hidden_size] in BF16.
    moe_output: DevicePtr,
    /// Logits: [M, vocab_size] in BF16.
    logits: DevicePtr,
    /// SSM QKVZ projection: [M, ssm_qkvz_size] in BF16.
    ssm_qkvz: DevicePtr,
    /// SSM beta-alpha projection: [M, ssm_ba_size] in BF16.
    ssm_ba: DevicePtr,
    /// SSM deinterleaved QKVZ: [M, ssm_qkvz_size] in BF16 (sequential [Q|K|V|Z]).
    ssm_deinterleaved: DevicePtr,
    /// SSM FP32 gates: [num_v_heads * 2] as FP32 (gate + beta for GDN).
    ssm_gates: DevicePtr,
    /// SSM conv1d output in FP32: [M, conv_dim] as FP32.
    /// Prevents BF16 truncation in the SSM recurrent path (conv → GDN).
    /// Without this, ~7 bits of precision are lost every token, causing
    /// coherence degradation after 8k+ tokens.
    ssm_conv_out_f32: DevicePtr,
    /// Scratch space for kernel metadata (positions, slot_mapping, block_tables).
    scratch: DevicePtr,
    /// Expert gate projection output: [k2 * top_k, moe_intermediate_size] BF16.
    expert_gate_out: DevicePtr,
    /// Expert up projection output: [k2 * top_k, moe_intermediate_size] BF16.
    expert_up_out: DevicePtr,
    /// Expert down projection output: [k2 * top_k, hidden_size] BF16.
    expert_down_out: DevicePtr,
    /// Split-K decode attention workspace: partials from split CTAs (F32).
    splitk_workspace: DevicePtr,
    /// Grouped O-projection latent: [M, o_groups*o_lora_rank] BF16 (V4-Flash).
    o_latent: DevicePtr,
    /// Zero-filled BF16 weight (max_dim) for unweighted RMSNorm under the
    /// offset-from-1 kernel convention (scale = 1+weight → 1.0). Used by q_b_norm.
    norm_unit_w: DevicePtr,
    /// HC residual streams: [M, hc_mult, hidden] BF16 (DeepSeek-V4 mHC).
    hc_streams: DevicePtr,
    /// HC `post` mixing weights: [M, hc_mult] F32.
    hc_post: DevicePtr,
    /// HC `comb` Sinkhorn matrix: [M, hc_mult, hc_mult] F32.
    hc_comb: DevicePtr,
    /// GDN FLA chunked-prefill scratch (W|U|S|uc sub-divided). NULL unless the
    /// model is a 128-dim-linear-head GDN model (ATLAS_GDN_FLA path).
    gdn_fla_scratch: DevicePtr,
    /// Mamba-2 SSD chunked-scan scratch (dt | dA_cumsum | CB). NULL unless the model
    /// has Mamba-2 SSM layers.
    ssd_scratch: DevicePtr,
    /// Token IDs `[M]` u32 — stable across the layer loop so DeepSeek-V4
    /// hash-MoE layers can read `tid2eid[token_id]`.
    token_ids: DevicePtr,
    /// Shared FFN activation-quant scratch (dense-FFN MMQ/int8 prefill path).
    /// Allocated once here instead of per-DenseFfnLayer (64× would leak ~18GB).
    /// NULL unless the model is dense (`num_experts == 0`).
    /// `ffn_act_q8`: q8_1 activations for the Q4_K MMQ gate/up GEMM.
    /// `ffn_act_a` / `ffn_act_scale`: int8 (a_i8 / a_scale) — reused for NVFP4 packed/scale.
    ffn_act_q8: DevicePtr,
    ffn_act_a: DevicePtr,
    ffn_act_scale: DevicePtr,
    /// Persistent FP8 block-scaled activation scratch for prefill projections.
    fp8_act: DevicePtr,
    /// Persistent per-128-block FP32 scales paired with `fp8_act`.
    fp8_act_scale: DevicePtr,
    /// LoRA shrink scratch `xa = x@Aᵀ`: [M, adapter_max_rank] BF16.
    /// NULL when no adapter is configured.
    lora_xa: DevicePtr,
    /// LoRA expand scratch `delta = xa@Bᵀ`: [M, max(hidden, intermediate)]
    /// BF16. NULL when no adapter is configured.
    lora_delta: DevicePtr,
    /// LoRA hidden-activation scratch: [M, intermediate_size] BF16 for the
    /// runtime FFN delta path. NULL when no adapter is configured.
    lora_hact: DevicePtr,
    /// LoRA per-request routing slots `[M]` i32 for the prefill path (one
    /// adapter SLOT index per prefilling token). NULL when no adapter.
    lora_seq_slot: DevicePtr,
    /// Maximum batch tokens this arena was sized for.
    max_batch_tokens: usize,
    /// Sizes in bytes for each buffer (for debug/logging).
    sizes: BufferSizes,
}

impl BufferArena {
    /// Allocate all intermediate buffers on the GPU.
    pub fn new(
        config: &ModelConfig,
        max_batch_tokens: usize,
        max_seq_len: usize,
        kv_block_size: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let sizes = BufferSizes::from_config(config, max_batch_tokens, max_seq_len, kv_block_size);

        let hidden_states = gpu.alloc(sizes.hidden_states)?;
        let residual = gpu.alloc(sizes.residual)?;
        let norm_output = gpu.alloc(sizes.norm_output)?;
        let qkv_output = gpu.alloc(sizes.qkv_output)?;
        let attn_output = gpu.alloc(sizes.attn_output)?;
        let gate_logits = gpu.alloc(sizes.gate_logits)?;
        let gate_logits_f32 = gpu.alloc(sizes.gate_logits_f32)?;
        let moe_router_in_f32 = gpu.alloc(sizes.moe_router_in_f32)?;
        let moe_output = gpu.alloc(sizes.moe_output)?;
        let logits = gpu.alloc(sizes.logits)?;
        let ssm_qkvz = gpu.alloc(sizes.ssm_qkvz)?;
        let ssm_ba = gpu.alloc(sizes.ssm_ba)?;
        let ssm_deinterleaved = gpu.alloc(sizes.ssm_deinterleaved)?;
        let ssm_gates = gpu.alloc(sizes.ssm_gates)?;
        let ssm_conv_out_f32 = gpu.alloc(sizes.ssm_conv_out_f32)?;
        let scratch = gpu.alloc(sizes.scratch)?;
        let expert_gate_out = gpu.alloc(sizes.expert_gate_out)?;
        let expert_up_out = gpu.alloc(sizes.expert_up_out)?;
        let expert_down_out = gpu.alloc(sizes.expert_down_out)?;
        let splitk_workspace = gpu.alloc(sizes.splitk_workspace)?;
        let o_latent = gpu.alloc(sizes.o_latent)?;
        // Zero-filled "weight" for unweighted RMSNorm under the offset-from-1
        // convention used by the rms_norm kernel (scale = 1 + weight). Weight = 0
        // → scale = 1.0, i.e. a pure normalize (DeepSeek-V4 q_b_norm).
        let norm_unit_w = gpu.alloc(sizes.norm_unit_w)?;
        gpu.memset(norm_unit_w, 0, sizes.norm_unit_w)?;
        let hc_streams = gpu.alloc(sizes.hc_streams)?;
        let hc_post = gpu.alloc(sizes.hc_post)?;
        let hc_comb = gpu.alloc(sizes.hc_comb)?;
        // GDN FLA scratch: only allocate for the 128-dim-linear-head GDN path
        // (size 0 → NULL → ATLAS_GDN_FLA dispatch stays disabled).
        let ssd_scratch = if sizes.ssd_scratch > 0 {
            gpu.alloc(sizes.ssd_scratch)?
        } else {
            DevicePtr::NULL
        };
        let gdn_fla_scratch = if sizes.gdn_fla_scratch > 0 {
            gpu.alloc(sizes.gdn_fla_scratch)?
        } else {
            DevicePtr::NULL
        };
        let token_ids = gpu.alloc(sizes.token_ids)?;
        // Shared dense-FFN activation-quant scratch (MMQ/int8 prefill). Sized 0
        // for MoE models → NULL → per-layer ensure_* path stays inert.
        let ffn_act_q8 = if sizes.ffn_act_q8 > 0 {
            gpu.alloc(sizes.ffn_act_q8)?
        } else {
            DevicePtr::NULL
        };
        let ffn_act_a = if sizes.ffn_act_a > 0 {
            gpu.alloc(sizes.ffn_act_a)?
        } else {
            DevicePtr::NULL
        };
        let ffn_act_scale = if sizes.ffn_act_scale > 0 {
            gpu.alloc(sizes.ffn_act_scale)?
        } else {
            DevicePtr::NULL
        };
        let fp8_act = gpu.alloc(sizes.fp8_act)?;
        let fp8_act_scale = gpu.alloc(sizes.fp8_act_scale)?;
        // LoRA scratch: only allocate when an adapter is configured
        // (size 0 → NULL; cuMemAlloc rejects 0-byte allocations).
        let lora_xa = if sizes.lora_xa > 0 {
            gpu.alloc(sizes.lora_xa)?
        } else {
            DevicePtr::NULL
        };
        let lora_delta = if sizes.lora_delta > 0 {
            gpu.alloc(sizes.lora_delta)?
        } else {
            DevicePtr::NULL
        };
        let lora_hact = if sizes.lora_hact > 0 {
            gpu.alloc(sizes.lora_hact)?
        } else {
            DevicePtr::NULL
        };
        let lora_seq_slot = if sizes.lora_seq_slot > 0 {
            gpu.alloc(sizes.lora_seq_slot)?
        } else {
            DevicePtr::NULL
        };

        tracing::info!(
            "Buffer arena: {} tokens × {:.1} MB total (attn_out={:.1}MB, ssm_deint={:.1}MB, kv_lora_rank={})",
            max_batch_tokens,
            sizes.total_bytes() as f64 / (1024.0 * 1024.0),
            sizes.attn_output as f64 / (1024.0 * 1024.0),
            sizes.ssm_deinterleaved as f64 / (1024.0 * 1024.0),
            config.kv_lora_rank,
        );

        Ok(Self {
            hidden_states,
            residual,
            norm_output,
            qkv_output,
            attn_output,
            gate_logits,
            gate_logits_f32,
            moe_router_in_f32,
            moe_output,
            logits,
            ssm_qkvz,
            ssm_ba,
            ssm_deinterleaved,
            ssm_gates,
            ssm_conv_out_f32,
            scratch,
            expert_gate_out,
            expert_up_out,
            expert_down_out,
            splitk_workspace,
            o_latent,
            norm_unit_w,
            hc_streams,
            hc_post,
            hc_comb,
            gdn_fla_scratch,
            ssd_scratch,
            token_ids,
            ffn_act_q8,
            ffn_act_a,
            ffn_act_scale,
            fp8_act,
            fp8_act_scale,
            lora_xa,
            lora_delta,
            lora_hact,
            lora_seq_slot,
            max_batch_tokens,
            sizes,
        })
    }
}

#[cfg(test)]
mod tests;
