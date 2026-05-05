// SPDX-License-Identifier: AGPL-3.0-only

//! Pre-allocated GPU buffer arena for intermediate tensors.
//!
//! All buffer sizes derive from [`ModelConfig`] (SSOT). The arena is
//! allocated once during initialization and reused across decode steps.

use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::Result;
use atlas_core::config::ModelConfig;

mod sizes;
pub use sizes::BufferSizes;

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
            max_batch_tokens,
            sizes,
        })
    }

    pub fn hidden_states(&self) -> DevicePtr {
        self.hidden_states
    }
    pub fn residual(&self) -> DevicePtr {
        self.residual
    }
    pub fn norm_output(&self) -> DevicePtr {
        self.norm_output
    }
    pub fn qkv_output(&self) -> DevicePtr {
        self.qkv_output
    }
    pub fn attn_output(&self) -> DevicePtr {
        self.attn_output
    }
    pub fn gate_logits(&self) -> DevicePtr {
        self.gate_logits
    }
    pub fn moe_output(&self) -> DevicePtr {
        self.moe_output
    }
    pub fn logits(&self) -> DevicePtr {
        self.logits
    }
    pub fn ssm_qkvz(&self) -> DevicePtr {
        self.ssm_qkvz
    }
    pub fn ssm_ba(&self) -> DevicePtr {
        self.ssm_ba
    }
    /// Sequential [Q|K|V|Z] after deinterleaving.
    pub fn ssm_deinterleaved(&self) -> DevicePtr {
        self.ssm_deinterleaved
    }
    /// FP32 [gate, beta] for GDN (num_v_heads * 2 floats).
    pub fn ssm_gates(&self) -> DevicePtr {
        self.ssm_gates
    }
    /// FP32 conv1d output for SSM recurrent path (prevents BF16 precision drift).
    pub fn ssm_conv_out_f32(&self) -> DevicePtr {
        self.ssm_conv_out_f32
    }
    /// Scratch buffer for MoE routing + kernel metadata uploads.
    pub fn scratch(&self) -> DevicePtr {
        self.scratch
    }
    /// Batched expert gate projection output.
    pub fn expert_gate_out(&self) -> DevicePtr {
        self.expert_gate_out
    }
    /// Batched expert up projection output.
    pub fn expert_up_out(&self) -> DevicePtr {
        self.expert_up_out
    }
    /// Batched expert down projection output.
    pub fn expert_down_out(&self) -> DevicePtr {
        self.expert_down_out
    }
    /// Split-K decode attention workspace (F32 partials).
    pub fn splitk_workspace(&self) -> DevicePtr {
        self.splitk_workspace
    }
    pub fn max_batch_tokens(&self) -> usize {
        self.max_batch_tokens
    }
    pub fn sizes(&self) -> &BufferSizes {
        &self.sizes
    }

    /// Zero only buffers that carry residual state between requests.
    ///
    /// During prefill, every buffer except hidden_states and residual is fully
    /// overwritten before being read within the layer loop:
    /// - norm_output, qkv_output, attn_output: written by each layer's projection
    /// - gate_logits, moe_output: written by MoE gate/output
    /// - ssm_*: written by SSM projection
    /// - expert_*: written by expert compute
    /// - logits: written by LM head on last token
    /// - scratch: overwritten by metadata upload and MoE routing
    /// - splitk_workspace: written by attention kernel
    ///
    /// This reduces per-chunk memset from 17 calls to 2, saving ~15 memset
    /// launches × bandwidth on the LPDDR5X bus per prefill chunk.
    pub fn zero_prefill_essentials(&self, gpu: &dyn GpuBackend, stream: u64) -> anyhow::Result<()> {
        gpu.memset_async(self.hidden_states, 0, self.sizes.hidden_states, stream)?;
        gpu.memset_async(self.residual, 0, self.sizes.residual, stream)?;
        // MoE buffers: gate_logits may carry stale expert indices from a prior
        // request with different token count, causing out-of-bounds expert access
        // (CUDA error 700 at layer 38+ on 122B). Zero to prevent.
        gpu.memset_async(self.gate_logits, 0, self.sizes.gate_logits, stream)?;
        gpu.memset_async(self.expert_gate_out, 0, self.sizes.expert_gate_out, stream)?;
        gpu.memset_async(self.expert_up_out, 0, self.sizes.expert_up_out, stream)?;
        gpu.memset_async(self.expert_down_out, 0, self.sizes.expert_down_out, stream)?;
        gpu.memset_async(self.moe_output, 0, self.sizes.moe_output, stream)?;
        Ok(())
    }

    /// Zero all reusable buffers to eliminate stale data between requests.
    /// Ensures deterministic computation regardless of request history.
    pub fn zero_all(&self, gpu: &dyn GpuBackend, stream: u64) -> anyhow::Result<()> {
        gpu.memset_async(self.hidden_states, 0, self.sizes.hidden_states, stream)?;
        gpu.memset_async(self.residual, 0, self.sizes.residual, stream)?;
        gpu.memset_async(self.norm_output, 0, self.sizes.norm_output, stream)?;
        gpu.memset_async(self.qkv_output, 0, self.sizes.qkv_output, stream)?;
        gpu.memset_async(self.attn_output, 0, self.sizes.attn_output, stream)?;
        gpu.memset_async(self.gate_logits, 0, self.sizes.gate_logits, stream)?;
        gpu.memset_async(self.moe_output, 0, self.sizes.moe_output, stream)?;
        gpu.memset_async(self.ssm_qkvz, 0, self.sizes.ssm_qkvz, stream)?;
        gpu.memset_async(self.ssm_ba, 0, self.sizes.ssm_ba, stream)?;
        gpu.memset_async(
            self.ssm_deinterleaved,
            0,
            self.sizes.ssm_deinterleaved,
            stream,
        )?;
        gpu.memset_async(self.ssm_gates, 0, self.sizes.ssm_gates, stream)?;
        gpu.memset_async(
            self.ssm_conv_out_f32,
            0,
            self.sizes.ssm_conv_out_f32,
            stream,
        )?;
        gpu.memset_async(
            self.splitk_workspace,
            0,
            self.sizes.splitk_workspace,
            stream,
        )?;
        gpu.memset_async(self.expert_gate_out, 0, self.sizes.expert_gate_out, stream)?;
        gpu.memset_async(self.expert_up_out, 0, self.sizes.expert_up_out, stream)?;
        gpu.memset_async(self.expert_down_out, 0, self.sizes.expert_down_out, stream)?;
        gpu.memset_async(self.logits, 0, self.sizes.logits, stream)?;
        gpu.memset_async(self.scratch, 0, self.sizes.scratch, stream)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
