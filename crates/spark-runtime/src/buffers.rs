// SPDX-License-Identifier: AGPL-3.0-only

//! Pre-allocated GPU buffer arena for intermediate tensors.
//!
//! All buffer sizes derive from [`ModelConfig`] (SSOT). The arena is
//! allocated once during initialization and reused across decode steps.

use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::Result;
use atlas_core::config::ModelConfig;

mod sizes;
pub use sizes::{BufferSizes, Q12_SIZING_STREAMS, q12_batched_scratch_bytes};

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
    /// Token IDs `[M]` u32 — stable across the layer loop so DeepSeek-V4
    /// hash-MoE layers can read `tid2eid[token_id]`.
    token_ids: DevicePtr,
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
        let gdn_fla_scratch = if sizes.gdn_fla_scratch > 0 {
            gpu.alloc(sizes.gdn_fla_scratch)?
        } else {
            DevicePtr::NULL
        };
        let token_ids = gpu.alloc(sizes.token_ids)?;

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
            token_ids,
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
    pub fn gate_logits_f32(&self) -> DevicePtr {
        self.gate_logits_f32
    }
    pub fn moe_router_in_f32(&self) -> DevicePtr {
        self.moe_router_in_f32
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
    /// Token IDs `[M]` u32 — stable across the layer loop (DeepSeek-V4 hash-MoE
    /// reads `tid2eid[token_id]`). Upload the pass's token IDs here before the
    /// layer loop; under CUDA-graph decode upload before each replay.
    pub fn token_ids(&self) -> DevicePtr {
        self.token_ids
    }
    /// Allocated byte size of the scratch buffer (#110: bounds-check
    /// batched metadata-staging uploads against this).
    pub fn scratch_bytes(&self) -> usize {
        self.sizes.scratch
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
    /// GDN FLA chunked-prefill scratch base (W|U|S|uc sub-divided by the caller).
    /// `DevicePtr::NULL` unless this is a 128-dim-linear-head GDN model.
    pub fn gdn_fla_scratch(&self) -> DevicePtr {
        self.gdn_fla_scratch
    }
    pub fn splitk_workspace(&self) -> DevicePtr {
        self.splitk_workspace
    }
    /// Grouped O-projection latent [M, o_groups*o_lora_rank] BF16 (V4-Flash).
    pub fn o_latent(&self) -> DevicePtr {
        self.o_latent
    }
    /// All-ones BF16 vector (max_dim) — weight for unweighted RMSNorm (q_b_norm).
    pub fn norm_unit_w(&self) -> DevicePtr {
        self.norm_unit_w
    }
    /// HC residual streams [M, hc_mult, hidden] BF16 (DeepSeek-V4 mHC).
    pub fn hc_streams(&self) -> DevicePtr {
        self.hc_streams
    }
    /// HC `post` mixing weights [M, hc_mult] F32.
    pub fn hc_post(&self) -> DevicePtr {
        self.hc_post
    }
    /// HC `comb` Sinkhorn matrix [M, hc_mult, hc_mult] F32.
    pub fn hc_comb(&self) -> DevicePtr {
        self.hc_comb
    }
    pub fn max_batch_tokens(&self) -> usize {
        self.max_batch_tokens
    }
    pub fn sizes(&self) -> &BufferSizes {
        &self.sizes
    }

    /// Env-gated (`ATLAS_SSM_SAVE_DUMP`) per-buffer checksum probe.
    ///
    /// CBD: localize a stale/uninitialized decode-scratch buffer on the
    /// prefix-cache skip path. Dumps sum/ssq/sabs over the FULL allocation
    /// (so leftover-from-prior-occupant bytes in unwritten rows are visible)
    /// for every reusable buffer. Treats raw bytes as f32 lanes — exact
    /// numeric meaning is irrelevant; we only need a stable fingerprint that
    /// differs iff the bytes differ. Synchronizes the stream first.
    pub fn debug_buffer_checksum(&self, gpu: &dyn GpuBackend, stream: u64, tag: &str) {
        gpu.synchronize(stream).ok();
        let probe = |name: &str, ptr: DevicePtr, bytes: usize| {
            let mut hb = vec![0u8; bytes];
            if gpu.copy_d2h(ptr, &mut hb).is_err() {
                return;
            }
            let (mut sum, mut ssq, mut sabs) = (0f64, 0f64, 0f64);
            for c in hb.chunks_exact(4) {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64;
                if v.is_finite() {
                    sum += v;
                    ssq += v * v;
                    sabs += v.abs();
                }
            }
            tracing::warn!(
                "ATLAS_BUF_CKSUM[{tag}] {name} bytes={bytes} sum={sum:.6} ssq={ssq:.6} sabs={sabs:.6}"
            );
        };
        probe(
            "hidden_states",
            self.hidden_states,
            self.sizes.hidden_states,
        );
        probe("residual", self.residual, self.sizes.residual);
        probe("norm_output", self.norm_output, self.sizes.norm_output);
        probe("qkv_output", self.qkv_output, self.sizes.qkv_output);
        probe("attn_output", self.attn_output, self.sizes.attn_output);
        probe("gate_logits", self.gate_logits, self.sizes.gate_logits);
        probe("moe_output", self.moe_output, self.sizes.moe_output);
        probe("ssm_qkvz", self.ssm_qkvz, self.sizes.ssm_qkvz);
        probe("ssm_ba", self.ssm_ba, self.sizes.ssm_ba);
        probe(
            "ssm_deinterleaved",
            self.ssm_deinterleaved,
            self.sizes.ssm_deinterleaved,
        );
        probe("ssm_gates", self.ssm_gates, self.sizes.ssm_gates);
        probe(
            "ssm_conv_out_f32",
            self.ssm_conv_out_f32,
            self.sizes.ssm_conv_out_f32,
        );
        probe(
            "expert_gate_out",
            self.expert_gate_out,
            self.sizes.expert_gate_out,
        );
        probe(
            "expert_up_out",
            self.expert_up_out,
            self.sizes.expert_up_out,
        );
        probe(
            "expert_down_out",
            self.expert_down_out,
            self.sizes.expert_down_out,
        );
        probe(
            "splitk_workspace",
            self.splitk_workspace,
            self.sizes.splitk_workspace,
        );
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
