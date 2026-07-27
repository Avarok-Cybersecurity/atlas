// SPDX-License-Identifier: AGPL-3.0-only

//! `BufferArena` accessors. Split from `buffers.rs` (500-LoC cap).

use super::{BufferArena, sizes::BufferSizes};
use crate::gpu::{DevicePtr, GpuBackend};

impl BufferArena {
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
    /// Mamba-2 SSD chunked-scan scratch (dt | dA_cumsum | CB). NULL if unused.
    pub fn ssd_scratch(&self) -> DevicePtr {
        self.ssd_scratch
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
    /// Shared dense-FFN q8_1 activation scratch (Q4_K MMQ gate/up). NULL for MoE.
    pub fn ffn_act_q8(&self) -> DevicePtr {
        self.ffn_act_q8
    }
    /// Shared dense-FFN int8/NVFP4 activation scratch (a_i8 / packed). NULL for MoE.
    pub fn ffn_act_a(&self) -> DevicePtr {
        self.ffn_act_a
    }
    /// Shared dense-FFN int8/NVFP4 activation-scale scratch. NULL for MoE.
    pub fn ffn_act_scale(&self) -> DevicePtr {
        self.ffn_act_scale
    }
    /// Persistent FP8 block-scaled activation scratch for prefill projections.
    /// Replaces a per-projection alloc/sync/free in the W8A8+FP32-epilogue path.
    pub fn fp8_act(&self) -> DevicePtr {
        self.fp8_act
    }
    /// Allocated byte size of `fp8_act` (debug bounds-check at call sites).
    pub fn fp8_act_bytes(&self) -> usize {
        self.sizes.fp8_act
    }
    /// Persistent per-128-block FP32 scales paired with `fp8_act`.
    pub fn fp8_act_scale(&self) -> DevicePtr {
        self.fp8_act_scale
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
    /// LoRA compressed activation scratch `xa = x@Aᵀ` [M, max_rank] BF16.
    /// `DevicePtr::NULL` when no adapter is configured.
    pub fn lora_xa(&self) -> DevicePtr {
        self.lora_xa
    }
    /// Allocated byte size of `lora_xa` (0 when no adapter).
    pub fn lora_xa_bytes(&self) -> usize {
        self.sizes.lora_xa
    }
    /// LoRA expand scratch `delta = xa@Bᵀ` [M, max(hidden, intermediate)]
    /// BF16. `DevicePtr::NULL` when no adapter is configured.
    pub fn lora_delta(&self) -> DevicePtr {
        self.lora_delta
    }
    /// Allocated byte size of `lora_delta` (0 when no adapter).
    pub fn lora_delta_bytes(&self) -> usize {
        self.sizes.lora_delta
    }
    /// LoRA hidden-activation scratch [M, intermediate_size] BF16 for the
    /// runtime FFN delta path. `DevicePtr::NULL` when no adapter.
    pub fn lora_hact(&self) -> DevicePtr {
        self.lora_hact
    }
    /// Allocated byte size of `lora_hact` (0 when no adapter).
    pub fn lora_hact_bytes(&self) -> usize {
        self.sizes.lora_hact
    }
    /// LoRA per-request routing slots `[max_batch_tokens]` i32 for the prefill
    /// path — one adapter SLOT index per prefilling token. `DevicePtr::NULL`
    /// when no adapter is configured.
    pub fn lora_seq_slot(&self) -> DevicePtr {
        self.lora_seq_slot
    }

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
