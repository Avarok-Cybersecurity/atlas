// SPDX-License-Identifier: AGPL-3.0-only

//! Byte sizes for the per-pass GPU buffer arena.

use atlas_core::config::ModelConfig;

/// Byte sizes of each buffer, derived from ModelConfig.
#[derive(Debug, Clone)]
pub struct BufferSizes {
    pub hidden_states: usize,
    pub residual: usize,
    pub norm_output: usize,
    pub qkv_output: usize,
    pub attn_output: usize,
    pub gate_logits: usize,
    pub moe_output: usize,
    pub logits: usize,
    pub ssm_qkvz: usize,
    pub ssm_ba: usize,
    pub ssm_deinterleaved: usize,
    pub ssm_gates: usize,
    pub ssm_conv_out_f32: usize,
    pub scratch: usize,
    pub expert_gate_out: usize,
    pub expert_up_out: usize,
    pub expert_down_out: usize,
    pub splitk_workspace: usize,
}

impl BufferSizes {
    /// Compute all buffer sizes from model config and max batch tokens.
    ///
    /// All sizes in bytes. BF16 = 2 bytes per element.
    /// Logits buffer is capped: only needed for decode (1 token) or
    /// speculative verification (K tokens), never for full prefill.
    ///
    /// `max_seq_len` and `kv_block_size` are needed to size the scratch
    /// buffer for block table metadata during batched decode / verify.
    pub fn from_config(
        config: &ModelConfig,
        max_batch_tokens: usize,
        max_seq_len: usize,
        kv_block_size: usize,
    ) -> Self {
        let bf16 = 2;
        let m = max_batch_tokens;
        let h = config.hidden_size;

        // Q projection output: gated models produce [Q, gate] (2× nq*hd),
        // ungated models (VL) produce only [Q] (nq*hd).
        let q_heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let hd = config.head_dim;
        let q_proj_mul = if config.attn_gated { 2 } else { 1 };
        let qkv_dim = (q_heads * q_proj_mul + 2 * kv_heads) * hd;

        let top_k = config.num_experts_per_tok;

        // Scratch layout (two users, take max):
        //
        // A) Prefill chunk metadata (after MoE routing data):
        //   [0 .. moe_scratch): MoE topK routing indices+weights
        //   [moe_scratch .. ): positions(m*4) + slots(m*8) + block_table(max_blocks*4) + seq_len(4)
        //
        // B) Batched decode/verify metadata:
        //   [0 .. 32768): fixed metadata region
        //   [32768 .. 32768+768): decode metadata (positions, slots, seq_lens)
        //   [32768+768 .. ): block table (padded_n × max_blocks × 4 bytes)
        //
        // MoE scratch: 2 * M * top_k * 4 (indices [M*top_k] u32 + weights [M*top_k] f32)
        let moe_scratch = 2 * m * top_k * 4;
        let max_blocks = if kv_block_size > 0 {
            max_seq_len / kv_block_size + 1
        } else {
            256
        };
        // Prefill metadata: mirrors exact layout in prefill_chunk(). MRoPE
        // (Qwen3-VL / Qwen3.6) uploads THREE u32 position streams packed
        // back-to-back (T, H, W); every other model uploads ONE. Sizing the
        // scratch region for 1× with MRoPE active caused `cuMemcpyHtoDAsync_v2
        // status 1` failures on long-context prefills (observed: 16k Qwen3.6
        // failed, 8k passed because the extra 64 KB of write overflow happened
        // to still land inside the over-provisioned `moe_scratch + meta`
        // aggregate).
        let pos_streams = if config.mrope_interleaved { 3 } else { 1 };
        let pos_bytes = m * 4 * pos_streams;
        let slot_offset = (pos_bytes + 7) & !7;
        let slot_end = slot_offset + m * 8;
        let bt_offset = (slot_end + 3) & !3;
        let bt_end = bt_offset + max_blocks * 4;
        let sl_offset = (bt_end + 3) & !3;
        let prefill_meta = sl_offset + 4;
        // Block table metadata: max(batch_size=8, K=4 verify, K=γ DFlash verify)
        // rows × max_blocks × 4 bytes. DFlash γ-block verify uses up to γ+1=17
        // rows (γ=16 for Qwen3.6-DFlash), so size for the worst case.
        let bt_rows = 32usize; // headroom for K=γ DFlash verify (typical γ=16, K=17)
        let bt_meta = 32768 + 768 + bt_rows * max_blocks * 4;
        let scratch_min = 64 * 1024;
        let scratch = scratch_min.max(moe_scratch + prefill_meta).max(bt_meta);

        // Batched expert output buffers for MoE (or dense FFN).
        // Sized for max(K=3 verify, prefill chunk) × top_k experts.
        let k_max = m.max(3); // prefill chunk or K=3 verify, whichever larger
        let expert_inter = if config.num_experts > 0 {
            k_max * config.num_experts_per_tok * config.moe_intermediate_size
        } else {
            k_max * config.intermediate_size
        };
        let expert_gate_out = expert_inter * bf16;
        let expert_up_out = expert_inter * bf16;
        // Routed expert down output: [k_max * top_k, moe_input_size].
        // For LatentMoE (Super 120B), routed experts output in latent space.
        let moe_out_dim = config.moe_input_size();
        let expert_down_out = if config.num_experts > 0 {
            k_max * config.num_experts_per_tok * moe_out_dim * bf16
        } else {
            k_max * h * bf16
        };

        // Logits: only last token used during prefill. Cap at 32 tokens
        // (sufficient for decode=1, batched_decode=8, spec_verify≤5,
        // DFlash K=γ verify with γ=16 → K=17 tokens — bumped from 16
        // for DFlash K=γ headroom; matches `bt_rows` cap above).
        let logits_tokens = m.min(32);

        // Mamba-2 d_inner may exceed hidden_size; norm_output and attn_output must fit.
        let mamba2_d_inner = config.mamba2_d_inner();
        let max_dim = h.max(mamba2_d_inner);

        // Split-K decode workspace: NUM_SMS * (head_dim + 2) * sizeof(f32).
        // Partials from split CTAs are stored as [o[head_dim], m, l] per split.
        // Total slots = num_seqs * num_splits ≤ NUM_SMS, so this is constant ~48 KB.
        let splitk_workspace = 48 * (hd + 2) * 4;

        // Residual dtype controlled by config.use_fp32_residual().
        // FP32 prevents BF16 truncation across 48 layers but costs 2x bandwidth.
        let residual_elem = if config.use_fp32_residual() { 4 } else { bf16 };

        Self {
            hidden_states: m * h * residual_elem,
            residual: m * h * residual_elem,
            norm_output: m * max_dim * bf16,
            qkv_output: m * qkv_dim * bf16,
            attn_output: (m * config.num_attention_heads * config.head_dim * bf16)
                .max(m * mamba2_d_inner * bf16)
                // MLA absorbed: attention output is [M, nq, mla_cache_dim=kv_lora+rope]
                .max(if config.kv_lora_rank > 0 {
                    m * config.num_attention_heads
                        * (config.kv_lora_rank + config.qk_rope_head_dim)
                        * bf16
                } else {
                    0
                }),
            gate_logits: if config.num_experts > 0 {
                m * config.num_experts * bf16
            } else {
                256
            },
            moe_output: m * h * bf16,
            logits: logits_tokens * config.vocab_size * bf16, // BF16 from LM head kernel
            // SSM buffers are also reused by attention prefill/multi-seq as scratch:
            //   ssm_qkvz: K+V contiguous storage in prefill [M, 2*kv_dim]
            //             Mamba-2 in_proj output [M, in_proj_size]
            //   ssm_deinterleaved: Q contiguous copy [M, nq*hd]
            //                      Mamba-2 conv1d output [M, d_xBC]
            // Use max across all uses with minimum 256 to avoid 0-byte alloc.
            ssm_qkvz: (m * config.ssm_qkvz_size() * bf16)
                .max(m * config.mamba2_in_proj_size() * bf16)
                .max(m * 2 * kv_heads * hd * bf16)
                .max(m * config.shared_expert_intermediate_size * bf16) // MoE shared up scratch
                .max(256),
            ssm_ba: (m * config.ssm_ba_size() * bf16)
                .max(m * config.moe_latent_size * bf16) // LatentMoE latent buffer
                // MLA reuses ssm_ba for two separate buffers:
                //   - q_latent    [M, q_lora_rank]    BF16 — output of wq_a GEMM
                //   - k_rope_buf  [M, qk_rope_head_dim] BF16 — output of wkv_a_rope GEMM
                // Both are written sequentially (q_latent is consumed before
                // k_rope_buf is allocated). Size for the larger of the two.
                .max(if config.kv_lora_rank > 0 {
                    (m * config.qk_rope_head_dim * bf16).max(m * config.q_lora_rank * bf16)
                } else {
                    0
                })
                .max(256),
            ssm_deinterleaved: (m * config.ssm_qkvz_size() * bf16)
                .max(m * config.mamba2_d_xbc() * bf16)
                .max(m * q_heads * hd * bf16)
                // MLA absorbed: Q_absorbed buffer is [M, nq, mla_cache_dim=kv_lora+rope]
                .max(if config.kv_lora_rank > 0 {
                    m * q_heads * (config.kv_lora_rank + config.qk_rope_head_dim) * bf16
                } else {
                    0
                })
                .max(256),
            ssm_gates: (m * config.linear_num_value_heads * 2 * 4).max(256),
            // FP32 conv output for SSM recurrent path precision (4 bytes/element).
            // Uses ssm_qkvz_size as upper bound (includes Q+K+V+Z).
            // Also reused by MLA as q_rope contiguous buffer: [M, nq * qk_rope_head_dim] BF16.
            ssm_conv_out_f32: (m * config.ssm_qkvz_size() * 4)
                .max(if config.kv_lora_rank > 0 {
                    m * q_heads * config.qk_rope_head_dim * bf16
                } else {
                    0
                })
                .max(256),
            scratch,
            expert_gate_out,
            expert_up_out,
            expert_down_out,
            splitk_workspace,
        }
    }

    /// Total bytes across all buffers.
    pub fn total_bytes(&self) -> usize {
        self.hidden_states
            + self.residual
            + self.norm_output
            + self.qkv_output
            + self.attn_output
            + self.gate_logits
            + self.moe_output
            + self.logits
            + self.ssm_qkvz
            + self.ssm_ba
            + self.ssm_deinterleaved
            + self.ssm_gates
            + self.ssm_conv_out_f32
            + self.scratch
            + self.expert_gate_out
            + self.expert_up_out
            + self.expert_down_out
            + self.splitk_workspace
    }
}
