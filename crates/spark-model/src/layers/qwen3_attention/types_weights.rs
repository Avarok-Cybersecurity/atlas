// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3 attention weight component structs (MLA / compressor / hyper-connection).
//! Split from `types.rs` (500-LoC cap).

use spark_runtime::gpu::DevicePtr;

use crate::weight_map::{DenseWeight, QuantizedWeight};

/// MLA (Multi-head Latent Attention) weight components for 2-step decode.
///
/// Instead of a single Q GEMV: `input × Q_expanded → Q[n_heads*hd]`,
/// MLA does: `input × wq_a → latent[q_lora]` → `norm` → `latent × wq_b → Q`.
/// This preserves the latent normalization that's critical for output quality.
pub struct MlaWeights {
    pub wq_a: DenseWeight, // [q_lora, h] — Q down-projection (BF16)
    pub wq_a_nvfp4: Option<QuantizedWeight>, // NVFP4 for fast decode
    /// Native block-scaled FP8 weight (the checkpoint ships these projections as
    /// FP8-E4M3 + 128×128 block scales). Used by the decode GEMV (w8a16_gemv) so
    /// the hot path reads 1 byte/elem instead of the BF16-dequant's 2 — lossless
    /// (the in-kernel dequant keeps F32 precision before the BF16 activation MAC).
    pub wq_a_fp8: Option<crate::weight_map::Fp8Weight>,
    pub wq_b: DenseWeight, // [n_heads*hd, q_lora] — Q up-projection (BF16)
    pub wq_b_nvfp4: Option<QuantizedWeight>, // NVFP4 for fast decode
    pub wq_b_fp8: Option<crate::weight_map::Fp8Weight>,
    pub q_a_norm: DenseWeight,                // [q_lora] — RMS norm weight
    pub wkv_a: DenseWeight,                   // [kv_lora, h] — KV down-projection (BF16)
    pub wkv_a_nvfp4: Option<QuantizedWeight>, // NVFP4 for fast decode
    pub wkv_a_fp8: Option<crate::weight_map::Fp8Weight>,
    pub wkv_b: DenseWeight, // [n_kv*(nope+v), kv_lora] — KV up-projection (BF16)
    pub kv_a_norm: DenseWeight, // [kv_lora] — RMS norm weight
    pub wkv_a_rope: DenseWeight, // [rope, h] — K RoPE projection (BF16)
    /// Merged wkv_a + wkv_a_rope for prefill: [kv_lora+rope, h] — single GEMM replaces 2
    pub wkv_a_merged: DenseWeight,
    pub wo: DenseWeight, // [h, n_heads*v_dim] — O projection BF16 (for prefill accuracy)
    pub wo_nvfp4: Option<QuantizedWeight>, // O projection NVFP4 (for fast decode GEMV)
    /// Grouped low-rank O down-projection (wo_a → wo_b) for DeepSeek-V4-Flash.
    /// When `o_lora_rank > 0`, the decode/prefill paths use wo_a→wo_b instead of `wo`.
    pub wo_a: DenseWeight, // [o_lora_rank, n_heads*v_dim]
    pub wo_a_nvfp4: Option<QuantizedWeight>,
    /// Native block-scaled FP8 wo_a for the grouped decode O-projection. Sliced
    /// per o_group (block-diagonal) into w8a16_gemv calls.
    pub wo_a_fp8: Option<crate::weight_map::Fp8Weight>,
    pub wo_b: DenseWeight, // [h, o_lora_rank]
    pub wo_b_nvfp4: Option<QuantizedWeight>,
    pub wo_b_fp8: Option<crate::weight_map::Fp8Weight>,
    /// Absorbed MLA weights for decode (avoid full K/V expansion, preserve precision).
    /// W_UK_T: [n_heads, nope, kv_lora] — Q_nope absorption: Q_absorbed = Q_nope @ W_UK_T
    pub w_uk_t: DenseWeight,
    /// W_UV: [n_heads, kv_lora, v_dim] — V extraction: v_out = attn_latent @ W_UV
    pub w_uv: DenseWeight,
    /// Q rope projection: wq_b_rope[nq*rope, q_lora] — Q_rope = wq_b_rope @ Q_latent
    /// Extracted from wq_b rows [n*hd+nope .. n*hd+nope+rope] for each head.
    pub wq_b_rope: DenseWeight,
    /// Fused Q absorption: `W_QK_absorbed[nq*kv_lora, q_lora]` — Q_absorbed = W_QK @ Q_latent
    /// Precomputed as: `W_QK[n, lkv, l] = sum_p wq_b_nope[n, p, l] * W_UK[n, p, lkv]`
    /// Enables single GEMV: `Q_absorbed[nq*kv_lora] = W_QK[nq*kv_lora, q_lora] @ Q_latent[q_lora]`
    pub w_qk_absorbed: DenseWeight,
    /// Block-diagonal W_UK for prefill batched GEMM: [nq*kv_lora, nq*nope]
    /// Single GEMM replaces 32*N per-head GEMV calls for Q absorption in prefill.
    pub w_uk_block_diag: DenseWeight,
    /// Block-diagonal W_UV for prefill batched GEMM: [nq*v_dim, nq*kv_lora]
    /// Single GEMM replaces 32*N per-head GEMV calls for V extraction in prefill.
    pub w_uv_block_diag: DenseWeight,
    /// Precomputed YaRN inv_freq table [rotary_dim/2] FP32 on GPU.
    /// NULL = use standard theta computation in the RoPE kernel.
    pub yarn_inv_freq: spark_runtime::gpu::DevicePtr,
    /// Plain θ=10000 inv_freq [rotary_dim/2] FP32 on GPU, NO YaRN. Used for the
    /// raw-arm Q/K rope on `sliding_attention` layers (compressor==None): the
    /// reference gives sliding layers the "main" rope (θ=rope_theta=10000, no
    /// yarn) while CSA/HCA layers use "compress" (θ=compress_rope_theta=160000
    /// + yarn). Atlas previously applied the single yarn table to every layer.
    pub main_inv_freq: spark_runtime::gpu::DevicePtr,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub o_lora_rank: usize,
    pub nope: usize,
    pub rope: usize,
    pub v_dim: usize,
    /// DeepSeek Sparse Attention compressor (CSA ratio-4 / HCA ratio-128).
    /// `None` for full-attention layers (`compress_ratios[L]` == 0).
    pub compressor: Option<CompressorWeights>,
    /// Per-head attention sink logit `[num_q_heads]` BF16 (DeepSeek-V4 s_aux).
    /// NULL if the checkpoint has no attn_sink for this layer.
    pub attn_sink: spark_runtime::gpu::DevicePtr,
}

/// DeepSeek-V4 compressed-attention compressor weights (one per compressed layer).
/// Produces `n_win = usable/ratio` compressed KV entries that are concatenated to
/// the raw sliding-window KV before core attention. CSA (ratio 4) uses a 2×ratio
/// overlap window (Ca/Cb); HCA (ratio 128) uses a single non-overlapping window.
#[derive(Debug, Clone, Copy)]
pub struct CompressorWeights {
    /// kv_proj: [proj_dim, hidden]. proj_dim = 2*head_dim (CSA) or head_dim (HCA).
    pub wkv: DenseWeight,
    /// gate_proj: same shape as wkv.
    pub wgate: DenseWeight,
    /// kv_norm weight `[head_dim]` — HF-vanilla RMSNorm (loaded exactly).
    pub norm: DenseWeight,
    /// position_bias / ape: [ratio, proj_dim] BF16, added to the gate before softmax.
    pub ape: spark_runtime::gpu::DevicePtr,
    /// compress_rate for this layer (4 = CSA, 128 = HCA).
    pub ratio: usize,
    /// proj_dim of wkv/wgate output (2*head_dim for CSA, head_dim for HCA).
    pub proj_dim: usize,
    /// true = CSA (2×ratio overlap window); false = HCA (single window).
    pub is_csa: bool,
    /// 4b: persistent flat compressed-KV pool (decode reads it; inc-3 appends).
    /// Layout `[pool_blocks × hd_mla]` FP8-E4M3, each block = one rope'd `comp_k`
    /// entry quantized at the raw KV arm's scale (k_scale=1.0 for V4) so decode
    /// reads raw+compressed at one dtype/scale (single online softmax). Flat
    /// per-seq (V4 serves max_batch=1), NOT paged — mirrors the reference
    /// `Compressor.kv_cache` contiguous buffer so `block_idx = pos/ratio` matches
    /// prefill's index set exactly (no ring, no block-table remap).
    /// Prefill fills blocks `[0, n_win)`; decode appends after.
    pub pool: spark_runtime::gpu::DevicePtr,
    /// Capacity in compressed blocks = `max_position_embeddings.div_ceil(ratio)`.
    pub pool_blocks: usize,
    /// 4b inc-3: persistent decode-time normed-x ring `[ratio × hidden]` BF16.
    /// Each decode token's compressor input (`normed`, the layer-input RMSNorm
    /// output — the SAME tensor prefill's `cache_skip_v4` feeds `wkv`/`wgate`) is
    /// written to slot `pos % ratio`. At a window boundary the ring holds the
    /// `ratio` tokens of the just-completed window in order, and decode reruns the
    /// prefill compress pipeline over it to append one pool block. BF16 (not FP8):
    /// quantize only at the pool write, so decode's compressor input matches
    /// prefill's bit-for-bit (fp8-ing the input would add a stage prefill never
    /// sees and make the golden-vector gate uninterpretable).
    pub ring: spark_runtime::gpu::DevicePtr,
    /// 4b inc-3 (CSA only): previous completed window's normed-x `[ratio × hidden]`
    /// BF16. CSA reads a 2×ratio overlap (prev window's Ca + current window's Cb);
    /// after each append the ring is copied here to feed the next window's Ca.
    /// `DevicePtr::NULL` for HCA (no overlap). The first decode window has no valid
    /// prev (it would be a prefill window absent from the decode ring) → Ca masked.
    pub prev_win: spark_runtime::gpu::DevicePtr,
    /// 4b inc-3 (CSA only): concat staging `[2×ratio × hidden]` BF16 = prev_win ‖
    /// ring, the 2×ratio-token input the CSA compress kernel indexes for one
    /// overlapped window. `DevicePtr::NULL` for HCA.
    pub stage: spark_runtime::gpu::DevicePtr,
}

/// Per-block Manifold-Constrained Hyper-Connection (mHC) parameters for one
/// site (attention or FFN). All buffers are float32 device pointers, matching
/// the checkpoint dtype. See `ops::hc_pre` / `ops::hc_post`.
pub struct HcSiteWeights {
    /// Mix projection `fn`: `[mix_hc, hc_mult*hidden]` f32, where
    /// `mix_hc = (2 + hc_mult) * hc_mult`.
    pub hc_fn: DevicePtr,
    /// Mix bias `base`: `[mix_hc]` f32.
    pub hc_base: DevicePtr,
    /// Mix scale: `[3]` f32 (pre / post / comb scalars).
    pub hc_scale: DevicePtr,
}

/// Both HC sites for a DeepSeek-V4 block: the attention site runs before/after
/// attention, the FFN site before/after the MoE FFN.
/// Model-level HC head parameters (final collapse before LM head).
/// Loaded once, attached to every layer, but only used by the last layer.
#[derive(Clone)]
pub struct HcHeadWeights {
    /// Mix projection `head_fn`: `[hc_mult, hc_mult*hidden]` f32.
    pub hc_fn: DevicePtr,
    /// Mix bias `head_base`: `[hc_mult]` f32.
    pub hc_base: DevicePtr,
    /// Mix scale: `[1]` f32.
    pub hc_scale: DevicePtr,
}

pub struct HcWeights {
    pub attn: HcSiteWeights,
    pub ffn: HcSiteWeights,
    /// Model-level head weights. `Some` on all layers (replicated pointer),
    /// consumed only by the last layer's `hc_head` call.
    pub head: Option<HcHeadWeights>,
    pub hc_mult: usize,
    pub sinkhorn_iters: usize,
    pub hc_eps: f32,
}
