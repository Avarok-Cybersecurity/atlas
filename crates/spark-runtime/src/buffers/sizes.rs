// SPDX-License-Identifier: AGPL-3.0-only

//! Byte sizes for the per-pass GPU buffer arena.

use atlas_core::config::ModelConfig;
use atlas_core::device::sm121::NUM_SMS;

use super::sizes_q12::{Q12_SIZING_STREAMS, q12_batched_scratch_bytes};

/// Byte sizes of each buffer, derived from ModelConfig.
#[derive(Debug, Clone)]
pub struct BufferSizes {
    pub hidden_states: usize,
    pub residual: usize,
    pub norm_output: usize,
    pub qkv_output: usize,
    pub attn_output: usize,
    pub gate_logits: usize,
    /// FP32 gate logits [m, num_experts] for the ATLAS_FP32_GATE routing path.
    /// Keeps the router GEMM accumulator unrounded into top-K so near-tied
    /// experts don't flip on a BF16 store. Allocated whenever num_experts > 0.
    pub gate_logits_f32: usize,
    /// FP32 MoE-input norm output [m, hidden] for ATLAS_FP32_ROUTING — the
    /// full-precision router_in the gate GEMM consumes. Allocated when experts > 0.
    pub moe_router_in_f32: usize,
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
    /// GDN FLA chunked-prefill scratch (single buffer, sub-divided W|U|S|uc).
    /// 0 unless the model is a 128-dim-linear-head GDN model (ATLAS_GDN_FLA path).
    pub gdn_fla_scratch: usize,
    /// Mamba-2 SSD chunked-scan scratch (single buffer, sub-divided dt | dA_cumsum | CB).
    /// 0 unless the model has Mamba-2 SSM layers. Shared across layers: they run
    /// sequentially on one stream, so one allocation serves all 40.
    pub ssd_scratch: usize,
    /// Grouped O-projection latent: `[M, o_groups*o_lora_rank]` BF16 (V4-Flash).
    /// 256 (placeholder) when `o_groups == 0`.
    pub o_latent: usize,
    /// Zero-filled BF16 weight (length max_dim) for unweighted RMSNorm under the
    /// offset-from-1 kernel convention (scale = 1+weight → 1.0). DeepSeek-V4 q_b_norm.
    pub norm_unit_w: usize,
    /// HC residual streams: `[M, hc_mult, hidden]` BF16 (DeepSeek-V4 mHC).
    /// 256 (placeholder) when `hc_mult == 0`.
    pub hc_streams: usize,
    /// HC `post` mixing weights: `[M, hc_mult]` F32.
    pub hc_post: usize,
    /// HC `comb` Sinkhorn matrix: `[M, hc_mult, hc_mult]` F32.
    pub hc_comb: usize,
    /// Low-rank mHC split-collapse scratch (Qwen3.8-Flash-Next): the staged
    /// normed vector `[T, hc_mult*hidden]` F32 plus the rank vector
    /// `[T, hc_lowrank]` F32, for SMALL T only — decode runs the collapse as
    /// three multi-block launches because grid=[1] starves the fused kernel
    /// (measured 2.0 ms/call, one SM's bandwidth). Sized for 64 tokens; the
    /// dispatcher falls back to the fused kernel above that.
    pub hc_lowrank_scratch: usize,
    /// QSA stage-2 prefill-selection scratch (Qwen3.8-Flash-Next), SHARED
    /// across the 12 indexer layers (they run serially). Layout, slabbed at
    /// 2048 selective rows: qk [2048, (n_heads+1)*hd] BF16, q_post
    /// [2048, n_heads, hd] F32, scores [2048, max_seq/ratio] F32, lists
    /// [2048, topk] i32. 256 (placeholder) when no indexer.
    pub qsa_select_scratch: usize,
    /// Token IDs `[M]` u32 for the current pass — stable across the layer loop
    /// so DeepSeek-V4 hash-MoE layers can read `tid2eid[token_id]`. Always
    /// allocated (small); unused by models without hash routing.
    pub token_ids: usize,
    /// Dense-FFN activation-quant scratch, SHARED across all layers by the
    /// MMQ (Q4_K), int8 (W4A8), and NVFP4 (W4A4) prefill paths. Was previously a
    /// per-`DenseFfnLayer` field → 64× duplication (18 GB on Qwen3.6-27B) that
    /// OOM'd chunked prefill layer-by-layer. Sized for the largest projection K.
    /// `ffn_act_q8`: q8_1_mmq activations `m*kpad*4 + 1MB` (Q4_K path).
    /// `ffn_act_a`: int8 `[m,K]` / NVFP4 packed `[m,K/2]` activations.
    /// `ffn_act_scale`: int8 `[m,K/32]*4` / NVFP4 `[m,K/16]` group scales.
    /// 0 for MoE models (dense FFN prefill path is Dense-only).
    pub ffn_act_q8: usize,
    pub ffn_act_a: usize,
    pub ffn_act_scale: usize,
    /// FP8 block-scaled activation scratch for prefill projections (qkv / o /
    /// ssm-qkvz). Persistent so the W8A8+FP32-epilogue path stops doing a
    /// per-projection cuMemAlloc + cuStreamSynchronize + cuMemFree. 1 byte/elem.
    pub fp8_act: usize,
    /// Per-128-block FP32 scales paired with `fp8_act` (one f32 per 128 elems).
    pub fp8_act_scale: usize,
    /// LoRA shrink output `xa = x@Aᵀ`: [m, adapter_max_rank] BF16.
    /// 0 (→ NULL alloc) when no adapter is configured (adapter_max_rank == 0).
    pub lora_xa: usize,
    /// LoRA expand output `delta = xa@Bᵀ`: [m, max target n_out] BF16, where
    /// max n_out = max(hidden, intermediate) — covers k/v/o/gate/up/down in
    /// v0 (q_proj is excluded). 0 (→ NULL) when no adapter.
    pub lora_delta: usize,
    /// LoRA hidden-activation scratch [m, intermediate_size] BF16 for the
    /// runtime delta path on FFN projections. 0 (→ NULL) when no adapter.
    pub lora_hact: usize,
    /// LoRA per-request routing slots `[m]` i32 — one adapter SLOT index per
    /// prefilling token (all equal for a single-request prefill; resolves
    /// `-1`→active before upload). Dedicated buffer (not a packed meta offset)
    /// so the m-element prefill slot array never collides with the per-path
    /// positions/slots/block_table region. 0 (→ NULL) when no adapter
    /// (adapter_max_rank == 0).
    pub lora_seq_slot: usize,
    /// Native keep-packed Q2_0 prefill transient-dequant scratch
    /// (`ATLAS_GGUF_NATIVE_Q2=1`). ONE persistent BF16 `[N,K]` buffer sized to
    /// the LARGEST keep-packed projection, REUSED for every per-projection
    /// dequant so prefill stops doing a per-matmul cuMemAlloc +
    /// cuStreamSynchronize + cuMemFree (the multi-second fixed cost behind the
    /// 3.7 s / 28-token TTFT regression). 0 (→ NULL) unless the flag is set.
    pub q2_dequant_scratch: usize,
    /// Native Q2_0 MMQ prefill q8_1 activation scratch (`ATLAS_GGUF_NATIVE_Q2_MMQ=1`).
    /// ONE persistent q8_1_mmq buffer (`m*kpad*4 + 1MB`) shared by every kept-packed
    /// projection (FFN gate/up/down, attn q/k/v/o, GDN qkvz): each seam quantizes
    /// its BF16 activation into this buffer then runs the packed MMQ GEMM — so the
    /// 2-bit weight is never dequantized to a BF16 scratch (kills the ~2s dequant
    /// tax AND the shared-`q2_dequant_scratch` co-dispatch race). Sized to the
    /// widest projection K = max(hidden, intermediate, q_heads*head_dim).
    /// 0 (→ NULL) unless the MMQ sub-flag is set.
    pub q2_act_q8: usize,
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
        max_batch_size: usize,
    ) -> Self {
        // Derived batched-decode metadata layout (rows = max(32, bs)).
        // Byte-identical sizing for every bs <= 32; see `decode_meta.rs`.
        let decode_meta = super::DecodeMetaLayout::for_max_batch_size(max_batch_size);
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
        //   [32768 .. 32768+24R): decode metadata (positions, seq_slot,
        //     slots, seq_lens; R = decode-meta rows, `decode_meta.rs` —
        //     24R = 768 at the 32-row floor)
        //   [32768+24R .. ): decode block table (padded_n × max_blocks × 4 B)
        //   Batched MTP verify (verify_e.rs) overlays the SAME base with
        //   96-row gaps: meta [32768, 32768+1920), bt at +2048 (bt_rows=96).
        //   Each path re-uploads its own layout pre-dispatch; sizing takes
        //   the wider (verify) envelope.
        //
        // MoE scratch: 2 * M * top_k * 4 (indices [M*top_k] u32 + weights [M*top_k] f32)
        let moe_scratch = 2 * m * top_k * 4;
        let max_blocks = max_seq_len
            .checked_div(kv_block_size)
            .map(|q| q + 1)
            .unwrap_or(256);
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
        // Block table metadata: the widest user is the batched MTP verify
        // (verify_e.rs) at R = 96 rows (n=32 × k=3 rows, the wave-11
        // depth-at-width envelope — 32:2), whose bt staging sits at
        // meta_base+2048 (wider 96-row gaps: positions 384 | seq_slot 384 |
        // slots 768 | seq_lens 384). Batched decode (padded_n ≤ 32) and
        // DFlash K=γ+1=17 verify keep the narrow +768 layout — strictly
        // inside this envelope.
        let bt_rows = 96usize; // batched verify R cap (VERIFY_ROW_CAP, verify_e2.rs)
        // Envelope = max(verify 96-row overlay, DERIVED decode layout).
        // The decode layout (`decode_meta.rs`, rows = max(32, bs)) sits
        // strictly inside the verify overlay for every rows <= 64 (bt at
        // 24R <= 1536 < 2048, rows <= 96), so this max() changes NOTHING
        // for bs <= 64; it only grows the scratch once rows > ~85.
        let bt_meta =
            32768 + (2048 + bt_rows * max_blocks * 4).max(decode_meta.meta_bytes(max_blocks));
        let scratch_min = 64 * 1024;
        // Q12 kernel-batched prefill stages N per-stream meta blocks plus a
        // stacked BatchedAttnMetadata block — a strictly larger footprint than
        // the single-stream `prefill_meta`. Provision for `Q12_SIZING_STREAMS`
        // streams splitting the full token arena so the fast path stays
        // available for deep-context concurrent prefills without overrunning
        // scratch (#110: the unprovisioned N-stream multiplication overran the
        // buffer, producing an out-of-range HtoD → sticky CUDA-700).
        let q12_chunk = m.div_ceil(Q12_SIZING_STREAMS).max(1);
        let q12_batched = q12_batched_scratch_bytes(
            Q12_SIZING_STREAMS,
            q12_chunk,
            top_k,
            config.mrope_interleaved,
        );
        let scratch = scratch_min
            .max(moe_scratch + prefill_meta)
            .max(bt_meta)
            .max(q12_batched);

        // Batched expert output buffers for MoE (or dense FFN).
        // Sized for max(K=3 verify, prefill chunk) × top_k experts.
        let k_max = m.max(3); // prefill chunk or K=3 verify, whichever larger
        let expert_inter = if config.num_experts > 0 {
            let routed = config.num_experts_per_tok * config.moe_intermediate_size;
            k_max * routed.max(config.intermediate_size)
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

        // Logits: only last token used during prefill. Cap at 96 tokens —
        // the batched MTP verify's R = Σ ks row cap (n=32 × k=3 rows, the
        // wave-11 depth-at-width envelope; VERIFY_ROW_CAP in verify_e2.rs).
        // This also covers decode=1, batched decode padded_n<=32 PLUS the
        // run_standard mixed path (`decode_b2`) parking prefill logits at
        // row `padded_n` = 32 (the old 33-row bound), spec_verify≤5, and
        // DFlash K=γ+1=17. ~45 MB at vocab 248320 (was ~30 MB at 64 rows,
        // ~16 MB at 33).
        // Derived floor for wide native batches: the run_standard mixed path
        // (`decode_b2`) parks prefill logits at row `padded_n`, which can be
        // as high as `decode_meta.rows()` — so the arena must hold rows+1.
        // Inert (96) for every rows <= 95, i.e. all bs <= 95.
        let logits_tokens = m.min(96.max(decode_meta.rows() + 1));

        // Mamba-2 d_inner may exceed hidden_size; norm_output and attn_output must fit.
        let mamba2_d_inner = config.mamba2_d_inner();
        let max_dim = h.max(mamba2_d_inner);

        // Split-K decode workspace: NUM_SMS * (head_dim + 2) * sizeof(f32).
        // Partials from split CTAs are stored as [o[head_dim], m, l] per split.
        // Total slots = num_seqs * num_splits ≤ NUM_SMS, so this is constant ~48 KB.
        // Read NUM_SMS rather than repeating its value: run_paged_decode derives
        // num_splits from the same constant, so a literal here is a second source
        // of truth that under-allocates — silently, into out-of-bounds device
        // writes — the moment the constant moves.
        let splitk_workspace = NUM_SMS as usize * (hd + 2) * 4;

        // The residual stream is always BF16.
        let residual_elem = bf16;

        // FP8 block-scaled activation scratch for prefill projections. The
        // widest contract dim across call sites is hidden (qkv / ssm-qkvz) or
        // q_heads*head_dim (o_proj). 1 byte/elem fp8 + one f32 per 128-block.
        // Mamba-2 out_proj contracts over d_inner (may exceed hidden), and its
        // prefill input is FP8-precast into this buffer.
        let max_proj_k = h.max(q_heads * hd).max(mamba2_d_inner);
        let fp8_act = m * max_proj_k;
        let fp8_act_scale = m * max_proj_k.div_ceil(128) * 4;
        // LoRA scratch — only when an adapter is configured (adapter_max_rank
        // set programmatically pre-build). Widest target n_out =
        // max(hidden, intermediate, q_proj): covers k/v, o/down (hidden),
        // gate/up (intermediate), and gated q_proj (2*q_heads*head_dim, which
        // can exceed both — e.g. 35B 2*16*256=8192 > hidden 4096).
        let (lora_xa, lora_delta, lora_hact, lora_seq_slot) = if config.adapter_max_rank > 0 {
            let max_n = h
                .max(config.intermediate_size)
                .max(q_proj_mul * q_heads * hd);
            (
                m * config.adapter_max_rank * bf16,
                m * max_n * bf16,
                m * config.intermediate_size * bf16,
                m * 4, // [m] i32 per-request routing slots (prefill path)
            )
        } else {
            (0, 0, 0, 0)
        };

        // GDN FLA chunked-prefill scratch — ONE buffer holding W|U|S|uc back-to-back,
        // sized for the chunked-prefill arena (nt = ceil(max_batch_tokens / CHUNK)).
        // Only the 128-dim-linear-head GDN path uses it (the FLA kernels are compiled
        // for K_DIM=V_DIM=128); 0 otherwise so BufferArena allocs NULL and the
        // ATLAS_GDN_FLA dispatch stays disabled. Layout per region:
        //   W  [nt*nv][CHUNK][kd] bf16 ; U,uc [nt*nv][CHUNK][vd] bf16 ;
        //   S  [nt*nv][kd][vd] bf16 ; gc [nt*nv][CHUNK] f32.
        const FLA_CHUNK: usize = 64;
        // SSD chunked scan (mamba2_ssd_*): dt[H][nc][L] f32 + dA_cs[H][nc][L] f32
        //                                 + CB[nc][G][L][L] f32,  L = 64.
        const SSD_L: usize = 64;
        let ssd_scratch = if config.mamba_num_heads > 0 && config.ssm_state_size > 0 {
            let nc = m.div_ceil(SSD_L) + 1;
            let hh = config.mamba_num_heads;
            let gg = config.n_groups.max(1);
            (hh * nc * SSD_L * 4) * 2 + nc * gg * SSD_L * SSD_L * 4
        } else {
            0
        };

        let gdn_fla_scratch = if config.linear_num_value_heads > 0
            && config.linear_key_head_dim == 128
            && config.linear_value_head_dim == 128
        {
            // +margin: the batched FLA path (ATLAS_GDN_BATCHED_FLA) sizes its
            // regions by total_nt = batch*ceil(chunk_len/64), which can exceed
            // ceil(m/64) by up to `batch` chunks due to per-stream last-chunk
            // rounding. 16 covers the co-dispatch max-seqs.
            let nt = m.div_ceil(FLA_CHUNK) + 16;
            let nv = config.linear_num_value_heads;
            let kd = config.linear_key_head_dim;
            let vd = config.linear_value_head_dim;
            let w = nt * nv * FLA_CHUNK * kd * bf16;
            let u = nt * nv * FLA_CHUNK * vd * bf16;
            let s = nt * nv * kd * vd * bf16;
            let uc = nt * nv * FLA_CHUNK * vd * bf16;
            let gc = nt * nv * FLA_CHUNK * 4;
            w + u + s + uc + gc
        } else {
            0
        };

        // Native keep-packed Q2_0 prefill scratch (Tier-1 transient-dequant +
        // Tier-2 MMQ q8_1 activation); env-gated, 0 unless the flags are set.
        // Sizing rationale + bounds live on `sizes_q2::q2_scratch_sizes`.
        let (q2_dequant_scratch, q2_act_q8) = super::sizes_q2::q2_scratch_sizes(config, m, h, hd);

        // Dense-FFN activation-quant scratch, shared across all layers (SSOT).
        // Sized for the largest projection K = max(hidden, intermediate); the
        // dense_ffn prefill paths pass `h.max(inter)` to the requant kernels.
        // 0 for MoE (num_experts>0) — those never take the dense_ffn MMQ path.
        let (ffn_act_q8, ffn_act_a, ffn_act_scale) = if config.num_experts == 0 {
            let kmax = h.max(config.intermediate_size);
            let kpad = kmax.div_ceil(256) * 256;
            (
                m * kpad * 4 + (1 << 20), // q8_1_mmq: m*kpad*4 + 1MB (matches q8_1_scratch_bytes)
                m * kmax,                 // int8 a_i8 [m,K] ≥ NVFP4 packed [m,K/2]
                m * (kmax / 32) * 4,      // int8 a_scale [m,K/32]*4 ≥ NVFP4 scale [m,K/16]
            )
        } else {
            (0, 0, 0)
        };

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
                // LongCat zero-experts: the router scores (routed + zero)
                // logits even though only `num_experts` expert FFNs exist.
                m * (config.num_experts + config.zero_expert_num) * bf16
            } else {
                256
            },
            gate_logits_f32: if config.num_experts > 0 {
                m * (config.num_experts + config.zero_expert_num) * 4
            } else {
                256
            },
            moe_router_in_f32: if config.num_experts > 0 {
                m * h * 4
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
            gdn_fla_scratch,
            ssd_scratch,
            // Grouped O-projection latent (V4-Flash): [M, o_groups*o_lora_rank].
            o_latent: (m * config.o_groups * config.o_lora_rank * bf16).max(256),
            // Zero-filled weight for unweighted RMSNorm (q_b_norm).
            norm_unit_w: max_dim * bf16,
            // HC buffers: only allocated for DeepSeek-V4 (hc_mult > 0).
            hc_streams: if config.hc_mult > 0 {
                // FP32 mHC highway: the residual streams grow large across the
                // blocks (the manifold-mixing is norm-preserving, eigenvalue 1),
                // so BF16 storage swamps the small per-layer signal at scale and
                // collapses generation. Store the streams in FP32 (4 bytes).
                m * config.hc_mult * h * 4
            } else {
                256
            },
            hc_post: if config.hc_mult > 0 {
                (m * config.hc_mult * 4).max(256)
            } else {
                256
            },
            hc_comb: if config.hc_mult > 0 {
                (m * config.hc_mult * config.hc_mult * 4).max(256)
            } else {
                256
            },
            hc_lowrank_scratch: if config.hc_mult > 0 && config.hc_lowrank > 0 {
                // Two exclusive layouts share this region:
                // - decode split path (T <= 64): normed FP32 [64, hc*H] then
                //   low FP32 [64, rank];
                // - prefill GEMM path (T > 64, slabbed at <= 2048 tokens):
                //   normed BF16 [Ts, hc*H], up_pre BF16 [Ts, hc*H],
                //   low BF16 [Ts, rank], inj_pre BF16 [Ts, hc].
                let t = m.min(64);
                let split = t * (config.hc_mult * h + config.hc_lowrank) * 4;
                let ts = m.min(2048);
                let gemm = ts * (2 * config.hc_mult * h + config.hc_lowrank + config.hc_mult) * 2;
                split.max(gemm)
            } else {
                256
            },
            qsa_select_scratch: if config.index_topk > 0 && config.index_compress_ratio > 0 {
                const ROWS: usize = 2048;
                let qkw = (config.index_n_heads + 1) * config.index_head_dim;
                let n_blocks = max_seq_len.div_ceil(config.index_compress_ratio);
                let topk = config.index_topk / config.index_compress_ratio;
                ROWS * qkw * 2
                    + ROWS * config.index_n_heads * config.index_head_dim * 4
                    + ROWS * n_blocks * 4
                    + ROWS * topk * 4
            } else {
                256
            },
            // Token IDs [M] u32 (stable across the layer loop for hash-MoE).
            token_ids: (m * 4).max(256),
            ffn_act_q8,
            ffn_act_a,
            ffn_act_scale,
            fp8_act,
            fp8_act_scale,
            lora_xa,
            lora_delta,
            lora_hact,
            lora_seq_slot,
            q2_dequant_scratch,
            q2_act_q8,
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
            + self.gate_logits_f32
            + self.moe_router_in_f32
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
            + self.hc_lowrank_scratch
            + self.qsa_select_scratch
            + self.expert_down_out
            + self.splitk_workspace
            + self.gdn_fla_scratch
            + self.ssd_scratch
            + self.hc_streams
            + self.hc_post
            + self.hc_comb
            + self.token_ids
            + self.ffn_act_q8
            + self.ffn_act_a
            + self.ffn_act_scale
            + self.fp8_act
            + self.fp8_act_scale
            + self.lora_xa
            + self.lora_delta
            + self.lora_hact
            + self.lora_seq_slot
            + self.q2_dequant_scratch
            + self.q2_act_q8
    }
}
