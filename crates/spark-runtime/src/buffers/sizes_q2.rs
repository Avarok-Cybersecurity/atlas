// SPDX-License-Identifier: AGPL-3.0-only

//! Native keep-packed Q2_0 buffer sizing, split out of `sizes.rs` (≤500 LoC
//! cap). Both scratch buffers are env-gated — 0 (→ NULL) unless the respective
//! `ATLAS_GGUF_NATIVE_Q2*` flag is set, so non-Q2 models pay nothing.

use atlas_core::config::ModelConfig;

/// Bytes for the native keep-packed Q2_0 prefill transient-dequant scratch: the
/// LARGEST keep-packed projection `[N, K]` expanded to BF16 (2 bytes/elem). The
/// prefill dequant writes `N*K` BF16 elements into this buffer, which is then
/// consumed by the same-stream GEMM and reused by the next projection.
///
/// Every keep-packed projection has exactly one dimension equal to
/// `hidden_size` (FFN gate/up `[inter, h]`, FFN down `[h, inter]`, attention
/// q/k/v/o `[·, h]` or `[h, ·]`, fused GDN `in_proj_qkvz [qkvz, h]`), so
/// `N*K = max_other_dim * hidden` — an EXACT bound, not an over-estimate, for
/// the covered families (the k/v and gated-q terms are safe upper bounds).
/// Independent of batch tokens (this dequants the WEIGHT, not activations).
pub fn q2_dequant_scratch_bytes(config: &ModelConfig) -> usize {
    let bf16 = 2;
    let hd = config.head_dim;
    let q_proj_mul = if config.attn_gated { 2 } else { 1 };
    let max_n = config
        .intermediate_size
        .max(config.ssm_qkvz_size())
        .max(config.mamba2_in_proj_size())
        .max(config.num_attention_heads * q_proj_mul * hd)
        .max(2 * config.num_key_value_heads * hd);
    max_n * config.hidden_size * bf16
}

/// `(q2_dequant_scratch, q2_act_q8)` sizes for the arena. `m` = max batch
/// tokens, `h` = hidden_size, `hd` = head_dim.
///
/// - `q2_dequant_scratch` (Tier-1, `ATLAS_GGUF_NATIVE_Q2=1`): the widest
///   keep-packed projection expanded to BF16 (see [`q2_dequant_scratch_bytes`]).
/// - `q2_act_q8` (Tier-2 MMQ, `ATLAS_GGUF_NATIVE_Q2_MMQ=1`): the q8_1 activation
///   scratch. Widest INPUT dim K — FFN gate/up (h) or down (intermediate), attn
///   qkv (h) or o (q_heads*head_dim), GDN qkvz (h). q8_1_mmq is 4 bytes/elem
///   over kpad (K rounded to 256), + 1MB margin — matches `q8_1_scratch_bytes`.
pub fn q2_scratch_sizes(config: &ModelConfig, m: usize, h: usize, hd: usize) -> (usize, usize) {
    let q2_dequant_scratch = if std::env::var("ATLAS_GGUF_NATIVE_Q2").ok().as_deref() == Some("1") {
        q2_dequant_scratch_bytes(config)
    } else {
        0
    };

    let q2_act_q8 = if std::env::var("ATLAS_GGUF_NATIVE_Q2_MMQ").ok().as_deref() == Some("1") {
        let kmax = h
            .max(config.intermediate_size)
            .max(config.num_attention_heads * hd);
        let kpad = kmax.div_ceil(256) * 256;
        m * kpad * 4 + (1 << 20)
    } else {
        0
    };

    (q2_dequant_scratch, q2_act_q8)
}
