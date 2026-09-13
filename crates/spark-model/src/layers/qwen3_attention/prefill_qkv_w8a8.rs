// SPDX-License-Identifier: AGPL-3.0-only

//! W8A8 block-scaled cuBLASLt arm for the **cache-skip Q/K/V PREFILL** chain —
//! the `ATLAS_CUBLAS_GEMM=attn` route for chunk 0 of a prefill, which
//! `prefill_w8a8.rs` (o_proj) and `paged_qkv.rs` (later chunks) left behind.
//!
//! WHY THIS EXISTS (#917 / #928). nsys on 1xH100, 2026-09-11 round 9,
//! `Qwen/Qwen3.8-27B-FP8`, recipe `ATLAS_CUBLAS_GEMM=ffn,ssm,attn`
//! (`nsys-r9-prefill`). In the 368.263 ms GPU-busy union of a 1193-token
//! prefill, `w8a16_gemm_pipelined` is 100.582 ms over 112 launches and
//! `w8a16_gemm_t_m128` a further 12.012 ms over 32. Resolving them by launch
//! geometry — `w8a16_gemm_pipelined`'s grid is `(ceil(N/32), ceil(M/128))`,
//! `w8a16_gemm_n128_m128`'s is `(ceil(N/128), ceil(M/128))` — names the sites:
//!
//! | launches | grid | shape | site | µs |
//! |---|---|---|---|---|
//! | 16 | 384x10 | N=12288 K=5120 M=1168 | **`q_proj`, this chain** | 32 466.8 |
//! | 32 | 8x10 | N=1024 K=5120 M=1168 | **`k_proj`+`v_proj`, this chain** | 12 012.2 |
//!
//! i.e. one pass per attention layer (16 of them) over the FIRST prefill chunk,
//! **44.5 ms** of the 1193-token TTFT. The *later* chunks of the same prefill
//! go through `paged_qkv.rs`, whose W8A8 arm the trace does show
//! (`fp8_gemm_t_blockscaled`, 48 launches at M=25) — so the split was never
//! "attention has no W8A8", it was "chunk 0 has no W8A8".
//!
//! WHY IT WAS EXCLUDED, and what changed. `cache_skip_one_proj` documented the
//! reason in place: the three projections write into two arena buffers as
//! back-to-back regions (`q` alone in `qkv_output`; `k` at `ssm_qkvz` + 0 and
//! `v` at `ssm_qkvz + num_tokens*kv_dim`), and cuBLASLt writes `ceil16(M)`
//! rows, so `k`'s padded rows land inside `v`'s region. THE CHOICE MADE HERE is
//! per-region capacity accounting rather than a padded scratch + copy, because
//! the overlap is provably benign and a copy would cost a full `[M, kv_dim]`
//! D2D per projection per layer:
//!
//!   * `q` is alone in `qkv_output`; it only needs `ceil16(M) * q_proj_dim`
//!     BF16 of room, which `sizes.rs` now allocates.
//!   * `k` runs BEFORE `v` on the same stream, so `k`'s phantom rows
//!     (`M..ceil16(M)`, at most 15) land in `v`'s region and are then
//!     OVERWRITTEN by `v`'s own write. With the `m >= 16` clause below,
//!     `ceil16(M) - M <= 15 < M`, so `v`'s REAL rows cover every one of them —
//!     no reader ever sees `k`'s pad.
//!   * `v`'s own phantom rows are the only ones that survive, at the tail, so
//!     the buffer must hold `(M + ceil16(M)) * kv_dim` BF16. That is the clause
//!     `ssm_qkvz` is checked against, and the extent `sizes.rs` now allocates.
//!
//! Both buffers are bounds-checked at dispatch: if either is too small the
//! whole chain falls back to the W8A16 kernels, which is what an un-armed serve
//! runs byte for byte.
//!
//! WHAT ELSE IT BUYS. The three projections share one `normed` input, so the
//! activation is quantized ONCE for all three instead of once per projection —
//! `paged_qkv.rs` re-quantizes per projection, which the same trace shows as
//! three `per_token_group_quant_fp8` launches per attention layer.
//!
//! ACCURACY. W8A8 quantizes the activation to E4M3 per 128-wide K group and is
//! lossier than W8A16 by construction — the same deliberate trade the o_proj,
//! SSM and dense-FFN prefills already made.
//! `native_fp8_prefill_proj_w8a8_microtest` gates it at cosine >= 0.999 /
//! rel_rms <= 3e-2 against the W8A16 reference at these shapes.
//! `ATLAS_ATTN_QKV_W8A16_ONLY` (presence) restores W8A16.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, KernelHandle};

use super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{Fp8Weight, WeightQuantFormat};

/// `ATLAS_ATTN_QKV_W8A16_ONLY` kill switch: PRESENCE (any value, including
/// empty) keeps the cache-skip Q/K/V prefill on the W8A16 kernels. Presence
/// rather than `=1`, matching `ATLAS_FFN_W8A16_ONLY` — an escape hatch whose
/// `=0` spelling must not mean "on".
pub(super) fn attn_qkv_w8a16_only() -> bool {
    static ONLY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONLY.get_or_init(|| std::env::var_os("ATLAS_ATTN_QKV_W8A16_ONLY").is_some())
}

/// The three destination extents this chain writes, in BF16 elements, given
/// `m` real rows and the cuBLASLt pad. SSOT for both the selector's capacity
/// clauses and the module header's argument above.
///
/// Returns `(q_elems_in_qkv_output, kv_elems_in_ssm_qkvz)`.
pub(super) fn cache_skip_qkv_extents(m: u32, q_proj_dim: u32, kv_dim: u32) -> (usize, usize) {
    let m_pad = ops::cublas_fp8_m_pad(m) as usize;
    // `v` starts at row `m` of its buffer and writes `m_pad` rows from there.
    (
        m_pad * q_proj_dim as usize,
        (m as usize + m_pad) * kv_dim as usize,
    )
}

/// Whether the WHOLE cache-skip Q/K/V chain takes the cuBLASLt W8A8 arm, as a
/// pure function so the CPU tests can pin each clause without a GPU.
///
/// All three projections or none: the phantom-row argument in the module header
/// depends on `k` and `v` both going through the same arm in the same order, so
/// a per-projection decision would be a correctness hazard, not a refinement.
///
/// Clauses, each load-bearing:
///
/// * `!w8a16_only` — the kill switch above.
/// * `cublas_attn` — `ATLAS_CUBLAS_GEMM` must name the `attn` family.
/// * `fp8_blockscaled_prefill` — the `ATLAS_FP8_SINGLE_SCALE` kill switch.
/// * `m >= 16` — the covering argument for `k`'s phantom rows (header). It is
///   stricter than the `m > 4` floor the other prefill arms use, and it is the
///   clause that makes the shared-buffer overlap safe rather than merely
///   in-bounds.
/// * all three weights `Fp8BlockScaled` — cuBLASLt is told the weight scales
///   are a `[N/128, K/128]` grid; a per-row `row_scale` would be read as
///   garbage.
/// * `q_n % 128 == 0`, `kv_n % 128 == 0`, `k % 128 == 0` — that grid, and the
///   activation quantizer's one-scale-per-128-of-K contract.
/// * `blk128x128_stride_ok(k)` (`k % 512 == 0`) — cuBLAS requires the weight
///   scale column stride `K/128` to be a multiple of 4.
/// * room in BOTH destination buffers for the padded extents above.
/// * activation scratch room for `ceil16(M) * K` FP8 and its scales —
///   `cublas_fp8_proj_prequant` zeroes and reads the pad rows of both.
/// * the quantizer kernel, and the VEC128 scale-layout adapter (kernel,
///   scratch, capacity). Without the transpose the GEMM is fast and WRONG
///   (H100 2026-09-11: rel_rms 7.7e-2 on identical FP8 bytes), so a missing
///   adapter must fall back, never proceed.
#[allow(clippy::too_many_arguments)]
pub(super) fn cache_skip_qkv_cublas_selected(
    cublas_attn: bool,
    fp8_blockscaled_prefill: bool,
    w8a16_only: bool,
    formats: [Option<WeightQuantFormat>; 3],
    m: u32,
    q_n: u32,
    kv_n: u32,
    k: u32,
    qkv_output_capacity_bytes: usize,
    ssm_qkvz_capacity_bytes: usize,
    act_capacity_bytes: usize,
    act_scale_capacity_bytes: usize,
    quant_k: KernelHandle,
    scale_kmajor_k: KernelHandle,
    scale_kmajor_buf: DevicePtr,
    scale_kmajor_capacity_bytes: usize,
) -> bool {
    let m_pad = ops::cublas_fp8_m_pad(m) as usize;
    let kg = k as usize / 128;
    let (q_elems, kv_elems) = cache_skip_qkv_extents(m, q_n, kv_n);
    let kmajor_ready = scale_kmajor_k.0 != 0
        && scale_kmajor_buf.0 != 0
        && scale_kmajor_capacity_bytes >= m_pad * kg * 4;
    !w8a16_only
        && cublas_attn
        && fp8_blockscaled_prefill
        && m >= 16
        && formats
            .iter()
            .all(|f| *f == Some(WeightQuantFormat::Fp8BlockScaled))
        && q_n.is_multiple_of(128)
        && kv_n.is_multiple_of(128)
        && k.is_multiple_of(128)
        && spark_runtime::cublaslt::scale_layout::blk128x128_stride_ok(k as usize)
        && q_elems * 2 <= qkv_output_capacity_bytes
        && kv_elems * 2 <= ssm_qkvz_capacity_bytes
        && m_pad * (k as usize) <= act_capacity_bytes
        && m_pad * kg * 4 <= act_scale_capacity_bytes
        && quant_k.0 != 0
        && (kmajor_ready || !ops::cublas_scale_layout_kmajor())
}

impl Qwen3AttentionLayer {
    /// Whether this layer's cache-skip Q/K/V chain runs on cuBLASLt W8A8.
    ///
    /// `m` is the chunk's token count, `k` the hidden size all three contract
    /// over. Called ONCE per chain, not per projection — see the all-or-none
    /// note on [`cache_skip_qkv_cublas_selected`].
    pub(super) fn cache_skip_qkv_w8a8_selected(
        &self,
        ctx: &ForwardContext,
        m: u32,
        q_proj_dim: u32,
        kv_dim: u32,
        k: u32,
    ) -> bool {
        let fmt = |w: Option<&crate::weight_map::QuantWeight>| {
            w.and_then(|w| w.as_fp8()).map(|f| f.scale_format)
        };
        cache_skip_qkv_cublas_selected(
            ctx.dispatch.cublas.attn,
            ctx.dispatch.fp8_blockscaled_prefill,
            attn_qkv_w8a16_only(),
            [
                fmt(self.q_weight.as_ref()),
                fmt(self.k_weight.as_ref()),
                fmt(self.v_weight.as_ref()),
            ],
            m,
            q_proj_dim,
            kv_dim,
            k,
            ctx.buffers.qkv_output_bytes(),
            ctx.buffers.ssm_qkvz_bytes(),
            ctx.buffers.fp8_act_bytes(),
            ctx.buffers.fp8_act_scale_bytes(),
            self.per_token_group_quant_fp8_k,
            self.fp8_act_scale_kmajor_k,
            ctx.buffers.fp8_act_scale_kmajor(),
            ctx.buffers.fp8_act_scale_kmajor_bytes(),
        )
    }

    /// Quantize `normed[m, k]` ONCE into the arena's `fp8_act` /
    /// `fp8_act_scale` scratch for all three projections.
    ///
    /// Q, K and V share one input, so one quantization serves all three. The
    /// chain is same-stream ordered, so the three GEMMs read it without a host
    /// sync and nothing else writes it in between.
    pub(super) fn cache_skip_qkv_w8a8_quant(
        &self,
        ctx: &ForwardContext,
        normed: DevicePtr,
        m: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let m_pad = ops::cublas_fp8_m_pad(m) as usize;
        debug_assert!(m_pad * k as usize <= ctx.buffers.fp8_act_bytes());
        debug_assert!(m_pad * (k as usize / 128) * 4 <= ctx.buffers.fp8_act_scale_bytes());
        ops::per_token_group_quant_fp8(
            ctx.gpu,
            self.per_token_group_quant_fp8_k,
            normed,
            ctx.buffers.fp8_act(),
            ctx.buffers.fp8_act_scale(),
            m,
            k,
            stream,
        )
    }

    /// One projection of the chain: `out[m, n] = a_fp8[m, k] @ weight[n, k]ᵀ`
    /// with both block-scale sets folded in an FP32 epilogue, through cuBLASLt.
    ///
    /// Consumes the activation [`Self::cache_skip_qkv_w8a8_quant`] produced;
    /// allocates nothing. Callers MUST have checked
    /// [`Self::cache_skip_qkv_w8a8_selected`] — this does not re-check and does
    /// not fall back.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn cache_skip_qkv_w8a8_gemm(
        &self,
        ctx: &ForwardContext,
        fp8w: &Fp8Weight,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        ops::cublas_fp8_proj_prequant(
            ctx.gpu,
            self.fp8_act_scale_kmajor_k,
            ctx.buffers.fp8_act(),
            ctx.buffers.fp8_act_scale(),
            ctx.buffers.fp8_act_scale_kmajor(),
            fp8w,
            out,
            m,
            n,
            k,
            stream,
        )
    }

    /// Log-once route line for the cache-skip Q/K/V prefill, emitted from BOTH
    /// arms so the absence of the W8A8 line is never ambiguous between "not
    /// armed" and "log lost".
    pub(super) fn log_cache_skip_qkv_route(&self, ctx: &ForwardContext, cublas: bool) {
        if ctx.stats.once("log:attn_cache_skip_qkv_prefill") {
            if cublas {
                tracing::info!(
                    "[atlas] attention Q/K/V prefill (chunk 0, cache-skip): W8A8 block-scaled \
                     via cuBLASLt, activation quantized once for all three \
                     (per-token 1x128 act scales x 128x128 weight scales, FP32 epilogue). \
                     ATLAS_CUBLAS_GEMM=attn selected it; ATLAS_ATTN_QKV_W8A16_ONLY restores \
                     W8A16. This arm allocates nothing."
                );
            } else {
                tracing::info!(
                    "[atlas] attention Q/K/V prefill (chunk 0, cache-skip): W8A16 \
                     (BF16 act x FP8 weight). W8A8 cuBLASLt not selected — add `attn` to \
                     ATLAS_CUBLAS_GEMM; see #917/#928."
                );
            }
        }
    }
}

#[cfg(test)]
#[path = "prefill_qkv_w8a8_tests.rs"]
mod tests;
