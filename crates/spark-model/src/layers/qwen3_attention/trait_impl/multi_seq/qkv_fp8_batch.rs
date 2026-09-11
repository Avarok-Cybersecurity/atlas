// SPDX-License-Identifier: AGPL-3.0-only

//! Batched native-FP8 (W8A16 block-scaled) Q/K/V projections for multi-seq
//! decode — the FP8 sibling of `qkv::ms_qkv_batchm_bf16`.
//!
//! WHY (issue #927, O13). The FP8 attention projections had no batched tier:
//! `w8a16_gemv_batch4`/`_batch16` write a CONTIGUOUS `[M, N]` output, while the
//! multi-seq QKV buffer is `[n, per_seq_qkv]` (14336 BF16 elements per row on
//! Qwen3.8-27B) with Q at offset 0, K after Q and V after K inside each row. So
//! every concurrent decode row re-read the whole q/k/v weight through three
//! scalar `w8a16_gemv` launches — 3n launches and n full weight passes per
//! attention layer per step. In the C=4 decode profile that bucket
//! (qkvz/out_proj) is ~6.3 ms of a 44.7 ms step, the third largest.
//!
//! The `_strided` kernel entry points added alongside this module take the A
//! and C row pitches, so one launch per projection writes all `n` rows straight
//! into their slots. Pure launch-count change: same kernel body, same
//! K-iteration order, same reduction tree, hence bit-identical to the scalar
//! path it replaces (oracle: `examples/native_fp8_qkv_batch_microtest`).

use anyhow::Result;

use super::ctx::MultiSeqCtx;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use crate::weight_map::{Fp8Weight, WeightQuantFormat};

/// The shared shape of `ops::w8a16_gemv_batch{4,16}_strided`, so the MAX_M
/// choice is one branch instead of two duplicated call sites.
type StridedBatchGemv = fn(
    &dyn GpuBackend,
    KernelHandle,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    u32,
    u32,
    u32,
    u32,
    u32,
    u64,
) -> Result<()>;

/// Kill switch for this tier: PRESENCE of `ATLAS_NO_FP8_QKV_BATCH` (any value)
/// restores the per-sequence scalar loop. Read ONCE, never per layer per step,
/// so it cannot vary across CUDA-graph replays — same contract as
/// `qkv::bf16_batchm_enabled`.
pub(super) fn fp8_batchm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_NO_FP8_QKV_BATCH").is_none())
}

impl Qwen3AttentionLayer {
    /// Tier selection predicate. `enabled` is the kill switch, passed in rather
    /// than read here so tests can drive both arms without racing the
    /// process-global `OnceLock` in [`fp8_batchm_enabled`].
    ///
    /// GRAPH-CAPTURE: `c.n` is the ctx n, which IS `padded_n` (the ladder in
    /// `traits::model::padded_batch_n`, which the graph cache is keyed by), so
    /// branching on it bakes exactly the value the graph is keyed by — the
    /// same contract the n==2 / n==3 NVFP4 branches rely on. Never branch on
    /// the unpadded `seqs.len()` here.
    ///
    /// The band is 2..=16, the MAX_M of `w8a16_gemv_batch16_strided`. It read
    /// 2..=8 when this module landed, against a comment that the ladder was
    /// [2,4,8]; the ladder has had rungs 12 and 16 since the C=[1,2,4,8,16]
    /// concurrency work, so padded_n 12 and 16 were dropping back to the
    /// per-sequence scalar `w8a16_gemv` loop — 3n launches and n full weight
    /// passes per attention layer per step, which is the #927 cliff on the
    /// projection side. 17+ still falls through: the kernel's MAX_M is 16 and
    /// it CLAMPS rather than erroring, so the band's upper edge is the
    /// template bound and not a tuning choice.
    pub(super) fn ms_qkv_batchm_fp8_selected(&self, c: &MultiSeqCtx<'_>, enabled: bool) -> bool {
        if !enabled || !(2..=16).contains(&c.n) {
            return false;
        }
        if self.w8a16_gemv_batch4_strided_k.0 == 0 || self.w8a16_gemv_batch16_strided_k.0 == 0 {
            return false;
        }
        let Some((q, k, v)) = self.qkv_fp8_block_scaled() else {
            return false;
        };
        // The kernel indexes block_scale[(n/128) * ceil(K/128) + k/128]; the
        // o_proj FP8 tier carries the same `% 128` guards for the same reason.
        let kv_dim = c.nkv * c.hd;
        let dims_ok = c.h.is_multiple_of(128)
            && c.q_proj_dim.is_multiple_of(128)
            && kv_dim.is_multiple_of(128);
        // Row pitches must be whole BF16 elements and 16B-aligned on the A side
        // (uint4 activation loads); `normed` rows are `h` elements apart.
        let strides_ok = c.per_seq_qkv.is_multiple_of(c.bf16) && c.h.is_multiple_of(8);
        let shapes_ok = q.n == c.q_proj_dim && k.n == kv_dim && v.n == kv_dim;
        dims_ok && strides_ok && shapes_ok
    }

    /// q/k/v all present as native FP8 with 2D block scales (the only format
    /// `w8a16_gemv` and its batched siblings can read; `Fp8PerRow` weights are
    /// dequantized/requantized at load and never reach here).
    fn qkv_fp8_block_scaled(&self) -> Option<(&Fp8Weight, &Fp8Weight, &Fp8Weight)> {
        // A free fn, not a closure: closure lifetime inference cannot tie the
        // borrow of `w` to the returned reference here.
        fn block_scaled(w: &Option<crate::weight_map::QuantWeight>) -> Option<&Fp8Weight> {
            w.as_ref()
                .and_then(|w| w.as_fp8())
                .filter(|w| w.scale_format == WeightQuantFormat::Fp8BlockScaled)
        }
        Some((
            block_scaled(&self.q_weight)?,
            block_scaled(&self.k_weight)?,
            block_scaled(&self.v_weight)?,
        ))
    }

    /// ONE strided launch per projection for all `n` rows, writing straight
    /// into `qkv_buf` at the SAME per-seq offsets the scalar loop uses
    /// (Q at 0, K at `q_proj_bytes`, V after K), then the same post-GEMV step
    /// the per-sequence FP8 branch applies.
    ///
    /// Post-GEMV parity with `ms_qkv_seq_q`/`ms_qkv_seq_kv`:
    /// - gated Q gets the inline `deinterleave_qg` unless a q adapter is
    ///   resident, in which case it is deferred to `ms_qkv_deinterleave_q` past
    ///   the LoRA fold — identical condition, and the strided deinterleave does
    ///   all `n` tokens in one launch (a pure permutation, so bit-identical to
    ///   `n` single-token launches at the same addresses);
    /// - LoRA and the q/k RMS norms are NOT done here: `ms_phase_qkv` runs
    ///   `ms_qkv_apply_lora` / `ms_qkv_norms` for every branch.
    pub(super) fn ms_qkv_batchm_fp8(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            bf16,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        let (q, k, v) = self
            .qkv_fp8_block_scaled()
            .expect("ms_qkv_batchm_fp8 entered without block-scaled FP8 q/k/v");

        // Input rows: `normed` is contiguous [n, h]. Output rows: qkv_buf is
        // strided by per_seq_qkv BYTES; the kernel wants BF16 ELEMENTS.
        debug_assert_eq!(per_seq_qkv % bf16, 0);
        let a_stride = h as u32;
        let c_stride = (per_seq_qkv / bf16) as u32;
        let kv_dim = nkv * hd;

        // ── W8A8 block-scaled cuBLASLt at 5..16 rows (#927) ──
        // Round 7 (H100, 2026-09-11, n=16) put these three at 3.74 ms = 8.6%
        // of the 43.595 ms step on `w8a16_gemv_batch16_strided`, while the
        // dense FFN ran the SAME 16 rows through cuBLASLt W8A8 at ~128
        // us/layer. `ldc = per_seq_qkv` writes each row straight into its slot;
        // the phantom rows and the write-extent bound are argued in
        // `w8a8_decode.rs`. Declining for ANY reason keeps the GEMV tier below
        // — and the post-GEMV deinterleave runs either way.
        if !self.try_ms_qkv_decode_w8a8(c, q, k, v, kv_dim)? {
            self.ms_qkv_batchm_fp8_gemv(c, q, k, v, kv_dim, a_stride, c_stride)?;
        }

        if self.gated && !self.q_lora_active() {
            ops::deinterleave_qg(
                fwd.gpu,
                self.deinterleave_qg_k,
                qkv_buf,
                n as u32,
                nq,
                hd,
                c_stride,
                stream,
            )?;
        }
        Ok(())
    }

    /// The unchanged one-strided-launch-per-projection GEMV tier: `batch4`
    /// below 5 rows, then the `ATLAS_FFN_M16_TC` MMA arm, the bit-exact
    /// N-column arm, and `batch16`.
    #[allow(clippy::too_many_arguments)]
    fn ms_qkv_batchm_fp8_gemv(
        &self,
        c: &MultiSeqCtx<'_>,
        q: &Fp8Weight,
        k: &Fp8Weight,
        v: &Fp8Weight,
        kv_dim: u32,
        a_stride: u32,
        c_stride: u32,
    ) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            qkv_buf,
            normed,
            ..
        } = *c;
        let kv_bytes = kv_dim as usize * bf16;

        // batch4 for n<=4, batch16 for 5..=16 — one launch either way; the only
        // difference is the kernel's compile-time register-array bound (and so
        // the MAX_M the wrapper enforces).
        //
        // 5..=16 ALSO has a tensor-core tier (#927): `w8a16_gemm_m16_strided`
        // is the same one-weight-pass shape but replaces the batch16 GEMV's 16
        // scalar FFMA per weight byte with one m16n8k16 MMA lane-slot, which is
        // what the H100 measured as the difference between 342 GB/s and the
        // HBM3 roofline (SSOT + numbers: `layers::dense_ffn::m16_tc`). It
        // REASSOCIATES the K reduction, so it is behind `ATLAS_ATTN_M16_TC`
        // (or the `ATLAS_M16_TC` umbrella) and off by default; `h % 128 == 0`
        // is already guaranteed by `dims_ok` in the selector, and the row pitch
        // guard by `strides_ok`. That lever is SEPARATE from the FFN's since
        // round 6: this tier measured -21.7% on the H100 in the same serve
        // where the FFN arm measured +13.7%.
        let tc = self.m16_tc && self.w8a16_gemm_m16_strided_k.0 != 0 && h.is_multiple_of(128);
        let (launch, kernel): (StridedBatchGemv, KernelHandle) = if n <= 4 {
            (
                ops::w8a16_gemv_batch4_strided,
                self.w8a16_gemv_batch4_strided_k,
            )
        } else if tc {
            (ops::w8a16_gemm_m16_strided, self.w8a16_gemm_m16_strided_k)
        } else if let Some(route) = self.ncol_strided_route(n) {
            // N-COLUMN-BLOCKED, BIT-EXACT (#927, `attn_ncol_gemv.rs`). Same
            // one-weight-pass shape as the batch16 GEMV below and the same
            // per-row reduction order — one thread just owns N_COLS adjacent
            // output columns, so the 32 `uint4` activation loads and 256
            // BF16->FP32 converts it pays per 16 weight bytes amortise over
            // N_COLS of them. BELOW the `tc` arm on purpose: an operator who
            // sets `ATLAS_FFN_M16_TC` is asking for the MMA route explicitly,
            // and that lever reaches the FFN too.
            route
        } else {
            (
                ops::w8a16_gemv_batch16_strided,
                self.w8a16_gemv_batch16_strided_k,
            )
        };
        let gemv = |w: &Fp8Weight, out: DevicePtr, n_out: u32| {
            launch(
                fwd.gpu,
                kernel,
                normed,
                w.weight,
                w.row_scale,
                out,
                n as u32,
                n_out,
                h as u32,
                a_stride,
                c_stride,
                stream,
            )
        };

        // `q_proj_dim == q_dim` when ungated, so this one width covers both the
        // gated ([Q|gate], width 2*q_dim) and ungated arms of `ms_qkv_seq_q`.
        gemv(q, qkv_buf, q_proj_dim)?;
        gemv(k, qkv_buf.offset(q_proj_bytes), kv_dim)?;
        gemv(v, qkv_buf.offset(q_proj_bytes + kv_bytes), kv_dim)?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "qkv_fp8_batch_tests.rs"]
mod tests;
