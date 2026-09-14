// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 6: gate multiply + O projection. Split from `attn.rs` (500-LoC cap);
//! a child module of `attn` so `pub(super)` ctx internals stay reachable via
//! the shared `multi_seq` ancestry.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::HeadGateActivation;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use crate::weight_map::WeightQuantFormat;

/// The shared shape of `ops::w8a16_gemv_batch{4,16}` (contiguous A and C), so
/// the MAX_M choice in the FP8 o_proj tier is one branch instead of two
/// duplicated call sites — the same pattern `qkv_fp8_batch.rs` uses for the
/// `_strided` pair.
type BatchGemv = fn(
    &dyn GpuBackend,
    KernelHandle,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    u32,
    u32,
    u32,
    u64,
) -> Result<()>;

impl Qwen3AttentionLayer {
    /// Phase 6: gate multiply (when gated) + O projection. Writes to
    /// `o_out`. Returns the o_out buffer pointer.
    pub(in super::super) fn ms_phase_o_proj(
        &self,
        c: &MultiSeqCtx<'_>,
        attn_out: DevicePtr,
    ) -> Result<DevicePtr> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            hd,
            bf16,
            q_dim,
            per_seq_qkv,
            qkv_buf,
            normed,
            ..
        } = *c;
        if self.gated {
            // ONE launch for all n sequences. `attn_out` is contiguous [n, q_dim]
            // and the gate lives at a fixed offset inside each sequence's slice of
            // `qkv_buf`, i.e. strided by per_seq_qkv — which is exactly the layout
            // `sigmoid_gate_mul_batched` takes (`gate[t * gate_stride + d]`, stride
            // in ELEMENTS). The PREFILL path already drives this kernel on these
            // same buffers (prefill/paged.rs); multi-seq decode was looping the
            // single-token variant instead, n launches per layer x 16 layers.
            debug_assert_eq!(
                per_seq_qkv % bf16,
                0,
                "gate stride must be whole bf16 elements"
            );
            ops::sigmoid_gate_mul_batched(
                fwd.gpu,
                self.sigmoid_gate_mul_batched_k,
                attn_out,
                qkv_buf.offset(q_dim as usize * bf16),
                attn_out,
                q_dim,
                (per_seq_qkv / bf16) as u32,
                n as u32,
                stream,
            )?;
        }

        if let Some(ref g_proj) = self.head_gate_weight {
            let gate_buf = qkv_buf;
            // See the decode-path note: N = nq = 72 gives dense_gemm_tc only
            // ceil(72/64) = 2 CTAs. Use the batched GEMV (ceil(N/4) CTAs), which
            // also keeps this consistent with the single-sequence decode path.
            // ★ n MUST be in 2..=8: dense_gemv_bf16_batchm caps rows at a
            // compile-time MAX_M 8 and CLAMPS silently (`m = M > MAX_M ? MAX_M
            // : M`), so a larger n leaves gate rows 8..n unwritten and the
            // broadcast below multiplies attn_out by whatever stale bytes sit
            // in `qkv_buf` — silently wrong hidden states on every head-gated
            // model (Laguna-S-2.1, Step3.7) at decode concurrency >= 9.
            // `padded_batch_n`'s ladder is [2,4,8,12,16,24,32,48,64,96,128],
            // so n > 8 is routine, not hypothetical. dense_gemm_tc below
            // handles any M, so the fallback is correct, just slower.
            if (2..=8).contains(&n)
                && self.dense_gemv_batchm_k.0 != 0
                && super::super::qkv::bf16_batchm_enabled()
            {
                ops::dense_gemv_batchm(
                    fwd.gpu,
                    self.dense_gemv_batchm_k,
                    normed,
                    g_proj,
                    gate_buf,
                    n as u32,
                    nq,
                    h as u32,
                    nq, // gate rows are nq BF16 elements apart
                    stream,
                )?;
            } else {
                ops::dense_gemm_tc(
                    fwd.gpu,
                    self.dense_gemm_tc_k,
                    normed,
                    g_proj,
                    gate_buf,
                    n as u32,
                    nq,
                    h as u32,
                    stream,
                )?;
            }
            match self.head_gate_activation {
                HeadGateActivation::Sigmoid => ops::sigmoid_gate_mul_head_broadcast(
                    fwd.gpu,
                    self.sigmoid_gate_head_broadcast_k,
                    attn_out,
                    gate_buf,
                    attn_out,
                    nq,
                    hd,
                    n as u32,
                    stream,
                )?,
                HeadGateActivation::Softplus => ops::softplus_gate_mul_head_broadcast(
                    fwd.gpu,
                    self.softplus_gate_head_broadcast_k,
                    attn_out,
                    gate_buf,
                    attn_out,
                    nq,
                    hd,
                    n as u32,
                    stream,
                )?,
            }
        }

        let o_out = fwd.buffers.moe_output();
        if let Some(q2) = self.o_weight.as_ref().and_then(|w| w.as_packed_q2()) {
            // Keep-packed Q2_0 (Tier-1c): per-token 2-bit o_proj GEMV.
            for i in 0..n {
                let attn_out_i = attn_out.offset(i * q_dim as usize * bf16);
                let o_out_i = o_out.offset(i * h * bf16);
                ops::q2_0_gemv_vec(fwd.gpu, self.q2_0_gemv_k, attn_out_i, q2, o_out_i, stream)?;
            }
        } else if let Some(o_bf16) = self.o_dense_bf16.as_ref() {
            // ATLAS_FP8_DEQUANT_ATTN_TO_BF16: O-proj dequanted to BF16 at load.
            // attn_out is contiguous [n, q_dim] and o_out is [n, h], so a single
            // batched GEMM reads the BF16 o_proj weight ONCE for all n sequences
            // instead of once per sequence (per-seq dense_gemv re-read it N×).
            //
            // At small n the batched GEMV beats dense_gemm: dense_gemm's grid is
            // [ceil(N/16), ceil(M/16)] with a 16-row tile, so M<=8 wastes >=50%
            // of every tile and it is a scalar FFMA kernel (~89 GB/s measured)
            // against a ~274 GB/s streaming GEMV. o_proj is N=h=3072, K=nq*hd,
            // i.e. the same weight bytes as q_proj -- worth the branch.
            if (2..=8).contains(&n)
                && self.dense_gemv_batchm_k.0 != 0
                && super::super::qkv::bf16_batchm_enabled()
            {
                ops::dense_gemv_batchm(
                    fwd.gpu,
                    self.dense_gemv_batchm_k,
                    attn_out,
                    o_bf16,
                    o_out,
                    n as u32,
                    h as u32,
                    nq * hd,
                    h as u32, // o_out rows are h BF16 elements apart
                    stream,
                )?;
            } else {
                ops::dense_gemm(
                    fwd.gpu,
                    self.dense_gemm_k,
                    attn_out,
                    o_bf16,
                    o_out,
                    n as u32,
                    h as u32,
                    nq * hd,
                    stream,
                )?;
            }
        } else if let Some(o_fp8) = self.o_weight.as_ref().and_then(|w| w.as_fp8()) {
            // ── W8A8 block-scaled cuBLASLt at 5..16 rows (#927) ──
            // Round 7 (H100, 2026-09-11, n=16): the 16 `o_proj` launches inside
            // the GrdX=1280 `w8a16_gemv_batch16` group are ~1.70 ms of the
            // 43.595 ms step, while the dense FFN ran the SAME 16 rows through
            // cuBLASLt W8A8 at ~128 us/layer. Both operands are contiguous
            // here, so this is the plain `ldc == N` case. Declining for ANY
            // reason keeps the GEMV group loop below, untouched.
            if self.try_ms_o_proj_decode_w8a8(c, o_fp8, attn_out, o_out)? {
                return self.ms_o_proj_lora(c, attn_out, o_out);
            }
            // Both matrices are contiguous. Share each block-scaled weight
            // pass across as many rows as one kernel instantiation covers,
            // without staging or requantization. Keep the scalar route for
            // single-row decode and older bundles.
            //
            // #927: `step` used to be a flat 4, so 5..=16 concurrent decode
            // rows read the o_proj weight ceil(n/4) times — 4 full passes at
            // n=16. `w8a16_gemv_batch16` is the MAX_M=16 instantiation of the
            // SAME template (identical K order and per-row reduction tree, so
            // bit-identical per row to both `w8a16_gemv_batch4` and the scalar
            // `w8a16_gemv`), which makes that ONE pass. Rows stay contiguous
            // either way, so the group loop below is unchanged apart from its
            // stride — n > 16 still walks in 16-row groups.
            let block_scaled = o_fp8.scale_format == WeightQuantFormat::Fp8BlockScaled
                && h % 128 == 0
                && q_dim % 128 == 0;
            let wide = n > 4 && self.w8a16_gemv_batch16_k.0 != 0;
            // #927 tensor-core tier, same 16-row group, `ATLAS_ATTN_M16_TC`
            // (or the `ATLAS_M16_TC` umbrella) only — NOT the FFN's lever; the
            // two split in round 6 because the H100 measured this tier -21.7%
            // and the FFN arm +13.7% in one serve:
            // `w8a16_gemm_m16` replaces the batch16 GEMV's 16 scalar FFMA per
            // weight byte with one m16n8k16 MMA lane-slot. It REASSOCIATES the
            // K reduction (<= 2 BF16 ULP), which is why it is levered and off by
            // default — SSOT + the H100 numbers: `layers::dense_ffn::m16_tc`.
            // K here is `nq * hd`, and the kernel folds
            // `block_scale[n_block * (K/128) + k/128]`, so it needs whole
            // 128-wide scale blocks on BOTH axes.
            let tc = wide
                && self.m16_tc
                && self.w8a16_gemm_m16_k.0 != 0
                && (nq * hd).is_multiple_of(128);
            let batched = n > 1 && block_scaled && (self.w8a16_gemv_batch4_k.0 != 0 || wide);
            let (gemv, kernel, step) = if !batched {
                (
                    ops::w8a16_gemv_batch4 as BatchGemv,
                    self.w8a16_gemv_batch4_k,
                    1,
                )
            } else if tc {
                crate::layers::qwen3_attention::attn_m16_tc_route::log_o_proj_m16_tc_route(
                    fwd.stats,
                );
                (ops::w8a16_gemm_m16 as BatchGemv, self.w8a16_gemm_m16_k, 16)
            } else if let Some((gemv, kernel)) = self.ncol_contiguous_route(n) {
                // N-COLUMN-BLOCKED, BIT-EXACT (#927, `attn_ncol_gemv.rs`): the
                // batch16 GEMV's one weight pass and its exact per-row
                // reduction order, with the activation loads and BF16->FP32
                // converts amortised over N_COLS adjacent output columns. Same
                // 16-row group, so `step` is unchanged. Reachable only past the
                // `!batched` arm, so `block_scaled` — which this kernel needs
                // for the same `block_scale[(n/128) * (K/128) + k/128]` fold as
                // every rung of this family — already holds.
                (gemv as BatchGemv, kernel, 16)
            } else if wide {
                (
                    ops::w8a16_gemv_batch16 as BatchGemv,
                    self.w8a16_gemv_batch16_k,
                    16,
                )
            } else {
                (
                    ops::w8a16_gemv_batch4 as BatchGemv,
                    self.w8a16_gemv_batch4_k,
                    4,
                )
            };
            for i in (0..n).step_by(step) {
                let attn_out_i = attn_out.offset(i * q_dim as usize * bf16);
                let o_out_i = o_out.offset(i * h * bf16);
                if batched {
                    gemv(
                        fwd.gpu,
                        kernel,
                        attn_out_i,
                        o_fp8.weight,
                        o_fp8.row_scale,
                        o_out_i,
                        (n - i).min(step) as u32,
                        h as u32,
                        nq * hd,
                        stream,
                    )?;
                } else {
                    ops::w8a16_gemv(
                        fwd.gpu,
                        self.w8a16_gemv_k,
                        attn_out_i,
                        o_fp8.weight,
                        o_fp8.row_scale,
                        o_out_i,
                        h as u32,
                        nq * hd,
                        stream,
                    )?;
                }
            }
        } else if n == 3 && !self.attn.o_proj.is_null() {
            ops::w4a16_gemv_batch3(
                fwd.gpu,
                self.w4a16_gemv_batch3_k,
                attn_out,
                &self.attn.o_proj,
                o_out,
                h as u32,
                nq * hd,
                stream,
            )?;
        } else if n == 2 && !self.attn.o_proj.is_null() {
            ops::w4a16_gemv_batch2(
                fwd.gpu,
                self.w4a16_gemv_batch2_k,
                attn_out,
                &self.attn.o_proj,
                o_out,
                h as u32,
                nq * hd,
                stream,
            )?;
        } else if !self.attn.o_proj.is_null() {
            // WIDE-VERIFY BATCHED O-PROJ (DFlash γ=16, n>3). One GEMM reads
            // the o_proj weight ONCE for all n rows instead of the per-row
            // GEMV loop below. attn_out is contiguous [n, q_dim]; o_out is
            // contiguous [n, h]; both already laid out for a single M=n GEMM
            // (no scatter). Uses the pipelined m128_v2 kernel when the
            // transposed weight is present (base M64 GEMM is the slow path).
            self.wide_verify_gemm(
                c,
                attn_out,
                &self.attn.o_proj,
                self.o_nvfp4_t.as_ref(),
                o_out,
                n as u32,
                h as u32,
                nq * hd,
            )?;
        } else {
            for i in 0..n {
                let attn_out_i = attn_out.offset(i * q_dim as usize * bf16);
                let o_out_i = o_out.offset(i * h * bf16);
                self.nvfp4_decode_gemv(
                    fwd.gpu,
                    fwd.levers.gemv_sw,
                    attn_out_i,
                    &self.attn.o_proj,
                    o_out_i,
                    h as u32,
                    nq * hd,
                    stream,
                )?;
            }
        }

        self.ms_o_proj_lora(c, attn_out, o_out)
    }

    /// Per-request O LoRA delta (batched bgmv). x = attn_out (post-gate,
    /// contiguous `[n, q_dim]`); base_out = o_out (contiguous `[n, h]`) folded
    /// in place — matches the single-seq `apply_lora_delta` on o after o_proj.
    /// No-op unless a routing table is installed AND `seq_slot` is non-null.
    ///
    /// A method rather than a tail block so the W8A8 arm above can hand back
    /// through it: every o_proj route must fold the adapter, and an early
    /// return that skipped it would be a silent correctness bug on any served
    /// LoRA.
    fn ms_o_proj_lora(
        &self,
        c: &MultiSeqCtx<'_>,
        attn_out: DevicePtr,
        o_out: DevicePtr,
    ) -> Result<DevicePtr> {
        let MultiSeqCtx {
            fwd,
            n,
            h,
            q_dim,
            stream,
            ..
        } = *c;
        if let Some(ref lw) = self.lora
            && c.seq_slot.0 != 0
            && let Some(ref route) = lw.o_route
        {
            ops::lora_delta::apply_lora_bgmv(
                fwd.gpu,
                &lw.kernels,
                route,
                attn_out,
                o_out,
                c.seq_slot,
                n as u32,
                q_dim,    // x row stride (elements): attn_out is [n, q_dim]
                h as u32, // out row stride (elements): o_out is [n, h] contiguous
                fwd.buffers.lora_xa(),
                stream,
            )?;
        }
        Ok(o_out)
    }
}

#[cfg(test)]
#[path = "o_proj_tests.rs"]
mod tests;
