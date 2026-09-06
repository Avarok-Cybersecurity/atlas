// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 6: gate multiply + O projection. Split from `attn.rs` (500-LoC cap);
//! a child module of `attn` so `pub(super)` ctx internals stay reachable via
//! the shared `multi_seq` ancestry.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::HeadGateActivation;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

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
        if let Some(x) = self.exl3_attn_arm(fwd, "decode_multi_seq o_proj")? {
            // Native EXL3 (ATLAS_EXL3_NATIVE_DENSE=1): attn_out is contiguous
            // [n, q_dim] and o_out contiguous [n, h] — ONE packed o_proj call
            // for all n rows (GEMV tier n <= 8, GEMM above). The LoRA delta
            // below still applies.
            x.o_proj_linear(fwd.gpu, attn_out, o_out, n, stream)?;
        } else if let Some(q2) = self.o_weight.as_ref().and_then(|w| w.as_packed_q2()) {
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
            // FP8 native: per-token w8a16_gemv for O projection.
            for i in 0..n {
                let attn_out_i = attn_out.offset(i * q_dim as usize * bf16);
                let o_out_i = o_out.offset(i * h * bf16);
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

        // ── Per-request O LoRA delta (batched bgmv). x = attn_out (post-gate,
        // contiguous [n, q_dim]); base_out = o_out (contiguous [n, h]) folded in
        // place — matches the single-seq apply_lora_delta on o after o_proj.
        // No-op unless a routing table is installed AND seq_slot is non-null.
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
