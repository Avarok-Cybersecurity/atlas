// SPDX-License-Identifier: AGPL-3.0-only

//! Phases 3-6: per-sequence RoPE, KV-cache write, batched paged
//! attention, gate multiply + O projection.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::ctx::MultiSeqCtx;
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// Phase 3: per-token RoPE (each sequence has its own position).
    pub(super) fn ms_phase_rope(&self, c: &MultiSeqCtx<'_>, meta: AttnMetadataDev) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            q_proj_bytes,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let pos_i = meta.positions.offset(i * 4); // u32 per position
            ops::rope(
                fwd.gpu,
                self.rope_k,
                q_out_i,
                k_out_i,
                pos_i,
                1,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(fwd.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(fwd.config.rope_theta as f32),
                stream,
            )?;
        }
        Ok(())
    }

    /// Phase 4: per-token KV cache write.
    pub(super) fn ms_phase_cache_write(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nkv,
            hd,
            bs,
            bf16,
            q_proj_bytes,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        let kv_stride = nkv * hd;
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset((nkv * hd) as usize * bf16);
            let slot_i = meta.slot.offset(i * 8); // i64 per slot
            self.write_kv_cache(
                fwd.gpu,
                k_out_i,
                v_out_i,
                kv_cache,
                slot_i,
                1,
                nkv,
                hd,
                bs,
                kv_stride,
                kv_stride,
                stream,
                fwd.graph_capture,
            )?;
        }
        Ok(())
    }

    /// Phase 5: build contiguous Q buffer + run BATCHED paged decode.
    /// Returns the attn_out buffer pointer for downstream phases.
    pub(super) fn ms_phase_paged_decode(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<DevicePtr> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            bs,
            bf16,
            q_dim,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        // Build contiguous Q buffer [N, nq*hd] for batched attention.
        let q_contiguous = fwd.buffers.ssm_qkvz();
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            fwd.gpu.copy_d2d_async(
                q_out_i,
                q_contiguous.offset(i * q_dim as usize * bf16),
                q_dim as usize * bf16,
                stream,
            )?;
        }
        let attn_out = fwd.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);

        // TurboQuant WHT bookends (mirrors decode/attention_forward.rs).
        // The cache holds WHT(K)/WHT(V) for turbo dtypes: rotate the batched
        // Q rows before the paged decode and rotate the output back after —
        // without these the multi-seq batched decode scores raw Q against
        // rotated K and returns output in the rotated-V basis.
        let (wht_k_dtype, wht_v_dtype) = self.kv_dtype.kv_pair();
        let k_is_turbo = wht_k_dtype.is_wht_rotated();
        let v_is_turbo = wht_v_dtype.is_wht_rotated();
        let weight_pre_rotated = std::env::var("TQ_PLUS_WEIGHT_ROTATION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let wht_runtime_active = !weight_pre_rotated && (hd == 128 || hd == 256 || hd == 512);
        if k_is_turbo && self.innerq_apply_q_k.0 != 0 && hd == 128 {
            use spark_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(fwd.gpu, self.innerq_apply_q_k)
                .grid([n as u32 * nq, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(q_contiguous)
                .arg_u32(hd)
                .launch(stream)?;
        }
        if k_is_turbo && wht_runtime_active && self.wht_bf16_k.0 != 0 {
            use spark_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(fwd.gpu, self.wht_bf16_k)
                .grid([n as u32 * nq, 1, 1]) // one warp per (seq, q_head)
                .block([32, 1, 1])
                .arg_ptr(q_contiguous)
                .arg_u32(hd)
                .launch(stream)?;
        }
        self.run_paged_decode(
            fwd.gpu,
            q_contiguous,
            kv_cache,
            attn_out,
            meta.block_table,
            meta.seq_len,
            meta.max_blocks_per_seq,
            n as u32,
            nq,
            nkv,
            hd,
            bs,
            inv_sqrt_d,
            nq * hd,
            fwd.buffers.splitk_workspace(),
            stream,
        )?;
        if v_is_turbo && wht_runtime_active && self.wht_bf16_k_inv.0 != 0 {
            use spark_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(fwd.gpu, self.wht_bf16_k_inv)
                .grid([n as u32 * nq, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(attn_out)
                .arg_u32(hd)
                .launch(stream)?;
        }
        Ok(attn_out)
    }

    /// Phase 6: gate multiply (when gated) + O projection. Writes to
    /// `o_out`. Returns the o_out buffer pointer.
    pub(super) fn ms_phase_o_proj(
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
            ..
        } = *c;
        if self.gated {
            for i in 0..n {
                let gate_i = qkv_buf.offset(i * per_seq_qkv + q_dim as usize * bf16);
                let attn_out_i = attn_out.offset(i * q_dim as usize * bf16);
                ops::sigmoid_gate_mul(
                    fwd.gpu,
                    self.sigmoid_gate_mul_k,
                    attn_out_i,
                    gate_i,
                    attn_out_i,
                    q_dim,
                    stream,
                )?;
            }
        }

        let o_out = fwd.buffers.moe_output();
        if let Some(o_bf16) = self.o_dense_bf16.as_ref() {
            // ATLAS_FP8_DEQUANT_ATTN_TO_BF16: O-proj dequanted to BF16 at load.
            // attn_out is contiguous [n, q_dim] and o_out is [n, h], so a single
            // batched GEMM reads the BF16 o_proj weight ONCE for all n sequences
            // instead of once per sequence (per-seq dense_gemv re-read it N×).
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
                ops::w4a16_gemv(
                    fwd.gpu,
                    self.w4a16_gemv_k,
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
