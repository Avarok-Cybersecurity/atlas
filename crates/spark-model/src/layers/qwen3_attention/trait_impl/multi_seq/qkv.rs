// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 2: per-token Q/K/V projection. Three branches:
//! - n=3 + NVFP4 → batch3 GEMV path
//! - n=2 + NVFP4 → batch2 GEMV path
//! - else        → sequential per-token GEMV (FP8/NVFP4/BF16 fallback)
//!
//! Both batch paths read each weight once for N tokens and then scatter
//! into the per-seq QKV layout. The sequential path repeats the GEMV per
//! token but supports every weight encoding.

use anyhow::Result;

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    pub(super) fn ms_phase_qkv(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_dim,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;

        if n == 3
            && self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
        {
            self.ms_qkv_batch3(c)?;
        } else if n == 2
            && self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
        {
            self.ms_qkv_batch2(c)?;
        } else {
            for i in 0..n {
                let normed_i = normed.offset(i * h * bf16);
                let q_out_i = qkv_buf.offset(i * per_seq_qkv);
                let k_out_i = q_out_i.offset(q_proj_bytes);
                let v_out_i = k_out_i.offset((nkv * hd) as usize * bf16);

                self.ms_qkv_seq_q(fwd, normed_i, q_out_i, q_proj_dim, q_dim, nq, hd, h, stream)?;
                self.ms_qkv_seq_kv(fwd, normed_i, k_out_i, v_out_i, nkv, hd, h, stream)?;

                if !self.attn.q_norm.weight.is_null() {
                    ops::rms_norm(
                        fwd.gpu,
                        self.rms_norm_k,
                        q_out_i,
                        &self.attn.q_norm,
                        q_out_i,
                        nq,
                        hd,
                        eps,
                        stream,
                    )?;
                }
                if !self.attn.k_norm.weight.is_null() {
                    ops::rms_norm(
                        fwd.gpu,
                        self.rms_norm_k,
                        k_out_i,
                        &self.attn.k_norm,
                        k_out_i,
                        nkv,
                        hd,
                        eps,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// n=3 NVFP4 batched path.
    fn ms_qkv_batch3(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let q_nvfp4 = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let k_nvfp4 = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let v_nvfp4 = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();

        let q_scratch = fwd.buffers.ssm_qkvz();
        if self.gated {
            ops::w4a16_gemv_qg_batch3(
                fwd.gpu,
                self.w4a16_gemv_qg_batch3_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                nq,
                hd,
                stream,
            )?;
        } else {
            ops::w4a16_gemv_batch3(
                fwd.gpu,
                self.w4a16_gemv_batch3_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                stream,
            )?;
        }

        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(3 * kv_bytes);
        ops::w4a16_gemv_dual_batch3(
            fwd.gpu,
            self.w4a16_gemv_dual_batch3_k,
            normed,
            k_nvfp4,
            k_scratch,
            v_nvfp4,
            v_scratch,
            kv_dim,
            h as u32,
            stream,
        )?;

        for i in 0..3usize {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset(kv_bytes);
            fwd.gpu.copy_d2d_async(
                q_scratch.offset(i * q_proj_bytes),
                q_out_i,
                q_proj_bytes,
                stream,
            )?;
            fwd.gpu
                .copy_d2d_async(k_scratch.offset(i * kv_bytes), k_out_i, kv_bytes, stream)?;
            fwd.gpu
                .copy_d2d_async(v_scratch.offset(i * kv_bytes), v_out_i, kv_bytes, stream)?;
        }

        for i in 0..3usize {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    q_out_i,
                    &self.attn.q_norm,
                    q_out_i,
                    nq,
                    hd,
                    eps,
                    stream,
                )?;
            }
            if !self.attn.k_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    k_out_i,
                    &self.attn.k_norm,
                    k_out_i,
                    nkv,
                    hd,
                    eps,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// n=2 NVFP4 batched path.
    fn ms_qkv_batch2(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let q_nvfp4 = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let k_nvfp4 = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let v_nvfp4 = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();

        let q_scratch = fwd.buffers.ssm_qkvz();
        if self.gated {
            ops::w4a16_gemv_qg_batch2(
                fwd.gpu,
                self.w4a16_gemv_qg_batch2_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                nq,
                hd,
                stream,
            )?;
        } else {
            ops::w4a16_gemv_batch2(
                fwd.gpu,
                self.w4a16_gemv_batch2_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                stream,
            )?;
        }

        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(2 * kv_bytes);
        ops::w4a16_gemv_dual_batch2(
            fwd.gpu,
            self.w4a16_gemv_dual_batch2_k,
            normed,
            k_nvfp4,
            k_scratch,
            v_nvfp4,
            v_scratch,
            kv_dim,
            h as u32,
            stream,
        )?;

        for i in 0..2usize {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset(kv_bytes);
            fwd.gpu.copy_d2d_async(
                q_scratch.offset(i * q_proj_bytes),
                q_out_i,
                q_proj_bytes,
                stream,
            )?;
            fwd.gpu
                .copy_d2d_async(k_scratch.offset(i * kv_bytes), k_out_i, kv_bytes, stream)?;
            fwd.gpu
                .copy_d2d_async(v_scratch.offset(i * kv_bytes), v_out_i, kv_bytes, stream)?;
        }

        for i in 0..2usize {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    q_out_i,
                    &self.attn.q_norm,
                    q_out_i,
                    nq,
                    hd,
                    eps,
                    stream,
                )?;
            }
            if !self.attn.k_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    k_out_i,
                    &self.attn.k_norm,
                    k_out_i,
                    nkv,
                    hd,
                    eps,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// Sequential per-token Q projection (handles gated and ungated).
    #[allow(clippy::too_many_arguments)]
    fn ms_qkv_seq_q(
        &self,
        fwd: &crate::layer::ForwardContext<'_>,
        normed_i: spark_runtime::gpu::DevicePtr,
        q_out_i: spark_runtime::gpu::DevicePtr,
        q_proj_dim: u32,
        q_dim: u32,
        nq: u32,
        hd: u32,
        h: usize,
        stream: u64,
    ) -> Result<()> {
        if self.gated {
            if let Some(fp8) = self.q_weight.as_ref().and_then(|w| w.as_fp8()) {
                ops::w8a16_gemv(
                    fwd.gpu,
                    self.w8a16_gemv_k,
                    normed_i,
                    fp8.weight,
                    fp8.row_scale,
                    q_out_i,
                    q_proj_dim,
                    h as u32,
                    stream,
                )?;
                ops::deinterleave_qg(
                    fwd.gpu,
                    self.deinterleave_qg_k,
                    q_out_i,
                    1,
                    nq,
                    hd,
                    q_proj_dim,
                    stream,
                )?;
            } else if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                ops::w4a16_gemv_qg(
                    fwd.gpu,
                    self.w4a16_gemv_qg_k,
                    normed_i,
                    nvfp4,
                    q_out_i,
                    q_proj_dim,
                    h as u32,
                    nq,
                    hd,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    fwd.gpu,
                    self.dense_gemv_k,
                    normed_i,
                    &self.attn.q_proj,
                    q_out_i,
                    q_proj_dim,
                    h as u32,
                    stream,
                )?;
                ops::deinterleave_qg(
                    fwd.gpu,
                    self.deinterleave_qg_k,
                    q_out_i,
                    1,
                    nq,
                    hd,
                    q_proj_dim,
                    stream,
                )?;
            }
        } else if let Some(fp8) = self.q_weight.as_ref().and_then(|w| w.as_fp8()) {
            ops::w8a16_gemv(
                fwd.gpu,
                self.w8a16_gemv_k,
                normed_i,
                fp8.weight,
                fp8.row_scale,
                q_out_i,
                q_dim,
                h as u32,
                stream,
            )?;
        } else if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
            ops::w4a16_gemv(
                fwd.gpu,
                self.w4a16_gemv_k,
                normed_i,
                nvfp4,
                q_out_i,
                q_dim,
                h as u32,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                fwd.gpu,
                self.dense_gemv_k,
                normed_i,
                &self.attn.q_proj,
                q_out_i,
                q_dim,
                h as u32,
                stream,
            )?;
        }
        Ok(())
    }

    /// Sequential per-token K + V projections.
    #[allow(clippy::too_many_arguments)]
    fn ms_qkv_seq_kv(
        &self,
        fwd: &crate::layer::ForwardContext<'_>,
        normed_i: spark_runtime::gpu::DevicePtr,
        k_out_i: spark_runtime::gpu::DevicePtr,
        v_out_i: spark_runtime::gpu::DevicePtr,
        nkv: u32,
        hd: u32,
        h: usize,
        stream: u64,
    ) -> Result<()> {
        if let (Some(k_fp8), Some(v_fp8)) = (
            self.k_weight.as_ref().and_then(|w| w.as_fp8()),
            self.v_weight.as_ref().and_then(|w| w.as_fp8()),
        ) {
            ops::w8a16_gemv(
                fwd.gpu,
                self.w8a16_gemv_k,
                normed_i,
                k_fp8.weight,
                k_fp8.row_scale,
                k_out_i,
                nkv * hd,
                h as u32,
                stream,
            )?;
            ops::w8a16_gemv(
                fwd.gpu,
                self.w8a16_gemv_k,
                normed_i,
                v_fp8.weight,
                v_fp8.row_scale,
                v_out_i,
                nkv * hd,
                h as u32,
                stream,
            )?;
        } else if let (Some(k_fp4), Some(v_fp4)) = (
            self.k_weight.as_ref().and_then(|w| w.as_nvfp4()),
            self.v_weight.as_ref().and_then(|w| w.as_nvfp4()),
        ) {
            ops::w4a16_gemv_dual(
                fwd.gpu,
                self.w4a16_gemv_dual_k,
                normed_i,
                k_fp4,
                k_out_i,
                v_fp4,
                v_out_i,
                nkv * hd,
                h as u32,
                stream,
            )?;
        } else {
            if let Some(nvfp4) = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                ops::w4a16_gemv(
                    fwd.gpu,
                    self.w4a16_gemv_k,
                    normed_i,
                    nvfp4,
                    k_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    fwd.gpu,
                    self.dense_gemv_k,
                    normed_i,
                    &self.attn.k_proj,
                    k_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            }
            if let Some(nvfp4) = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                ops::w4a16_gemv(
                    fwd.gpu,
                    self.w4a16_gemv_k,
                    normed_i,
                    nvfp4,
                    v_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    fwd.gpu,
                    self.dense_gemv_k,
                    normed_i,
                    &self.attn.v_proj,
                    v_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
