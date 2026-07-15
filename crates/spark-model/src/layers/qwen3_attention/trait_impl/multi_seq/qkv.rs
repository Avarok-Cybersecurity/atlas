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
        } else if n > 3
            && self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
        {
            // WIDE-VERIFY BATCHED QKV (DFlash γ=16, n=17). The last per-token
            // weight loop in the wide verify — one GEMM per Q/K/V reads each
            // weight ONCE for all n rows instead of n× (mirrors batch3 with M=n).
            self.ms_qkv_batchn(c)?;
        } else {
            for i in 0..n {
                let normed_i = normed.offset(i * h * bf16);
                let q_out_i = qkv_buf.offset(i * per_seq_qkv);
                let k_out_i = q_out_i.offset(q_proj_bytes);
                let v_out_i = k_out_i.offset((nkv * hd) as usize * bf16);

                self.ms_qkv_seq_q(fwd, normed_i, q_out_i, q_proj_dim, q_dim, nq, hd, h, stream)?;
                self.ms_qkv_seq_kv(fwd, normed_i, k_out_i, v_out_i, nkv, hd, h, stream)?;
            }
        }

        // ── Per-request Q/K/V LoRA delta (batched bgmv), pre-norm. No-op when no
        // routing table is installed or `seq_slot` is null (base model / n==1).
        // For gated Q this folds onto the RAW interleaved [Q|gate] segment; the
        // deferred deinterleave below then splits it (the projection branches
        // skipped their inline deinterleave when a q adapter is resident).
        self.ms_qkv_apply_lora(c)?;
        self.ms_qkv_deinterleave_q(c)?;

        // ── Shared q/k RMS-norm pass (all projection branches). HF computes
        // k_norm(k_proj(x) + Δ), so norms run AFTER the pre-norm LoRA delta.
        let _ = eps; // consumed by ms_qkv_norms via `c`
        self.ms_qkv_norms(c)?;
        Ok(())
    }

    /// `true` when a q_proj adapter is resident on the ACTIVE slot — the
    /// load-fixed (graph-stable) branch that makes the gated projection
    /// branches emit RAW interleaved `[Q|gate]` (deferring `deinterleave_qg`
    /// past the q LoRA fold) instead of the fused gemv+deinterleave fast path.
    fn q_lora_active(&self) -> bool {
        self.lora.as_ref().and_then(|lw| lw.q.as_ref()).is_some()
    }

    /// Deferred Q deinterleave over the per-seq `qkv_buf` Q segments. Runs ONLY
    /// when a q adapter is resident on a gated model: the projection wrote RAW
    /// interleaved `[Q|gate]` and the q LoRA fold in [`Self::ms_qkv_apply_lora`]
    /// has since landed on that raw basis, so the split happens here (in place,
    /// per token) — identical to the inline `deinterleave_qg` the no-adapter
    /// fast path runs during projection.
    fn ms_qkv_deinterleave_q(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        if !self.gated || !self.q_lora_active() {
            return Ok(());
        }
        for i in 0..c.n {
            let q_out_i = c.qkv_buf.offset(i * c.per_seq_qkv);
            ops::deinterleave_qg(
                c.fwd.gpu,
                self.deinterleave_qg_k,
                q_out_i,
                1,
                c.nq,
                c.hd,
                c.q_proj_dim,
                c.stream,
            )?;
        }
        Ok(())
    }

    /// Per-request K/V LoRA routing on the batched decode path. Applies each
    /// sequence's own adapter delta to the strided `qkv_buf` K and V regions
    /// via the fused bgmv (byte-identical to N single-seq `apply_lora_delta`).
    /// No-op unless a routing table is installed AND `seq_slot` is non-null.
    fn ms_qkv_apply_lora(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let Some(ref lw) = self.lora else {
            return Ok(());
        };
        if c.seq_slot.0 == 0 {
            return Ok(());
        }
        let bf16 = c.bf16;
        let out_row_stride = (c.per_seq_qkv / bf16) as u32; // strided [Q|K|V] layout
        let x_row_stride = c.h as u32; // normed rows are contiguous [n, h]
        let kv_bytes = (c.nkv * c.hd) as usize * bf16;
        // Q delta: base_out = the RAW interleaved [Q|gate] segment at qkv_buf
        // offset 0 (width q_proj_dim). Folded BEFORE the deferred deinterleave
        // (`ms_qkv_deinterleave_q`), matching the interleaved basis PEFT
        // trained against. Route is present iff a q adapter is resident.
        if let Some(ref route) = lw.q_route {
            let q_out0 = c.qkv_buf; // Q segment starts at offset 0
            ops::lora_delta::apply_lora_bgmv(
                c.fwd.gpu,
                &lw.kernels,
                route,
                c.normed,
                q_out0,
                c.seq_slot,
                c.n as u32,
                x_row_stride,
                out_row_stride,
                c.fwd.buffers.lora_xa(),
                c.stream,
            )?;
        }
        // K delta: base_out = k_out region (after Q), fold in place.
        if let Some(ref route) = lw.k_route {
            let k_out0 = c.qkv_buf.offset(c.q_proj_bytes);
            ops::lora_delta::apply_lora_bgmv(
                c.fwd.gpu,
                &lw.kernels,
                route,
                c.normed,
                k_out0,
                c.seq_slot,
                c.n as u32,
                x_row_stride,
                out_row_stride,
                c.fwd.buffers.lora_xa(),
                c.stream,
            )?;
        }
        // V delta: base_out = v_out region (after Q and K).
        if let Some(ref route) = lw.v_route {
            let v_out0 = c.qkv_buf.offset(c.q_proj_bytes + kv_bytes);
            ops::lora_delta::apply_lora_bgmv(
                c.fwd.gpu,
                &lw.kernels,
                route,
                c.normed,
                v_out0,
                c.seq_slot,
                c.n as u32,
                x_row_stride,
                out_row_stride,
                c.fwd.buffers.lora_xa(),
                c.stream,
            )?;
        }
        Ok(())
    }

    /// Shared q/k RMS-norm pass over the per-seq `qkv_buf` regions. Extracted
    /// so all projection branches (seq / batch2 / batch3 / batchn) defer norms
    /// to one place, after the pre-norm K/V LoRA delta.
    fn ms_qkv_norms(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            eps,
            q_proj_bytes,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_w_k,
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
                    self.rms_norm_w_k,
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
        if self.gated && !self.q_lora_active() {
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
            // Ungated, OR gated with a q adapter: emit RAW interleaved [Q|gate]
            // (the fused deinterleave is deferred to `ms_qkv_deinterleave_q` so
            // the q LoRA fold lands on the interleaved basis).
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

        // q/k RMS norms deferred to `ms_qkv_norms` (after the pre-norm LoRA
        // delta in `ms_phase_qkv`).
        let _ = (nq, eps);
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
        if self.gated && !self.q_lora_active() {
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
            // Ungated, OR gated with a q adapter: RAW interleaved [Q|gate]
            // (deinterleave deferred past the q LoRA fold).
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

        // q/k RMS norms deferred to `ms_qkv_norms` (after the pre-norm LoRA
        // delta in `ms_phase_qkv`).
        let _ = (nq, eps);
        Ok(())
    }

    /// Best available NVFP4 GEMM for a wide (M>3) verify projection. Prefers
    /// the pipelined `w4a16_gemm_t_m128_v2` (8-warp, cp.async double-buffered
    /// — the same fast kernel the dense FFN prefill uses) when the transposed
    /// weight and v2 kernel are present, then the 4-warp m128, then the N128,
    /// falling back to the base M64 `w4a16_gemm` (the "~10 TFLOP flat
    /// bottleneck") only when no transposed copy exists. B is read once either
    /// way; this just picks the faster tiling/pipelining at M=17.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn wide_verify_gemm(
        &self,
        c: &MultiSeqCtx<'_>,
        input: spark_runtime::gpu::DevicePtr,
        w_base: &crate::weight_map::QuantizedWeight,
        w_t: Option<&crate::weight_map::QuantizedWeight>,
        output: spark_runtime::gpu::DevicePtr,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let gpu = c.fwd.gpu;
        let stream = c.stream;
        if let Some(wt) = w_t {
            // Small-M routing (w4a16_m17_bench): at M<=64 the M64-tile
            // `w4a16_gemm_t` beats the M128-tile kernels (87% of an M128
            // tile is padding at M=17), and `w4a16_gemm_t_k64` wins deep-K
            // shapes. Mirrors dense_ffn::w4a16_prefill_gemm; same
            // ATLAS_FFN_SMALLM=0 kill-switch.
            fn small_m_enabled() -> bool {
                static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                *ON.get_or_init(|| std::env::var("ATLAS_FFN_SMALLM").ok().as_deref() != Some("0"))
            }
            if m <= 64 && k.is_multiple_of(32) && small_m_enabled() {
                if k >= 8192 && k.is_multiple_of(64) && self.w4a16_gemm_t_k64_k.0 != 0 {
                    return ops::w4a16_gemm_n128(
                        gpu,
                        self.w4a16_gemm_t_k64_k,
                        input,
                        wt,
                        output,
                        m,
                        n,
                        k,
                        stream,
                    );
                }
                if self.w4a16_gemm_t_k.0 != 0 {
                    return ops::w4a16_gemm_n128(
                        gpu,
                        self.w4a16_gemm_t_k,
                        input,
                        wt,
                        output,
                        m,
                        n,
                        k,
                        stream,
                    );
                }
            }
            if self.w4a16_gemm_t_m128_v2_k.0 != 0 {
                return ops::w4a16_gemm_n128_m128_v2(
                    gpu,
                    self.w4a16_gemm_t_m128_v2_k,
                    input,
                    wt,
                    output,
                    m,
                    n,
                    k,
                    stream,
                );
            }
            if self.w4a16_gemm_t_m128_k.0 != 0 {
                return ops::w4a16_gemm_n128_m128(
                    gpu,
                    self.w4a16_gemm_t_m128_k,
                    input,
                    wt,
                    output,
                    m,
                    n,
                    k,
                    stream,
                );
            }
        }
        ops::w4a16_gemm(
            gpu,
            self.w4a16_gemm_k,
            input,
            w_base,
            output,
            m,
            n,
            k,
            stream,
        )
    }

    /// Wide-verify (n>3) NVFP4 batched QKV. Reads each of Q/K/V ONCE for all
    /// n rows via `w4a16_gemm`, then scatters into the per-seq interleaved
    /// layout — a direct generalization of `ms_qkv_batch3` to arbitrary n
    /// (the fused batch3 GEMV only exists for n=3). The scatter + per-head
    /// norm loops are cheap (D2D + norm, no weight reads), so they stay.
    fn ms_qkv_batchn(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
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

        // Q projection: single GEMM → contiguous [n, q_proj_dim] in q_scratch
        // (interleaved Q|Gate when gated). q_proj_bytes = q_proj_dim*bf16, so
        // the row stride matches the batch3 output — the scatter below is
        // identical.
        let q_scratch = fwd.buffers.ssm_qkvz();
        self.wide_verify_gemm(
            c,
            normed,
            q_nvfp4,
            self.q_nvfp4_t.as_ref(),
            q_scratch,
            n as u32,
            q_proj_dim,
            h as u32,
        )?;
        if self.gated && !self.q_lora_active() {
            // Split interleaved [Q|Gate] → deinterleaved, in place, all n rows
            // (grid is per-token). Matches what w4a16_gemv_qg_batch3 does inline.
            // Deferred to `ms_qkv_deinterleave_q` (post-fold) when a q adapter is
            // resident, so the delta folds on the raw interleaved basis first.
            ops::deinterleave_qg(
                fwd.gpu,
                self.deinterleave_qg_k,
                q_scratch,
                n as u32,
                nq,
                hd,
                q_proj_dim,
                stream,
            )?;
        }

        // K, V projections: one GEMM each (weights read once).
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(n * kv_bytes);
        self.wide_verify_gemm(
            c,
            normed,
            k_nvfp4,
            self.k_nvfp4_t.as_ref(),
            k_scratch,
            n as u32,
            kv_dim,
            h as u32,
        )?;
        self.wide_verify_gemm(
            c,
            normed,
            v_nvfp4,
            self.v_nvfp4_t.as_ref(),
            v_scratch,
            n as u32,
            kv_dim,
            h as u32,
        )?;

        // Scatter contiguous Q/K/V into the per-seq interleaved qkv_buf.
        for i in 0..n {
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

        // q/k RMS norms deferred to `ms_qkv_norms` (after the pre-norm LoRA
        // delta in `ms_phase_qkv`).
        let _ = (nq, eps);
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
                // Deinterleave deferred past the q LoRA fold when a q adapter is
                // resident (see `ms_qkv_deinterleave_q`).
                if !self.q_lora_active() {
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
            } else if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                if self.q_lora_active() {
                    // Split the FUSED gemv+deinterleave into a raw interleaved
                    // gemv; the deinterleave is deferred past the q LoRA fold.
                    ops::w4a16_gemv(
                        fwd.gpu,
                        self.w4a16_gemv_k,
                        normed_i,
                        nvfp4,
                        q_out_i,
                        q_proj_dim,
                        h as u32,
                        stream,
                    )?;
                } else {
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
                }
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
                if !self.q_lora_active() {
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
