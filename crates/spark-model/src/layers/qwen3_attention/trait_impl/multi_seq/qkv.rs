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

/// Kill-switch for the batched dense-BF16 decode projections (q/k/v here, and
/// o_proj in `attn.rs`). Read ONCE (never per layer per step) so it cannot vary
/// across CUDA-graph replays.
pub(super) fn bf16_batchm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_BF16_QKV_BATCHM").ok().as_deref() != Some("0"))
}

/// Fused [q|k|v] projection GEMM (one N=14336 launch instead of three).
/// Kill switch: `ATLAS_NO_FUSED_QKV=1` restores the three separate GEMMs.
fn fused_qkv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_NO_FUSED_QKV").ok().as_deref() != Some("1"))
}

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

        // Native EXL3 q/k/v (ATLAS_EXL3_NATIVE_DENSE=1) FIRST: every arm below
        // reads projection slots the loader left null for such a layer (the
        // dense-BF16 predicate would otherwise admit an ungated layer).
        if self.ms_qkv_exl3(c)? {
            // qkv_buf written (+ Q deinterleaved unless a q adapter defers it).
        } else if n == 3
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
        } else if (2..=8).contains(&n)
            && !self.gated
            && self.dense_gemv_batchm_k.0 != 0
            && self.qkv_is_dense_bf16()
            && bf16_batchm_enabled()
        {
            // BATCHED DENSE-BF16 QKV. Models whose attention weights are plain
            // BF16 (Laguna, native-BF16 Qwen3/Gemma-4/Mistral/...) had NO
            // batched tier here and fell into the per-sequence loop below, so
            // each concurrent sequence re-read the whole weight matrix. That
            // made 54% of the decode step scale linearly with concurrency
            // (C=1 16.1 -> C=4 21.9 tok/s aggregate, i.e. ~1/n per stream).
            //
            // Graph-capture safe: `n` here is the ctx n, which IS `padded_n`
            // (decode_a2.rs pads to [2,4,8] and keys the graph cache on it), so
            // branching on it bakes exactly the value the graph is keyed by --
            // the same contract the n==2 / n==3 branches above already rely on.
            // Never branch on the unpadded seqs.len() here.
            self.ms_qkv_batchm_bf16(c)?;
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
    /// True when q/k/v are plain dense BF16 — no NVFP4 and no FP8 sidecar — so
    /// the projection actually reads `self.attn.{q,k,v}_proj`. Laguna ships
    /// attention unquantized in the checkpoint and it stays that way.
    fn qkv_is_dense_bf16(&self) -> bool {
        let dense = |w: &Option<crate::weight_map::QuantWeight>| {
            w.as_ref()
                .is_none_or(|w| w.as_nvfp4().is_none() && w.as_fp8().is_none())
        };
        dense(&self.q_weight) && dense(&self.k_weight) && dense(&self.v_weight)
    }

    /// Batched dense-BF16 q/k/v for multi-seq decode: ONE pass over each weight
    /// matrix produces all `n` rows, writing straight into the interleaved
    /// `qkv_buf` via the kernel's `out_stride` (so no scratch + scatter, and
    /// `ms_qkv_apply_lora` / `ms_qkv_norms` see an unchanged layout).
    ///
    /// Bit-identical to the per-sequence loop it replaces: `dense_gemv_batchm`
    /// keeps `dense_gemv_bf16`'s per-row K-iteration order and reduction tree,
    /// and the kernel dir builds with --fmad=false. Verified by
    /// examples/dense_gemv_bf16_batchm_microtest.
    fn ms_qkv_batchm_bf16(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nkv,
            hd,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;

        // Output rows are `per_seq_qkv` bytes apart; the kernel wants the stride
        // in BF16 ELEMENTS.
        debug_assert_eq!(per_seq_qkv % bf16, 0);
        let out_stride = (per_seq_qkv / bf16) as u32;
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;

        let gemv = |w, out, n_out| {
            ops::dense_gemv_batchm(
                fwd.gpu,
                self.dense_gemv_batchm_k,
                normed,
                w,
                out,
                n as u32,
                n_out,
                h as u32,
                out_stride,
                stream,
            )
        };

        gemv(&self.attn.q_proj, qkv_buf, q_proj_dim)?;
        gemv(&self.attn.k_proj, qkv_buf.offset(q_proj_bytes), kv_dim)?;
        gemv(
            &self.attn.v_proj,
            qkv_buf.offset(q_proj_bytes + kv_bytes),
            kv_dim,
        )?;
        Ok(())
    }

    pub(super) fn q_lora_active(&self) -> bool {
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
            bf16,
            q_proj_bytes,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        // ONE launch per norm for all n sequences. Each sequence's head-rows are
        // packed at `hd` inside its own [Q|K|V|gate] block, and the blocks sit
        // `per_seq_qkv` apart — exactly the (rows_per_group, num_groups,
        // row_stride) shape `rms_norm_strided` takes. The per-sequence loop below
        // was 516 launches/step across the 16 attention layers (0.76 ms).
        // Bit-identical: one block per row either way. Kill: ATLAS_NO_QK_NORM_STRIDED=1.
        fn qk_norm_strided_enabled() -> bool {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                std::env::var("ATLAS_NO_QK_NORM_STRIDED").ok().as_deref() != Some("1")
            })
        }
        if n > 1
            && self.rms_norm_strided_k.0 != 0
            && qk_norm_strided_enabled()
            && per_seq_qkv.is_multiple_of(bf16)
        {
            let stride_e = (per_seq_qkv / bf16) as u32;
            if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm_strided(
                    fwd.gpu,
                    self.rms_norm_strided_k,
                    qkv_buf,
                    &self.attn.q_norm,
                    qkv_buf,
                    nq,
                    n as u32,
                    hd,
                    eps,
                    stride_e,
                    stream,
                )?;
            }
            if !self.attn.k_norm.weight.is_null() {
                let k0 = qkv_buf.offset(q_proj_bytes);
                ops::rms_norm_strided(
                    fwd.gpu,
                    self.rms_norm_strided_k,
                    k0,
                    &self.attn.k_norm,
                    k0,
                    nkv,
                    n as u32,
                    hd,
                    eps,
                    stride_e,
                    stream,
                )?;
            }
            return Ok(());
        }
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
        // K=4 MTP verify (M<=4) and K=5..8 chain verify (M=5..8): the batched
        // GEMV reads the non-transposed weight ONCE for all rows at near-peak
        // stream bandwidth. nsys (2026-07-18, drafts=3): the M64-tile
        // w4a16_gemm_t this bypasses cost 16.3 ms/verify-step across the 16
        // attention layers' q/k/v/o at M=4 (94% tile padding) vs ~4.5 ms via
        // the GEMV. m=5..8 rides the narrow batch{5,6,7,8} tiers (batchm_bench:
        // same weight-streaming bandwidth, no M>4 cliff). The family caps at
        // M=8, so the DFlash wide verify (M=17) keeps the GEMM.
        let batchm = self.w4a16_batchm.kernel(m);
        if batchm.0 != 0 {
            return ops::w4a16_gemv_batchm(gpu, batchm, input, w_base, output, m, n, k, stream);
        }
        if let Some(wt) = w_t {
            // Small-M routing (w4a16_m17_bench): at M<=64 the M64-tile
            // `w4a16_gemm_t` beats the M128-tile kernels (87% of an M128
            // tile is padding at M=17), and `w4a16_gemm_t_k64` wins deep-K
            // shapes. Mirrors dense_ffn::w4a16_prefill_gemm; same
            // ATLAS_FFN_SMALLM=0 kill-switch.
            // The `OnceLock<bool>` static that lived here is now a field on
            // `layers::ops::ModelLevers` — resolved when the model is built and carried
            // on `ForwardContext`, because a static outlives the model whose flags it
            // encodes.
            if m <= 64 && k.is_multiple_of(32) && c.fwd.levers.ffn_small_m {
                if k >= crate::layers::w4a16_k64_min_k()
                    && k.is_multiple_of(64)
                    && self.w4a16_gemm_t_k64_k.0 != 0
                {
                    // Narrow-N deep-K twin: bit-identical, and 1.42x at the
                    // o_proj shape (N=5120, K=6144 -> 40 CTAs on 48 SMs).
                    if self.w4a16_gemm_t_k64_n64_k.0 != 0 && crate::layers::k64_n64_wins(m, n) {
                        return ops::w4a16_gemm(
                            gpu,
                            self.w4a16_gemm_t_k64_n64_k,
                            input,
                            wt,
                            output,
                            m,
                            n,
                            k,
                            stream,
                        );
                    }
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
        let kv_dim_e = (nkv * hd) as usize;
        // FUSED [q|k|v]: one N=14336 GEMM instead of three (96/8/8 CTAs). The
        // k/v shapes are N=1024 => 8 CTAs on 48 SMs, the worst-utilised kernels
        // in the model (23.6 GB/s, 9.75x off floor). Bit-identical — same dot
        // products, relocated along N — and the loader only builds the twin when
        // q/k/v share one `weight_scale_2`.
        // Kill switch: ATLAS_NO_FUSED_QKV=1.
        let fused_n = q_proj_dim as usize + 2 * kv_dim_e;
        // n > 8 is REQUIRED, not an optimisation: `wide_verify_gemm` early-returns
        // on the batched-GEMV arms for m <= 8 using the BASE (non-transposed)
        // weight and ignoring `w_t` entirely, so a fused N would read past the
        // q_proj weight. Only at m > 8 is the transposed tile GEMM guaranteed.
        // n varies as sequences finish, so this is hit at every concurrency.
        let use_fused = fused_qkv_enabled() && self.qkv_nvfp4_t.is_some() && n > 8;
        if use_fused {
            // per_seq_qkv == q_proj_bytes + 2*kv_bytes == fused_n*bf16, so the
            // fused GEMM's [n, fused_n] output IS the qkv_buf layout byte for
            // byte. Write straight into it and skip the scatter entirely —
            // that removes 3 GEMMs AND 48 D2D copies per attention layer.
            self.wide_verify_gemm(
                c,
                normed,
                q_nvfp4,
                self.qkv_nvfp4_t.as_ref(),
                qkv_buf,
                n as u32,
                fused_n as u32,
                h as u32,
            )?;
        } else {
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
        }
        if self.gated && !self.q_lora_active() {
            // Split interleaved [Q|Gate] → deinterleaved, in place, all n rows
            // (grid is per-token). Matches what w4a16_gemv_qg_batch3 does inline.
            // Deferred to `ms_qkv_deinterleave_q` (post-fold) when a q adapter is
            // resident, so the delta folds on the raw interleaved basis first.
            ops::deinterleave_qg(
                fwd.gpu,
                self.deinterleave_qg_k,
                if use_fused { qkv_buf } else { q_scratch },
                n as u32,
                nq,
                hd,
                if use_fused {
                    fused_n as u32
                } else {
                    q_proj_dim
                },
                stream,
            )?;
        }

        // K, V projections: one GEMM each (weights read once).
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(n * kv_bytes);
        if !use_fused {
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
        }

        // Scatter contiguous Q/K/V into the per-seq interleaved qkv_buf.
        // Not needed when fused: the GEMM already wrote that exact layout.
        for i in (0..n).take_while(|_| !use_fused) {
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
            if let Some(q2) = self.q_weight.as_ref().and_then(|w| w.as_packed_q2()) {
                ops::q2_0_gemv_vec(fwd.gpu, self.q2_0_gemv_k, normed_i, q2, q_out_i, stream)?;
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
            } else if let Some(fp8) = self.q_weight.as_ref().and_then(|w| w.as_fp8()) {
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
                    self.nvfp4_decode_gemv(
                        fwd.gpu,
                        fwd.levers.gemv_sw,
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
        } else if let Some(q2) = self.q_weight.as_ref().and_then(|w| w.as_packed_q2()) {
            ops::q2_0_gemv_vec(fwd.gpu, self.q2_0_gemv_k, normed_i, q2, q_out_i, stream)?;
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
            self.nvfp4_decode_gemv(
                fwd.gpu,
                fwd.levers.gemv_sw,
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
        if let (Some(k_q2), Some(v_q2)) = (
            self.k_weight.as_ref().and_then(|w| w.as_packed_q2()),
            self.v_weight.as_ref().and_then(|w| w.as_packed_q2()),
        ) {
            ops::q2_0_gemv_vec(fwd.gpu, self.q2_0_gemv_k, normed_i, k_q2, k_out_i, stream)?;
            ops::q2_0_gemv_vec(fwd.gpu, self.q2_0_gemv_k, normed_i, v_q2, v_out_i, stream)?;
        } else if let (Some(k_fp8), Some(v_fp8)) = (
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
                self.nvfp4_decode_gemv(
                    fwd.gpu,
                    fwd.levers.gemv_sw,
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
                self.nvfp4_decode_gemv(
                    fwd.gpu,
                    fwd.levers.gemv_sw,
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
