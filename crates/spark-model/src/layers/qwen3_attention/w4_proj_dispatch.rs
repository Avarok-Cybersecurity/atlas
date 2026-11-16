// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the dispatch helpers the packed (W4) attention arms call (port lane L22b,
//! child of `w4_proj.rs`): row-windowed decode, the shared-input k/v pair, the prefill route with
//! its BF16 decline fallback, and the batched q/k/v tier that decodes into a staging region and
//! scatters into the per-sequence `[Q | K | V]` layout.
//!
//! Every helper checks the installed weight's shape against what its call site multiplies by
//! before the first launch, so a mis-built weight is an error naming the projection, never a
//! wrong answer. Decode helpers make no allocation and no host read once the layer's regions are
//! reserved, so they are graph-capturable.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::Qwen3AttentionLayer;
use super::{BF16, STAGE_ROWS, W4Slot, decode_chunks};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::port::w4::scratch;
use crate::port::w4::select::{self, ProjCall};
use crate::port::w4::{W4Box, W4Linear, W4Scratch};
use crate::weight_map::DenseWeight;

/// Once-per-backend key of the first declined prefill chunk's log line.
const ONCE_DECLINE_LOGGED: &str = "port.w4.attn.decline.logged";

/// `rows` decode rows of `w`, chunked by its window. No shape check, no reservation.
#[allow(clippy::too_many_arguments)]
fn decode_into(
    w: &dyn W4Linear,
    gpu: &dyn GpuBackend,
    x: DevicePtr,
    rows: usize,
    out: DevicePtr,
    n_out: usize,
    k_in: usize,
    s: &W4Scratch,
    stream: u64,
) -> Result<()> {
    for (first, len) in decode_chunks(rows, w.max_decode_rows()) {
        let (xi, oi) = (
            x.offset(first * k_in * BF16),
            out.offset(first * n_out * BF16),
        );
        w.decode(gpu, xi, len, oi, s, stream)?;
    }
    Ok(())
}

/// Two weights of one shape sharing `x` (k and v), through `decode_pair` in chunks both decode.
#[allow(clippy::too_many_arguments)]
fn pair_into(
    a: &dyn W4Linear,
    b: &dyn W4Linear,
    gpu: &dyn GpuBackend,
    x: DevicePtr,
    rows: usize,
    out_a: DevicePtr,
    out_b: DevicePtr,
    n_out: usize,
    k_in: usize,
    s: &W4Scratch,
    stream: u64,
) -> Result<()> {
    let window = a.max_decode_rows().min(b.max_decode_rows());
    for (first, len) in decode_chunks(rows, window) {
        let xi = x.offset(first * k_in * BF16);
        let (oa, ob) = (
            out_a.offset(first * n_out * BF16),
            out_b.offset(first * n_out * BF16),
        );
        debug_assert_eq!(
            select::pair_call(len, a.max_decode_rows(), b.max_decode_rows()),
            select::PairCall::Pair
        );
        a.decode_pair(b, gpu, xi, len, oa, ob, s, stream)?;
    }
    Ok(())
}

impl Qwen3AttentionLayer {
    fn w4_slot(&self, slot: W4Slot) -> Result<&W4Box> {
        match self.w4_proj.get(slot) {
            Some(w) => Ok(w),
            None => anyhow::bail!(
                "attention layer {}: no packed {slot:?} projection installed",
                self.attn_layer_idx
            ),
        }
    }

    /// `rows` contiguous decode rows (`x` `[rows, k_in]`, `out` `[rows, n_out]`), in chunks of the
    /// weight's decode window. Row-invariant bits, graph-capturable once reserved.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn w4_decode_rows(
        &self,
        w: &W4Box,
        what: &str,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        rows: usize,
        out: DevicePtr,
        n_out: usize,
        k_in: usize,
        stream: u64,
    ) -> Result<()> {
        select::check_shape(w.as_ref(), n_out, k_in, what)?;
        let r = self.w4_ready(gpu, stream)?;
        decode_into(w.as_ref(), gpu, x, rows, out, n_out, k_in, &r.s, stream)
    }

    /// The single-sequence k and v (one shared input row) through `decode_pair`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn w4_decode_kv(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        k_out: DevicePtr,
        v_out: DevicePtr,
        kv_dim: usize,
        h: usize,
        stream: u64,
    ) -> Result<()> {
        let (k, v) = (self.w4_slot(W4Slot::K)?, self.w4_slot(W4Slot::V)?);
        select::check_shape(k.as_ref(), kv_dim, h, "attention k_proj")?;
        select::check_shape(v.as_ref(), kv_dim, h, "attention v_proj")?;
        let r = self.w4_ready(gpu, stream)?;
        let (ka, va) = (k.as_ref(), v.as_ref());
        pair_into(ka, va, gpu, x, 1, k_out, v_out, kv_dim, h, &r.s, stream)
    }

    /// A prefill projection: `m` contiguous rows. `m` inside the decode window decodes; wider
    /// prefills go through `W4Linear::prefill` chunk by chunk, and a declined chunk is recomputed
    /// from the BF16 dequant of the weight (made once per call) with the layer's BF16 GEMM.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn w4_prefill_rows(
        &self,
        w: &W4Box,
        what: &str,
        ctx: &ForwardContext,
        x: DevicePtr,
        m: usize,
        out: DevicePtr,
        n_out: usize,
        k_in: usize,
        stream: u64,
    ) -> Result<()> {
        let max = w.max_decode_rows();
        if select::proj_call(m, max) == ProjCall::Decode {
            return self.w4_decode_rows(w, what, ctx.gpu, x, m, out, n_out, k_in, stream);
        }
        select::check_shape(w.as_ref(), n_out, k_in, what)?;
        let gpu = ctx.gpu;
        let r = self.w4_ready(gpu, stream)?;
        let mut bf16: Option<DevicePtr> = None;
        for (first, len) in select::prefill_chunks(m, select::prefill_chunk_cap(max), max) {
            let (xi, oi) = (
                x.offset(first * k_in * BF16),
                out.offset(first * n_out * BF16),
            );
            if w.prefill(gpu, xi, len, oi, &r.s, stream)? {
                continue;
            }
            crate::port::w4::note_prefill_decline();
            if gpu.op_cache().once(ONCE_DECLINE_LOGGED) {
                tracing::info!(
                    "[avarok] attention W4: {what} {} prefill declined at m={len}; BF16 dequant \
                     fallback (logged once; later declines are counted only)",
                    w.format()
                );
            }
            let wb = match bf16 {
                Some(p) => p,
                None => {
                    let p = scratch::bf16_dequant(gpu, &r.spec)?;
                    w.dequant_bf16(gpu, p, stream)?;
                    bf16 = Some(p);
                    p
                }
            };
            self.w4_bf16_gemm(ctx, xi, wb, oi, len, n_out, k_in, stream)?;
        }
        Ok(())
    }

    /// The BF16 GEMM of the decline fallback: the same ladder as the layer's BF16 prefill arms.
    #[allow(clippy::too_many_arguments)]
    fn w4_bf16_gemm(
        &self,
        ctx: &ForwardContext,
        x: DevicePtr,
        weight: DevicePtr,
        out: DevicePtr,
        m: usize,
        n_out: usize,
        k_in: usize,
        stream: u64,
    ) -> Result<()> {
        let (m, n, k) = (m as u32, n_out as u32, k_in as u32);
        if ctx.dispatch.cublas.attn && m > 1 {
            return ops::cublas_bf16_proj_dense(x, weight, out, m, n, k, stream);
        }
        ops::dense_gemm_prefill(
            ctx.gpu,
            self.dense_gemm_k,
            self.dense_gemm_pipelined_k,
            x,
            &DenseWeight { weight },
            out,
            m,
            n,
            k,
            stream,
        )
    }

    /// The batched-decode q/k/v tier (`trait_impl/multi_seq/qkv.rs`): `n` input rows (`normed`
    /// `[n, h]`) into the per-sequence `[Q | K | V]` layout of `qkv_buf` (rows `per_seq_qkv` bytes
    /// apart). One row writes in place; wider batches decode into the staging region (contiguous
    /// rows) and scatter with one pitched copy per projection per block of [`STAGE_ROWS`] rows.
    /// A gated q is then split per row exactly as the batched BF16 tier splits it, deferred past
    /// the q LoRA fold when a q adapter is resident.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn w4_qkv_strided(
        &self,
        gpu: &dyn GpuBackend,
        normed: DevicePtr,
        n: usize,
        h: usize,
        q_proj_dim: usize,
        kv_dim: usize,
        nq: u32,
        hd: u32,
        qkv_buf: DevicePtr,
        per_seq_qkv: usize,
        stream: u64,
    ) -> Result<()> {
        let q = self.w4_slot(W4Slot::Q)?;
        let (k, v) = (self.w4_slot(W4Slot::K)?, self.w4_slot(W4Slot::V)?);
        select::check_shape(q.as_ref(), q_proj_dim, h, "attention q_proj")?;
        select::check_shape(k.as_ref(), kv_dim, h, "attention k_proj")?;
        select::check_shape(v.as_ref(), kv_dim, h, "attention v_proj")?;
        let (q_bytes, kv_bytes) = (q_proj_dim * BF16, kv_dim * BF16);
        ensure!(
            per_seq_qkv == q_bytes + 2 * kv_bytes,
            "attention layer {}: per-sequence q/k/v stride {per_seq_qkv} is not {q_bytes} + 2 x {kv_bytes}",
            self.attn_layer_idx
        );
        let r = self.w4_ready(gpu, stream)?;
        let (qa, ka, va) = (q.as_ref(), k.as_ref(), v.as_ref());
        if n == 1 {
            let (k_out, v_out) = (qkv_buf.offset(q_bytes), qkv_buf.offset(q_bytes + kv_bytes));
            decode_into(qa, gpu, normed, 1, qkv_buf, q_proj_dim, h, &r.s, stream)?;
            pair_into(
                ka, va, gpu, normed, 1, k_out, v_out, kv_dim, h, &r.s, stream,
            )?;
        } else {
            ensure!(
                r.stage_bytes >= STAGE_ROWS * per_seq_qkv,
                "attention layer {}: staging region {} B under {} B",
                self.attn_layer_idx,
                r.stage_bytes,
                STAGE_ROWS * per_seq_qkv
            );
            let sq = r.stage;
            let sk = sq.offset(STAGE_ROWS * q_bytes);
            let sv = sk.offset(STAGE_ROWS * kv_bytes);
            for b0 in (0..n).step_by(STAGE_ROWS) {
                let rows = (n - b0).min(STAGE_ROWS);
                let x = normed.offset(b0 * h * BF16);
                decode_into(qa, gpu, x, rows, sq, q_proj_dim, h, &r.s, stream)?;
                pair_into(ka, va, gpu, x, rows, sk, sv, kv_dim, h, &r.s, stream)?;
                let dst = qkv_buf.offset(b0 * per_seq_qkv);
                let pitch = per_seq_qkv;
                gpu.copy_d2d_2d_async(sq, q_bytes, dst, pitch, q_bytes, rows, stream)?;
                let dk = dst.offset(q_bytes);
                gpu.copy_d2d_2d_async(sk, kv_bytes, dk, pitch, kv_bytes, rows, stream)?;
                let dv = dk.offset(kv_bytes);
                gpu.copy_d2d_2d_async(sv, kv_bytes, dv, pitch, kv_bytes, rows, stream)?;
            }
        }
        let q_lora = self.lora.as_ref().and_then(|lw| lw.q.as_ref()).is_some();
        if self.gated && !q_lora {
            for i in 0..n {
                ops::deinterleave_qg(
                    gpu,
                    self.deinterleave_qg_k,
                    qkv_buf.offset(i * per_seq_qkv),
                    1,
                    nq,
                    hd,
                    q_proj_dim as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
