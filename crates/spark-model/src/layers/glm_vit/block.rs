// SPDX-License-Identifier: AGPL-3.0-only

//! One GLM ViT block, plus the GEMM/norm/attention primitives it is built from.
//!
//! Block shape (`GlmOcrVisionBlock`, inherited unchanged by `Glm5NextVisionBlock`
//! except for the MLP swap):
//!   `h += attn(rmsnorm(h))` then `h += swiglu_mlp(rmsnorm(h))`
//! — plain pre-norm with unscaled residual adds.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::{GlmVit, GlmVitBlock};

/// Shared-memory floats the block-wide reductions need, beyond any per-kernel
/// payload: one slot per warp of a 1024-thread block.
const RED_SLOTS: u32 = 32;

impl GlmVit {
    /// `C[M,N] = A[M,K] · B[N,K]^T (+ bias[N])` on tensor cores.
    ///
    /// `dense_gemm_bf16_pipelined` has no bias epilogue, so the bias is a
    /// second launch on the same stream. `bias = None` is the merger's case —
    /// every one of its Linear layers is `bias=False`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gemm(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        b: DevicePtr,
        bias: Option<DevicePtr>,
        c: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_gemm)
            .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(b)
            .arg_ptr(c)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream)?;
        let Some(bias) = bias else { return Ok(()) };
        KernelLaunch::new(gpu, self.k_add_bias)
            .grid([div_ceil(m * n, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(c)
            .arg_ptr(bias)
            .arg_u32(m)
            .arg_u32(n)
            .launch(stream)
    }

    /// Weight-only RMSNorm, in place over `rows × dim`.
    pub(super) fn rmsnorm(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        w: DevicePtr,
        rows: u32,
        dim: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_rmsnorm)
            .grid([rows, 1, 1])
            .block([dim.min(1024), 1, 1])
            .shared_mem(RED_SLOTS * 4)
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_u32(rows)
            .arg_u32(dim)
            .arg_f32(self.rms_norm_eps)
            .launch(stream)
    }

    /// Mean-subtracting LayerNorm with bias, in place. Used once, for the
    /// merger's `post_projection_norm`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn layernorm(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        w: DevicePtr,
        b: DevicePtr,
        rows: u32,
        dim: u32,
        eps: f32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_layernorm)
            .grid([rows, 1, 1])
            .block([dim.min(1024), 1, 1])
            .shared_mem(RED_SLOTS * 4)
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(b)
            .arg_u32(rows)
            .arg_u32(dim)
            .arg_f32(eps)
            .launch(stream)
    }

    /// `out = silu(min(gate, limit)) * clamp(up, ±limit)`, elementwise.
    pub(super) fn swiglu(
        &self,
        gpu: &dyn GpuBackend,
        gate: DevicePtr,
        up: DevicePtr,
        out: DevicePtr,
        n: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_swiglu)
            .grid([div_ceil(n, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(gate)
            .arg_ptr(up)
            .arg_ptr(out)
            .arg_u32(n)
            .arg_f32(self.swiglu_limit)
            .launch(stream)
    }

    pub(super) fn copy(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        dst: DevicePtr,
        n: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src)
            .arg_ptr(dst)
            .arg_u32(n)
            .launch(stream)
    }

    pub(super) fn add_inplace(
        &self,
        gpu: &dyn GpuBackend,
        dst: DevicePtr,
        src: DevicePtr,
        n: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(dst)
            .arg_ptr(src)
            .arg_u32(n)
            .launch(stream)
    }

    /// Bidirectional SDPA over ONE image's `[seq, 3*H*D]` QKV slice, with the
    /// per-head QK-RMSNorm and the axial RoPE folded into the deinterleave.
    ///
    /// The attention is full within an image and never across images: GLM's
    /// `cu_seqlens` puts one segment per (image, frame), so the caller loops
    /// over disjoint row ranges rather than passing a mask.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        gpu: &dyn GpuBackend,
        blk: &GlmVitBlock,
        qkv: DevicePtr,
        o: DevicePtr,
        cos: DevicePtr,
        sin: DevicePtr,
        seq: u32,
        stream: u64,
    ) -> Result<()> {
        let d = self.head_dim as u32;
        let h_n = self.num_heads as u32;
        let hd = self.hidden_size as u32;

        // (1) QK-norm + rope + head-contiguous Qr/Kr + transposed V, all heads.
        KernelLaunch::new(gpu, self.k_qknorm_rope)
            .grid([seq, h_n, 1])
            .block([d, 1, 1])
            .shared_mem((2 * d + RED_SLOTS) * 4)
            .arg_ptr(qkv)
            .arg_ptr(blk.q_norm_w)
            .arg_ptr(blk.k_norm_w)
            .arg_ptr(self.scratch().buf_qr)
            .arg_ptr(self.scratch().buf_kr)
            .arg_ptr(self.scratch().buf_vt)
            .arg_ptr(cos)
            .arg_ptr(sin)
            .arg_u32(seq)
            .arg_u32(h_n)
            .arg_u32(d)
            .arg_f32(self.rms_norm_eps)
            .launch(stream)?;

        let qk_head = (seq * d) as usize;
        for head in 0..self.num_heads {
            let qr = self.scratch().buf_qr.offset(head * qk_head * 2);
            let kr = self.scratch().buf_kr.offset(head * qk_head * 2);
            let vt = self.scratch().buf_vt.offset(head * qk_head * 2);
            let o_h = o.offset(head * self.head_dim * 2);

            // (2) raw S[seq,seq] = Qr · Kr^T, f32 out (TILE 16).
            KernelLaunch::new(gpu, self.k_gemm_f32)
                .grid([div_ceil(seq, 16), div_ceil(seq, 16), 1])
                .block([16, 16, 1])
                .arg_ptr(qr)
                .arg_ptr(kr)
                .arg_ptr(self.scratch().buf_scores)
                .arg_u32(seq)
                .arg_u32(seq)
                .arg_u32(d)
                .launch(stream)?;
            // (3) row softmax with scale = rsqrt(head_dim) folded in.
            KernelLaunch::new(gpu, self.k_softmax)
                .grid([seq, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.scratch().buf_scores)
                .arg_ptr(self.scratch().buf_probs)
                .arg_u32(seq)
                .arg_u32(d)
                .launch(stream)?;
            // (4) O_stage[seq,D] = P[seq,seq] · Vt[D,seq]^T.
            KernelLaunch::new(gpu, self.k_gemm)
                .grid([div_ceil(d, 128), div_ceil(seq, 128), 1])
                .block([256, 1, 1])
                .arg_ptr(self.scratch().buf_probs)
                .arg_ptr(vt)
                .arg_ptr(self.scratch().buf_o_stage)
                .arg_u32(seq)
                .arg_u32(d)
                .arg_u32(seq)
                .launch(stream)?;
            // (5) scatter into this head's slot of the interleaved output.
            KernelLaunch::new(gpu, self.k_scatter_head)
                .grid([div_ceil(seq * d, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.scratch().buf_o_stage)
                .arg_ptr(o_h)
                .arg_u32(seq)
                .arg_u32(d)
                .arg_u32(hd)
                .launch(stream)?;
        }
        Ok(())
    }

    /// One block over `p_total` packed rows: the M-agnostic GEMMs and norms run
    /// once over the whole batch, attention loops per image.
    pub(super) fn block(
        &self,
        blk: &GlmVitBlock,
        p_total: usize,
        p_i: &[usize],
        p_off: &[usize],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size as u32;
        let pt = p_total as u32;
        let qkv_n = 3 * h;
        let inter = self.intermediate_size as u32;
        let n_h = (p_total * self.hidden_size) as u32;
        let s = self.scratch();

        // ── attention sub-block ──
        self.copy(gpu, s.buf_h1, s.buf_h2, n_h, stream)?; // residual
        self.rmsnorm(gpu, s.buf_h1, blk.norm1_w, pt, h, stream)?;
        self.gemm(
            gpu,
            s.buf_h1,
            blk.qkv_w,
            Some(blk.qkv_b),
            s.buf_gate,
            pt,
            qkv_n,
            h,
            stream,
        )?;
        for (i, &p) in p_i.iter().enumerate() {
            let qkv = s.buf_gate.offset(p_off[i] * qkv_n as usize * 2);
            let o = s.buf_h1.offset(p_off[i] * self.hidden_size * 2);
            let cos = s.buf_rope_cos.offset(p_off[i] * self.head_dim * 2);
            let sin = s.buf_rope_sin.offset(p_off[i] * self.head_dim * 2);
            self.attention(gpu, blk, qkv, o, cos, sin, p as u32, stream)?;
        }
        self.gemm(
            gpu,
            s.buf_h1,
            blk.proj_w,
            Some(blk.proj_b),
            s.buf_gate,
            pt,
            h,
            h,
            stream,
        )?;
        self.add_inplace(gpu, s.buf_gate, s.buf_h2, n_h, stream)?;
        self.copy(gpu, s.buf_gate, s.buf_h1, n_h, stream)?;

        // ── SwiGLU MLP sub-block ──
        self.copy(gpu, s.buf_h1, s.buf_h2, n_h, stream)?; // residual
        self.rmsnorm(gpu, s.buf_h1, blk.norm2_w, pt, h, stream)?;
        self.gemm(
            gpu,
            s.buf_h1,
            blk.gate_w,
            Some(blk.gate_b),
            s.buf_gate,
            pt,
            inter,
            h,
            stream,
        )?;
        self.gemm(
            gpu,
            s.buf_h1,
            blk.up_w,
            Some(blk.up_b),
            s.buf_up,
            pt,
            inter,
            h,
            stream,
        )?;
        self.swiglu(gpu, s.buf_gate, s.buf_up, s.buf_gate, pt * inter, stream)?;
        self.gemm(
            gpu,
            s.buf_gate,
            blk.down_w,
            Some(blk.down_b),
            s.buf_h1,
            pt,
            h,
            inter,
            stream,
        )?;
        self.add_inplace(gpu, s.buf_h1, s.buf_h2, n_h, stream)
    }
}
