// SPDX-License-Identifier: AGPL-3.0-only

//! One GLM ViT block, batched over Σpatches.
//!
//! `RMSNorm → fused QKV(+bias) → per-head q/k RMSNorm → full non-causal
//! attention → proj(+bias) → +residual → RMSNorm → clamped SwiGLU MLP(+biases)
//! → +residual`.
//!
//! Only the attention loops per image; every other op is M-agnostic and runs
//! once over `p_total`, which is the whole point of the batched form.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::{GLM_VISION_RMS_EPS, Glm5NextVisionEncoder, GlmViTBlock};

impl Glm5NextVisionEncoder {
    /// In-place RMSNorm over `rows` rows of `d` elements.
    pub(super) fn rms_norm(
        &self,
        gpu: &dyn GpuBackend,
        x: spark_runtime::gpu::DevicePtr,
        w: spark_runtime::gpu::DevicePtr,
        rows: u32,
        d: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_rms_norm)
            .grid([rows, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_u32(rows)
            .arg_u32(d)
            .arg_f32(GLM_VISION_RMS_EPS)
            .launch(stream)
    }

    /// BF16 element copy, `n` elements.
    pub(super) fn copy_bf16(
        &self,
        gpu: &dyn GpuBackend,
        src: spark_runtime::gpu::DevicePtr,
        dst: spark_runtime::gpu::DevicePtr,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src)
            .arg_ptr(dst)
            .arg_u32(n as u32)
            .launch(stream)
    }

    /// Clamped SwiGLU over a `[gate | up]` pair that is CONTIGUOUS BY BLOCK
    /// rather than interleaved per row.
    ///
    /// GLM ships `gate_proj` and `up_proj` as two separate tensors, so the two
    /// GEMMs land as `[rows, n]` blocks back to back, not as one `[rows, 2n]`
    /// row-interleaved buffer. `vision_swiglu_clamp` reads
    /// `SRC[r*2N + n]` / `SRC[r*2N + N + n]`, so passing `rows = 1` and
    /// `N = rows*n` makes those exactly `gate[i]` and `up[i]` over the whole
    /// block. The op is elementwise, so treating the block as one long row is
    /// not an approximation — it is the same arithmetic in the same order.
    ///
    /// The alternative was concatenating the weights into a fused
    /// `[2*inter, hidden]` at load, which would duplicate 402 MB of the block
    /// MLP for no numerical difference.
    pub(super) fn swiglu_clamp_block(
        &self,
        gpu: &dyn GpuBackend,
        src: spark_runtime::gpu::DevicePtr,
        dst: spark_runtime::gpu::DevicePtr,
        elems: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_swiglu_clamp)
            .grid([div_ceil(elems as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src)
            .arg_ptr(dst)
            .arg_u32(1)
            .arg_u32(elems as u32)
            .arg_f32(self.swiglu_limit)
            .launch(stream)
    }

    /// Run one block over `p_total` packed rows; `p_i`/`p_off` locate each
    /// image's disjoint slice for the attention step.
    pub(super) fn glm_block_batched(
        &self,
        blk: &GlmViTBlock,
        p_total: usize,
        p_i: &[usize],
        p_off: &[usize],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size as u32;
        let pt = p_total as u32;
        let inter = self.intermediate_size as u32;
        let qkv_n = (3 * self.num_heads * self.head_dim) as u32;
        let n_h = p_total * self.hidden_size;
        let s = self.scratch();

        // ── attention ──
        self.copy_bf16(gpu, s.buf_h1, s.buf_h2, n_h, stream)?; // save residual
        self.rms_norm(gpu, s.buf_h1, blk.norm1_w, pt, h, stream)?;
        self.glm_gemm_bias(
            gpu, s.buf_h1, blk.qkv_w, blk.qkv_b, s.buf_wide, pt, qkv_n, h, stream,
        )?;

        // Per-head q/k RMSNorm, BETWEEN the QKV GEMM and RoPE. Verified twice
        // upstream: `attn.py:743-760`'s ordering and `rope.cu:245-249`'s
        // `load_head → apply_norm → apply_rope`. ONE `[64]` weight vector is
        // broadcast across all 16 heads, and V is never normed.
        let sec = (self.num_heads * self.head_dim) as u32;
        for (w, sec_off) in [(blk.q_norm_w, 0u32), (blk.k_norm_w, sec)] {
            KernelLaunch::new(gpu, self.k_rms_norm_heads)
                .grid([pt, self.num_heads as u32, 1])
                .block([self.head_dim as u32, 1, 1])
                .arg_ptr(s.buf_wide)
                .arg_ptr(w)
                .arg_u32(pt)
                .arg_u32(self.num_heads as u32)
                .arg_u32(self.head_dim as u32)
                .arg_u32(sec_off)
                .arg_u32(qkv_n)
                .arg_f32(GLM_VISION_RMS_EPS)
                .launch(stream)?;
        }

        // Attention per image over its disjoint slice: buf_wide (QKV) is
        // read-only here and each image writes a disjoint buf_h1 row range.
        for (i, &p) in p_i.iter().enumerate() {
            let qkv = s.buf_wide.offset(p_off[i] * qkv_n as usize * 2);
            let o = s.buf_h1.offset(p_off[i] * self.hidden_size * 2);
            let cos = s.buf_rope_cos.offset(p_off[i] * self.head_dim * 2);
            let sin = s.buf_rope_sin.offset(p_off[i] * self.head_dim * 2);
            self.glm_attention_gemm(gpu, qkv, o, cos, sin, p as u32, stream)?;
        }

        // proj → buf_wide (QKV already consumed), residual add, back to buf_h1.
        self.glm_gemm_bias(
            gpu, s.buf_h1, blk.proj_w, blk.proj_b, s.buf_wide, pt, h, h, stream,
        )?;
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.buf_wide)
            .arg_ptr(s.buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)?;
        self.copy_bf16(gpu, s.buf_wide, s.buf_h1, n_h, stream)?;

        // ── MLP ──
        self.copy_bf16(gpu, s.buf_h1, s.buf_h2, n_h, stream)?; // save residual
        self.rms_norm(gpu, s.buf_h1, blk.norm2_w, pt, h, stream)?;

        // buf_wide layout for this half: [ gate(pt*inter) | up(pt*inter) |
        // swiglu(pt*inter) ]. The third block exists because the SwiGLU kernel's
        // threads are unordered — writing its output over its own source would
        // corrupt rows that have not been read yet (C-T4).
        let blk_elems = p_total * self.intermediate_size;
        let up_off = blk_elems * 2; // bytes
        let out_off = 2 * blk_elems * 2;
        self.glm_gemm_bias(
            gpu, s.buf_h1, blk.gate_w, blk.gate_b, s.buf_wide, pt, inter, h, stream,
        )?;
        self.glm_gemm_bias(
            gpu,
            s.buf_h1,
            blk.up_w,
            blk.up_b,
            s.buf_wide.offset(up_off),
            pt,
            inter,
            h,
            stream,
        )?;
        self.swiglu_clamp_block(
            gpu,
            s.buf_wide,
            s.buf_wide.offset(out_off),
            blk_elems,
            stream,
        )?;
        self.glm_gemm_bias(
            gpu,
            s.buf_wide.offset(out_off),
            blk.down_w,
            blk.down_b,
            s.buf_h1,
            pt,
            h,
            inter,
            stream,
        )?;
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.buf_h1)
            .arg_ptr(s.buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)
    }
}
