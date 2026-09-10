// SPDX-License-Identifier: AGPL-3.0-only

//! Everything after the last block, for one image:
//! `post_layernorm → 2x2 spatial merge → conv2d downsample → proj → LayerNorm →
//! exact-erf GELU → clamped SwiGLU (gate/up) → down`.
//!
//! The conv2d `downsample` IS the spatial merge's projection: `vision_spatial_merge`
//! only gathers the 2×2 block into one 4096-wide row, and the downsample turns
//! that row into the 4096-wide output token. `out_hidden_size == 4096 == the
//! text hidden size`, so merger rows drop straight onto the hidden state.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::{GLM_MERGER_LN_EPS, Glm5NextVisionEncoder};

impl Glm5NextVisionEncoder {
    /// Merge and project one image's `p` patch rows at `hidden_src` into
    /// `merged_p` rows at `out_slice`.
    ///
    /// Buffer routing, and every step of it is an aliasing decision:
    /// `buf_merge_in` holds the gathered rows, `buf_wide` takes the downsample
    /// output, `buf_merge_in` is then REUSED for the projection (its gathered
    /// rows are dead by then), and `buf_wide` is reused again for the SwiGLU
    /// output because `buf_merge_fc1` is the SwiGLU's source. Nothing here is
    /// written while it is still being read.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_merger(
        &self,
        p: usize,
        grid_h: usize,
        grid_w: usize,
        hidden_src: DevicePtr,
        out_slice: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let m = &self.merger;
        let s = self.scratch();
        let ms = self.spatial_merge_size as u32;
        let hidden = self.hidden_size as u32;
        let merged_in = ms * ms * hidden; // 4096
        let out_h = self.out_hidden_size as u32; // 4096
        let interm = self.proj_intermediate_size as u32; // 10240
        let sms2 = (self.spatial_merge_size * self.spatial_merge_size).max(1);
        let mp = (p / sms2) as u32;

        // 1. post_layernorm (RMS, weight only) over the raw patch rows.
        self.rms_norm(
            gpu,
            hidden_src,
            m.post_layernorm_w,
            p as u32,
            hidden,
            stream,
        )?;

        // 2. 2×2 gather. The kernel reads RASTER rows and emits channel-concat
        //    `j = (mh*2 + mw)*hidden + c`, which is the order the downsample
        //    weight was permuted into at load.
        KernelLaunch::new(gpu, self.k_merge)
            .grid([mp, 1, 1])
            .block([merged_in.min(1024), 1, 1])
            .arg_ptr(hidden_src)
            .arg_ptr(s.buf_merge_in)
            .arg_u32(grid_h as u32)
            .arg_u32(grid_w as u32)
            .arg_u32(hidden)
            .arg_u32(ms)
            .launch(stream)?;

        // 3. downsample (HAS a bias) → buf_wide.
        self.glm_gemm_bias(
            gpu,
            s.buf_merge_in,
            m.downsample_w,
            m.downsample_b,
            s.buf_wide,
            mp,
            out_h,
            merged_in,
            stream,
        )?;
        // 4. merger.proj (NO bias) → back into buf_merge_in.
        self.glm_gemm_bias(
            gpu,
            s.buf_wide,
            m.proj_w,
            s.buf_zero_bias,
            s.buf_merge_in,
            mp,
            out_h,
            out_h,
            stream,
        )?;
        // 5. post_projection_norm: a torch LayerNorm with weight AND bias, so
        //    it uses the shared `vision_layer_norm`, the one place this tower
        //    still calls a Qwen-shaped norm. eps forks to 1e-6 — see A2.
        KernelLaunch::new(gpu, self.k_layer_norm)
            .grid([mp, 1, 1])
            .block([out_h.min(1024), 1, 1])
            .arg_ptr(s.buf_merge_in)
            .arg_ptr(m.ln_w)
            .arg_ptr(m.ln_b)
            .arg_u32(mp)
            .arg_u32(out_h)
            .arg_f32(GLM_MERGER_LN_EPS)
            .launch(stream)?;
        // 6. EXACT erf GELU. NOT `vision_gelu`, which is the tanh approximation
        //    Qwen's tower wants by design; GLM declares gelu_approx="none"
        //    (glm4v.py:111) and reusing the tanh one is a silent ~1e-4/value drift.
        let ln_elems = mp * out_h;
        KernelLaunch::new(gpu, self.k_gelu_exact)
            .grid([div_ceil(ln_elems, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.buf_merge_in)
            .arg_u32(ln_elems)
            .launch(stream)?;

        // 7. gate/up (NO biases) as two blocks, then the clamped SwiGLU over
        //    them; see `swiglu_clamp_block` for why the block layout works.
        let gate_elems = (mp * interm) as usize;
        self.glm_gemm_bias(
            gpu,
            s.buf_merge_in,
            m.gate_w,
            s.buf_zero_bias,
            s.buf_merge_fc1,
            mp,
            interm,
            out_h,
            stream,
        )?;
        self.glm_gemm_bias(
            gpu,
            s.buf_merge_in,
            m.up_w,
            s.buf_zero_bias,
            s.buf_merge_fc1.offset(gate_elems * 2),
            mp,
            interm,
            out_h,
            stream,
        )?;
        self.swiglu_clamp_block(gpu, s.buf_merge_fc1, s.buf_wide, gate_elems, stream)?;

        // 8. down (NO bias) → the caller's packed output slice.
        self.glm_gemm_bias(
            gpu,
            s.buf_wide,
            m.down_w,
            s.buf_zero_bias,
            out_slice,
            mp,
            out_h,
            interm,
            stream,
        )
    }
}
