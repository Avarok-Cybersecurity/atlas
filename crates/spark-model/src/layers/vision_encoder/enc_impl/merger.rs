// SPDX-License-Identifier: AGPL-3.0-only

//! `apply_merger`: per-patch norm → spatial merge → fc1 → GELU → fc2.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::{MergerLayer, VisionEncoder};

impl VisionEncoder {
    /// Apply one DeepStack/final merger: norm (per-patch) → spatial merge → fc1 → GELU → fc2.
    /// Writes output to `out_slice` (pre-offset into buf_out for this merger index).
    ///
    /// `hidden_src` is the source of per-patch hidden states (the merger
    /// normalizes it in place). For deepstack mergers (called mid-loop
    /// between ViT blocks) callers MUST pass a scratch copy of `buf_h1`
    /// (e.g. `buf_h2`) rather than `buf_h1` directly — otherwise the
    /// in-place LayerNorm corrupts the residual stream that feeds the
    /// next ViT block. The final merger runs after block 27 so it can
    /// safely normalize `buf_h1` itself.
    ///
    /// Note on operation order: the merger's `norm_w/norm_b` tensors are
    /// sized `[hidden_size]` (e.g. 1152), not `[spatial_merge_size² ×
    /// hidden_size]` (4608). HF Qwen3-VL / Qwen3.6 apply this LayerNorm
    /// PER-PATCH *before* spatially concatenating the 2×2 block into a
    /// single 4608-dim token.
    pub(super) fn apply_merger(
        &self,
        m: &MergerLayer,
        p: usize,
        grid_h: usize,
        grid_w: usize,
        hidden_src: DevicePtr,
        out_slice: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let ms = self.spatial_merge_size as u32;
        let hidden = self.hidden_size as u32; // 1152
        let merged_in = ms * ms * self.hidden_size as u32; // 4608
        let out_h_size = self.out_hidden_size as u32; // 2048
        let merged_p = (p / (self.spatial_merge_size * self.spatial_merge_size)) as u32;

        // 1. norm each of the p patches in place on hidden_src (D = hidden_size).
        KernelLaunch::new(gpu, self.k_norm)
            .grid([p as u32, 1, 1])
            .block([hidden.min(1024), 1, 1])
            .arg_ptr(hidden_src)
            .arg_ptr(m.norm_w)
            .arg_ptr(m.norm_b)
            .arg_u32(p as u32)
            .arg_u32(hidden)
            .arg_f32(1e-6)
            .launch(stream)?;
        // 2. spatial merge: hidden_src[p, 1152] → buf_merge_in[merged_p, 4608]
        KernelLaunch::new(gpu, self.k_merge)
            .grid([merged_p, 1, 1])
            .block([merged_in.min(1024), 1, 1])
            .arg_ptr(hidden_src)
            .arg_ptr(self.buf_merge_in)
            .arg_u32(grid_h as u32)
            .arg_u32(grid_w as u32)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(ms)
            .launch(stream)?;
        // fc1 GEMM → buf_merge_fc1
        KernelLaunch::new(gpu, self.k_gemm)
            .grid([div_ceil(merged_in, 32), div_ceil(merged_p, 32), 1])
            .block([32, 32, 1])
            .arg_ptr(self.buf_merge_in)
            .arg_ptr(m.fc1_w)
            .arg_ptr(m.fc1_b)
            .arg_ptr(self.buf_merge_fc1)
            .arg_u32(merged_p)
            .arg_u32(merged_in)
            .arg_u32(merged_in)
            .launch(stream)?;
        // GELU in-place
        KernelLaunch::new(gpu, self.k_gelu)
            .grid([div_ceil(merged_p * merged_in, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_merge_fc1)
            .arg_u32(merged_p * merged_in)
            .launch(stream)?;
        // fc2 GEMM → out_slice
        KernelLaunch::new(gpu, self.k_gemm)
            .grid([div_ceil(out_h_size, 32), div_ceil(merged_p, 32), 1])
            .block([32, 32, 1])
            .arg_ptr(self.buf_merge_fc1)
            .arg_ptr(m.fc2_w)
            .arg_ptr(m.fc2_b)
            .arg_ptr(out_slice)
            .arg_u32(merged_p)
            .arg_u32(out_h_size)
            .arg_u32(merged_in)
            .launch(stream)
    }
}
