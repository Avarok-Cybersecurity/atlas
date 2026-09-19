// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! The decode step split into a host half and a capturable device half:
//! the projections and the attend-and-project body as device-only work, the
//! per-token uploads as the host half, and the checks that say whether a
//! layer's decode can be captured into a CUDA graph at all. Split from
//! `attn_v41.rs` (500-LoC cap) when the line was re-cut onto main.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::{AttnV41, AttnV41LayerState, AttnV41LayerWeights, SharedV41, upload_i32_async};
use crate::layers::deepseek_v41_ref::attn::window_topk_idxs;

impl AttnV41 {
    /// q (low-rank, normed, up-projected, rotated) and kv (one latent row per
    /// token, normed, rotated, fp8) from the normed input `x`. Device work
    /// only: the positions are read from `pos` / `head_pos`.
    pub(super) fn q_kv_projections(
        &self,
        gpu: &dyn GpuBackend,
        w: &AttnV41LayerWeights,
        x: DevicePtr,
        m: usize,
        yarn: bool,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (nh, hd, dim) = (c.n_heads, c.head_dim, c.dim);
        self.gemm(gpu, x, w.wq_a, self.qr_raw, m, c.q_rank, dim, stream)?;
        self.rmsnorm(
            gpu,
            false,
            self.qr_raw,
            w.q_norm,
            self.qr,
            m,
            c.q_rank,
            stream,
        )?;
        self.gemm(gpu, self.qr, w.wq_b, self.q, m, nh * hd, c.q_rank, stream)?;
        self.rope(gpu, self.q, self.head_pos, m * nh, hd, yarn, false, stream)?;
        self.gemm(gpu, x, w.wkv, self.kv_raw, m, hd, dim, stream)?;
        self.rmsnorm(gpu, false, self.kv_raw, w.kv_norm, self.kv, m, hd, stream)?;
        self.rope(gpu, self.kv, self.pos, m, hd, yarn, false, stream)?;
        self.act_quant(gpu, self.kv, m * hd, stream)
    }

    /// Sparse attention over the selection in `idx_dev` (`topk` entries a
    /// token, -1 = absent), the inverse rotation and the grouped output
    /// projection into `self.out`. Device work only.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attend_and_project(
        &self,
        gpu: &dyn GpuBackend,
        w: &AttnV41LayerWeights,
        rows_a: DevicePtr,
        rows_a_len: usize,
        rows_b: Option<DevicePtr>,
        topk: usize,
        m: usize,
        yarn: bool,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (nh, hd, dim) = (c.n_heads, c.head_dim, c.dim);
        // sparse attention with the sink, then the inverse rotation
        let scale = (hd as f32).powf(-0.5);
        KernelLaunch::new(gpu, self.k.sparse_attn)
            .grid([m as u32, nh as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(self.q)
            .arg_ptr(rows_a)
            .arg_ptr(rows_b.unwrap_or(rows_a))
            .arg_u32(rows_a_len as u32)
            .arg_ptr(self.idx_dev)
            .arg_ptr(w.sink)
            .arg_ptr(self.o)
            .arg_u32(nh as u32)
            .arg_u32(hd as u32)
            .arg_u32(topk as u32)
            .arg_f32(scale)
            .launch(stream)?;
        // `run.o` is the pre-rotation output (the reference's `sa_o`); the
        // inverse rotation runs on a copy
        let o_copy = self.o_rot;
        gpu.copy_d2d_async(self.o, o_copy, m * nh * hd * 2, stream)?;
        self.rope(gpu, o_copy, self.head_pos, m * nh, hd, yarn, true, stream)?;

        // grouped low-rank output projection: og[t, g*o_rank + r] = o_g . wo_a[g*o_rank + r]
        let gw = c.gw();
        for g in 0..c.groups {
            KernelLaunch::new(gpu, self.k.slice_cols)
                .grid([m as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(o_copy)
                .arg_ptr(self.slice_in)
                .arg_u32((nh * hd) as u32)
                .arg_u32((g * gw) as u32)
                .arg_u32(gw as u32)
                .launch(stream)?;
            self.gemm(
                gpu,
                self.slice_in,
                w.wo_a.at_rows(g * c.o_rank, gw),
                self.slice_out,
                m,
                c.o_rank,
                gw,
                stream,
            )?;
            KernelLaunch::new(gpu, self.k.scatter_cols)
                .grid([m as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.slice_out)
                .arg_ptr(self.og)
                .arg_u32((c.groups * c.o_rank) as u32)
                .arg_u32((g * c.o_rank) as u32)
                .arg_u32(c.o_rank as u32)
                .launch(stream)?;
        }
        self.gemm(
            gpu,
            self.og,
            w.wo_b,
            self.out,
            m,
            dim,
            c.groups * c.o_rank,
            stream,
        )?;
        Ok(())
    }

    /// Can this layer's single-token step be captured into a CUDA graph?
    /// The kv and index sources compute the position's compressor group and
    /// the index top-k on the host in the middle of the forward (position
    /// parity, a score download); every other layer is kernels and copies
    /// from end to end once its per-token inputs are read from the device.
    pub fn decode_capturable(&self, w: &AttnV41LayerWeights) -> bool {
        !w.role.is_kv_source && !w.role.is_index_source
    }

    /// The `topk` a captured step launches `sparse_attn` with: the widest
    /// selection the layer can ever see, so the launch argument is constant
    /// and the selection is padded with -1 (which the kernel skips: a padded
    /// entry adds a zero to the denominator sum and nothing to the output,
    /// so the result is the eager step's bit for bit).
    pub fn decode_fixed_topk(&self, w: &AttnV41LayerWeights) -> usize {
        let c = &self.cfg;
        if w.role.ratio > 0 {
            (c.window + c.index_topk).min(2048)
        } else {
            c.window
        }
    }

    /// Forget what the captured decode step last uploaded; the next
    /// [`Self::decode_prep`] re-uploads. Call after any eager `forward` (it
    /// writes `pos` / `head_pos` / `idx_dev` for itself) and at a new step.
    pub fn invalidate_decode_uploads(&mut self) {
        self.decode_pos = None;
        self.decode_idx = None;
    }

    /// The HOST half of a capturable layer's single-token step at `start_pos`:
    /// the position into `pos` / `head_pos` and the padded selection (window
    /// slots, then the shared index selection, then -1) into `idx_dev`, each
    /// only when it differs from what is already there. Returns the compressed
    /// rows the captured `sparse_attn` reads (`None` on ratio-0 layers) — a
    /// pointer the graph bakes, for the caller to verify on every replay.
    pub fn decode_prep(
        &mut self,
        w: &AttnV41LayerWeights,
        shared: &SharedV41,
        gpu: &dyn GpuBackend,
        start_pos: usize,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        let c = &self.cfg;
        ensure!(
            self.decode_capturable(w),
            "attn_v41: decode_prep on a kv/index source layer"
        );
        ensure!(
            start_pos >= 1 && start_pos < c.max_seq,
            "attn_v41: position {start_pos} outside the decode range 1..{}",
            c.max_seq
        );
        if self.decode_pos != Some(start_pos) {
            let p = start_pos as i32;
            upload_i32_async(gpu, self.pos, &[p], stream)?;
            upload_i32_async(gpu, self.head_pos, &vec![p; c.n_heads], stream)?;
            self.decode_pos = Some(start_pos);
        }
        let fixed = self.decode_fixed_topk(w);
        let (mut idx, topk) = window_topk_idxs(c.window, 1, start_pos);
        debug_assert_eq!(topk, c.window);
        let rows_b = if w.role.ratio > 0 {
            ensure!(
                shared.topk_idxs.len() == shared.topk,
                "attn_v41: shared index selection is {} for one token x {}",
                shared.topk_idxs.len(),
                shared.topk
            );
            idx.extend_from_slice(&shared.topk_idxs);
            Some(
                shared
                    .compress_kv
                    .context("compressed layer before any kv source published")?,
            )
        } else {
            None
        };
        ensure!(
            idx.len() <= fixed,
            "attn_v41: selection {} exceeds the fixed {fixed}",
            idx.len()
        );
        idx.resize(fixed, -1);
        if self.decode_idx.as_deref() != Some(&idx[..]) {
            upload_i32_async(gpu, self.idx_dev, &idx, stream)?;
            self.decode_idx = Some(idx);
        }
        Ok(rows_b)
    }

    /// The DEVICE half of a capturable layer's single-token step: the
    /// projections, the window ring write at `pos % window` (slot read on
    /// the device), sparse attention over the padded selection and the output
    /// projection into the returned buffer. No host work, no per-token
    /// scalar arguments: capturable into a CUDA graph and replayed for any
    /// position once `decode_prep` has updated `pos` / `head_pos` / `idx_dev`.
    pub fn decode_body(
        &self,
        gpu: &dyn GpuBackend,
        w: &AttnV41LayerWeights,
        st: &AttnV41LayerState,
        x: DevicePtr,
        rows_b: Option<DevicePtr>,
        stream: u64,
    ) -> Result<DevicePtr> {
        let c = &self.cfg;
        ensure!(
            self.decode_capturable(w),
            "attn_v41: decode_body on a kv/index source layer"
        );
        ensure!(
            (w.role.ratio > 0) == rows_b.is_some(),
            "attn_v41: compressed rows {} on a ratio-{} layer",
            if rows_b.is_some() { "given" } else { "missing" },
            w.role.ratio
        );
        let yarn = w.role.ratio > 0;
        self.q_kv_projections(gpu, w, x, 1, yarn, stream)?;
        KernelLaunch::new(gpu, self.k.ring_put)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.kv)
            .arg_ptr(st.window)
            .arg_ptr(self.pos)
            .arg_u32(c.window as u32)
            .arg_u32(c.head_dim as u32)
            .launch(stream)?;
        self.attend_and_project(
            gpu,
            w,
            st.window,
            c.window,
            rows_b,
            self.decode_fixed_topk(w),
            1,
            yarn,
            stream,
        )?;
        Ok(self.out)
    }
}
