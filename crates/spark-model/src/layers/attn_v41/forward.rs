// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! [`AttnV41::forward`]: one layer's attention for `m` tokens — projections,
//! the window ring, the compressed rows and index selection, sparse attention
//! with the sink, and the grouped output projection. Split from
//! `attn_v41.rs` (500-LoC cap).

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::{
    AttnV41, AttnV41LayerState, AttnV41LayerWeights, AttnV41Run, SharedV41, at, upload_i32,
};
use crate::layers::deepseek_v41_ref::attn::window_topk_idxs;
use crate::layers::deepseek_v41_ref::compress::LATENT_BLOCK;

impl AttnV41 {
    /// One layer's attention for `m` tokens at `start_pos`: `x` is the normed
    /// input `[m, dim]` bf16; the output is `[m, dim]` bf16 in `run.out`.
    pub fn forward(
        &self,
        gpu: &dyn GpuBackend,
        w: &AttnV41LayerWeights,
        st: &mut AttnV41LayerState,
        shared: &mut SharedV41,
        x: DevicePtr,
        m: usize,
        start_pos: usize,
        stream: u64,
    ) -> Result<AttnV41Run> {
        let c = &self.cfg;
        ensure!(
            m >= 1 && m <= c.max_tokens,
            "attn_v41: {m} tokens outside 1..={}",
            c.max_tokens
        );
        ensure!(
            start_pos + m <= c.max_seq,
            "attn_v41: position {} beyond max_seq {}",
            start_pos + m,
            c.max_seq
        );
        let (nh, hd, dim) = (c.n_heads, c.head_dim, c.dim);
        let yarn = w.role.ratio > 0;
        let pos: Vec<i32> = (0..m).map(|t| (start_pos + t) as i32).collect();
        let hpos: Vec<i32> = pos
            .iter()
            .flat_map(|&p| std::iter::repeat_n(p, nh))
            .collect();
        upload_i32(gpu, self.pos, &pos)?;
        upload_i32(gpu, self.head_pos, &hpos)?;

        // q: low-rank, normed, up-projected, rotated
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

        // kv: one latent row per token, normed, rotated, fp8
        self.gemm(gpu, x, w.wkv, self.kv_raw, m, hd, dim, stream)?;
        self.rmsnorm(gpu, false, self.kv_raw, w.kv_norm, self.kv, m, hd, stream)?;
        self.rope(gpu, self.kv, self.pos, m, hd, yarn, false, stream)?;
        self.act_quant(gpu, self.kv, m * hd, stream)?;

        // the window ring (stream-ordered copies, no host sync)
        let win = c.window;
        let row = hd * 2;
        let (rows_a, rows_a_len) = if start_pos == 0 {
            if m <= win {
                gpu.copy_d2d_async(self.kv, st.window, m * row, stream)?;
            } else {
                let cutoff = m % win;
                let tail = at(self.kv, (m - win) * row);
                gpu.copy_d2d_async(
                    tail,
                    at(st.window, cutoff * row),
                    (win - cutoff) * row,
                    stream,
                )?;
                gpu.copy_d2d_async(
                    at(tail, (win - cutoff) * row),
                    st.window,
                    cutoff * row,
                    stream,
                )?;
            }
            (self.kv, m)
        } else {
            let slot = start_pos % win;
            gpu.copy_d2d_async(self.kv, at(st.window, slot * row), row, stream)?;
            (st.window, win)
        };
        let (mut idx, mut topk) = window_topk_idxs(win, m, start_pos);

        let (mut rows_b, mut rows_b_len) = (None, 0usize);
        if w.role.ratio > 0 {
            let ratio = w.role.ratio;
            let offset = rows_a_len;
            let compress_len = (start_pos + m) / ratio;
            let latent_groups = if w.role.is_kv_source {
                let g = self.compressor(gpu, w, st, x, m, start_pos, stream)?;
                shared.compress_kv = st.compress_kv;
                g
            } else {
                None
            };
            let (cidx, ctopk) = if !w.role.is_index_source {
                (shared.topk_idxs.clone(), shared.topk)
            } else if compress_len == 0 {
                (Vec::new(), 0)
            } else {
                let r = self.indexer(
                    gpu,
                    w,
                    st,
                    shared,
                    x,
                    latent_groups,
                    m,
                    start_pos,
                    offset,
                    stream,
                )?;
                shared.topk_idxs = r.0.clone();
                shared.topk = r.1;
                r
            };
            if let Some(groups) = latent_groups {
                let cache = st.compress_kv.context("kv source without a latent cache")?;
                upload_i32(
                    gpu,
                    self.grp_pos,
                    &Self::group_positions(groups, ratio, start_pos),
                )?;
                self.rope(
                    gpu,
                    self.latent,
                    self.grp_pos,
                    groups,
                    hd,
                    true,
                    false,
                    stream,
                )?;
                self.fp4_quant(gpu, self.latent, groups * hd, LATENT_BLOCK, true, stream)?;
                gpu.copy_d2d_async(
                    self.latent,
                    at(cache, (start_pos / ratio) * row),
                    groups * row,
                    stream,
                )?;
                shared.compress_kv = Some(cache);
                shared.compress_len = shared.compress_len.max(start_pos / ratio + groups);
            }
            rows_b = Some(
                shared
                    .compress_kv
                    .context("compressed layer before any kv source published")?,
            );
            rows_b_len = compress_len;
            ensure!(
                cidx.len() == m * ctopk,
                "index selection is {} for {m} x {ctopk}",
                cidx.len()
            );
            let mut merged = Vec::with_capacity(m * (topk + ctopk));
            for t in 0..m {
                merged.extend_from_slice(&idx[t * topk..(t + 1) * topk]);
                merged.extend_from_slice(&cidx[t * ctopk..(t + 1) * ctopk]);
            }
            idx = merged;
            topk += ctopk;
        }
        ensure!(
            topk <= win + c.index_topk && topk <= 2048,
            "attn_v41: topk {topk} exceeds the workspace"
        );
        upload_i32(gpu, self.idx_dev, &idx)?;

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
        // no host sync here: everything downstream runs on the same stream,
        // and the routing download in the MoE block drains it before any
        // expert slot can be rewritten
        Ok(AttnV41Run {
            q: self.q,
            rows_a,
            rows_a_len,
            rows_b,
            rows_b_len,
            idx,
            topk,
            o: self.o,
            out: self.out,
        })
    }
}
