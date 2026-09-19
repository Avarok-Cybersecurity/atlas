// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! What the source layers publish for the layers below them: the kv-source
//! compressor (latent pooling) and the index-source indexer (index keys,
//! scores, candidate blocks and the CPU top-k). Split from `attn_v41.rs`
//! (500-LoC cap).

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::{AttnMat, AttnV41, AttnV41LayerState, AttnV41LayerWeights, SharedV41, at, upload_i32};
use crate::layers::deepseek_v41_ref::compress::{
    FP4_BLOCK, select_candidate_blocks, torch_cpu_topk_set,
};

impl AttnV41 {
    /// The compressor's latent for this call (pre-RoPE, bf16 in `self.latent`),
    /// or `None` while a ratio-2 group is still filling. Returns the group count.
    pub(super) fn compressor(
        &self,
        gpu: &dyn GpuBackend,
        w: &AttnV41LayerWeights,
        st: &AttnV41LayerState,
        x: DevicePtr,
        m: usize,
        start_pos: usize,
        stream: u64,
    ) -> Result<Option<usize>> {
        let c = &self.cfg;
        let comp = w
            .comp
            .as_ref()
            .context("kv source without compressor weights")?;
        let hd = c.head_dim;
        let ratio = w.role.ratio;
        if ratio == 1 {
            self.gemm(
                gpu,
                x,
                AttnMat::Bf16(comp.kv),
                self.latent_raw,
                m,
                hd,
                c.dim,
                stream,
            )?;
            self.rmsnorm(
                gpu,
                false,
                self.latent_raw,
                comp.norm,
                self.latent,
                m,
                hd,
                stream,
            )?;
            return Ok(Some(m));
        }
        let gate = comp.gate.context("ratio > 1 compressor without wgate")?;
        let (kv_state, score_state) = st.comp_state.context("compressor state")?;
        let gemm_f32 = |wt: DevicePtr, out: DevicePtr| {
            KernelLaunch::new(gpu, self.k.gemm_f32)
                .grid([(hd as u32).div_ceil(16), (m as u32).div_ceil(16), 1])
                .block([16, 16, 1])
                .arg_ptr(x)
                .arg_ptr(wt)
                .arg_ptr(out)
                .arg_u32(m as u32)
                .arg_u32(hd as u32)
                .arg_u32(c.dim as u32)
                .launch(stream)
        };
        gemm_f32(comp.kv, self.ckv)?;
        gemm_f32(gate, self.cscore)?;
        let row = hd * 4;
        let (src_kv, src_score, groups) = if start_pos == 0 {
            let remainder = m % ratio;
            let cutoff = m - remainder;
            if remainder > 0 {
                gpu.copy_d2d_async(
                    at(self.ckv, cutoff * row),
                    kv_state,
                    remainder * row,
                    stream,
                )?;
                gpu.copy_d2d_async(
                    at(self.cscore, cutoff * row),
                    score_state,
                    remainder * row,
                    stream,
                )?;
            }
            if m < ratio {
                return Ok(None);
            }
            (self.ckv, self.cscore, cutoff / ratio)
        } else {
            let slot = start_pos % ratio;
            gpu.copy_d2d_async(self.ckv, at(kv_state, slot * row), row, stream)?;
            gpu.copy_d2d_async(self.cscore, at(score_state, slot * row), row, stream)?;
            if !(start_pos + 1).is_multiple_of(ratio) {
                return Ok(None);
            }
            (kv_state, score_state, 1)
        };
        KernelLaunch::new(gpu, self.k.pool)
            .grid([groups as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src_kv)
            .arg_ptr(src_score)
            .arg_ptr(self.pooled)
            .arg_u32(ratio as u32)
            .arg_u32(hd as u32)
            .launch(stream)?;
        self.rmsnorm(
            gpu,
            true,
            self.pooled,
            comp.norm,
            self.latent,
            groups,
            hd,
            stream,
        )?;
        Ok(Some(groups))
    }

    /// Group positions of `groups` new latents: `g * ratio` on prefill, the
    /// group's first position on decode.
    pub(super) fn group_positions(groups: usize, ratio: usize, start_pos: usize) -> Vec<i32> {
        if start_pos == 0 {
            (0..groups).map(|g| (g * ratio) as i32).collect()
        } else {
            vec![(start_pos + 1 - ratio) as i32]
        }
    }

    /// The indexer: publishes index keys (kv sources), scores the compressed
    /// positions, selects candidates and the top-k on the CPU. Returns
    /// `([tokens][topk], topk)` offset by `offset`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn indexer(
        &self,
        gpu: &dyn GpuBackend,
        w: &AttnV41LayerWeights,
        st: &AttnV41LayerState,
        shared: &mut SharedV41,
        x: DevicePtr,
        latent_groups: Option<usize>,
        m: usize,
        start_pos: usize,
        offset: usize,
        stream: u64,
    ) -> Result<(Vec<i32>, usize)> {
        let c = &self.cfg;
        let iw = w
            .idx
            .as_ref()
            .context("index source without indexer weights")?;
        let ratio = w.role.ratio;
        let (nhi, ihd, hd) = (c.index_heads, c.index_hd, c.head_dim);
        let end_pos = start_pos + m;
        if let (Some(groups), Some(cache)) = (latent_groups, st.index_k) {
            let wk = iw.wk.context("kv source without index wk")?;
            let k_norm = iw.k_norm.context("kv source without index k_norm")?;
            self.gemm(
                gpu,
                self.latent,
                AttnMat::Bf16(wk),
                self.ik_raw,
                groups,
                ihd,
                hd,
                stream,
            )?;
            self.rmsnorm(
                gpu,
                false,
                self.ik_raw,
                k_norm,
                self.ik,
                groups,
                ihd,
                stream,
            )?;
            upload_i32(
                gpu,
                self.grp_pos,
                &Self::group_positions(groups, ratio, start_pos),
            )?;
            self.rope(gpu, self.ik, self.grp_pos, groups, ihd, true, false, stream)?;
            self.fp4_quant(gpu, self.ik, groups * ihd, FP4_BLOCK, false, stream)?;
            gpu.copy_d2d_async(
                self.ik,
                at(cache, (start_pos / ratio) * ihd * 2),
                groups * ihd * 2,
                stream,
            )?;
            shared.index_k = Some(cache);
        }
        let index_k = shared
            .index_k
            .context("indexer before any index keys were published")?;
        // queries
        self.gemm(
            gpu,
            self.qr,
            AttnMat::Bf16(iw.wq_b),
            self.iq,
            m,
            nhi * ihd,
            c.q_rank,
            stream,
        )?;
        let ihpos: Vec<i32> = (0..m)
            .flat_map(|t| std::iter::repeat_n((start_pos + t) as i32, nhi))
            .collect();
        upload_i32(gpu, self.idx_pos, &ihpos)?;
        self.rope(
            gpu,
            self.iq,
            self.idx_pos,
            m * nhi,
            ihd,
            true,
            false,
            stream,
        )?;
        self.fp4_quant(gpu, self.iq, m * nhi * ihd, FP4_BLOCK, false, stream)?;
        // head weights: bf16(bf16(x . wproj) * ihd^-0.5 * nh^-0.5)
        self.gemm(
            gpu,
            x,
            AttnMat::Bf16(iw.weights_proj),
            self.iw_raw,
            m,
            nhi,
            c.dim,
            stream,
        )?;
        let wscale = (ihd as f32).powf(-0.5) * (nhi as f32).powf(-0.5);
        KernelLaunch::new(gpu, self.k.scale_bf16)
            .grid([((m * nhi) as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.iw_raw)
            .arg_ptr(self.iw)
            .arg_u32((m * nhi) as u32)
            .arg_f32(wscale)
            .launch(stream)?;
        let width = end_pos / ratio;
        ensure!(
            width * m <= c.max_seq * c.max_tokens,
            "index width {width} x {m} exceeds the workspace"
        );
        KernelLaunch::new(gpu, self.k.index_score)
            .grid([(width as u32).div_ceil(256), m as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(self.iq)
            .arg_ptr(index_k)
            .arg_ptr(self.iw)
            .arg_ptr(self.score)
            .arg_u32(width as u32)
            .arg_u32(nhi as u32)
            .arg_u32(ihd as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut bytes = vec![0u8; m * width * 4];
        gpu.copy_d2h(self.score, &mut bytes)?;
        let mut score: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let compress_lens: Vec<usize> = if start_pos == 0 {
            (0..m).map(|t| (t + 1) / ratio).collect()
        } else {
            vec![width; m]
        };
        if start_pos == 0 {
            for t in 0..m {
                for p in compress_lens[t]..width {
                    score[t * width + p] = f32::NEG_INFINITY;
                }
            }
        }
        if w.role.is_candidate_source {
            shared.candidates = select_candidate_blocks(
                &score,
                m,
                width,
                &compress_lens,
                c.cand_topk_blocks,
                c.cand_block,
            );
            shared.cand_width = width;
        } else if w.role.uses_candidates {
            ensure!(
                shared.candidates.len() == m * width,
                "candidate mask is {} for {} x {width}",
                shared.candidates.len(),
                m
            );
            for (s, &keep) in score.iter_mut().zip(&shared.candidates) {
                if !keep {
                    *s = f32::NEG_INFINITY;
                }
            }
        }
        let topk = c.index_topk.min(end_pos / ratio);
        let mut out = Vec::with_capacity(m * topk);
        for t in 0..m {
            let row = &score[t * width..(t + 1) * width];
            let mut picked = torch_cpu_topk_set(row, topk);
            picked.sort_unstable();
            for i in picked {
                out.push(if i < compress_lens[t] {
                    (i + offset) as i32
                } else {
                    -1
                });
            }
        }
        Ok((out, topk))
    }
}
