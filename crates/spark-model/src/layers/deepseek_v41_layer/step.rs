// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! `DeepSeekV41Layer::step`, one block for `m` tokens on the mHC highway, with
//! its hc-mix, engram and diagnostics helpers.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::{DeepSeekV41Layer, GraphMode, V41LayerState, graph_mode};
use crate::layer::{ForwardContext, LayerState};
use crate::layers::attn_v41::SharedV41;
use crate::layers::engram_v41::ENGRAM_ROW_BYTES;
use crate::layers::ops;
use crate::layers::qwen3_attention::HcSiteWeights;

pub(super) fn diag_on() -> bool {
    std::env::var("ATLAS_DS41_DIAG").is_ok_and(|v| v == "1")
}

/// RMS of the first `n` f32 values at `p` (diagnostics only, synchronises).
fn diag_rms_f32(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> f32 {
    let mut b = vec![0u8; n * 4];
    if gpu.copy_d2h(p, &mut b).is_err() {
        return f32::NAN;
    }
    let v: Vec<f32> = b
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (v.iter().map(|x| x * x).sum::<f32>() / n.max(1) as f32).sqrt()
}

pub(super) fn diag_rms_bf16(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> f32 {
    let mut b = vec![0u8; n * 2];
    if gpu.copy_d2h(p, &mut b).is_err() {
        return f32::NAN;
    }
    let v: Vec<f32> = b
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect();
    (v.iter().map(|x| x * x).sum::<f32>() / n.max(1) as f32).sqrt()
}

impl DeepSeekV41Layer {
    pub(super) fn mixes(
        &self,
        gpu: &dyn GpuBackend,
        site: &HcSiteWeights,
        streams: DevicePtr,
        pre: DevicePtr,
        post: DevicePtr,
        comb: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let mix_hc = (2 + rt.hc_mult) * rt.hc_mult;
        // one block per (token, mix) for the 24 dot products over hc * H, then
        // the tiny epilogue; bit-identical to the one-block hc_v41_mixes
        KernelLaunch::new(gpu, self.k_mixes_dot)
            .grid([m as u32, mix_hc as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(streams)
            .arg_ptr(site.hc_fn)
            .arg_ptr(rt.mixes_s)
            .arg_u32(rt.hidden as u32)
            .arg_u32(rt.hc_mult as u32)
            .arg_f32(rt.norm_eps)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.k_mixes_finish)
            .grid([m as u32, 1, 1])
            .block([32, 1, 1])
            .arg_ptr(rt.mixes_s)
            .arg_ptr(site.hc_scale)
            .arg_ptr(site.hc_base)
            .arg_ptr(pre)
            .arg_ptr(post)
            .arg_ptr(comb)
            .arg_u32(rt.hc_mult as u32)
            .arg_u32(rt.sinkhorn_iters as u32)
            .arg_f32(rt.hc_eps)
            .launch(stream)
    }

    pub(super) fn collapse(
        &self,
        gpu: &dyn GpuBackend,
        streams: DevicePtr,
        pre: DevicePtr,
        y: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_collapse)
            .grid([m as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(streams)
            .arg_ptr(pre)
            .arg_ptr(y)
            .arg_u32(self.rt.hidden as u32)
            .arg_u32(self.rt.hc_mult as u32)
            .launch(stream)
    }

    pub(super) fn hc_post(
        &self,
        gpu: &dyn GpuBackend,
        block_out: DevicePtr,
        streams: DevicePtr,
        post: DevicePtr,
        comb: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        ops::hc_post(
            gpu,
            self.k_hc_post,
            block_out,
            streams,
            post,
            comb,
            streams,
            m as u32,
            self.rt.hidden as u32,
            self.rt.hc_mult as u32,
            stream,
        )
    }

    /// The token ids of this step, for the engram hash.
    pub(super) fn step_token_ids(&self, ctx: &ForwardContext, m: usize) -> Result<Vec<u32>> {
        if let Some(ids) = ctx.host_token_ids {
            ensure!(
                ids.len() >= m,
                "host token ids: {} for {m} tokens",
                ids.len()
            );
            return Ok(ids[..m].to_vec());
        }
        let dev = ctx
            .token_ids
            .context("engram needs the step's token ids (none in the context)")?;
        let mut bytes = vec![0u8; m * 4];
        ctx.gpu.copy_d2h(dev, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    pub(super) fn engram(
        &self,
        hi: usize,
        streams: DevicePtr,
        m: usize,
        start_pos: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let gpu = ctx.gpu;
        let hashes = {
            let mut cache = rt.step_hashes.lock().unwrap();
            if hi == 0 || cache.is_none() {
                let ids = self.step_token_ids(ctx, m)?;
                let mut hasher = rt.hasher.lock().unwrap();
                let h = hasher.hash(&ids, start_pos)?;
                *cache = Some(h.clone());
                h
            } else {
                cache.clone().unwrap()
            }
        };
        let hasher = rt.hasher.lock().unwrap();
        let row_ids = hasher.layer_row_ids(&hashes, m, hi);
        drop(hasher);
        let mut raw = vec![0u8; row_ids.len() * ENGRAM_ROW_BYTES];
        rt.rows.read_rows(self.idx, &row_ids, &mut raw)?;
        let engram = rt.engram.lock().unwrap();
        engram.rows_from_q2k(gpu, &raw, row_ids.len(), stream)?;
        engram.apply(gpu, self.idx, streams, m, stream)
    }

    /// One block for `m` tokens at `start_pos`, on the highway.
    pub(super) fn step(
        &self,
        hidden: DevicePtr,
        m: usize,
        start_pos: usize,
        state: &mut dyn LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let gpu = ctx.gpu;
        ensure!(
            m >= 1 && m <= rt.max_tokens,
            "deepseek-v4.1: {m} tokens exceeds the {}-token workspace",
            rt.max_tokens
        );
        ensure!(
            start_pos == 0 || m == 1,
            "deepseek-v4.1: chunked prefill is not supported (start {start_pos}, {m} tokens); raise --max-prefill-tokens"
        );
        let st = state
            .as_any_mut()
            .downcast_mut::<V41LayerState>()
            .context("deepseek-v4.1 layer given a foreign state")?;
        let streams = ctx.buffers.hc_streams();
        let (h, hc) = (rt.hidden, rt.hc_mult);

        if self.idx == 0 {
            if start_pos == 0 {
                // a new sequence: fresh shared slots, hasher, delayed mix
                *rt.shared.lock().unwrap() = SharedV41::default();
                rt.hasher.lock().unwrap().reset();
                *rt.step_hashes.lock().unwrap() = None;
            }
            // a new step: the captured step's device-side position and
            // selection are stale
            rt.attn.lock().unwrap().invalidate_decode_uploads();
            // the initial pre-mix is one-hot on stream 0
            let mut onehot = vec![0u8; m * hc * 4];
            for t in 0..m {
                onehot[t * hc * 4..t * hc * 4 + 4].copy_from_slice(&1f32.to_le_bytes());
            }
            gpu.copy_h2d(&onehot, rt.pre_prev)?;
            ops::hc_expand(
                gpu,
                self.k_hc_expand,
                hidden,
                streams,
                m as u32,
                h as u32,
                hc as u32,
                stream,
            )?;
        }
        if self.idx == 0 {
            *rt.step_start.lock().unwrap() = Some(std::time::Instant::now());
            *rt.step_moe.lock().unwrap() = Default::default();
            *rt.step_attn_ms.lock().unwrap() = 0.0;
            *rt.step_engram_ms.lock().unwrap() = 0.0;
        }
        let te = std::time::Instant::now();
        if let Some(hi) = self.engram_index
            && !std::env::var("ATLAS_DS41_NO_ENGRAM").is_ok_and(|v| v == "1")
        {
            self.engram(hi, streams, m, start_pos, ctx, stream)?;
        }
        *rt.step_engram_ms.lock().unwrap() += te.elapsed().as_secs_f64() * 1e3;
        let diag = diag_on();
        if diag {
            gpu.synchronize(stream)?;
            tracing::info!(
                "DS41 L{} in: streams rms {:.4} (token 0)",
                self.idx,
                diag_rms_f32(gpu, streams, hc * h)
            );
        }

        // the single-token step at a position > 0 is the captured one; prefill
        // (m > 1, or the one-token prompt at position 0) stays eager
        let mode = graph_mode();
        let graph = mode != GraphMode::Off && m == 1 && start_pos > 0 && !gpu.debug_sync_kernels();
        match (graph, mode) {
            (false, _) => self.step_eager(hidden, m, start_pos, st, ctx, stream),
            (true, GraphMode::Oracle) => self.step_oracle(hidden, start_pos, st, ctx, stream),
            (true, _) => self.step_graph(hidden, start_pos, st, ctx, stream),
        }
    }

    /// The eager step after the engram: every launch issued from the host,
    /// the attention and the MoE with their host work inline. This is the
    /// path `ATLAS_DS41_GRAPH` unset (or `0`) takes, and prefill always.
    pub(super) fn step_eager(
        &self,
        hidden: DevicePtr,
        m: usize,
        start_pos: usize,
        st: &mut V41LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let gpu = ctx.gpu;
        let streams = ctx.buffers.hc_streams();
        let (h, hc) = (rt.hidden, rt.hc_mult);
        let diag = diag_on();

        // attention
        self.mixes(
            gpu,
            &self.hc_attn,
            streams,
            rt.pre_a,
            rt.post_s,
            rt.comb_s,
            m,
            stream,
        )?;
        self.collapse(gpu, streams, rt.pre_prev, hidden, m, stream)?;
        let normed = ctx.buffers.norm_output();
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            hidden,
            &self.attn_norm,
            normed,
            m as u32,
            h as u32,
            rt.norm_eps,
            stream,
        )?;
        let ta = std::time::Instant::now();
        let attn_out = {
            let attn = rt.attn.lock().unwrap();
            let mut shared = rt.shared.lock().unwrap();
            let run = attn.forward(
                gpu,
                &self.attn_w,
                &mut st.attn,
                &mut shared,
                normed,
                m,
                start_pos,
                stream,
            )?;
            run.out
        };
        *rt.step_attn_ms.lock().unwrap() += ta.elapsed().as_secs_f64() * 1e3;
        if diag {
            gpu.synchronize(stream)?;
            tracing::info!(
                "DS41 L{} attn: in rms {:.4} normed rms {:.4} out rms {:.4}",
                self.idx,
                diag_rms_bf16(gpu, hidden, h),
                diag_rms_bf16(gpu, normed, h),
                diag_rms_bf16(gpu, attn_out, h)
            );
        }
        self.hc_post(gpu, attn_out, streams, rt.post_s, rt.comb_s, m, stream)?;

        // ffn
        self.mixes(
            gpu,
            &self.hc_ffn,
            streams,
            rt.pre_f,
            rt.post_s,
            rt.comb_s,
            m,
            stream,
        )?;
        self.collapse(gpu, streams, rt.pre_a, hidden, m, stream)?;
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            hidden,
            &self.ffn_norm,
            normed,
            m as u32,
            h as u32,
            rt.norm_eps,
            stream,
        )?;
        let moe_out = {
            let moe = rt.moe.lock().unwrap();
            let mut lru = rt.lru.lock().unwrap();
            let (out, _w, _i) = moe.forward(
                gpu,
                &self.moe_w,
                &mut lru,
                &rt.slices,
                normed,
                m,
                rt.reader_threads,
                stream,
            )?;
            rt.step_moe.lock().unwrap().add(&moe.last.get());
            out
        };
        if diag {
            gpu.synchronize(stream)?;
            tracing::info!(
                "DS41 L{} ffn: in rms {:.4} normed rms {:.4} out rms {:.4}",
                self.idx,
                diag_rms_bf16(gpu, hidden, h),
                diag_rms_bf16(gpu, normed, h),
                diag_rms_bf16(gpu, moe_out, h)
            );
        }
        self.hc_post(gpu, moe_out, streams, rt.post_s, rt.comb_s, m, stream)?;
        gpu.copy_d2d_async(rt.pre_f, rt.pre_prev, m * hc * 4, stream)?;

        if self.idx + 1 == rt.n_layers {
            self.step_line(m, start_pos, "eager");
            // no learned head on V4.1: the final collapse uses the last ffn pre
            self.collapse(gpu, streams, rt.pre_prev, hidden, m, stream)?;
            gpu.synchronize(stream)?;
        }
        Ok(())
    }

}
