// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! `DeepSeekV41Layer::step`, one block for `m` tokens on the mHC highway, with
//! its hc-mix, engram and diagnostics helpers.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{DeepSeekV41Layer, GraphMode, V41LayerState, graph_mode};
use crate::layer::{ForwardContext, LayerState};
use crate::layers::attn_v41::SharedV41;
use crate::layers::engram_v41::ENGRAM_ROW_BYTES;
use crate::layers::ops;

pub(super) fn diag_on() -> bool {
    std::env::var("ATLAS_DS41_DIAG").is_ok_and(|v| v == "1")
}

/// `ATLAS_DS41_TRACE=1`, read once: after every layer's attention and MoE
/// halves, synchronise and log a checksum of the half's output and of the
/// highway, so two runs of the same request can be diffed to the first
/// (position, layer, half) where they part. Diagnostics only.
fn trace_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ATLAS_DS41_TRACE").is_ok_and(|v| v == "1"))
}

/// FNV-1a over `bytes` device bytes at `p` (synchronises).
fn trace_hash(gpu: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> u64 {
    let mut b = vec![0u8; bytes];
    if gpu.copy_d2h(p, &mut b).is_err() {
        return 0;
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
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

    /// `ATLAS_DS41_PREDICT_TRACE`: before this layer's routing, run its
    /// router on the MoE inputs of layers L-1 and L-2 (kept in `rt.pred_x`),
    /// keep the 12 best of each, note which of the layer's experts are
    /// resident now, then save this layer's input for the layers after it.
    /// Diagnostics: synchronises twice a layer.
    #[allow(clippy::type_complexity)]
    fn predict_before_moe(
        &self,
        gpu: &dyn GpuBackend,
        normed: DevicePtr,
        m: usize,
        start_pos: usize,
        stream: u64,
    ) -> Result<Option<(Vec<bool>, Vec<Vec<usize>>)>> {
        let rt = &self.rt;
        if m != 1 || start_pos == 0 || rt.pred_trace.lock().unwrap().is_none() {
            return Ok(None);
        }
        let h = rt.hidden;
        let slot = |l: usize| DevicePtr(rt.pred_x.0 + ((l % 3) * h * 2) as u64);
        let moe = rt.moe.lock().unwrap();
        let mut preds = Vec::new();
        for d in 1..=2 {
            if self.idx >= d {
                preds.push(moe.route_predict(gpu, &self.moe_w, slot(self.idx - d), 12, stream)?);
            }
        }
        gpu.copy_d2d_async(normed, slot(self.idx), h * 2, stream)?;
        let lru = rt.lru.lock().unwrap();
        let resident: Vec<bool> = (0..rt.moe_cfg.n_routed)
            .map(|e| lru.contains(self.idx as u32, e as u32))
            .collect();
        Ok(Some((resident, preds)))
    }

    fn predict_log(
        &self,
        start_pos: usize,
        resident: &[bool],
        preds: &[Vec<usize>],
        actual: &[usize],
    ) {
        use std::io::Write;
        let mut g = self.rt.pred_trace.lock().unwrap();
        let Some(w) = g.as_mut() else { return };
        let ids = |v: &[usize]| {
            v.iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        let misses: Vec<usize> = actual.iter().copied().filter(|&e| !resident[e]).collect();
        let _ = write!(
            w,
            "pos={start_pos} L{} act={} miss={}",
            self.idx,
            ids(actual),
            ids(&misses)
        );
        for (i, p) in preds.iter().enumerate() {
            // the prediction, and the part of it a prefetch would have to read
            let cold: Vec<usize> = p.iter().copied().filter(|&e| !resident[e]).collect();
            let _ = write!(w, " d{}={} c{}={}", i + 1, ids(p), i + 1, ids(&cold));
        }
        let _ = writeln!(w);
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
        self.mixes_collapse(
            gpu,
            &self.hc_attn,
            streams,
            rt.pre_a,
            rt.pre_prev,
            hidden,
            m,
            stream,
        )?;
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
        if trace_on() {
            gpu.synchronize(stream)?;
            tracing::info!(
                "DS41 trace pos={} L{} attn={:016x} normed={:016x}",
                start_pos,
                self.idx,
                trace_hash(gpu, attn_out, m * h * 2),
                trace_hash(gpu, normed, m * h * 2)
            );
        }
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
        self.mixes_collapse(
            gpu,
            &self.hc_ffn,
            streams,
            rt.pre_f,
            rt.pre_a,
            hidden,
            m,
            stream,
        )?;
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
        if trace_on() {
            gpu.synchronize(stream)?;
            tracing::info!(
                "DS41 trace pos={} L{} moein={:016x} hwyin={:016x}",
                start_pos,
                self.idx,
                trace_hash(gpu, normed, m * h * 2),
                trace_hash(gpu, streams, m * hc * h * 4)
            );
        }
        let pred = self.predict_before_moe(gpu, normed, m, start_pos, stream)?;
        let moe_out = {
            let moe = rt.moe.lock().unwrap();
            let mut lru = rt.lru.lock().unwrap();
            let before = lru.stats();
            let (out, w_, i_) = moe.forward(
                gpu,
                &self.moe_w,
                &mut lru,
                &*rt.slices,
                normed,
                m,
                rt.reader_threads,
                self.next_router.as_ref(),
                stream,
            )?;
            rt.step_moe.lock().unwrap().add(&moe.last.get());
            if let Some((resident, preds)) = pred {
                self.predict_log(start_pos, &resident, &preds, &i_);
            }
            if trace_on() {
                let after = lru.stats();
                let wb: Vec<u8> = w_.iter().flat_map(|x| x.to_le_bytes()).collect();
                let mut hw: u64 = 0xcbf2_9ce4_8422_2325;
                for x in wb {
                    hw ^= x as u64;
                    hw = hw.wrapping_mul(0x0000_0100_0000_01b3);
                }
                tracing::info!(
                    "DS41 trace pos={} L{} route={:?} w={:016x} hits={} misses={} evict={}",
                    start_pos,
                    self.idx,
                    &i_[..i_.len().min(12)],
                    hw,
                    after.hits - before.hits,
                    after.misses - before.misses,
                    after.evictions - before.evictions
                );
            }
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
        if trace_on() {
            gpu.synchronize(stream)?;
            tracing::info!(
                "DS41 trace pos={} L{} moe={:016x} hwy={:016x}",
                start_pos,
                self.idx,
                trace_hash(gpu, moe_out, m * h * 2),
                trace_hash(gpu, streams, m * hc * h * 4)
            );
        }

        if self.idx + 1 == rt.n_layers {
            self.step_line(m, start_pos, "eager");
            // no learned head on V4.1: the final collapse uses the last ffn pre
            self.collapse(gpu, streams, rt.pre_prev, hidden, m, stream)?;
            gpu.synchronize(stream)?;
        }
        Ok(())
    }
}
