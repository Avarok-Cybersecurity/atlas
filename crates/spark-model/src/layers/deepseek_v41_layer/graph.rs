// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! The captured single-token step (`ATLAS_DS41_GRAPH=1`): the step split into
//! graph segments around the host work in the middle of every layer, the
//! segment runner, the capture and replay, and the eager oracle that checks
//! a replayed step against the eager path. Split from
//! `deepseek_v41_layer/step.rs` (500-LoC cap) when the line was re-cut onto main.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle};

use std::sync::atomic::Ordering;

use super::step::{diag_on, diag_rms_bf16};
use super::{Baked, DeepSeekV41Layer, LayerGraphs, V41LayerState};
use crate::layer::ForwardContext;
use crate::layers::attn_v41::{AttnV41, AttnV41LayerState};
use crate::layers::moe_v41::MoeV41;
use crate::layers::ops;

impl DeepSeekV41Layer {
    /// Segment A's opening: the attention site's mixes, the delayed-pre
    /// collapse and the attention norm into `normed`. Device work only.
    pub(super) fn seg_attn_in(
        &self,
        gpu: &dyn GpuBackend,
        streams: DevicePtr,
        hidden: DevicePtr,
        normed: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        self.mixes_collapse(
            gpu,
            &self.hc_attn,
            streams,
            rt.pre_a,
            rt.pre_prev,
            hidden,
            1,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            hidden,
            &self.attn_norm,
            normed,
            1,
            rt.hidden as u32,
            rt.norm_eps,
            stream,
        )
    }

    /// Segment A's close: `hc_post` of the attention, the ffn site's mixes,
    /// the collapse, the ffn norm into `normed`, the router GEMV into the MoE's
    /// logits. Device work only.
    pub(super) fn seg_ffn_in(
        &self,
        gpu: &dyn GpuBackend,
        moe: &MoeV41,
        attn_out: DevicePtr,
        streams: DevicePtr,
        hidden: DevicePtr,
        normed: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        self.hc_post(gpu, attn_out, streams, rt.post_s, rt.comb_s, 1, stream)?;
        self.mixes_collapse(
            gpu,
            &self.hc_ffn,
            streams,
            rt.pre_f,
            rt.pre_a,
            hidden,
            1,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            hidden,
            &self.ffn_norm,
            normed,
            1,
            rt.hidden as u32,
            rt.norm_eps,
            stream,
        )?;
        moe.route_launch(gpu, &self.moe_w, normed, 1, stream)
    }

    /// Segment B: the expert compute from the staged plan, `hc_post`, the
    /// delayed-mix copy, and on the last layer the final collapse. Device
    /// work only.
    pub(super) fn seg_ffn_out(
        &self,
        gpu: &dyn GpuBackend,
        moe: &MoeV41,
        ne: usize,
        streams: DevicePtr,
        hidden: DevicePtr,
        normed: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let moe_out = moe.compute_m1(gpu, &self.moe_w, normed, ne, stream)?;
        self.hc_post(gpu, moe_out, streams, rt.post_s, rt.comb_s, 1, stream)?;
        gpu.copy_d2d_async(rt.pre_f, rt.pre_prev, rt.hc_mult * 4, stream)?;
        if self.idx + 1 == rt.n_layers {
            // no learned head on V4.1: the final collapse uses the last ffn pre
            self.collapse(gpu, streams, rt.pre_prev, hidden, 1, stream)?;
        }
        Ok(())
    }

    /// Replay `slot`'s graph, or capture `body` into it (and run it) on the
    /// first pass. A capture that cannot begin or end runs `body` eagerly and
    /// disables further captures; an error INSIDE the body ends the capture
    /// (so the stream is usable again) and propagates.
    pub(super) fn run_segment(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        slot: &mut Option<GraphHandle>,
        body: impl Fn() -> Result<()>,
    ) -> Result<()> {
        let rt = &self.rt;
        if let Some(g) = slot {
            return gpu.launch_graph(*g, stream);
        }
        if rt.graph_disabled.load(Ordering::Relaxed) {
            return body();
        }
        if let Err(e) = gpu.begin_capture(stream) {
            tracing::warn!(
                "DS41 L{}: CUDA graph begin_capture failed ({e:#}); running eagerly and disabling capture",
                self.idx
            );
            rt.graph_disabled.store(true, Ordering::Relaxed);
            return body();
        }
        if let Err(e) = body() {
            gpu.abort_capture_if_active(stream);
            let msg = format!("{e:#}");
            let poison = msg.contains("status 900")
                || msg.contains("status 901")
                || msg.contains("STREAM_CAPTURE");
            if !poison {
                return Err(e.context(format!("DS41 L{} under CUDA graph capture", self.idx)));
            }
            // a capture RECORDS: nothing ran yet, so the eager body is the step
            tracing::warn!(
                "DS41 L{}: segment failed under capture ({msg}); running eagerly and disabling capture",
                self.idx
            );
            rt.graph_disabled.store(true, Ordering::Relaxed);
            return body();
        }
        match gpu.end_capture(stream) {
            Ok(g) => {
                *slot = Some(g);
                gpu.launch_graph(g, stream)
            }
            Err(e) => {
                tracing::warn!(
                    "DS41 L{}: CUDA graph end_capture failed ({e:#}); running eagerly and disabling capture",
                    self.idx
                );
                rt.graph_disabled.store(true, Ordering::Relaxed);
                body()
            }
        }
    }

    /// The captured single-token step: segment A (replayed or captured),
    /// the host span (routing download, expert fetch, plan uploads), segment
    /// B. Bit for bit the eager step: the kernels and their arguments are the
    /// same, the per-token inputs (position, selection, expert pointers,
    /// routing weights) are read from device buffers the host refills before
    /// each replay, and `sparse_attn` runs at a fixed, -1-padded `topk`.
    pub(super) fn step_graph(
        &self,
        hidden: DevicePtr,
        start_pos: usize,
        st: &mut V41LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let gpu = ctx.gpu;
        let streams = ctx.buffers.hc_streams();
        let normed = ctx.buffers.norm_output();
        let h = rt.hidden;
        let diag = diag_on();
        let V41LayerState {
            attn: st_attn,
            graphs,
        } = st;

        let ta = std::time::Instant::now();
        let mut attn = rt.attn.lock().unwrap();
        let mut shared = rt.shared.lock().unwrap();
        let moe = rt.moe.lock().unwrap();
        let capturable = attn.decode_capturable(&self.attn_w);
        // the host half of the attention: position and padded selection onto
        // the device (only what changed since the last upload)
        let rows_b = if capturable {
            attn.decode_prep(&self.attn_w, &shared, gpu, start_pos, stream)?
        } else {
            None
        };
        let baked = Baked {
            hidden,
            streams,
            normed,
            window: st_attn.window(),
            rows_b,
            attn_out: attn.out_ptr(),
            moe_out: moe.out_ptr(),
        };
        if let Some(g) = graphs
            && g.baked != baked
        {
            tracing::warn!(
                "DS41 L{}: captured step bakes other buffers ({:?} vs {:?}); recapturing",
                self.idx,
                g.baked,
                baked
            );
            if let Some(g) = graphs.take() {
                g.destroy(gpu)?;
            }
        }
        let graphs = graphs.get_or_insert(LayerGraphs {
            a: [None, None],
            b: None,
            baked,
        });

        // ── segment A ──
        if capturable {
            let attn_ref: &AttnV41 = &attn;
            let st_ref: &AttnV41LayerState = st_attn;
            self.run_segment(gpu, stream, &mut graphs.a[0], || {
                self.seg_attn_in(gpu, streams, hidden, normed, stream)?;
                let attn_out =
                    attn_ref.decode_body(gpu, &self.attn_w, st_ref, normed, rows_b, stream)?;
                self.seg_ffn_in(gpu, &moe, attn_out, streams, hidden, normed, stream)
            })?;
        } else {
            self.run_segment(gpu, stream, &mut graphs.a[0], || {
                self.seg_attn_in(gpu, streams, hidden, normed, stream)
            })?;
            // a kv/index source: the compressor group and the index top-k
            // are host work in the middle of the attention, so it runs eagerly
            let run = attn.forward(
                gpu,
                &self.attn_w,
                st_attn,
                &mut shared,
                normed,
                1,
                start_pos,
                stream,
            )?;
            // it wrote pos / head_pos / idx_dev for itself
            attn.invalidate_decode_uploads();
            let attn_out = run.out;
            self.run_segment(gpu, stream, &mut graphs.a[1], || {
                self.seg_ffn_in(gpu, &moe, attn_out, streams, hidden, normed, stream)
            })?;
        }
        *rt.step_attn_ms.lock().unwrap() += ta.elapsed().as_secs_f64() * 1e3;
        if diag {
            gpu.synchronize(stream)?;
            tracing::info!(
                "DS41 L{} attn: out rms {:.4}; ffn: in rms {:.4} normed rms {:.4} ({})",
                self.idx,
                diag_rms_bf16(gpu, attn.out_ptr(), h),
                diag_rms_bf16(gpu, hidden, h),
                diag_rms_bf16(gpu, normed, h),
                if capturable {
                    "graph"
                } else {
                    "graph+eager attention"
                }
            );
        }
        drop(shared);
        drop(attn);

        // ── the host span ──
        let stage = {
            let mut lru = rt.lru.lock().unwrap();
            moe.stage_m1(
                gpu,
                &self.moe_w,
                &mut lru,
                &rt.slices,
                rt.reader_threads,
                stream,
            )?
        };

        // ── segment B ──
        let tb = std::time::Instant::now();
        let ne = stage.ne;
        self.run_segment(gpu, stream, &mut graphs.b, || {
            self.seg_ffn_out(gpu, &moe, ne, streams, hidden, normed, stream)
        })?;
        if diag {
            gpu.synchronize(stream)?;
            tracing::info!(
                "DS41 L{} ffn: out rms {:.4} (graph)",
                self.idx,
                diag_rms_bf16(gpu, moe.out_ptr(), h)
            );
        }
        let mut timing = stage.timing;
        timing.compute_ms = tb.elapsed().as_secs_f64() * 1e3;
        rt.step_moe.lock().unwrap().add(&timing);
        drop(moe);

        if self.idx + 1 == rt.n_layers {
            self.step_line(1, start_pos, "graph");
            gpu.synchronize(stream)?;
        }
        Ok(())
    }

    /// `ATLAS_DS41_GRAPH_ORACLE=1`: the captured step, then the highway put
    /// back and the eager step, and the two outputs (the streams, the delayed
    /// mix, the collapsed hidden) compared bit for bit. Diagnostics: every
    /// layer runs twice.
    pub(super) fn step_oracle(
        &self,
        hidden: DevicePtr,
        start_pos: usize,
        st: &mut V41LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let gpu = ctx.gpu;
        let streams = ctx.buffers.hc_streams();
        let (h, hc) = (rt.hidden, rt.hc_mult);
        let d2h = |p: DevicePtr, n: usize| -> Result<Vec<u8>> {
            let mut b = vec![0u8; n];
            gpu.copy_d2h(p, &mut b)?;
            Ok(b)
        };
        gpu.synchronize(stream)?;
        let streams0 = d2h(streams, hc * h * 4)?;
        let pre0 = d2h(rt.pre_prev, hc * 4)?;

        self.step_graph(hidden, start_pos, st, ctx, stream)?;
        gpu.synchronize(stream)?;
        let streams_g = d2h(streams, hc * h * 4)?;
        let pre_g = d2h(rt.pre_prev, hc * 4)?;
        let hidden_g = d2h(hidden, h * 2)?;

        gpu.copy_h2d(&streams0, streams)?;
        gpu.copy_h2d(&pre0, rt.pre_prev)?;
        self.step_eager(hidden, 1, start_pos, st, ctx, stream)?;
        gpu.synchronize(stream)?;
        let streams_e = d2h(streams, hc * h * 4)?;
        let pre_e = d2h(rt.pre_prev, hc * 4)?;
        let hidden_e = d2h(hidden, h * 2)?;

        let diff = |a: &[u8], b: &[u8], w: usize| {
            a.chunks(w).zip(b.chunks(w)).filter(|(x, y)| x != y).count()
        };
        let ds = diff(&streams_g, &streams_e, 4);
        let dp = diff(&pre_g, &pre_e, 4);
        // the collapsed hidden is the layer's output only on the last layer;
        // elsewhere it is the ffn input scratch, which both paths write
        let dh = diff(&hidden_g, &hidden_e, 2);
        if ds + dp + dh == 0 {
            tracing::info!(
                "DS41 GRAPH ORACLE L{} pos {start_pos}: graph == eager (streams {} f32, pre {} f32, hidden {} bf16)",
                self.idx,
                hc * h,
                hc,
                h
            );
        } else {
            tracing::error!(
                "DS41 GRAPH ORACLE L{} pos {start_pos}: MISMATCH streams {ds}/{} pre {dp}/{} hidden {dh}/{}",
                self.idx,
                hc * h,
                hc,
                h
            );
        }
        Ok(())
    }

    /// The per-step diagnostic line (`ATLAS_DS41_DIAG=1`), from the last layer.
    pub(super) fn step_line(&self, m: usize, start_pos: usize, how: &str) {
        if !diag_on() {
            return;
        }
        let rt = &self.rt;
        let total = rt
            .step_start
            .lock()
            .unwrap()
            .map(|t| t.elapsed().as_secs_f64() * 1e3)
            .unwrap_or(0.0);
        let mo = *rt.step_moe.lock().unwrap();
        // one guard at a time: two `lru.lock()` temporaries in a single
        // statement deadlock on the std Mutex (the first guard lives to the
        // end of the statement)
        let (resident, n_slots) = {
            let lru = rt.lru.lock().unwrap();
            (lru.resident(), lru.n_slots())
        };
        let attn_ms = *rt.step_attn_ms.lock().unwrap();
        let engram_ms = *rt.step_engram_ms.lock().unwrap();
        tracing::info!(
            "DS41 step ({how}): {m} tok pos {start_pos}: total {total:.0} ms = attn {:.0} + engram {:.0} + moe(route {:.0} fetch {:.0} compute {:.0}) ms; experts hit {} miss {} read {:.2} GiB; cache {}/{} resident",
            attn_ms,
            engram_ms,
            mo.route_ms,
            mo.fetch_ms,
            mo.compute_ms,
            mo.hits,
            mo.misses,
            mo.bytes_read as f64 / 1073741824.0,
            resident,
            n_slots
        );
    }
}
