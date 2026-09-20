// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! The ffn half of the eager step: the attention's `hc_post`, the ffn
//! site's mixes and norm, the MoE
//! with its host work inline, `hc_post`, the delayed-mix copy and on the
//! last layer the final collapse. Split from `step.rs` (500-LoC cap) so the
//! segment graphs' miss path (`step_seg.rs`) can run one layer from its ffn
//! on, with the very code the eager step runs.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::DeepSeekV41Layer;
use super::step::{diag_on, diag_rms_bf16, trace_hash, trace_on};
use crate::layer::ForwardContext;
use crate::layers::ops;

impl DeepSeekV41Layer {
    /// The eager step from the attention's `hc_post` to the end of the layer,
    /// `m` tokens, on the attention output in the attention's `out` buffer
    /// (`AttnV41::out_ptr`, the buffer `forward` returns).
    pub(super) fn ffn_eager(
        &self,
        hidden: DevicePtr,
        m: usize,
        start_pos: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let gpu = ctx.gpu;
        let streams = ctx.buffers.hc_streams();
        let normed = ctx.buffers.norm_output();
        let (h, hc) = (rt.hidden, rt.hc_mult);
        let diag = diag_on();
        // the attention site's post mix of the attention output (in `out`)
        let attn_out = rt.attn.lock().unwrap().out_ptr();
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
