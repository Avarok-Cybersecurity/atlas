// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! The segment owner's work around a graph: the engram preload, the
//! device-only layer bodies appended under capture, the save / restore
//! nodes and the miss protocol. Split from `step_seg.rs` (500-LoC cap).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle};

use super::step_seg::step_log_on;
use super::{DeepSeekV41Layer, V41LayerState};
use crate::layer::ForwardContext;
use crate::layers::engram_v41::ENGRAM_ROW_BYTES;
use crate::layers::moe_v41::MoeV41;

impl DeepSeekV41Layer {
    /// Every engram layer's rows for this token, uploaded before any graph.
    pub(super) fn engram_preload(
        &self,
        ctx: &ForwardContext,
        start_pos: usize,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        if std::env::var("ATLAS_DS41_NO_ENGRAM").is_ok_and(|v| v == "1") {
            return Ok(());
        }
        let ids = self.step_token_ids(ctx, 1)?;
        let mut hasher = rt.hasher.lock().unwrap();
        let hashes = hasher.hash(&ids, start_pos)?;
        let engram = rt.engram.lock().unwrap();
        for &(layer, hi) in rt.engram_layers.lock().unwrap().iter() {
            let row_ids = hasher.layer_row_ids(&hashes, 1, hi);
            let mut raw = vec![0u8; row_ids.len() * ENGRAM_ROW_BYTES];
            rt.rows.read_rows(layer, &row_ids, &mut raw)?;
            engram.rows_from_q2k(ctx.gpu, Some(layer), &raw, row_ids.len(), stream)?;
        }
        *rt.step_hashes.lock().unwrap() = Some(hashes);
        Ok(())
    }

    /// This layer's device-only attention body under a capture.
    pub(super) fn body_attn(
        &self,
        gpu: &dyn GpuBackend,
        st: &V41LayerState,
        hidden: DevicePtr,
        streams: DevicePtr,
        normed: DevicePtr,
        rows_b: Option<DevicePtr>,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        if self.engram_index.is_some()
            && !std::env::var("ATLAS_DS41_NO_ENGRAM").is_ok_and(|v| v == "1")
        {
            rt.engram
                .lock()
                .unwrap()
                .apply(gpu, self.idx, streams, 1, stream)?;
        }
        self.seg_attn_in(gpu, streams, hidden, normed, stream)?;
        let attn = rt.attn.lock().unwrap();
        attn.decode_body(gpu, &self.attn_w, &st.attn, normed, rows_b, stream)?;
        Ok(())
    }

    /// This layer's device-only ffn body: the mixes and norm, the router,
    /// the device selection (header deferred), the experts, the post mix.
    pub(super) fn body_ffn(
        &self,
        gpu: &dyn GpuBackend,
        moe: &MoeV41,
        arena: (u64, u64, u64, u64, u64),
        hidden: DevicePtr,
        streams: DevicePtr,
        normed: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let attn_out = self.rt.attn.lock().unwrap().out_ptr();
        self.seg_ffn_in(gpu, moe, attn_out, streams, hidden, normed, stream)?;
        moe.route_select_deferred(gpu, &self.moe_w, arena, stream)?;
        self.seg_ffn_out(gpu, moe, moe.cfg.topk, streams, hidden, normed, stream)
    }

    /// The save nodes at a segment's start.
    pub(super) fn save_nodes(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        streams: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let (h, hc) = (rt.hidden, rt.hc_mult);
        let attn_out = rt.attn.lock().unwrap().out_ptr();
        let src = [hidden, streams, rt.pre_prev, rt.pre_a, attn_out];
        let n = [h * 2, hc * h * 4, hc * 4, hc * 4, h * 2];
        for i in 0..5 {
            gpu.copy_d2d_async(src[i], rt.seg_save[i], n[i], stream)?;
        }
        Ok(())
    }

    pub(super) fn restore(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        streams: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let (h, hc) = (rt.hidden, rt.hc_mult);
        let attn_out = rt.attn.lock().unwrap().out_ptr();
        let dst = [hidden, streams, rt.pre_prev, rt.pre_a, attn_out];
        let n = [h * 2, hc * h * 4, hc * 4, hc * 4, h * 2];
        for i in 0..5 {
            gpu.copy_d2d_async(rt.seg_save[i], dst[i], n[i], stream)?;
        }
        Ok(())
    }

    /// Launch a segment once, read its layers' headers, touch the picks in
    /// the cache (fetching the misses) and bring the slot table up to date.
    /// Returns whether any layer flagged a miss or any slot moved: the
    /// segment's outputs are then wrong (an absent expert read slot 0), the
    /// caller restores the saved input and runs the segment's layers
    /// eagerly for this token. A second graph run would not do: a wrong
    /// layer's output re-routes every layer after it, so each run can
    /// reveal new misses.
    pub(super) fn launch_once(
        &self,
        gpu: &dyn GpuBackend,
        g: GraphHandle,
        moe_layers: &[u32],
        stream: u64,
    ) -> Result<bool> {
        let rt = &self.rt;
        let moe = rt.moe.lock().unwrap();
        gpu.launch_graph(g, stream)?;
        let hdr = moe.read_headers(gpu, stream)?;
        let mut again = false;
        let mut lru = rt.lru.lock().unwrap();
        lru.begin_token();
        let misses0 = lru.stats().misses;
        let mut flagged: Vec<u32> = Vec::new();
        for &l in moe_layers {
            let (flag, picks, _) = moe.parse_header(&hdr, l)?;
            if flag {
                flagged.push(l);
                // the picks of a flagged layer are real; the picks of the
                // layers AFTER a flagged one are not (their input was wrong),
                // and fetching them would fill the cache with noise
                let keys: Vec<(u32, u32)> = picks.iter().map(|&e| (l, e as u32)).collect();
                lru.fetch_many_on(gpu, stream, &*rt.slices, &keys, &[], rt.reader_threads)?;
                let changes = lru.drain_slot_changes();
                moe.slot_table_update(gpu, &changes, stream)?;
                again = true;
                break;
            }
            let keys: Vec<(u32, u32)> = picks.iter().map(|&e| (l, e as u32)).collect();
            lru.fetch_many_on(gpu, stream, &*rt.slices, &keys, &[], rt.reader_threads)?;
            let changes = lru.drain_slot_changes();
            again |= !changes.is_empty();
            moe.slot_table_update(gpu, &changes, stream)?;
        }
        let misses = lru.stats().misses - misses0;
        drop(lru);
        if step_log_on() && again {
            tracing::info!(
                "DS41 step graph: segment {} missed (flagged {flagged:?}, {misses} fetched): its layers run eagerly this token",
                self.idx
            );
        }
        Ok(again)
    }

    /// The owner's preparation before a launch or a capture: the engram
    /// rows (layer 0), the eager attention (an index layer), the selection
    /// uploads for the classes the segment reads, the slot table.
    pub(super) fn prepare(
        &self,
        gpu: &dyn GpuBackend,
        st: &mut V41LayerState,
        ctx: &ForwardContext,
        start_pos: usize,
        classes: (bool, bool),
        normed: DevicePtr,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        let rt = &self.rt;
        if self.idx == 0 {
            self.engram_preload(ctx, start_pos, stream)?;
        } else {
            // an index layer: its attention is host work, run eagerly on the
            // normed input the closing graph left; it writes pos / idx_dev
            // for itself, so the captured uploads are stale after it
            let mut attn = rt.attn.lock().unwrap();
            let mut shared = rt.shared.lock().unwrap();
            attn.forward(
                gpu,
                &self.attn_w,
                &mut st.attn,
                &mut shared,
                normed,
                1,
                start_pos,
                stream,
            )?;
            attn.invalidate_decode_uploads();
        }
        let mut attn = rt.attn.lock().unwrap();
        let shared = rt.shared.lock().unwrap();
        let mut rows_b = None;
        if classes.0 {
            rows_b = attn.decode_prep_role(true, &shared, gpu, start_pos, stream)?;
        }
        if classes.1 {
            attn.decode_prep_role(false, &shared, gpu, start_pos, stream)?;
        }
        drop(shared);
        drop(attn);
        let pending = rt.lru.lock().unwrap().drain_slot_changes();
        rt.moe
            .lock()
            .unwrap()
            .slot_table_update(gpu, &pending, stream)?;
        Ok(rows_b)
    }
}
