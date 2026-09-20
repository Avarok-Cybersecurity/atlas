// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! The single-token step as a few CUDA graphs spanning layers
//! (`ATLAS_DS41_STEP_GRAPH=1`, needs `ATLAS_DS41_DEVICE_ROUTE=1`).
//!
//! With the routing on the device and every engram layer's rows uploaded
//! up front, the only host work left in a decode step is the index-source
//! layers' attention (the index score read-back and the host top-k, plus
//! the compressor's position parity): eight layers of forty. So the step is
//! captured as SEGMENTS between those points: segment 0 = layers 0 and 1
//! through layer 2's attention input; segment k = the ffn of index layer k
//! through the next index layer's attention input; the last runs to the
//! final collapse. Nine graph launches a token instead of ~1,900 kernel
//! launches, and one read-back per segment (the routing headers of its
//! layers) instead of one per layer.
//!
//! STATUS (2026-09-19): EXPERIMENTAL, default off, NOT KEPT. The capture,
//! the segment geometry (nine segments a token), the device selection with
//! deferred headers and the one read-back per segment all work on GB10 (the
//! first MinHeap request under the first protocol produced the oracle's text).
//! What does not yet work is the miss fallback. Inside a replayed segment
//! nothing fetches: the selection kernel points an absent expert at slot 0
//! and flags the layer, and every layer after it in the segment then routes
//! on a wrong input, so a second run of the graph reveals new misses (up to
//! one run per layer) and fills the cache with noise; and running the
//! flagged layer and the ones after it eagerly needs the layer driver to
//! revisit layers it has already passed, which the per-layer `step` cannot
//! do (a segment closed and launched by layer c has layers s..c-1 behind it).
//! The sound design, not yet built: (1) the graph snapshots hidden, streams,
//! pre_prev and pre_a after EVERY layer (~90 KB a layer, D2D), so a miss at
//! layer f restores the state after f-1 and marks f..end eager while s..f-1
//! stand (their picks were real); (2) the capture token runs every layer
//! eagerly on the compute stream while the bodies are recorded on a second
//! capture stream, so a segment is only ever launched at its owner's step,
//! never at its closing layer; (3) `eager_until` then covers f..end and the
//! driver reaches those layers in order. Until then a miss inside a segment
//! fails the request ("layer N reached outside any segment"), which is why
//! the flag stays off.
//!
//! Every layer's `step` lands here; the OWNER of a segment (layer 0, or an
//! index layer at its ffn) prepares, launches and runs the protocol, and
//! the layers a replayed segment covered return at once (`skip_until`).
//! Capture happens on the first token: the owner opens the capture, the
//! following layers append their bodies, the next index layer (or the last
//! layer) closes it.

use std::sync::OnceLock;

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle};

use super::{DeepSeekV41Layer, V41LayerState};
use crate::layer::ForwardContext;
use crate::layers::attn_v41::LayerRole;
use crate::layers::moe_v41::MoeV41;

/// `ATLAS_DS41_STEP_GRAPH=1`, read once.
pub fn step_graph_on() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("ATLAS_DS41_STEP_GRAPH").is_ok_and(|v| v == "1"))
}

/// One captured segment.
pub struct SegGraph {
    g: GraphHandle,
    /// The layer whose attention input closes the segment (`n_layers` for
    /// the last one); layers below it are covered.
    end: usize,
    /// Layers whose MoE ran inside (their headers to read after a launch).
    moe_layers: Vec<u32>,
    /// Selection classes the covered attention bodies read (ratio > 0,
    /// ratio 0) and the compressed rows they bake.
    classes: (bool, bool),
    rows_b: Option<DevicePtr>,
}

/// An open capture: the owner and the MoE layers appended so far.
pub struct SegCapture {
    owner: usize,
    moe_layers: Vec<u32>,
    classes: (bool, bool),
    rows_b: Option<DevicePtr>,
}

#[derive(Default)]
pub struct SegState {
    /// A capture or a replay failed: the mode is off for the process.
    pub disabled: bool,
    /// Layers below this were covered by the segment just launched.
    pub skip_until: usize,
    /// Layers below this run eagerly this token (a segment missed).
    pub eager_until: usize,
    pub capturing: Option<SegCapture>,
    /// By owner layer.
    pub graphs: Vec<Option<SegGraph>>,
    /// The buffers every graph bakes: hidden, streams, normed.
    pub baked: Option<[DevicePtr; 3]>,
    /// Graph launches and eager fallbacks this process, for the log.
    pub launches: u64,
    pub reruns: u64,
}

impl DeepSeekV41Layer {
    fn role_of(&self, l: usize) -> Result<LayerRole> {
        self.rt.roles.lock().unwrap()[l].context("step graph: a layer without a role")
    }

    fn capturable(r: &LayerRole) -> bool {
        !r.is_kv_source && !r.is_index_source
    }

    /// The segment an owner at `s` spans: its end and the selection classes
    /// of the capturable layers in it.
    fn geometry(&self, s: usize) -> Result<(usize, (bool, bool))> {
        let n = self.rt.n_layers;
        let mut classes = (false, false);
        let mut end = n;
        for l in s..n {
            let r = self.role_of(l)?;
            if l != s && !Self::capturable(&r) {
                end = l;
                break;
            }
            if Self::capturable(&r) {
                if r.ratio > 0 {
                    classes.0 = true;
                } else {
                    classes.1 = true;
                }
            }
        }
        Ok((end, classes))
    }

    /// The step of one layer in segment-graph mode.
    pub(super) fn step_seg(
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
        let n = rt.n_layers;
        let mut seg = rt.seg.lock().unwrap();
        if seg.disabled {
            drop(seg);
            return self.step_eager(hidden, 1, start_pos, st, ctx, stream);
        }
        if seg.graphs.len() != n {
            seg.graphs = (0..n).map(|_| None).collect();
        }
        if self.idx == 0 {
            // a new step: the baked buffers, the skip and fallback marks
            seg.skip_until = 0;
            seg.eager_until = 0;
            let baked = [hidden, streams, normed];
            if seg.baked != Some(baked) {
                if seg.baked.is_some() {
                    tracing::warn!("DS41 step graph: buffers changed; recapturing every segment");
                }
                for g in seg.graphs.iter_mut().flat_map(Option::take) {
                    gpu.destroy_graph(g.g)?;
                }
                seg.baked = Some(baked);
            }
        }
        if self.idx < seg.skip_until {
            return Ok(());
        }
        if self.idx < seg.eager_until {
            drop(seg);
            return self.step_eager(hidden, 1, start_pos, st, ctx, stream);
        }
        let role = self.role_of(self.idx)?;
        let owner = self.idx == 0 || !Self::capturable(&role);

        // a layer inside an open capture: append the body (an index layer
        // appends its attention input and closes the capture first)
        if let Some(cap) = seg.capturing.take() {
            ensure!(
                cap.owner < self.idx,
                "step graph: capture owned by a later layer"
            );
            let arena = MoeV41::arena_of(&rt.lru.lock().unwrap());
            if owner {
                if self.engram_index.is_some()
                    && !std::env::var("ATLAS_DS41_NO_ENGRAM").is_ok_and(|v| v == "1")
                {
                    rt.engram
                        .lock()
                        .unwrap()
                        .apply(gpu, self.idx, streams, 1, stream)?;
                }
                self.seg_attn_in(gpu, streams, hidden, normed, stream)?;
                let g = self.close_capture(gpu, &mut seg, cap, self.idx, stream)?;
                let owner_l = seg.graphs[g]
                    .as_ref()
                    .map(|s| s.moe_layers.clone())
                    .unwrap_or_default();
                let gh = seg.graphs[g]
                    .as_ref()
                    .map(|s| s.g)
                    .context("closed segment")?;
                let miss = self.launch_once(gpu, gh, &owner_l, stream)?;
                seg.launches += 1;
                if miss {
                    seg.reruns += 1;
                    seg.eager_until = self.idx;
                    drop(seg);
                    self.restore(gpu, hidden, streams, stream)?;
                    return self.step_eager(hidden, 1, start_pos, st, ctx, stream);
                }
                seg.skip_until = self.idx;
                // fall through to the owner path for the ffn segment
            } else {
                let moe = rt.moe.lock().unwrap();
                let mut cap = cap;
                self.body_attn(gpu, st, hidden, streams, normed, cap.rows_b, stream)?;
                self.body_ffn(gpu, &moe, arena, hidden, streams, normed, stream)?;
                cap.moe_layers.push(self.idx as u32);
                drop(moe);
                if self.idx + 1 == n {
                    let g = self.close_capture(gpu, &mut seg, cap, n, stream)?;
                    let layers = seg.graphs[g]
                        .as_ref()
                        .map(|s| s.moe_layers.clone())
                        .unwrap_or_default();
                    let gh = seg.graphs[g]
                        .as_ref()
                        .map(|s| s.g)
                        .context("closed segment")?;
                    let miss = self.launch_once(gpu, gh, &layers, stream)?;
                    seg.launches += 1;
                    if miss {
                        seg.reruns += 1;
                        seg.eager_until = n;
                        drop(seg);
                        self.restore(gpu, hidden, streams, stream)?;
                        return self.step_eager(hidden, 1, start_pos, st, ctx, stream);
                    }
                    seg.skip_until = n;
                    self.log_step(&seg, start_pos);
                } else {
                    seg.capturing = Some(cap);
                }
                return Ok(());
            }
        }
        ensure!(
            owner,
            "step graph: layer {} reached outside any segment",
            self.idx
        );

        // the owner: prepare, then replay or capture the segment from here
        let (end, classes) = self.geometry(self.idx)?;
        let rows_b = self.prepare(gpu, st, ctx, start_pos, classes, normed, stream)?;
        if let Some(sg) = seg.graphs[self.idx].as_ref()
            && (sg.end != end || sg.classes != classes || sg.rows_b != rows_b)
        {
            tracing::warn!(
                "DS41 step graph: segment {} bakes other rows; recapturing",
                self.idx
            );
            if let Some(sg) = seg.graphs[self.idx].take() {
                gpu.destroy_graph(sg.g)?;
            }
        }
        if let Some(sg) = seg.graphs[self.idx].as_ref() {
            let (gh, layers) = (sg.g, sg.moe_layers.clone());
            let miss = self.launch_once(gpu, gh, &layers, stream)?;
            seg.launches += 1;
            if miss {
                seg.reruns += 1;
                seg.eager_until = end;
                drop(seg);
                self.restore(gpu, hidden, streams, stream)?;
                return self.step_eager(hidden, 1, start_pos, st, ctx, stream);
            }
            seg.skip_until = end;
            if end == n {
                self.log_step(&seg, start_pos);
            }
            return Ok(());
        }
        // capture from here
        gpu.begin_capture(stream)
            .context("step graph: begin_capture")?;
        let arena = MoeV41::arena_of(&rt.lru.lock().unwrap());
        let body = || -> Result<()> {
            self.save_nodes(gpu, hidden, streams, stream)?;
            let moe = rt.moe.lock().unwrap();
            if self.idx == 0 {
                self.body_attn(gpu, st, hidden, streams, normed, rows_b, stream)?;
            }
            self.body_ffn(gpu, &moe, arena, hidden, streams, normed, stream)
        };
        if let Err(e) = body() {
            gpu.abort_capture_if_active(stream);
            seg.disabled = true;
            return Err(e.context(format!(
                "DS41 L{}: step graph capture failed; the mode is off",
                self.idx
            )));
        }
        let cap = SegCapture {
            owner: self.idx,
            moe_layers: vec![self.idx as u32],
            classes,
            rows_b,
        };
        if self.idx + 1 == n {
            let g = self.close_capture(gpu, &mut seg, cap, n, stream)?;
            let layers = seg.graphs[g]
                .as_ref()
                .map(|s| s.moe_layers.clone())
                .unwrap_or_default();
            let gh = seg.graphs[g]
                .as_ref()
                .map(|s| s.g)
                .context("closed segment")?;
            let miss = self.launch_once(gpu, gh, &layers, stream)?;
            seg.launches += 1;
            if miss {
                seg.reruns += 1;
                seg.eager_until = n;
                drop(seg);
                self.restore(gpu, hidden, streams, stream)?;
                return self.step_eager(hidden, 1, start_pos, st, ctx, stream);
            }
            seg.skip_until = n;
            self.log_step(&seg, start_pos);
        } else {
            seg.capturing = Some(cap);
        }
        Ok(())
    }

    /// End the open capture as the segment of `cap.owner` ending at `end`.
    fn close_capture(
        &self,
        gpu: &dyn GpuBackend,
        seg: &mut SegState,
        cap: SegCapture,
        end: usize,
        stream: u64,
    ) -> Result<usize> {
        let g = match gpu.end_capture(stream) {
            Ok(g) => g,
            Err(e) => {
                seg.disabled = true;
                return Err(e.context("step graph: end_capture failed; the mode is off"));
            }
        };
        tracing::info!(
            "DS41 step graph: segment {} = layers {}..{} captured ({} MoE layers)",
            cap.owner,
            cap.owner,
            end,
            cap.moe_layers.len()
        );
        seg.graphs[cap.owner] = Some(SegGraph {
            g,
            end,
            moe_layers: cap.moe_layers,
            classes: cap.classes,
            rows_b: cap.rows_b,
        });
        Ok(cap.owner)
    }

    fn log_step(&self, seg: &SegState, start_pos: usize) {
        if step_log_on() {
            tracing::info!(
                "DS41 step graph pos {start_pos}: {} launches, {} re-runs so far",
                seg.launches,
                seg.reruns
            );
        }
    }
}

/// `ATLAS_DS41_STEP_GRAPH_LOG=1`: one line per token with the running counts.
pub(super) fn step_log_on() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("ATLAS_DS41_STEP_GRAPH_LOG").is_ok_and(|v| v == "1"))
}
