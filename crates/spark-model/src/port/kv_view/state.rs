// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the bounded view's process state (port lane L32) behind the
//! selector's lock: the device buffers and core of the current backend, the
//! per-sequence arms of prefills in flight, the streams of the call in flight,
//! the conversation store, and how prefill hooks find their sequence. The
//! hooks' work is in `plans.rs`, the live views in `live.rs`, the per-turn
//! selection in `select.rs`.
//!
//! # Identifying a stream
//! Prefill hooks name their streams by the ids the scheduler published
//! (`sparse_seam::publish_step`, the pool slot). Where the forward splits a
//! batched call into single-stream calls the published count no longer
//! matches and the seam hands over no ids; the stream is then found among the
//! call's streams by its chunk geometry, and a call of one stream owns every
//! single-stream hook it reaches.
//!
//! # Lifetimes
//! A new arm for a slot (a new prefill there) drops the slot's earlier view.
//! A freed sequence drops every view bound to its blocks (`release_table`).
//! A model swap (a new backend) starts from an empty state.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::config::{CAPTURE_SLOTS, KvViewConfig};
use super::core_api::{Geom, ViewCore};
use super::device::{DESC_ENTRIES, DeviceState, backend_key};
use super::history::History;
use super::metrics::{Counters, SkipReason};
use super::view::{Desc, ViewShape, prefix_hash};
use super::{CallStream, ModelFacts, TurnInfo, TurnKind};
use crate::layers::qwen3_attention::sparse_seam::{PAD_SEQ_ID, SeqBatch};

/// Builds the core on a backend (L31's in a serve, a fake in tests).
pub(crate) type CoreFactory = fn(&dyn GpuBackend, Geom, u32, usize) -> Result<Box<dyn ViewCore>>;

/// Conversations remembered (least recently used dropped first).
pub(crate) const CONVERSATIONS: usize = 32;
/// An arm untouched for this many prefill calls may lose its capture slot.
pub(crate) const STALE_CALLS: u64 = 64;

/// A conversation's last selection.
#[derive(Clone, Debug)]
pub(crate) struct Conv {
    /// Prompt tokens a later turn must repeat (before the generation header).
    pub(crate) match_len: u32,
    pub(crate) match_hash: u64,
    /// The prompt length of the turn that selected (a retry repeats it).
    pub(crate) prompt_len: u32,
    pub(crate) shape: ViewShape,
    pub(crate) history: History,
    /// This round's ranking, committed once the conversation continues.
    pub(crate) pending: Option<Arc<[f32]>>,
    pub(crate) stamp: u64,
}

/// A sequence whose prefill is in flight.
#[derive(Clone, Debug)]
pub(crate) struct Arm {
    pub(crate) session: u64,
    pub(crate) prompt_len: u32,
    pub(crate) turn: TurnInfo,
    /// The query rows the capture keeps (`None`: not armed).
    pub(crate) kept_span: Option<Range<u32>>,
    /// Every capture slot was taken.
    pub(crate) busy: bool,
    /// Some K rows were written where the sketch could not follow.
    pub(crate) gap: bool,
    /// The conversation's earlier selection, when the prompt continues it.
    pub(crate) prior: Option<Conv>,
    pub(crate) last_call: u64,
}

/// A live view (one device descriptor).
#[derive(Clone, Debug)]
pub(crate) struct Live {
    pub(crate) owner: u64,
    pub(crate) desc: Desc,
    pub(crate) view_tokens: u32,
    pub(crate) stamp: u64,
}

/// One stream of the prefill call in flight.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CallGeom {
    pub(crate) slot: u64,
    pub(crate) start: u32,
    pub(crate) end: u32,
}

/// Everything behind the selector's lock.
pub(crate) struct Inner {
    backend: usize,
    pub(crate) geom: Option<Geom>,
    pub(crate) dev: Option<DeviceState>,
    pub(crate) core: Option<Box<dyn ViewCore>>,
    pub(crate) off: Option<SkipReason>,
    pub(crate) arms: HashMap<u64, Arm>,
    pub(crate) call: Vec<CallGeom>,
    pub(crate) live: Vec<Option<Live>>,
    pub(crate) convs: HashMap<u64, Conv>,
    pub(crate) clock: u64,
    /// `(slot, first_pos, extent, kv_len, table)` of the last prefill upload.
    pub(crate) prefill_cache: Option<(u64, u32, u32, u32, DevicePtr)>,
    pub(crate) noted_shapes: Vec<(usize, u32)>,
    pub(crate) noted_not_ready: bool,
    /// Slots the seam reported freed, dropped at the next call with a backend.
    freed: Vec<u64>,
    /// The view failed inside a graph capture: clear the device descriptors
    /// at the next eager call (replays then rebuild identity tables).
    clear_pending: bool,
}

impl Inner {
    pub(crate) fn new() -> Self {
        Self {
            backend: 0,
            geom: None,
            dev: None,
            core: None,
            off: None,
            arms: HashMap::new(),
            call: Vec::new(),
            live: vec![None; DESC_ENTRIES],
            convs: HashMap::new(),
            clock: 0,
            prefill_cache: None,
            noted_shapes: Vec::new(),
            noted_not_ready: false,
            freed: Vec::new(),
            clear_pending: false,
        }
    }

    /// The buffers and core exist on `gpu` and the view is not turned off.
    pub(crate) fn ready(&self, gpu: &dyn GpuBackend) -> bool {
        self.off.is_none()
            && self.core.is_some()
            && self
                .dev
                .as_ref()
                .is_some_and(|d| d.owner == backend_key(gpu))
    }

    fn disable(&mut self, r: SkipReason, counters: &Counters, why: &str) {
        tracing::warn!(
            "bounded KV view: turned off for this model ({}): {why}",
            r.label()
        );
        counters.skip(r);
        self.off = Some(r);
    }

    /// A hook failed after the view was serving: turn it off and clear every
    /// live view, so already captured graphs rebuild identity tables. Never
    /// fails the forward. Device writes wait for an eager call when the
    /// failure happened while capturing.
    pub(crate) fn fail(
        &mut self,
        gpu: &dyn GpuBackend,
        in_graph: bool,
        counters: &Counters,
        why: &str,
    ) {
        self.disable(SkipReason::ScoreError, counters, why);
        self.clear_pending = true;
        if !in_graph {
            self.clear_all(gpu, counters);
        }
    }

    fn clear_all(&mut self, gpu: &dyn GpuBackend, counters: &Counters) {
        if !self.clear_pending {
            return;
        }
        self.clear_pending = false;
        if let Err(e) = self.clear_where(gpu, counters, |_| true) {
            tracing::warn!("bounded KV view: clearing the views failed: {e:#}");
        }
    }

    /// Allocate the buffers and the core on the model's backend (outside any
    /// capture: the scheduler calls this before a prefill). A new backend
    /// starts from an empty state; its predecessor's sweep freed the old one.
    pub(crate) fn ensure(
        &mut self,
        cfg: &KvViewConfig,
        facts: &ModelFacts<'_>,
        make_core: CoreFactory,
        counters: &Counters,
    ) {
        let gpu = facts.gpu;
        let key = backend_key(gpu);
        if self.backend != key {
            *self = Self::new();
            self.backend = key;
        }
        self.clear_all(gpu, counters);
        if self.off.is_some() || self.dev.is_some() {
            return;
        }
        if facts.heterogeneous {
            self.disable(
                SkipReason::Geometry,
                counters,
                "per-layer KV head shapes are not served",
            );
            return;
        }
        let geom = facts.geom;
        let dev = match DeviceState::new(gpu, &geom, cfg.max_selected_pages(geom.block_size)) {
            Ok(Some(d)) => d,
            Ok(None) => {
                self.disable(
                    SkipReason::NoKernel,
                    counters,
                    "kv_view_table_ada is not in this build (Ada only)",
                );
                return;
            }
            Err(e) => {
                self.disable(SkipReason::Alloc, counters, &format!("{e:#}"));
                return;
            }
        };
        let core = match make_core(gpu, geom, cfg.query_tokens, CAPTURE_SLOTS) {
            Ok(c) => c,
            Err(e) => {
                let _ = dev.free(gpu);
                self.disable(SkipReason::Alloc, counters, &format!("{e:#}"));
                return;
            }
        };
        let bytes = dev.bytes() + core.device_bytes();
        counters.set_device_bytes(bytes);
        let reserved = cfg.reserve_estimate_bytes();
        tracing::info!(
            "bounded KV view: {:.1} MiB of device buffers on this model ({:.1} MiB reserved from the \
             KV budget; the per-page key sketch is sized by the KV pool)",
            bytes as f64 / f64::from(1u32 << 20),
            reserved as f64 / f64::from(1u32 << 20),
        );
        self.geom = Some(geom);
        self.dev = Some(dev);
        self.core = Some(core);
    }

    /// Before a prefill call: drop what the seam reported freed, record the
    /// call's streams, and arm every stream that starts a new prefill.
    pub(crate) fn begin_call(
        &mut self,
        gpu: &dyn GpuBackend,
        streams: &[CallStream<'_>],
        counters: &Counters,
    ) -> Result<()> {
        self.clock += 1;
        self.call.clear();
        self.prefill_cache = None;
        for slot in std::mem::take(&mut self.freed) {
            self.drop_arm(slot);
            self.deactivate_owner(gpu, slot, counters)?;
        }
        for st in streams {
            self.call.push(CallGeom {
                slot: st.slot,
                start: st.chunk_start,
                end: st.chunk_end,
            });
            let fresh = self
                .arms
                .get(&st.slot)
                .is_none_or(|a| a.session != st.session || a.prompt_len != st.prompt.len() as u32);
            if fresh {
                self.new_arm(gpu, st, counters)?;
            }
            if let Some(a) = self.arms.get_mut(&st.slot) {
                a.last_call = self.clock;
            }
        }
        Ok(())
    }

    /// After the call.
    pub(crate) fn end_call(&mut self) {
        self.call.clear();
    }

    fn new_arm(
        &mut self,
        gpu: &dyn GpuBackend,
        st: &CallStream<'_>,
        counters: &Counters,
    ) -> Result<()> {
        self.drop_arm(st.slot);
        self.deactivate_owner(gpu, st.slot, counters)?;
        let prior = self.take_prior(st);
        let kept_span = if st.turn.query.is_empty() {
            None
        } else {
            self.arm_capture(st.slot, st.turn.query.clone())
        };
        self.arms.insert(
            st.slot,
            Arm {
                session: st.session,
                prompt_len: st.prompt.len() as u32,
                turn: st.turn.clone(),
                busy: kept_span.is_none() && !st.turn.query.is_empty(),
                kept_span,
                gap: false,
                prior,
                last_call: self.clock,
            },
        );
        Ok(())
    }

    /// Claim a capture slot, taking one from a stale arm when all are held.
    fn arm_capture(&mut self, slot: u64, span: Range<u32>) -> Option<Range<u32>> {
        let core = self.core.as_ref()?;
        if let Some(kept) = core.arm(slot, span.clone()) {
            return Some(kept);
        }
        let clock = self.clock;
        let stale = self
            .arms
            .iter()
            .filter(|(_, a)| a.kept_span.is_some() && clock - a.last_call > STALE_CALLS)
            .min_by_key(|(_, a)| a.last_call)
            .map(|(&s, _)| s)?;
        tracing::debug!(
            "bounded KV view: slot {stale}'s capture went stale; reusing it for slot {slot}"
        );
        self.drop_arm(stale);
        self.core.as_ref()?.arm(slot, span)
    }

    /// The conversation's selection when this prompt continues it; an edit
    /// (a prompt that no longer repeats the selected prefix) drops it. The
    /// pending ranking is committed unless the prompt repeats the selecting
    /// turn (a retry of the same round).
    fn take_prior(&mut self, st: &CallStream<'_>) -> Option<Conv> {
        if st.turn.kind == TurnKind::Unmarked {
            return None;
        }
        let conv = self.convs.get(&st.session)?;
        let ml = conv.match_len as usize;
        if ml > st.prompt.len() || prefix_hash(&st.prompt[..ml]) != conv.match_hash {
            self.convs.remove(&st.session);
            return None;
        }
        let mut conv = conv.clone();
        let pending = conv.pending.take();
        if st.prompt.len() as u32 != conv.prompt_len
            && let Some(p) = pending
        {
            conv.history = conv.history.commit(p);
        }
        conv.stamp = self.clock;
        self.convs.insert(st.session, conv.clone());
        Some(conv)
    }

    /// Remember a conversation's selection (least recently used dropped).
    pub(crate) fn remember(&mut self, session: u64, mut conv: Conv) {
        conv.stamp = self.clock;
        self.convs.insert(session, conv);
        while self.convs.len() > CONVERSATIONS {
            let Some(&oldest) = self
                .convs
                .iter()
                .min_by_key(|(_, c)| c.stamp)
                .map(|(k, _)| k)
            else {
                break;
            };
            self.convs.remove(&oldest);
        }
    }

    /// Forget a sequence's arm and capture.
    pub(crate) fn drop_arm(&mut self, slot: u64) {
        if self.arms.remove(&slot).is_some()
            && let Some(core) = &self.core
        {
            core.disarm(slot);
        }
    }

    /// The seam reported a published sequence freed (no backend in hand).
    pub(crate) fn note_freed(&mut self, slot: u64) {
        if !self.freed.contains(&slot) {
            self.freed.push(slot);
        }
    }

    /// The slot each logical stream of a prefill call belongs to.
    pub(crate) fn stream_slots(&self, s: &SeqBatch) -> Vec<Option<u64>> {
        let n = s.first_pos.len().min(s.new_tokens.len());
        if !s.seq_ids.is_empty() {
            return s
                .seq_ids
                .iter()
                .take(n)
                .map(|&id| (id != PAD_SEQ_ID).then_some(id))
                .collect();
        }
        if self.call.len() == 1 && n == 1 {
            return vec![Some(self.call[0].slot)];
        }
        (0..n)
            .map(|i| {
                let (start, end) = (s.first_pos[i], s.first_pos[i] + s.new_tokens[i]);
                let mut hit = self
                    .call
                    .iter()
                    .filter(|g| g.start == start && g.end == end);
                match (hit.next(), hit.next()) {
                    (Some(g), None) => Some(g.slot),
                    _ => None,
                }
            })
            .collect()
    }
}
