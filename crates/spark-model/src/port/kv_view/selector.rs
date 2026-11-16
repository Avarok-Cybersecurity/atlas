// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the bounded view's `KvSelector` (port lane L32), the one
//! object the sparse-attention seam (lane L29) calls, and
//! the entry points the scheduler reaches through `ModelPortExt`.
//!
//! Seam hooks:
//! - `on_kv_written`: fold the written K rows into the per-page sketch (every
//!   phase; graph-safe in decode);
//! - `on_query` (prefill): capture the final user message's query rows;
//! - `plan` (decode, verify): always a `View` over the fixed arena once the
//!   buffers exist, rebuilt on the device each step, so captured graphs
//!   never change topology; rows without a view read their real table;
//! - `plan_prefill` (paged chunks of a continuing conversation): the
//!   conversation's view plus every page from its tail on;
//! - `on_seq_free`: forget the sequence.
//!
//! Every hook takes the lock only on eager steps and at capture time; a
//! replayed graph runs no host code.

use std::sync::{Mutex, MutexGuard};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::config::KvViewConfig;
use super::core_api::{Geom, L31Core, ViewCore};
use super::metrics::{self, Counters};
use super::state::{CoreFactory, Inner};
use super::{CallStream, ModelFacts, ViewOutcome, ViewSpec};
use crate::layers::qwen3_attention::sparse_seam::{
    KvSelector, PrefillPlan, SeamCtx, SeamPhase, SeqBatch, SparsePlan,
};

fn l31_core(
    gpu: &dyn GpuBackend,
    geom: Geom,
    query_tokens: u32,
    slots: usize,
) -> Result<Box<dyn ViewCore>> {
    Ok(Box::new(L31Core::new(gpu, geom, query_tokens, slots)?))
}

/// The bounded view's selector.
pub struct KvViewSelector {
    cfg: KvViewConfig,
    inner: Mutex<Inner>,
    counters: &'static Counters,
    make_core: CoreFactory,
}

impl KvViewSelector {
    /// The serve's selector (L31's core, the global counters).
    pub fn new(cfg: KvViewConfig) -> Self {
        Self::with_core(cfg, l31_core, &metrics::GLOBAL)
    }

    /// A selector over another core (tests).
    pub(crate) fn with_core(
        cfg: KvViewConfig,
        make_core: CoreFactory,
        counters: &'static Counters,
    ) -> Self {
        Self {
            cfg,
            inner: Mutex::new(Inner::new()),
            counters,
            make_core,
        }
    }

    pub fn config(&self) -> &KvViewConfig {
        &self.cfg
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The state, for the flow tests.
    #[cfg(test)]
    pub(crate) fn inner_for_test(&self) -> MutexGuard<'_, Inner> {
        self.lock()
    }

    /// Before a prefill call: allocate on first use, then arm its streams.
    pub fn begin_call(&self, facts: &ModelFacts<'_>, streams: &[CallStream<'_>]) -> Result<()> {
        let mut inner = self.lock();
        inner.ensure(&self.cfg, facts, self.make_core, self.counters);
        if !inner.ready(facts.gpu) {
            return Ok(());
        }
        inner.begin_call(facts.gpu, streams, self.counters)
    }

    /// After the prefill call.
    pub fn end_call(&self) {
        self.lock().end_call();
    }

    /// Select a view for `slot` (`Some`) or drop its arm and view (`None`).
    pub fn set_kv_view(
        &self,
        gpu: &dyn GpuBackend,
        slot: u64,
        spec: Option<ViewSpec<'_>>,
    ) -> Result<ViewOutcome> {
        let mut inner = self.lock();
        match spec {
            Some(spec) => inner.select(&self.cfg, gpu, slot, &spec, self.counters),
            None => {
                inner.drop_arm(slot);
                inner.deactivate_owner(gpu, slot, self.counters)?;
                Ok(ViewOutcome::Discarded)
            }
        }
    }

    /// A sequence is freed: drop its arm and the views bound to its blocks.
    pub fn release_sequence(&self, gpu: &dyn GpuBackend, slot: u64, table: &[u32], len: u32) {
        let mut inner = self.lock();
        inner.drop_arm(slot);
        if let Err(e) = inner.release_table(gpu, table, len, self.counters) {
            tracing::warn!("bounded KV view: releasing slot {slot}'s view failed: {e:#}");
        }
    }
}

impl KvSelector for KvViewSelector {
    fn name(&self) -> &'static str {
        "kv_view"
    }

    fn reserved_bytes(&self) -> u64 {
        self.cfg.reserve_estimate_bytes()
    }

    fn on_kv_written(
        &self,
        c: &SeamCtx,
        s: &SeqBatch,
        first_pos: &[u32],
        n_new: &[u32],
    ) -> Result<()> {
        self.lock()
            .record_written(c, s, first_pos, n_new, self.counters)
    }

    fn on_query(
        &self,
        c: &SeamCtx,
        s: &SeqBatch,
        q_pre_rope: Option<DevicePtr>,
        q: DevicePtr,
    ) -> Result<()> {
        if c.phase != SeamPhase::Prefill || c.in_graph {
            return Ok(());
        }
        self.lock().capture(c, s, q_pre_rope, q, self.counters)
    }

    fn plan(&self, c: &SeamCtx, s: &SeqBatch, _q: DevicePtr) -> Result<SparsePlan> {
        self.lock().decode_plan(c, s, self.counters)
    }

    fn plan_prefill(&self, c: &SeamCtx, s: &SeqBatch, _q: DevicePtr) -> Result<PrefillPlan> {
        self.lock().prefill_plan(c, s)
    }

    fn on_seq_free(&self, seq_id: u64) {
        let mut inner = self.lock();
        inner.drop_arm(seq_id);
        inner.note_freed(seq_id);
    }
}
