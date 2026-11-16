// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm: RetroInfer, arXiv 2505.02922 (VLDB 2026).

//! Responsibility: the RetroInfer `KvSelector` of the sparse-attention seam,
//! registered by `api::apply` when the `retro_attn` row or `--retro-shadow` is
//! on.
//!
//! Per decode or verify layer call: `on_kv_written` detects a new step, matches
//! the rows to live sequences and readies each served sequence's index for the
//! layer (cache install, build, append or update; `runtime.rs`); `plan`
//! samples shadow mode, then returns `Custom` (the retro attention,
//! `attend.rs`) when every sequence of the step is served, and `Dense`
//! otherwise, with the reason counted. Prefill chunks always run dense; when
//! the scheduler publishes the last chunk the index is built at the end of the
//! prefill, otherwise at the sequence's first decode step. Everything here
//! runs on eager steps: while a CUDA graph is captured the plan is `Dense`
//! (exact), and the selector vetoes decode graphs while it is registered
//! (`api::vetoes_decode_graphs`).
//!
//! Errors: a kernel that fails (or a module that is absent on this target)
//! turns retro attention off for the rest of the serve with one error line;
//! the decode keeps running dense.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::attend::RetroAttention;
use super::config::{HEAD_DIM, MAX_VERIFY_ROWS, RetroParams};
use super::metrics::{Fallback, GLOBAL, Metrics};
use super::runtime::{NotReady, Runtime};
use super::shadow::Shadow;
use crate::layers::qwen3_attention::sparse_seam::{
    KvSelector, SeamCtx, SeamPhase, SeqBatch, SparsePlan,
};

/// The retro selector.
pub struct RetroSelector {
    params: RetroParams,
    /// Retro attention replaces dense attention (false: shadow mode only).
    output_on: bool,
    shadow: Option<Shadow>,
    rt: Arc<Mutex<Runtime>>,
    failed: Arc<AtomicBool>,
    metrics: &'static Metrics,
    attention: Arc<RetroAttention>,
}

impl RetroSelector {
    pub fn new(params: RetroParams, output_on: bool) -> Self {
        Self::with_metrics(params, output_on, &GLOBAL)
    }

    /// A selector reporting into `metrics` (tests pass their own).
    pub fn with_metrics(params: RetroParams, output_on: bool, metrics: &'static Metrics) -> Self {
        let rt = Arc::new(Mutex::new(Runtime::default()));
        let failed = Arc::new(AtomicBool::new(false));
        let shadow = params.shadow.then(|| Shadow::new(params.clone(), metrics));
        Self {
            attention: Arc::new(RetroAttention {
                rt: rt.clone(),
                metrics,
                failed: failed.clone(),
            }),
            params,
            output_on,
            shadow,
            rt,
            failed,
            metrics,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Runtime> {
        self.rt
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn fail(&self, what: &str, e: &anyhow::Error) {
        if !self.failed.swap(true, Ordering::Relaxed) {
            tracing::error!(
                "retro: {what} failed ({e:#}); retro attention is off for the rest of the serve"
            );
        }
    }

    /// Whether retro work runs for this call at all (device side).
    fn device_live(&self) -> bool {
        self.output_on && !self.failed.load(Ordering::Relaxed)
    }

    fn decode_written(&self, c: &SeamCtx, s: &SeqBatch) -> Result<()> {
        let mut rt = self.lock();
        let new_step = rt.last_layer.is_none_or(|l| c.attn_layer <= l);
        rt.last_layer = Some(c.attn_layer);
        if self.device_live() {
            rt.ensure_device(&self.params, c, s, self.shadow.as_ref())?;
        }
        if new_step {
            rt.begin_step(s, self.shadow.as_ref());
        }
        if !self.device_live() || !rt.serves_shape(c) || c.kv.head_dim != HEAD_DIM {
            return Ok(());
        }
        for g in rt.groups.clone() {
            if g.rows > MAX_VERIFY_ROWS as usize {
                continue;
            }
            // A not-ready index (short sequence, budget) is not an error here: the
            // plan step sees no ready index for the group and serves it densely.
            let _not_ready = rt.prepare(
                &self.params,
                c,
                g.uid,
                g.len,
                &g.table,
                self.metrics,
                self.shadow.as_ref(),
            )?;
        }
        self.metrics.set_index_bytes(rt.used_bytes());
        Ok(())
    }

    /// Why the step cannot run retro attention for this layer, if it cannot.
    fn refuse(&self, rt: &Runtime, c: &SeamCtx, s: &SeqBatch) -> Option<Fallback> {
        if !rt.serves_shape(c)
            || c.kv.head_dim != HEAD_DIM
            || s.block_table.is_null()
            || s.seq_lens.is_null()
        {
            return Some(Fallback::Shape);
        }
        if rt.groups.is_empty() || rt.groups.iter().map(|g| g.rows).sum::<usize>() != s.n {
            return Some(Fallback::Shape);
        }
        let layer = u32::try_from(c.attn_layer).unwrap_or(u32::MAX);
        let mut why: Vec<Fallback> = Vec::new();
        for g in &rt.groups {
            if g.rows > MAX_VERIFY_ROWS as usize {
                why.push(Fallback::Shape);
            } else if rt.index_of(g.uid, layer).is_some() {
                continue;
            } else if super::update::initial_end(g.len, &self.params, c.kv.block_size).is_none() {
                why.push(Fallback::ShortContext);
            } else if rt
                .seqs
                .by_uid(g.uid)
                .is_some_and(|i| rt.seqs.get(i).dense_only)
            {
                why.push(Fallback::Budget);
            } else {
                why.push(Fallback::NoIndex);
            }
        }
        match why.len() {
            0 => None,
            n if n == rt.groups.len() => why.first().copied(),
            _ => Some(Fallback::MixedBatch),
        }
    }

    fn sample_shadow(&self, rt: &Runtime, c: &SeamCtx, s: &SeqBatch, q: DevicePtr) {
        let Some(sh) = self.shadow.as_ref() else {
            return;
        };
        let Some(g) = rt.groups.first() else {
            return;
        };
        let layer = u32::try_from(c.attn_layer).unwrap_or(u32::MAX);
        if !sh.due(rt.step, layer, g.len) {
            return;
        }
        // Row 0 of the step attends `[0, len - (rows - 1))`.
        let len = g.len - u32::try_from(g.rows - 1).unwrap_or(0);
        let q0 = q.offset(g.first_row * s.q_row_stride as usize * 2);
        if let Err(e) = sh.sample(
            c.gpu,
            &c.kv,
            &g.table,
            len,
            q0,
            c.num_q_heads,
            c.inv_sqrt_d,
            g.uid,
            layer,
            c.stream,
        ) {
            tracing::warn!("retro shadow: sample on attn layer {layer} failed ({e:#}); skipped");
        }
    }
}

impl KvSelector for RetroSelector {
    fn name(&self) -> &'static str {
        "retro"
    }

    fn reserved_bytes(&self) -> u64 {
        if self.output_on {
            u64::from(self.params.index_cache_mb) << 20
        } else {
            0
        }
    }

    fn on_kv_written(
        &self,
        c: &SeamCtx,
        s: &SeqBatch,
        first_pos: &[u32],
        n_new: &[u32],
    ) -> Result<()> {
        if c.in_graph {
            return Ok(());
        }
        match c.phase {
            SeamPhase::Prefill => {
                let mut rt = self.lock();
                let shadow = self.shadow.as_ref();
                rt.prefill_written(s, first_pos, n_new, c.kv.block_size, shadow);
                Ok(())
            }
            SeamPhase::Decode | SeamPhase::Verify => {
                if let Err(e) = self.decode_written(c, s) {
                    self.fail("index preparation", &e);
                }
                Ok(())
            }
        }
    }

    fn plan(&self, c: &SeamCtx, s: &SeqBatch, q: DevicePtr) -> Result<SparsePlan> {
        if c.in_graph {
            if self.device_live() {
                self.metrics.count_fallback(Fallback::Graph);
            }
            return Ok(SparsePlan::Dense);
        }
        if c.phase == SeamPhase::Prefill {
            return Ok(SparsePlan::Dense);
        }
        let rt = self.lock();
        self.sample_shadow(&rt, c, s, q);
        if !self.device_live() {
            if self.output_on {
                self.metrics.count_fallback(Fallback::Kernel);
            }
            return Ok(SparsePlan::Dense);
        }
        if let Some(why) = self.refuse(&rt, c, s) {
            self.metrics.count_fallback(why);
            return Ok(SparsePlan::Dense);
        }
        Ok(SparsePlan::Custom(self.attention.clone()))
    }

    fn on_prefill_layer_done(&self, c: &SeamCtx, s: &SeqBatch, last_chunk: bool) -> Result<()> {
        if !last_chunk
            || c.in_graph
            || !self.device_live()
            || s.n != 1
            || s.host_block_tables.len() != 1
            || c.kv.head_dim != HEAD_DIM
        {
            return Ok(());
        }
        let res = (|| -> Result<()> {
            let mut rt = self.lock();
            if s.max_blocks_per_seq == 0 {
                return Ok(());
            }
            rt.ensure_device(&self.params, c, s, self.shadow.as_ref())?;
            if !rt.serves_shape(c) {
                return Ok(());
            }
            let len = s.host_seq_lens.first().copied().unwrap_or(0);
            let table = s.host_block_tables[0];
            let id = s.seq_ids.first().copied();
            let step = rt.step;
            let sight = rt.seqs.sight(id, table, len, step, &[]);
            let uid = rt.seqs.get(sight.index()).uid;
            match rt.prepare(
                &self.params,
                c,
                uid,
                len,
                table,
                self.metrics,
                self.shadow.as_ref(),
            )? {
                Ok(()) | Err(NotReady::Short) | Err(NotReady::Budget) => Ok(()),
            }
        })();
        if let Err(e) = res {
            self.fail("index build at prefill", &e);
        }
        Ok(())
    }

    fn on_seq_free(&self, seq_id: u64) {
        let mut rt = self.lock();
        if let Some(s) = rt.seqs.remove_id(seq_id) {
            rt.release_seq(s, self.shadow.as_ref());
        }
    }
}
