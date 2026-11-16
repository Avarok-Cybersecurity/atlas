// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the parts of [`KvHostTier`] beyond the 3.7.5 contract that the glue (lane
//! L26b) needs to drive the tier: the per-step write-back budget, publishing finished write-backs,
//! the lazy-store drain, the restore cost gate and its prefill feed, the in-flight cap, stream
//! and namespace setup, accessors, the stats snapshot and teardown.

use super::cost::GateDecision;
use super::fingerprint::BlockKey;
use super::lazy::{DrainSignals, EvictionView};
use super::{KvHostTier, TierConfig, TierOptions, TierSnapshot};
use crate::gpu::GpuBackend;

impl KvHostTier {
    /// Mark the start of a scheduler step: resets the per-step write-back byte budget. A tier
    /// whose owner never calls this treats every store call as its own step.
    pub fn begin_step(&self) {
        let mut inner = self.inner.lock();
        inner.step_mode = true;
        inner.step_spent = 0;
    }

    /// Publish every write-back whose D2H completed and return the keys of every finished batch
    /// (published or failed), so the caller drops the source references it holds for them.
    /// Also finishes completed restores and reclaims released or expired ones.
    pub fn poll_writebacks(&self, gpu: &dyn GpuBackend) -> Vec<BlockKey> {
        let mut inner = self.inner.lock();
        inner.sweep_restores(&self.sh, gpu);
        let done = inner.poll_writebacks(&self.sh, gpu);
        inner.publish_gauges(&self.sh);
        done
    }

    /// Lazy store mode: start buffered stores whose blocks come within the eviction horizon.
    /// Returns the keys whose copy started (take a source reference for each, drop it when the
    /// key comes back from [`Self::poll_writebacks`]). A no-op in eager mode.
    pub fn drain_lazy(
        &self,
        gpu: &dyn GpuBackend,
        signals: DrainSignals,
        view: &dyn EvictionView,
    ) -> anyhow::Result<Vec<BlockKey>> {
        let mut inner = self.inner.lock();
        let started = inner.drain_lazy(&self.sh, gpu, signals, view);
        inner.publish_gauges(&self.sh);
        started
    }

    /// The restore cost gate: restore `tokens` tokens held in `blocks`
    /// blocks, or recompute them? Counts declines on `/metrics`.
    pub fn gate(&self, tokens: usize, blocks: usize) -> GateDecision {
        let inner = self.inner.lock();
        let bytes = blocks as u64 * self.sh.block_bytes as u64;
        let d = inner.cost.decide(
            tokens,
            bytes,
            self.sh.cfg.restore_gate_ratio,
            self.sh.cfg.min_tokens,
        );
        self.sh.stats.set_cost_valid(inner.cost.valid());
        if !d.restores() {
            self.sh.stats.add_gate_declined();
        }
        d
    }

    /// Feed one measured prefill (`tokens` computed in `secs`) to the cost model.
    pub fn observe_prefill(&self, tokens: usize, secs: f64) {
        let mut inner = self.inner.lock();
        inner.cost.observe_prefill(tokens as f64, secs);
        self.sh.stats.set_cost_valid(inner.cost.valid());
    }

    /// Whether the cost model passed its held-out check (and so gates restores).
    pub fn cost_model_valid(&self) -> bool {
        self.inner.lock().cost.valid()
    }

    /// How many blocks a new restore may lease right now (the in-flight cap). When nothing is
    /// leased at all, one block is always allowed, so a tiny budget never blocks restores.
    pub fn max_restore_blocks(&self) -> usize {
        let inner = self.inner.lock();
        if self.sh.block_bytes == 0 {
            return 0;
        }
        let leased = inner.index.leased_bytes();
        let left = self.sh.inflight_cap().saturating_sub(leased);
        let n = (left / self.sh.block_bytes as u64) as usize;
        if leased == 0 { n.max(1) } else { n }
    }

    /// The stream the stored KV is written on (default: `gpu.default_stream()`).
    pub fn set_source_stream(&self, stream: u64) {
        self.inner.lock().source_stream = Some(stream);
    }

    /// The model fingerprint written into cold-store records and used to adopt them.
    pub fn set_namespace(&self, ns: u64) {
        self.inner.lock().namespace = ns;
    }

    pub fn config(&self) -> &TierConfig {
        &self.sh.cfg
    }

    pub fn options(&self) -> &TierOptions {
        &self.sh.opts
    }

    pub fn block_bytes(&self) -> usize {
        self.sh.block_bytes
    }

    pub fn copy_stream(&self) -> u64 {
        self.sh.copy_stream
    }

    pub fn min_tokens(&self) -> u32 {
        self.sh.cfg.min_tokens
    }

    /// Counters and gauges of this tier (the `/metrics` values plus internal state).
    pub fn snapshot(&self) -> TierSnapshot {
        let inner = self.inner.lock();
        self.sh.stats.snapshot(
            inner.index.len(),
            inner.index.staging_bytes(),
            inner.index.leased_bytes(),
            inner.lazy.pending_blocks(),
        )
    }

    /// Quiesce before teardown: wait for the copy stream, finish every write-back and restore,
    /// and return every host run. Returns the keys of every write-back that finished or was
    /// abandoned here (drop their source references). Call it before the last handle (and the
    /// pool) is dropped; the tier serves nothing afterwards.
    pub fn shutdown(&self, gpu: &dyn GpuBackend) -> Vec<BlockKey> {
        if let Err(e) = gpu.synchronize(self.sh.copy_stream) {
            tracing::warn!("kv_host_tier shutdown: copy stream sync failed: {e:#}");
        }
        let mut inner = self.inner.lock();
        inner.sweep_restores(&self.sh, gpu);
        let mut done = inner.poll_writebacks(&self.sh, gpu);
        // After the sync nothing should still run; whatever does is abandoned, not reused.
        let abandoned: Vec<_> = inner.writebacks.drain(..).collect();
        for mut b in abandoned {
            b.timer.discard(gpu);
            let _ = gpu.destroy_event(b.event);
            done.extend(b.keys);
        }
        let running = inner.running_restores();
        if running > 0 {
            tracing::warn!(
                "kv_host_tier shutdown: {running} restore(s) still running after the sync"
            );
        }
        inner.restores.clear();
        inner.lazy.clear();
        inner.free_all(&self.sh);
        if let Some(f) = inner.fence.take() {
            let _ = gpu.destroy_event(f);
        }
        inner.events.destroy_all(gpu);
        inner.publish_gauges(&self.sh);
        done
    }
}

impl Drop for KvHostTier {
    fn drop(&mut self) {
        // Without a backend we cannot wait for in-flight copies, so only quiescent runs go back
        // to the arena; a run a DMA may still touch is leaked rather than handed out again.
        let inner = self.inner.get_mut();
        let leaked = inner.free_quiescent(&self.sh);
        if leaked > 0 {
            tracing::warn!(
                "kv_host_tier dropped with {leaked} block(s) in flight; call shutdown() first"
            );
        }
    }
}
