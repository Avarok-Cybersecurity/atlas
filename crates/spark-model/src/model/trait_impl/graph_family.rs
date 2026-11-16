// SPDX-License-Identifier: AGPL-3.0-only
// Design after Cinference (https://github.com/satellitedown/cinference, commit 383e5db),
// itself derived from NInfer by Neroued and contributors (https://github.com/Neroued/ninfer),
// Apache License 2.0: src/models/qwen3_5/program/graphs.cpp:45-104 (one executable per
// topology class, updated across profiles) and src/core/decode_graph.cpp:142-162 (update,
// instantiate on failure). Idea only; no code copied. The topology signature is the
// backend's (`GpuBackend::graph_topology_signature`, lane L08).

//! Responsibility: topology classes for the two LRU-bounded batched CUDA-graph
//! caches, behind the `graph_exec_update` row (port sprint lane L09).
//!
//! # What changes with the row on
//!
//! The batched-decode cache (`decode_graph_key.rs`, cap `16 + decode-meta
//! rows`) and the batched-verify cache (`verify_e.rs`, cap 32) key on the
//! per-row SSM slot vector, because a capture bakes each row's pool-state
//! pointers. Slot vectors churn as sequences finish, so a long serve keeps
//! missing at the cap, and today every such miss pays capture, then
//! `cuGraphInstantiate` for the new graph, then `cuGraphExecDestroy` for the
//! least-recently-used one (`speculative/verify_key.rs` priced the
//! instantiate and destroy at 23.2 ms per step on dgx2 at an 89% miss rate).
//!
//! Two slot vectors of the same width and depth shape capture the SAME
//! topology: the same kernels in the same order with different pointer
//! arguments. With the row on, a miss at the cap still captures (stream
//! capture records, it does not execute), but when the LRU victim's recorded
//! topology signature equals the new capture's, the victim's executable is
//! updated in place with `cuGraphExecUpdate` and moves to the new key; the
//! captured template is then destroyed. No instantiate, no exec destroy.
//!
//! # What does not change
//!
//! * The key, the hit test, the borrow path (`graph_borrow.rs`) and the LRU
//!   order: the victim is the entry `decode_graph_key::lru_insert_graph`
//!   would evict, chosen by the same rule, and the tick advances exactly as it
//!   does there. Row on or off, the cache holds the same keys with the same
//!   ticks after every step (`graph_family_tests.rs` drives both side by side).
//! * What a replay runs: an updated executable carries every node parameter
//!   of the new capture, which is what instantiating that capture would give.
//!   Greedy output is therefore bit-identical by construction; the GPU receipt
//!   (greedy SHA at C=1 and C=8) checks the driver keeps that promise.
//! * Below the cap nothing is shared: a miss instantiates, so a key that is
//!   still cached never loses its executable to another key.
//!
//! Any failure on the update path (unknown or different signature, the driver
//! refusing the update, an error) falls back to today's instantiate-and-evict.
//! The update is safe for the same reason today's destroy is: every batched
//! decode and verify step ends with a blocking D2H on this stream, so no
//! earlier replay of the victim is still in flight.
//!
//! Row off, or on a backend without `cuGraphExecUpdate`
//! (`GpuBackend::supports_graph_exec_update`), the call sites run their
//! pre-sprint code unchanged; the only addition there is the counter hooks
//! (`note_capture`, `note_eviction`), which move only when the row is on or
//! `AVAROK_GRAPH_COUNTERS=1` (`crate::port::graph_stats`).
//!
//! The single-sequence caches (`decode_a.rs`, `verify_b/c/c2/d/fused.rs`) are
//! keyed per slot with no cap, so they never evict at steady state and there
//! is no victim to recycle; they get the counter hook only.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};
use spark_runtime::gpu::{GpuBackend, GraphHandle, GraphTemplate};

use super::super::types::TransformerModel;
use crate::port::graph_stats::{GRAPH_COUNTERS, GraphCounters, counting_enabled};

/// The batched caches' shape: `key -> (exec, last-use tick)` plus the tick
/// counter (`types.rs` `batch_decode_graphs` / `verify_batched_graphs`).
pub(super) type LruGraphCache = (HashMap<Vec<u32>, (GraphHandle, u64)>, u64);

/// Which batched cache a capture belongs to (cap and signature table).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FamilyCache {
    /// `batch_decode_graphs` (`decode_a2.rs`).
    BatchDecode,
    /// `verify_batched_graphs` (`verify_e.rs`).
    BatchVerify,
}

/// How a capture ended up as an executable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Finish {
    /// Instantiated from the capture (below the cap, or the update path declined).
    Instantiated,
    /// The LRU victim's executable was updated in place and moved to the new key.
    Updated,
    /// The backend returned a null graph (the mock convention: the work already
    /// ran eagerly); nothing to insert or launch.
    Empty,
}

/// Whether the batched caches take the exec-update path: the row is on AND the
/// backend implements `cuGraphExecUpdate`. Row off returns before touching the
/// backend. Logs once which way it went.
pub(super) fn exec_update_active(gpu: &dyn GpuBackend) -> bool {
    if !crate::layers::ops::port_rows::graph_exec_update::row() {
        return false;
    }
    let supported = gpu.supports_graph_exec_update();
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        if supported {
            tracing::info!(
                "graph_exec_update on: a full batched decode or verify graph cache updates its \
                 same-topology LRU executable in place instead of instantiating"
            );
        } else {
            tracing::warn!(
                "graph_exec_update is on but this backend has no cuGraphExecUpdate; batched \
                 graphs instantiate as before"
            );
        }
    });
    supported
}

/// Counter hook for a capture that ended through the backend's plain
/// `end_capture` (which instantiates): one capture, one instantiation. A null
/// handle (the mock) counts nothing.
pub(super) fn note_capture(graph: GraphHandle) {
    if graph.0 != 0 && counting_enabled() {
        GRAPH_COUNTERS.add_capture();
        GRAPH_COUNTERS.add_instantiation();
    }
}

/// Counter hook for an LRU entry removed at the cap on the row-off path.
pub(super) fn note_eviction() {
    if counting_enabled() {
        GRAPH_COUNTERS.add_eviction();
    }
}

/// The driver operations the family path needs, so the cache state machine
/// runs against a fake in tests. Implemented for `dyn GpuBackend` below.
pub(super) trait FamilyOps {
    /// End the stream capture and keep the graph, without instantiating it.
    fn close_capture(&self, stream: u64) -> Result<GraphTemplate>;
    /// Node count plus a hash of node types and kernel functions.
    fn topology(&self, t: &GraphTemplate) -> Result<u64>;
    /// Update `exec` in place from `t`; `Ok(false)` when the driver refuses.
    fn exec_update(&self, exec: GraphHandle, t: &GraphTemplate) -> Result<bool>;
    /// Instantiate an executable from `t` (`t` stays alive).
    fn instantiate_template(&self, t: &GraphTemplate) -> Result<GraphHandle>;
    /// Destroy a template.
    fn drop_template(&self, t: GraphTemplate) -> Result<()>;
    /// Destroy an executable.
    fn drop_exec(&self, exec: GraphHandle) -> Result<()>;
}

impl FamilyOps for dyn GpuBackend {
    fn close_capture(&self, stream: u64) -> Result<GraphTemplate> {
        self.end_capture_graph(stream)
    }
    fn topology(&self, t: &GraphTemplate) -> Result<u64> {
        self.graph_topology_signature(t)
    }
    fn exec_update(&self, exec: GraphHandle, t: &GraphTemplate) -> Result<bool> {
        self.graph_exec_update(exec, t)
    }
    fn instantiate_template(&self, t: &GraphTemplate) -> Result<GraphHandle> {
        self.graph_instantiate(t)
    }
    fn drop_template(&self, t: GraphTemplate) -> Result<()> {
        self.graph_template_destroy(t)
    }
    fn drop_exec(&self, exec: GraphHandle) -> Result<()> {
        self.destroy_graph(exec)
    }
}

/// The key `decode_graph_key::lru_insert_graph` would evict to insert `key`:
/// the least-recently-used entry, and only when the cache is at `cap` and
/// `key` is new. Ticks are unique (every touch and insert takes a fresh one),
/// so the minimum is unambiguous.
pub(super) fn lru_victim(cache: &LruGraphCache, cap: usize, key: &[u32]) -> Option<Vec<u32>> {
    if cache.0.len() < cap || cache.0.contains_key(key) {
        return None;
    }
    cache
        .0
        .iter()
        .min_by_key(|(_, entry)| entry.1)
        .map(|(k, _)| k.clone())
}

/// Close the capture on `stream` and install it under `key`: update the LRU
/// victim in place when it has the same topology, else instantiate and insert
/// with `lru_insert_graph` (evicting and destroying the victim as today).
///
/// `sigs` maps executable handle to topology signature for this cache. The
/// template is always destroyed before returning. Returns the executable to
/// launch and how it was produced; the caller launches it.
#[allow(clippy::too_many_arguments)]
pub(super) fn finish_capture<O: FamilyOps + ?Sized>(
    ops: &O,
    sigs: &mut HashMap<u64, u64>,
    cache: &mut LruGraphCache,
    cap: usize,
    key: Vec<u32>,
    stream: u64,
    counters: &GraphCounters,
    label: &str,
) -> Result<(GraphHandle, Finish)> {
    let template = ops.close_capture(stream)?;
    if template.0 == 0 {
        return Ok((GraphHandle(0), Finish::Empty));
    }
    counters.add_capture();
    let sig = match ops.topology(&template) {
        Ok(s) => Some(s),
        Err(e) => {
            tracing::warn!("{label} graph topology signature failed ({e:#}); instantiating");
            None
        }
    };
    let victim = lru_victim(cache, cap, &key);
    if let (Some(sig), Some(vkey)) = (sig, victim.as_ref()) {
        let vexec = cache.0[vkey].0;
        if sigs.get(&vexec.0) == Some(&sig) {
            match ops.exec_update(vexec, &template) {
                Ok(true) => {
                    // The victim's executable now runs the new capture: move it
                    // to the new key, with the tick `lru_insert_graph` would give.
                    cache.0.remove(vkey);
                    cache.1 += 1;
                    let tick = cache.1;
                    cache.0.insert(key, (vexec, tick));
                    counters.add_update();
                    counters.add_eviction();
                    release_template(ops, template, label);
                    return Ok((vexec, Finish::Updated));
                }
                Ok(false) => {
                    counters.add_update_failure();
                    tracing::debug!("{label} graph exec update refused; instantiating");
                }
                Err(e) => {
                    counters.add_update_failure();
                    tracing::warn!("{label} graph exec update failed ({e:#}); instantiating");
                }
            }
        }
    }
    let exec = match ops.instantiate_template(&template) {
        Ok(g) => g,
        Err(e) => {
            release_template(ops, template, label);
            return Err(e);
        }
    };
    release_template(ops, template, label);
    if exec.0 == 0 {
        // A non-null capture whose instantiation is null would replay nothing:
        // fail the step loudly rather than skip its forward.
        bail!("{label} graph instantiate returned a null executable");
    }
    counters.add_instantiation();
    match sig {
        Some(s) => sigs.insert(exec.0, s),
        // A recycled handle value must not inherit a dead executable's class.
        None => sigs.remove(&exec.0),
    };
    if victim.is_some() {
        counters.add_eviction();
    }
    for old in super::decode_graph_key::lru_insert_graph(cache, cap, key, exec) {
        sigs.remove(&old.0);
        if let Err(e) = ops.drop_exec(old) {
            tracing::warn!("{label} graph evict: {e:#}");
        }
    }
    prune_signatures(sigs, cache, cap);
    Ok((exec, Finish::Instantiated))
}

fn release_template<O: FamilyOps + ?Sized>(ops: &O, t: GraphTemplate, label: &str) {
    if let Err(e) = ops.drop_template(t) {
        tracing::warn!("{label} graph template destroy: {e:#}");
    }
}

/// Drop entries for executables no longer in `cache` (destroyed by a LoRA
/// drain or a stale-slot removal outside this path) once the table holds more
/// than twice the cap.
pub(super) fn prune_signatures(sigs: &mut HashMap<u64, u64>, cache: &LruGraphCache, cap: usize) {
    if sigs.len() <= 2 * cap.max(1) {
        return;
    }
    let live: HashSet<u64> = cache.0.values().map(|(g, _)| g.0).collect();
    sigs.retain(|h, _| live.contains(h));
}

impl TransformerModel {
    /// Row-on end of a batched capture: close it, install it in `cache` under
    /// `key` through [`finish_capture`], and launch it. Returns whether the
    /// graph went into the cache (the verify path's `Capture` outcome).
    ///
    /// With no cache or no key (not reachable from the two call sites, whose
    /// guard and key are both `Some` whenever they capture) this is today's
    /// plain `end_capture` and launch.
    pub(super) fn graph_family_finish(
        &self,
        which: FamilyCache,
        cache: Option<&mut LruGraphCache>,
        key: Option<Vec<u32>>,
        stream: u64,
    ) -> Result<bool> {
        let (Some(cache), Some(key)) = (cache, key) else {
            let graph = self.gpu.end_capture(stream)?;
            note_capture(graph);
            if graph.0 != 0 {
                self.gpu.launch_graph(graph, stream)?;
            }
            return Ok(false);
        };
        let (cap, sigs, label) = match which {
            FamilyCache::BatchDecode => (
                super::decode_graph_key::batch_decode_graph_cap(self.buffers.decode_meta().rows()),
                &self.port.graph_family.decode_sigs,
                "batched-decode",
            ),
            FamilyCache::BatchVerify => (
                super::verify_e2::VERIFY_BATCHED_GRAPH_CAP,
                &self.port.graph_family.verify_sigs,
                "batched-verify",
            ),
        };
        let finished = finish_capture(
            self.gpu.as_ref(),
            &mut sigs.lock(),
            cache,
            cap,
            key,
            stream,
            &GRAPH_COUNTERS,
            label,
        );
        let (exec, how) = match finished {
            Ok(v) => v,
            Err(e) => {
                // A failed close leaves the stream recording; release it so the
                // next op on this stream is not poisoned (no-op otherwise).
                self.gpu.abort_capture_if_active(stream);
                return Err(e);
            }
        };
        if how == Finish::Empty {
            return Ok(false);
        }
        tracing::info!(
            "Captured CUDA graph for {label} ({}, {} live keys)",
            if how == Finish::Updated {
                "exec updated in place"
            } else {
                "instantiated"
            },
            cache.0.len()
        );
        self.gpu.launch_graph(exec, stream)?;
        Ok(true)
    }
}

#[cfg(test)]
#[path = "graph_family_tests.rs"]
mod tests;
