// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the mock backend's emulation of the port sprint's `GpuBackend`
//! extensions (contract 3.7.1), so lanes test their state machines without a GPU.
//! A child of `mock`, so it reads the mock's private launch and allocation tables.
//!
//! * **Graphs.** `begin_capture` marks the launch log; ending the capture turns the
//!   launches recorded on that stream since the mark into a template's node list
//!   (every node a kernel node whose function is the launch's handle). An exec
//!   remembers the signature it was built from; `graph_exec_update` succeeds only
//!   when the template's signature matches, as `cuGraphExecUpdate` does for a
//!   same-topology graph. Unknown or destroyed handles are errors, so a
//!   double-destroy or a use-after-destroy fails a test instead of passing.
//! * **Batched copies.** Off by default (`supports_copy_batch` false, the trait's
//!   per-copy loop runs). `set_copy_batch_supported(true)` makes one call move
//!   every entry under one lock and count as ONE batch, so a test can prove the
//!   batched shape while comparing bytes against the loop.
//! * **Events.** By default every event is complete at once, as before.
//!   `hold_events(true)` keeps each recorded event pending until
//!   `complete_event` / `complete_all_events` / `event_synchronize`, which is how
//!   a test drives "D2H still in flight" states. `set_unique_events(true)` makes
//!   `create_event` hand out distinct ids (default: 0, the trait default).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use anyhow::{Result, anyhow, bail};
use parking_lot::Mutex;

use super::MockGpuBackend;
use super::mock_mem::{find_alloc, find_alloc_mut};
use crate::gpu::port_ext::{self, NODE_TYPE_KERNEL, topology_signature};
use crate::gpu::{CopyEntry, GraphHandle, GraphTemplate};

/// First template/exec id. Far from 0 so a stale `GraphHandle(0)` from the old
/// `end_capture` default is never mistaken for a tracked exec.
const FIRST_GRAPH_ID: u64 = 0x6A00_0000;
/// First copy stream id: non-zero (never the legacy stream) and distinct per call.
const FIRST_COPY_STREAM: u64 = 0x5100;

pub(super) struct MockPortState {
    /// Launch-log length at `begin_capture`, per stream.
    capture_start: Mutex<HashMap<u64, usize>>,
    /// Live templates: id -> `(node type, function)` list.
    templates: Mutex<HashMap<u64, Vec<(u32, u64)>>>,
    /// Live execs: id -> signature of the graph it currently holds.
    execs: Mutex<HashMap<u64, u64>>,
    next_graph_id: AtomicU64,
    instantiations: AtomicUsize,
    updates: AtomicUsize,
    update_rejects: AtomicUsize,
    uploads: AtomicUsize,
    graph_update_supported: AtomicBool,
    fail_next_update: AtomicBool,
    copy_batch_supported: AtomicBool,
    batch_calls: AtomicUsize,
    batch_entries: AtomicUsize,
    batch_streams: Mutex<Vec<u64>>,
    hold_events: AtomicBool,
    unique_events: AtomicBool,
    pending_events: Mutex<HashSet<u64>>,
    next_event: AtomicU64,
    next_copy_stream: AtomicU64,
}

impl Default for MockPortState {
    fn default() -> Self {
        Self {
            capture_start: Mutex::new(HashMap::new()),
            templates: Mutex::new(HashMap::new()),
            execs: Mutex::new(HashMap::new()),
            next_graph_id: AtomicU64::new(FIRST_GRAPH_ID),
            instantiations: AtomicUsize::new(0),
            updates: AtomicUsize::new(0),
            update_rejects: AtomicUsize::new(0),
            uploads: AtomicUsize::new(0),
            graph_update_supported: AtomicBool::new(true),
            fail_next_update: AtomicBool::new(false),
            copy_batch_supported: AtomicBool::new(false),
            batch_calls: AtomicUsize::new(0),
            batch_entries: AtomicUsize::new(0),
            batch_streams: Mutex::new(Vec::new()),
            hold_events: AtomicBool::new(false),
            unique_events: AtomicBool::new(false),
            pending_events: Mutex::new(HashSet::new()),
            next_event: AtomicU64::new(1),
            next_copy_stream: AtomicU64::new(FIRST_COPY_STREAM),
        }
    }
}

// ── Trait-override bodies (called from `impl GpuBackend for MockGpuBackend`) ──
impl MockGpuBackend {
    pub(super) fn port_begin_capture(&self, stream: u64) {
        let mark = self.launches.lock().len();
        self.port.capture_start.lock().insert(stream, mark);
    }

    pub(super) fn port_end_capture_graph(&self, stream: u64) -> Result<GraphTemplate> {
        let start = self
            .port
            .capture_start
            .lock()
            .remove(&stream)
            .ok_or_else(|| anyhow!("end_capture_graph: stream {stream} is not capturing"))?;
        let nodes: Vec<(u32, u64)> = {
            let launches = self.launches.lock();
            launches
                .get(start..)
                .unwrap_or(&[])
                .iter()
                .filter(|l| l.stream == stream)
                .map(|l| (NODE_TYPE_KERNEL, l.func))
                .collect()
        };
        let id = self.port.next_graph_id.fetch_add(1, Ordering::Relaxed);
        self.port.templates.lock().insert(id, nodes);
        Ok(GraphTemplate(id))
    }

    pub(super) fn port_instantiate(&self, t: &GraphTemplate) -> Result<GraphHandle> {
        let sig = self.port_signature(t)?;
        let id = self.port.next_graph_id.fetch_add(1, Ordering::Relaxed);
        self.port.execs.lock().insert(id, sig);
        self.port.instantiations.fetch_add(1, Ordering::Relaxed);
        Ok(GraphHandle(id))
    }

    pub(super) fn port_end_capture_template(
        &self,
        stream: u64,
    ) -> Result<(GraphHandle, GraphTemplate)> {
        let t = self.port_end_capture_graph(stream)?;
        let exec = self.port_instantiate(&t)?;
        Ok((exec, t))
    }

    pub(super) fn port_template_destroy(&self, t: GraphTemplate) -> Result<()> {
        if self.port.templates.lock().remove(&t.0).is_none() {
            bail!("graph_template_destroy: template {:#x} is not live", t.0);
        }
        Ok(())
    }

    pub(super) fn port_signature(&self, t: &GraphTemplate) -> Result<u64> {
        let templates = self.port.templates.lock();
        let nodes = templates
            .get(&t.0)
            .ok_or_else(|| anyhow!("graph template {:#x} is not live", t.0))?;
        Ok(topology_signature(nodes.iter().copied()))
    }

    pub(super) fn port_exec_update(&self, exec: GraphHandle, t: &GraphTemplate) -> Result<bool> {
        let sig = self.port_signature(t)?;
        let execs = self.port.execs.lock();
        let held = *execs
            .get(&exec.0)
            .ok_or_else(|| anyhow!("graph_exec_update: exec {:#x} is not live", exec.0))?;
        let refused = !self.port.graph_update_supported.load(Ordering::Relaxed)
            || self.port.fail_next_update.swap(false, Ordering::Relaxed);
        if refused || held != sig {
            self.port.update_rejects.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        // Same topology: the exec now holds `t`'s parameters under the same signature.
        self.port.updates.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    pub(super) fn port_upload(&self, exec: GraphHandle) -> Result<()> {
        if !self.port.execs.lock().contains_key(&exec.0) {
            bail!("graph_upload: exec {:#x} is not live", exec.0);
        }
        self.port.uploads.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// `destroy_graph`: forget a tracked exec. Untracked handles (the `GraphHandle(0)`
    /// the old `end_capture` default returns) stay a no-op, as before.
    pub(super) fn port_destroy_exec(&self, exec: GraphHandle) {
        self.port.execs.lock().remove(&exec.0);
    }

    pub(super) fn port_graph_update_supported(&self) -> bool {
        self.port.graph_update_supported.load(Ordering::Relaxed)
    }

    pub(super) fn port_copy_batch_supported(&self) -> bool {
        self.port.copy_batch_supported.load(Ordering::Relaxed)
    }

    /// One batch: the trait's per-copy loop when batching is off, else every entry
    /// under one lock, counted as one call.
    pub(super) fn port_copy_batch(
        &self,
        e: &[CopyEntry],
        stream: u64,
        to_host: bool,
    ) -> Result<()> {
        if !self.port_copy_batch_supported() {
            return if to_host {
                port_ext::loop_d2h(self, e, stream)
            } else {
                port_ext::loop_h2d(self, e, stream)
            };
        }
        port_ext::validate_entries(e)?;
        let mut allocs = self.allocs.lock();
        for (i, c) in e.iter().enumerate().filter(|(_, c)| c.bytes > 0) {
            let (offset, alloc) = find_alloc(&allocs, c.dev)
                .ok_or_else(|| anyhow!("copy batch entry {i}: {} not allocated", c.dev))?;
            if offset + c.bytes > alloc.bytes {
                bail!(
                    "copy batch entry {i}: {} + {} overruns its allocation",
                    c.dev,
                    c.bytes
                );
            }
            if to_host {
                // SAFETY: the `copy_batch_d2h_async` caller contract: `host` is valid for
                // `bytes` writable bytes nothing else touches until the stream syncs.
                let dst = unsafe { std::slice::from_raw_parts_mut(c.host as *mut u8, c.bytes) };
                dst.copy_from_slice(&alloc.data[offset..offset + c.bytes]);
            } else {
                // SAFETY: the `copy_batch_h2d_async` caller contract: `host` is valid for
                // `bytes` readable bytes that stay unchanged until the stream syncs.
                let src = unsafe { std::slice::from_raw_parts(c.host as *const u8, c.bytes) };
                let (offset, alloc) = find_alloc_mut(&mut allocs, c.dev)
                    .ok_or_else(|| anyhow!("copy batch entry {i}: {} not allocated", c.dev))?;
                alloc.data[offset..offset + c.bytes].copy_from_slice(src);
            }
        }
        self.port.batch_calls.fetch_add(1, Ordering::Relaxed);
        let live = e.iter().filter(|c| c.bytes > 0).count();
        self.port.batch_entries.fetch_add(live, Ordering::Relaxed);
        self.port.batch_streams.lock().push(stream);
        Ok(())
    }

    pub(super) fn port_next_copy_stream(&self) -> u64 {
        self.port.next_copy_stream.fetch_add(1, Ordering::Relaxed)
    }

    pub(super) fn port_create_event(&self) -> u64 {
        if self.port.unique_events.load(Ordering::Relaxed) {
            self.port.next_event.fetch_add(1, Ordering::Relaxed)
        } else {
            0
        }
    }

    pub(super) fn port_record_event(&self, event: u64) {
        if self.port.hold_events.load(Ordering::Relaxed) {
            self.port.pending_events.lock().insert(event);
        }
    }

    pub(super) fn port_event_done(&self, event: u64) -> bool {
        !self.port.pending_events.lock().contains(&event)
    }
}

// ── Test knobs and counters ──
impl MockGpuBackend {
    /// Whether `supports_graph_exec_update` answers true (default true). Off, every
    /// `graph_exec_update` returns `Ok(false)`.
    pub fn set_graph_update_supported(&self, on: bool) {
        self.port
            .graph_update_supported
            .store(on, Ordering::Relaxed);
    }

    /// The next `graph_exec_update` returns `Ok(false)` even on a matching topology,
    /// like a driver that refuses an update for a reason the signature cannot see.
    pub fn fail_next_graph_update(&self) {
        self.port.fail_next_update.store(true, Ordering::Relaxed);
    }

    pub fn graph_instantiation_count(&self) -> usize {
        self.port.instantiations.load(Ordering::Relaxed)
    }

    pub fn graph_update_count(&self) -> usize {
        self.port.updates.load(Ordering::Relaxed)
    }

    pub fn graph_update_reject_count(&self) -> usize {
        self.port.update_rejects.load(Ordering::Relaxed)
    }

    pub fn graph_upload_count(&self) -> usize {
        self.port.uploads.load(Ordering::Relaxed)
    }

    pub fn live_graph_template_count(&self) -> usize {
        self.port.templates.lock().len()
    }

    pub fn live_graph_exec_count(&self) -> usize {
        self.port.execs.lock().len()
    }

    /// Whether `supports_copy_batch` answers true and batched copies take the
    /// one-call path (default false: the trait's per-copy loop).
    pub fn set_copy_batch_supported(&self, on: bool) {
        self.port.copy_batch_supported.store(on, Ordering::Relaxed);
    }

    /// Batched-copy calls that took the one-call path.
    pub fn copy_batch_call_count(&self) -> usize {
        self.port.batch_calls.load(Ordering::Relaxed)
    }

    /// Non-empty entries moved by one-call batches.
    pub fn copy_batch_entry_count(&self) -> usize {
        self.port.batch_entries.load(Ordering::Relaxed)
    }

    /// Streams of the one-call batches, in call order.
    pub fn copy_batch_streams(&self) -> Vec<u64> {
        self.port.batch_streams.lock().clone()
    }

    /// Keep each recorded event pending until it is completed (default off).
    pub fn hold_events(&self, on: bool) {
        self.port.hold_events.store(on, Ordering::Relaxed);
    }

    /// `create_event` hands out distinct non-zero ids (default off: always 0).
    pub fn set_unique_events(&self, on: bool) {
        self.port.unique_events.store(on, Ordering::Relaxed);
    }

    /// Complete one held event (also what `event_synchronize` does).
    pub fn complete_event(&self, event: u64) {
        self.port.pending_events.lock().remove(&event);
    }

    pub fn complete_all_events(&self) {
        self.port.pending_events.lock().clear();
    }

    pub fn pending_event_count(&self) -> usize {
        self.port.pending_events.lock().len()
    }
}
