// SPDX-License-Identifier: AGPL-3.0-only

//! The device-allocation ledger: what is live, how big it is, and which call
//! site asked for it.
//!
//! On GB10 every `cuMemAlloc` consumes host RAM, so an allocation outside the
//! util pledge is how the box ends up in swap — and before this existed the
//! KV budget inferred "Atlas-own" bytes from a free-memory delta, which counts
//! a co-tenant's pages as ours. The ledger replaces that inference with a
//! measurement.
//!
//! Split from `cuda_backend.rs` for the 500-LoC cap, which the file crossed
//! when this landed. Exact piecewise copy, with one correction: the doc
//! comment above `record_alloc` described the `registry` field, not the
//! method, on main.

use super::AtlasCudaBackend;

/// One live device allocation: how big it is and who asked for it.
///
/// `site` is the CALLER of `GpuBackend::alloc`, not this file, because both
/// allocating methods are `#[track_caller]`. Costs one pointer per live
/// allocation and nothing at all per kernel launch — allocation is a
/// load-time event, not a hot-path one.
#[derive(Clone, Copy)]
pub(super) struct AllocRecord {
    pub(super) bytes: usize,
    pub(super) site: &'static std::panic::Location<'static>,
}

impl AtlasCudaBackend {
    /// Enter an allocation in the ledger. `site` is the CALLER of
    /// `GpuBackend::alloc` (both allocating methods are `#[track_caller]`),
    /// which is what makes `alloc_report` name a file rather than this one.
    pub(crate) fn record_alloc(
        &self,
        ptr: crate::gpu::DevicePtr,
        bytes: usize,
        site: &'static std::panic::Location<'static>,
    ) {
        self.live_allocs
            .lock()
            .insert(ptr.0, AllocRecord { bytes, site });
    }

    pub(crate) fn forget_alloc(&self, ptr: crate::gpu::DevicePtr) {
        self.live_allocs.lock().remove(&ptr.0);
    }

    /// Total live device bytes this backend has allocated and not freed.
    pub fn live_bytes(&self) -> usize {
        self.live_allocs.lock().values().map(|r| r.bytes).sum()
    }

    /// Human-readable attribution of live device memory, biggest site first.
    ///
    /// Aggregated by allocating call site rather than by pointer: one site
    /// looping over 48 SSM layers is one line reading 9.7 GB across 48
    /// allocations, which is the shape that makes an over-sized pool obvious.
    /// Sites below `min_mb` are folded into a remainder line so the report
    /// stays readable while still summing to the true total.
    pub fn alloc_report(&self, top_n: usize, min_mb: usize) -> String {
        use std::collections::HashMap;
        let mut by_site: HashMap<String, (usize, usize)> = HashMap::new();
        let mut total = 0usize;
        for rec in self.live_allocs.lock().values() {
            total += rec.bytes;
            let key = format!("{}:{}", rec.site.file(), rec.site.line());
            let e = by_site.entry(key).or_insert((0, 0));
            e.0 += rec.bytes;
            e.1 += 1;
        }
        let mut rows: Vec<(String, usize, usize)> =
            by_site.into_iter().map(|(k, v)| (k, v.0, v.1)).collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1));

        let mut out = format!(
            "GPU allocation ledger: {:.2} GB live across {} sites\n",
            total as f64 / 1e9,
            rows.len()
        );
        let mut shown = 0usize;
        let mut folded_bytes = 0usize;
        let mut folded_sites = 0usize;
        for (site, bytes, count) in rows {
            if shown < top_n && bytes >= min_mb * 1024 * 1024 {
                out.push_str(&format!(
                    "  {:>9.1} MB  x{:<5} {}\n",
                    bytes as f64 / (1024.0 * 1024.0),
                    count,
                    site
                ));
                shown += 1;
            } else {
                folded_bytes += bytes;
                folded_sites += 1;
            }
        }
        if folded_sites > 0 {
            out.push_str(&format!(
                "  {:>9.1} MB  across {} smaller sites\n",
                folded_bytes as f64 / (1024.0 * 1024.0),
                folded_sites
            ));
        }

        // Per-FILE rollup. The by-site view above has a blind spot: a
        // subsystem that allocates many distinct buffers from many distinct
        // lines is split into pieces small enough to fall below the cut and
        // vanish into the remainder. The vision encoder is exactly that shape
        // — ~16 buffers (scores, probs, qr/kr/vt, merge, rope, ...) each from
        // its own line — so ~2.2 GB was invisible in a top-12 by site while
        // being the fourth-largest consumer in the process. Rolling up by
        // file costs one more pass over the same map and makes a subsystem
        // legible as a subsystem.
        let mut by_file: HashMap<&str, (usize, usize)> = HashMap::new();
        for rec in self.live_allocs.lock().values() {
            let e = by_file.entry(rec.site.file()).or_insert((0, 0));
            e.0 += rec.bytes;
            e.1 += 1;
        }
        let mut frows: Vec<(&str, usize, usize)> =
            by_file.into_iter().map(|(k, v)| (k, v.0, v.1)).collect();
        frows.sort_by(|a, b| b.1.cmp(&a.1));
        out.push_str("  ── by file ──\n");
        for (file, bytes, count) in frows.into_iter().take(top_n) {
            out.push_str(&format!(
                "  {:>9.1} MB  x{:<5} {}\n",
                bytes as f64 / (1024.0 * 1024.0),
                count,
                file
            ));
        }
        out
    }
}
