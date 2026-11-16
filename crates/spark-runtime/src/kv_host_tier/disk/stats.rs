// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the disk tier's counters and gauges and their `/metrics` text
//! (`atlas_kv_tier_disk_*`). Each [`super::DiskTier`] owns one [`DiskStats`];
//! [`render_prometheus`] sums every live tier's values. spark-runtime sits below spark-server, so
//! it formats its own Prometheus text with the same `# HELP` / `# TYPE` lines as
//! `api/port_metrics.rs`. With no disk tier opened every series renders as zero.

use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// One disk tier's counters (monotonic) and gauges (current values).
#[derive(Default, Debug)]
pub struct DiskStats {
    bytes: AtomicU64,
    capacity: AtomicU64,
    files: AtomicU64,
    direct_io: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    writes: AtomicU64,
    write_bytes: AtomicU64,
    skipped: AtomicU64,
    evictions: AtomicU64,
    adopted: AtomicU64,
    err_io: AtomicU64,
    err_corrupt: AtomicU64,
    stray_tmp: AtomicU64,
}

/// A copy of one tier's values (tests and logs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiskSnapshot {
    pub bytes: u64,
    pub capacity: u64,
    pub files: u64,
    pub direct_io: u64,
    pub hits: u64,
    pub misses: u64,
    pub writes: u64,
    pub write_bytes: u64,
    pub skipped: u64,
    pub evictions: u64,
    pub adopted: u64,
    pub err_io: u64,
    pub err_corrupt: u64,
    pub stray_tmp: u64,
}

fn add(a: &AtomicU64, n: u64) {
    a.fetch_add(n, Ordering::Relaxed);
}

impl DiskStats {
    pub(crate) fn set_usage(&self, bytes: u64, files: u64) {
        self.bytes.store(bytes, Ordering::Relaxed);
        self.files.store(files, Ordering::Relaxed);
    }
    pub(crate) fn set_capacity(&self, b: u64) {
        self.capacity.store(b, Ordering::Relaxed);
    }
    pub(crate) fn set_direct_io(&self, on: bool) {
        self.direct_io.store(on as u64, Ordering::Relaxed);
    }
    pub(crate) fn hit(&self) {
        add(&self.hits, 1);
    }
    pub(crate) fn miss(&self) {
        add(&self.misses, 1);
    }
    pub(crate) fn wrote(&self, bytes: u64) {
        add(&self.writes, 1);
        add(&self.write_bytes, bytes);
    }
    pub(crate) fn skipped(&self) {
        add(&self.skipped, 1);
    }
    pub(crate) fn evicted(&self, n: u64) {
        add(&self.evictions, n);
    }
    pub(crate) fn adopted(&self, n: u64) {
        add(&self.adopted, n);
    }
    pub(crate) fn io_error(&self) {
        add(&self.err_io, 1);
    }
    pub(crate) fn corrupt(&self) {
        add(&self.err_corrupt, 1);
    }
    pub(crate) fn stray(&self, n: u64) {
        add(&self.stray_tmp, n);
    }

    pub fn snapshot(&self) -> DiskSnapshot {
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        DiskSnapshot {
            bytes: g(&self.bytes),
            capacity: g(&self.capacity),
            files: g(&self.files),
            direct_io: g(&self.direct_io),
            hits: g(&self.hits),
            misses: g(&self.misses),
            writes: g(&self.writes),
            write_bytes: g(&self.write_bytes),
            skipped: g(&self.skipped),
            evictions: g(&self.evictions),
            adopted: g(&self.adopted),
            err_io: g(&self.err_io),
            err_corrupt: g(&self.err_corrupt),
            stray_tmp: g(&self.stray_tmp),
        }
    }
}

/// Every opened tier's stats, held weakly so a dropped tier stops counting toward the gauges.
/// Counters of dropped tiers are folded into `RETIRED` so `_total` series stay monotonic.
static LIVE: Mutex<Vec<Weak<DiskStats>>> = Mutex::new(Vec::new());
static RETIRED: Mutex<DiskSnapshot> = Mutex::new(DiskSnapshot {
    bytes: 0,
    capacity: 0,
    files: 0,
    direct_io: 0,
    hits: 0,
    misses: 0,
    writes: 0,
    write_bytes: 0,
    skipped: 0,
    evictions: 0,
    adopted: 0,
    err_io: 0,
    err_corrupt: 0,
    stray_tmp: 0,
});

pub(crate) fn register(s: &Arc<DiskStats>) {
    if let Ok(mut live) = LIVE.lock() {
        live.retain(|w| w.strong_count() > 0);
        live.push(Arc::downgrade(s));
    }
}

/// Fold a closing tier's counters into the retired totals (called from `DiskTier`'s `Drop`).
pub(crate) fn retire(s: &DiskStats) {
    let v = s.snapshot();
    if let Ok(mut r) = RETIRED.lock() {
        r.hits += v.hits;
        r.misses += v.misses;
        r.writes += v.writes;
        r.write_bytes += v.write_bytes;
        r.skipped += v.skipped;
        r.evictions += v.evictions;
        r.adopted += v.adopted;
        r.err_io += v.err_io;
        r.err_corrupt += v.err_corrupt;
        r.stray_tmp += v.stray_tmp;
    }
}

/// The sum over every live tier plus the retired counters.
pub fn totals() -> DiskSnapshot {
    let mut t = RETIRED.lock().map(|r| *r).unwrap_or_default();
    if let Ok(live) = LIVE.lock() {
        for s in live.iter().filter_map(Weak::upgrade) {
            let v = s.snapshot();
            t.bytes += v.bytes;
            t.capacity += v.capacity;
            t.files += v.files;
            t.direct_io = t.direct_io.max(v.direct_io);
            t.hits += v.hits;
            t.misses += v.misses;
            t.writes += v.writes;
            t.write_bytes += v.write_bytes;
            t.skipped += v.skipped;
            t.evictions += v.evictions;
            t.adopted += v.adopted;
            t.err_io += v.err_io;
            t.err_corrupt += v.err_corrupt;
            t.stray_tmp += v.stray_tmp;
        }
    }
    t
}

fn series(out: &mut String, name: &str, kind: &str, help: &str, v: u64) {
    let _ = writeln!(
        out,
        "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {v}"
    );
}

/// Append the `atlas_kv_tier_disk_*` series to the `/metrics` text.
pub fn render_prometheus(out: &mut String) {
    let t = totals();
    series(
        out,
        "atlas_kv_tier_disk_bytes",
        "gauge",
        "Bytes of KV block records held by the disk tier.",
        t.bytes,
    );
    series(
        out,
        "atlas_kv_tier_disk_capacity_bytes",
        "gauge",
        "Byte budget of the disk tier (--kv-tier-disk-gb).",
        t.capacity,
    );
    series(
        out,
        "atlas_kv_tier_disk_files",
        "gauge",
        "KV block records held by the disk tier.",
        t.files,
    );
    series(
        out,
        "atlas_kv_tier_disk_direct_io",
        "gauge",
        "1 while the disk tier still uses O_DIRECT, 0 after a buffered fallback or when off.",
        t.direct_io,
    );
    series(
        out,
        "atlas_kv_tier_disk_hits_total",
        "counter",
        "Blocks promoted from the disk tier back into the host tier.",
        t.hits,
    );
    series(
        out,
        "atlas_kv_tier_disk_misses_total",
        "counter",
        "Disk tier promotions that found no valid record.",
        t.misses,
    );
    series(
        out,
        "atlas_kv_tier_disk_writes_total",
        "counter",
        "Blocks demoted from the host tier into disk records.",
        t.writes,
    );
    series(
        out,
        "atlas_kv_tier_disk_write_bytes_total",
        "counter",
        "Bytes written to disk tier records.",
        t.write_bytes,
    );
    series(
        out,
        "atlas_kv_tier_disk_skipped_total",
        "counter",
        "Demotions not written (already on disk, or larger than the budget).",
        t.skipped,
    );
    series(
        out,
        "atlas_kv_tier_disk_evictions_total",
        "counter",
        "Disk tier records removed by least-recently-used eviction.",
        t.evictions,
    );
    series(
        out,
        "atlas_kv_tier_disk_adopted_total",
        "counter",
        "Records of a previous run adopted at startup for the served model's fingerprint.",
        t.adopted,
    );
    let _ = writeln!(
        out,
        "# HELP atlas_kv_tier_disk_errors_total Disk tier problems by kind (io, corrupt record, stray temp file).\n\
         # TYPE atlas_kv_tier_disk_errors_total counter\n\
         atlas_kv_tier_disk_errors_total{{kind=\"io\"}} {}\n\
         atlas_kv_tier_disk_errors_total{{kind=\"corrupt\"}} {}\n\
         atlas_kv_tier_disk_errors_total{{kind=\"stray_tmp\"}} {}",
        t.err_io, t.err_corrupt, t.stray_tmp
    );
}
