// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence memory trace — the instrument A76's runtime gates are measured on.
//!
//! # Why this exists
//!
//! Every prior leak measurement on this box rested on two point reads of
//! `MemAvailable` per leg. That instrument moves by GBs for reasons unrelated to
//! requests, mixes driver commitments with reclaimable page cache, and on the one
//! buffer set whose true size is computable from source it over-read by +34 %
//! (ANOMALIES A75/A76 follow-up). A slope needs a per-request series, and a leak
//! verdict needs the driver leg separated from the host leg.
//!
//! So each line carries three independent readings taken at the same instant:
//!
//! * `live` — live device allocations on the backend's own ledger. A COUNT, and the
//!   decisive one: it is unaffected by allocator caching, page cache, or driver
//!   book-keeping. A leaking request raises it by a fixed number every time.
//! * `cumemgetinfo` — the driver leg alone, no `max(.., MemAvailable)` (A73).
//! * `memavailable` — the host leg, for continuity with the older measurements.
//!
//! Off unless `ATLAS_SEQ_MEMTRACE` is set (presence — `=0` is NOT "off"), so the
//! serving path pays one `OnceLock` read per sequence when it is off.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use spark_runtime::gpu::GpuBackend;

static ENABLED: OnceLock<bool> = OnceLock::new();
static SEQ_NO: AtomicU64 = AtomicU64::new(0);

pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var("ATLAS_SEQ_MEMTRACE").is_ok())
}

fn mem_available_bytes() -> Option<usize> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: usize = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// One trace line. `phase` is `alloc` or `free`; the pair brackets one sequence.
///
/// 🪤 Emitted at `info!` on purpose. The two existing memory targets are
/// `debug!`, and a run that has to raise the whole crate to debug to see its own
/// instrument drowns the leak signal in NCCL and kernel chatter.
pub fn trace(gpu: &dyn GpuBackend, phase: &str) {
    if !enabled() {
        return;
    }
    let n = match phase {
        // The sequence number advances once per sequence, at admission, so the
        // `alloc` and `free` lines of one sequence carry the SAME index.
        "alloc" => SEQ_NO.fetch_add(1, Ordering::Relaxed),
        _ => SEQ_NO.load(Ordering::Relaxed).saturating_sub(1),
    };
    let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
    let dev = gpu.device_free_memory().unwrap_or(0);
    let host = mem_available_bytes().unwrap_or(0);
    tracing::info!(
        "seqmem: seq={n} phase={phase} live={} cumemgetinfo_mb={:.1} memavailable_mb={:.1}",
        gpu.live_alloc_count(),
        mb(dev),
        mb(host),
    );
}
