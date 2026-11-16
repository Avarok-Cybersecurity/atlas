// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache csrc/cuda/mp_mem_kernels.cu:251-345 (staged gather, then DMA in plan steps)
// and csrc/cuda/phase_timing_recorder.cu (per-phase GB/s), commit 3aefae1, Apache-2.0; no code
// copied. Changes: a two-buffer host staging that overlaps the copy of chunk i with the file write
// (or read) of chunk i-1, behind spark-model's `kv_batched_swap` row, by Atlas contributors.

//! Responsibility: the `atlas_kv_swap_*` `/metrics` series and the `AVAROK_KV_SWAP_TIMING=1` log
//! line of the batched swap image path (`block_io_swap.rs`, which re-exports every public item
//! here), which the per-region path reports into as well.

use std::fmt::Write as _;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use super::SwapReport;

// ── `/metrics` series and the timing log ──

/// Swap direction: `Out` = device to host (save), `In` = host to device (restore).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwapDir {
    Out,
    In,
}

impl SwapDir {
    fn idx(self) -> usize {
        match self {
            SwapDir::Out => 0,
            SwapDir::In => 1,
        }
    }

    /// The `dir` label value.
    pub fn label(self) -> &'static str {
        match self {
            SwapDir::Out => "d2h",
            SwapDir::In => "h2d",
        }
    }
}

static BYTES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static NANOS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Count one image save or restore (either path) into `atlas_kv_swap_*`.
pub fn record(dir: SwapDir, bytes: usize, elapsed: Duration) {
    let i = dir.idx();
    BYTES[i].fetch_add(bytes as u64, Ordering::Relaxed);
    let ns = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
    NANOS[i].fetch_add(ns, Ordering::Relaxed);
}

/// `(bytes, seconds)` counted so far for `dir`.
pub fn totals(dir: SwapDir) -> (u64, f64) {
    let i = dir.idx();
    (
        BYTES[i].load(Ordering::Relaxed),
        NANOS[i].load(Ordering::Relaxed) as f64 * 1e-9,
    )
}

/// `AVAROK_KV_SWAP_TIMING` (`1`/`true`/`on`/`yes`): log one line per image save or restore.
pub fn timing_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("AVAROK_KV_SWAP_TIMING").is_ok_and(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        })
    })
}

/// The `AVAROK_KV_SWAP_TIMING` line for one image (`path` = `batched` or `legacy`).
pub fn format_report(dir: SwapDir, path: &str, kv_bytes: usize, rep: &SwapReport) -> String {
    let what = match dir {
        SwapDir::Out => "out",
        SwapDir::In => "in",
    };
    format!(
        "KV swap {what} ({path}): {} B ({kv_bytes} KV + {} SSM), {} regions, {} chunks of <= {} B, \
         staging={} wait={}us io={}us total={}us = {:.2} GB/s",
        rep.bytes,
        rep.bytes.saturating_sub(kv_bytes),
        rep.regions,
        rep.chunks,
        rep.chunk_bytes,
        rep.staging,
        rep.wait_us,
        rep.io_us,
        rep.total_us,
        rep.gb_per_s()
    )
}

/// Log [`format_report`] when `AVAROK_KV_SWAP_TIMING` is on.
pub fn log_report(dir: SwapDir, path: &str, kv_bytes: usize, rep: &SwapReport) {
    if timing_enabled() {
        tracing::info!("{}", format_report(dir, path, kv_bytes, rep));
    }
}

/// Append `atlas_kv_swap_bytes_total{dir}` and `atlas_kv_swap_seconds_total{dir}`.
pub fn render_prometheus(out: &mut String) {
    let (ob, os) = totals(SwapDir::Out);
    let (ib, is) = totals(SwapDir::In);
    let _ = writeln!(
        out,
        "# HELP atlas_kv_swap_bytes_total Bytes of sequence swap images (KV blocks and SSM state) \
         moved between device and host, by direction.\n\
         # TYPE atlas_kv_swap_bytes_total counter\n\
         atlas_kv_swap_bytes_total{{dir=\"d2h\"}} {ob}\n\
         atlas_kv_swap_bytes_total{{dir=\"h2d\"}} {ib}\n\
         # HELP atlas_kv_swap_seconds_total Wall seconds of sequence swap image saves (d2h) and \
         restores (h2d), copies and file I/O included.\n\
         # TYPE atlas_kv_swap_seconds_total counter\n\
         atlas_kv_swap_seconds_total{{dir=\"d2h\"}} {os}\n\
         atlas_kv_swap_seconds_total{{dir=\"h2d\"}} {is}"
    );
}
