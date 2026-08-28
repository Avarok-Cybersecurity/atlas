// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_PROFILE=1` — per-section decode timing for the GLM-5.3 stack.
//!
//! Off unless the variable is set. Every span ends in a `synchronize`, so enabling it
//! SERIALISES the stream: read the split, not the total, and never quote a tok/s taken
//! with it on.
//!
//! Sections are chosen to separate the three things that can each explain a 10x decode
//! gap and look identical from the outside: weight bandwidth (the GEMM buckets), launch
//! and host-sync latency (`moe_hostsync`, call counts), and collectives (`reduce_*`).

use spark_runtime::gpu::GpuBackend;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

pub const MHC: usize = 0;
pub const NORM: usize = 1;
pub const KDA: usize = 2;
pub const DSA_PROJ: usize = 3;
pub const DSA_INDEXER: usize = 4;
pub const DSA_SELECT: usize = 5;
pub const DSA_ATTEND: usize = 6;
pub const REDUCE_ATTN: usize = 7;
pub const MLP_DENSE: usize = 8;
pub const MOE_ROUTER: usize = 9;
pub const MOE_HOSTSYNC: usize = 10;
pub const MOE_EXPERTS: usize = 11;
pub const MOE_SHARED: usize = 12;
pub const MOE_COMBINE: usize = 13;
pub const REDUCE_MLP: usize = 14;
/// Host-side ENQUEUE cost of the two collectives — the driver/NCCL calls only, no sync.
/// Paired with [`REDUCE_ATTN`]/[`REDUCE_MLP`], which then time ONLY the `synchronize` that
/// follows, i.e. the device + network + rank-skew wait. Splitting them is the difference
/// between "the fabric is slow" and "we call it 90 times a token".
pub const REDUCE_ATTN_ENQ: usize = 15;
pub const REDUCE_MLP_ENQ: usize = 16;
/// A 2-BYTE collective issued immediately before the real one, PROFILING ONLY. It is a
/// rendezvous: neither rank leaves it until both have arrived, so it absorbs the per-call
/// arrival jitter and charges the minimum-payload NCCL latency. The real 8 KB reduce that
/// follows therefore starts with both ranks synchronised, which is what makes
/// [`REDUCE_ATTN`]/[`REDUCE_MLP`] readable as network-and-kernel cost rather than "network
/// plus whatever the other rank was still doing".
///
/// 🪤 Aggregate rank skew being ~0 does NOT mean per-call wait is ~0 — the two ranks trade the
/// lead call by call, so the NET cancels while every individual call still pays |jitter|.
/// That is exactly why this probe exists and why the both-rank profile diff was not enough.
pub const REDUCE_ATTN_BAR: usize = 17;
pub const REDUCE_MLP_BAR: usize = 18;
/// mHC split by kernel: `hc_pre` (the mix + finish pair) vs `hc_post` (+ expand/head).
pub const MHC_POST: usize = 19;
const N: usize = 20;

const NAMES: [&str; N] = [
    "mhc",
    "norm",
    "kda_mixer",
    "dsa_proj",
    "dsa_indexer",
    "dsa_select",
    "dsa_attend",
    "reduce_attn",
    "mlp_dense",
    "moe_router",
    "moe_hostsync",
    "moe_experts",
    "moe_shared",
    "moe_combine",
    "reduce_mlp",
    "reduce_attn_enq",
    "reduce_mlp_enq",
    "reduce_attn_bar",
    "reduce_mlp_bar",
    "mhc_post",
];

static NANOS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static CALLS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static STEPS: AtomicU64 = AtomicU64::new(0);

pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_PROFILE").as_deref() == Ok("1"))
}

/// Open a span. `None` when profiling is off, which makes [`end`] a no-op.
pub fn start() -> Option<Instant> {
    on().then(Instant::now)
}

pub fn end(bucket: usize, t0: Option<Instant>, gpu: &dyn GpuBackend, stream: u64) {
    let Some(t0) = t0 else { return };
    let _ = gpu.synchronize(stream);
    NANOS[bucket].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    CALLS[bucket].fetch_add(1, Relaxed);
}

/// 4-byte device scratch for the rendezvous probe. Allocated once, PROFILING ONLY.
pub fn probe_buf(gpu: &dyn GpuBackend) -> u64 {
    static P: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *P.get_or_init(|| gpu.alloc(4).map(|p| p.0).unwrap_or(0))
}

/// Record a span WITHOUT synchronising — host-side wall time only.
pub fn end_nosync(bucket: usize, t0: Option<Instant>) {
    let Some(t0) = t0 else { return };
    NANOS[bucket].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    CALLS[bucket].fetch_add(1, Relaxed);
}

/// Close one token. Dumps a cumulative per-token split every 8 steps, then keeps going —
/// the totals are cumulative so a later dump is simply better averaged.
pub fn step() {
    if !on() {
        return;
    }
    let s = STEPS.fetch_add(1, Relaxed) + 1;
    if !s.is_multiple_of(8) {
        return;
    }
    let total: u64 = NANOS.iter().map(|n| n.load(Relaxed)).sum();
    let mut rows: Vec<(usize, u64, u64)> = (0..N)
        .map(|i| (i, NANOS[i].load(Relaxed), CALLS[i].load(Relaxed)))
        .collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    let mut out = format!(
        "GLM decode profile after {s} steps — {:.2} ms/token measured under profiling\n",
        total as f64 / 1e6 / s as f64
    );
    for (i, ns, calls) in rows {
        if calls == 0 {
            continue;
        }
        out += &format!(
            "  {:<13} {:>8.2} ms/tok  {:>6.1}%  {:>5} calls/tok  {:>7.1} us/call\n",
            NAMES[i],
            ns as f64 / 1e6 / s as f64,
            100.0 * ns as f64 / total.max(1) as f64,
            calls / s,
            ns as f64 / 1e3 / calls.max(1) as f64,
        );
    }
    tracing::warn!("{out}");
}
