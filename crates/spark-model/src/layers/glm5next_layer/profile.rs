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
/// `kda_mixer` split three ways. The parent bucket is 53.5% of prefill — by far the
/// largest — but it spans three very different things: batched dense projections, a
/// per-token recurrence that CANNOT be batched in the recurrent arm, and the output
/// norm + o_proj. Chunking the recurrence measured ~4% SLOWER, which only makes sense
/// if the recurrence is not the bulk; these say which part to actually attack.
pub const KDA_FRONT: usize = 20;
pub const KDA_RECUR: usize = 21;
pub const KDA_BACK: usize = 22;
/// `dsa_select` split by stage. The parent bucket is one number over four kernels with
/// very different shapes — a whole-context pool compression, a `P*Q`-block score pass, a
/// top-k, and an expansion — so it cannot say which to attack. Splitting `kda_mixer` the
/// same way is what found the two largest wins on this branch.
pub const DSA_KPOOL: usize = 23;
pub const DSA_SCORES: usize = 24;
pub const DSA_TOPK: usize = 25;
pub const DSA_EXPAND: usize = 26;
const N: usize = 27;

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
    "kda_front",
    "kda_recur",
    "kda_back",
    "dsa_kpool",
    "dsa_scores",
    "dsa_topk",
    "dsa_expand",
];

static NANOS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static CALLS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static STEPS: AtomicU64 = AtomicU64::new(0);

/// `ATLAS_GLM_PROFILE=1` full · `=2` COLLECTIVES ONLY.
///
/// 🔴 Level 2 exists because level 1 cannot answer its own biggest question. Every span ends in
/// a `synchronize`, ~15 of them per layer per rank, and any host-side scheduling difference
/// between the two ranks accumulates between rendezvous points and is then charged to the next
/// `reduce_*_bar`. Level 2 syncs ONLY the collective spans, so the bar it reports is arrival
/// jitter the model actually has — not jitter the profiler manufactured.
fn level() -> u8 {
    static L: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *L.get_or_init(|| match std::env::var("ATLAS_GLM_PROFILE").as_deref() {
        Ok("1") => 1,
        Ok("2") => 2,
        _ => 0,
    })
}

pub fn on() -> bool {
    level() != 0
}

/// True only at level 1 — the per-kernel spans.
pub fn full() -> bool {
    level() == 1
}

/// Open a span. `None` when profiling is off, which makes [`end`] a no-op.
pub fn start() -> Option<Instant> {
    full().then(Instant::now)
}

/// A span that survives level 2: the collectives and their rendezvous probe.
pub fn start_hot() -> Option<Instant> {
    on().then(Instant::now)
}

pub fn end(bucket: usize, t0: Option<Instant>, gpu: &dyn GpuBackend, stream: u64) {
    let _ = end_us(bucket, t0, gpu, stream);
}

/// Same, returning the measured microseconds (0.0 when profiling is off).
pub fn end_us(bucket: usize, t0: Option<Instant>, gpu: &dyn GpuBackend, stream: u64) -> f64 {
    let Some(t0) = t0 else { return 0.0 };
    let _ = gpu.synchronize(stream);
    let ns = t0.elapsed().as_nanos() as u64;
    NANOS[bucket].fetch_add(ns, Relaxed);
    CALLS[bucket].fetch_add(1, Relaxed);
    ns as f64 / 1e3
}

/// `ATLAS_GLM_ROUTE_TRACE=1` — emit one line per reduce site per layer per token carrying the
/// router's selected GLOBAL expert ids and the measured rendezvous (arrival-skew) time.
///
/// The router is REPLICATED and bit-identical on every rank, so rank 0's ids are the whole
/// picture: any static ownership map can be scored offline from this one trace without a
/// second run. Costly (one log line per MoE layer per token) — trace, then turn it off.
pub fn trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_ROUTE_TRACE").as_deref() == Ok("1"))
}

thread_local! {
    /// The ids `forward_moe` last read back to the host, for the trace line that follows.
    static ROUTE: std::cell::RefCell<Vec<i32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Called from `forward_moe` right after the routing D2H. No-op unless tracing.
pub fn stash_route(ids: &[i32]) {
    if !trace_on() {
        return;
    }
    ROUTE.with(|r| {
        let mut v = r.borrow_mut();
        v.clear();
        v.extend_from_slice(ids);
    });
}

/// Emit the joined line. `site` is `attn` or `mlp`; `moe` says whether THIS layer's MLP is
/// routed (the 3 dense layers are the natural control: no EP imbalance is possible there).
pub fn trace_bar(site: &str, layer: usize, moe: bool, us: f64) {
    if !trace_on() {
        return;
    }
    let step = STEPS.load(Relaxed);
    ROUTE.with(|r| {
        let v = r.borrow();
        let ids = v
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        tracing::warn!(
            "GLMTRACE step={step} site={site} L={layer} moe={} bar_us={us:.1} ids={ids}",
            u8::from(moe)
        );
    });
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
