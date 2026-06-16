// SPDX-License-Identifier: AGPL-3.0-only

//! Env-gated per-substep wall-time accumulators for the MTP K=2 verify path.
//!
//! Enabled only when `ATLAS_VERIFY_TIMING=1`. Each accumulator is a global
//! `AtomicU64` of nanoseconds. `decode_batched`/`decode_multi_seq` add their
//! synced substep durations; `verify_b` dumps and resets after each verify
//! step. Times are wall-clock around `gpu.synchronize(stream)` boundaries, so
//! they are only meaningful in eager mode (`ATLAS_DEBUG_NO_GRAPH=1`), where
//! each kernel runs un-fused into a CUDA graph.
//!
//! This is diagnostic-only: when the env var is unset, `enabled()` is false
//! and no syncs/timers are inserted (zero overhead on the hot path).

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Per-substep nanosecond accumulators, reset per verify step.
pub static SSM_RMSNORM_NS: AtomicU64 = AtomicU64::new(0);
pub static SSM_QKVZ_NS: AtomicU64 = AtomicU64::new(0);
pub static SSM_BA_GATES_NS: AtomicU64 = AtomicU64::new(0);
pub static SSM_CONV_GDN_NS: AtomicU64 = AtomicU64::new(0);
pub static SSM_GATEDRMS_NS: AtomicU64 = AtomicU64::new(0);
pub static SSM_OUTPROJ_NS: AtomicU64 = AtomicU64::new(0);
pub static SSM_MOE_NS: AtomicU64 = AtomicU64::new(0);
pub static ATTN_TOTAL_NS: AtomicU64 = AtomicU64::new(0);

/// K=1 single-token decode reference accumulators (non-spec path), used to
/// compare against the K=2 verify buckets and quantify amortization.
pub static K1_SSM_QKVZ_NS: AtomicU64 = AtomicU64::new(0);
pub static K1_SSM_MOE_NS: AtomicU64 = AtomicU64::new(0);
pub static K1_ATTN_TOTAL_NS: AtomicU64 = AtomicU64::new(0);
pub static K1_SSM_TOTAL_NS: AtomicU64 = AtomicU64::new(0);

/// Cached env gate. `OnceLock` avoids re-reading the env on every kernel.
static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

#[inline]
pub fn enabled() -> bool {
    *GATE.get_or_init(|| std::env::var("ATLAS_VERIFY_TIMING").as_deref() == Ok("1"))
}

#[inline]
pub fn add(acc: &AtomicU64, ns: u64) {
    acc.fetch_add(ns, Ordering::Relaxed);
}

/// Time a synced GPU substep, adding its wall-time (ns) to `acc`.
/// `sync` runs `gpu.synchronize(stream)` so the elapsed time reflects
/// only this substep's kernels (eager mode).
#[inline]
pub fn timed<F, S, R>(acc: &AtomicU64, sync: S, f: F) -> R
where
    F: FnOnce() -> R,
    S: FnOnce(),
{
    if !enabled() {
        return f();
    }
    let t0 = Instant::now();
    let r = f();
    sync();
    add(acc, t0.elapsed().as_nanos() as u64);
    r
}

fn take(acc: &AtomicU64) -> f64 {
    acc.swap(0, Ordering::Relaxed) as f64 / 1_000_000.0
}

/// Dump the accumulated substep breakdown for one verify step and reset all
/// accumulators. Called from `verify_b` after the layer loop + head.
pub fn dump_and_reset(head_ms: f64) {
    if !enabled() {
        return;
    }
    let rms = take(&SSM_RMSNORM_NS);
    let qkvz = take(&SSM_QKVZ_NS);
    let ba = take(&SSM_BA_GATES_NS);
    let convgdn = take(&SSM_CONV_GDN_NS);
    let grms = take(&SSM_GATEDRMS_NS);
    let outp = take(&SSM_OUTPROJ_NS);
    let moe = take(&SSM_MOE_NS);
    let attn = take(&ATTN_TOTAL_NS);
    let ssm = rms + qkvz + ba + convgdn + grms + outp + moe;
    let total = ssm + attn + head_ms;
    tracing::info!(
        "VERIFY_TIMING step total={:.2}ms | attn={:.2} ssm={:.2} head={:.2} || \
         ssm[rmsnorm={:.2} qkvz={:.2} ba_gates={:.2} conv_gdn={:.2} gatedrms={:.2} \
         outproj={:.2} moe={:.2}]",
        total,
        attn,
        ssm,
        head_ms,
        rms,
        qkvz,
        ba,
        convgdn,
        grms,
        outp,
        moe
    );
}

/// Dump the K=1 single-token reference buckets and reset. Called from the
/// single-token decode dispatch after the layer loop.
pub fn dump_and_reset_k1() {
    if !enabled() {
        return;
    }
    let qkvz = take(&K1_SSM_QKVZ_NS);
    let moe = take(&K1_SSM_MOE_NS);
    let attn = take(&K1_ATTN_TOTAL_NS);
    let ssm = take(&K1_SSM_TOTAL_NS);
    tracing::info!(
        "VERIFY_TIMING_K1 step | k1_attn={:.2} k1_ssm_total={:.2} \
         (k1_ssm_qkvz={:.2} k1_ssm_moe={:.2})",
        attn,
        ssm,
        qkvz,
        moe
    );
}
