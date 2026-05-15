// SPDX-License-Identifier: AGPL-3.0-only

//! MTP↔main side-by-side hidden-state comparator.
//!
//! When `ATLAS_MTP_DIVERGENCE_COMPARE=1` is set, the main-model decode
//! step records its pre-LM-head `final_normed` (BF16) and top-5 logits
//! into a process-wide snapshot; the MTP head's own divergence dump
//! reads that snapshot and emits a single `MTP_COMPARE` log line per
//! captured pair with L2 / cosine / top-5 Jaccard / argmax-agreement.
//!
//! Adjacent token positions in a trained transformer typically show
//! `cos(final_normed_t, final_normed_{t+1}) ≥ 0.9`. A healthy MTP head
//! should produce `final_normed` highly correlated with the main
//! model's previous-step `final_normed`. `cos < 0.5` indicates a
//! wiring bug (mis-loaded weight, wrong precision, broken residual).
//!
//! Cost: forces a default-stream sync + small D2H copy per capture.
//! Capped at the first 12 captures via the existing call counter.

use std::sync::Mutex;

/// One captured main-model decode step.
pub struct MainSnapshot {
    pub position: u64,
    pub final_normed: Vec<f32>,
    pub top5: Vec<(u32, f32)>,
    pub argmax: Option<u32>,
}

static SNAPSHOT: Mutex<Option<MainSnapshot>> = Mutex::new(None);

/// True iff `ATLAS_MTP_DIVERGENCE_COMPARE=1` is set in the environment.
/// Cached after first call.
pub fn enabled() -> bool {
    use std::sync::atomic::{AtomicI8, Ordering};
    static STATE: AtomicI8 = AtomicI8::new(-1);
    match STATE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = std::env::var("ATLAS_MTP_DIVERGENCE_COMPARE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            STATE.store(if on { 1 } else { 0 }, Ordering::Relaxed);
            on
        }
    }
}

/// Bound the number of paired captures so the D2H sync cost never
/// touches a steady-state hot path. CUDA graphs are auto-suppressed
/// when `ATLAS_MTP_DIVERGENCE_COMPARE=1`, so per-step D2H is already
/// the dominant cost and the cap is high enough to span the whole
/// response (max_tokens ~256-500 for typical comparator runs).
pub fn should_capture() -> bool {
    use std::sync::atomic::{AtomicU32, Ordering};
    const MAX_CAPS: u32 = 1024;
    static CALLS: AtomicU32 = AtomicU32::new(0);
    let n = CALLS.fetch_add(1, Ordering::Relaxed);
    n < MAX_CAPS
}

/// Store the main-model's pre-LM-head hidden state for the current
/// decode step. Called from `decode_dispatch` right after `rms_norm`.
pub fn record_main_hidden(position: u64, final_normed: Vec<f32>) {
    let mut g = SNAPSHOT.lock().unwrap();
    *g = Some(MainSnapshot {
        position,
        final_normed,
        top5: Vec::new(),
        argmax: None,
    });
}

/// Attach the top-5 logits + argmax to the most recent main snapshot.
/// Called from `decode_dispatch` after `self.lm_head(...)`.
pub fn record_main_top5(top5: Vec<(u32, f32)>, argmax: Option<u32>) {
    let mut g = SNAPSHOT.lock().unwrap();
    if let Some(s) = g.as_mut() {
        s.top5 = top5;
        s.argmax = argmax;
    }
}

/// Peek at the latest main snapshot without consuming it. The MTP
/// dump can run many times for the same main step (K draft tokens);
/// returning a clone lets each compare against the same reference.
pub fn peek_snapshot() -> Option<MainSnapshot> {
    let g = SNAPSHOT.lock().unwrap();
    g.as_ref().map(|s| MainSnapshot {
        position: s.position,
        final_normed: s.final_normed.clone(),
        top5: s.top5.clone(),
        argmax: s.argmax,
    })
}

/// L2 norm of a vector.
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt() as f32
}

/// Cosine similarity. Returns 0 if either vector is zero-norm.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::NAN;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let xf = *x as f64;
        let yf = *y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
    (dot / denom) as f32
}

/// Relative L2 difference `||a - b|| / ||a||`.
pub fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::NAN;
    }
    let mut diff_sq = 0.0f64;
    let mut a_sq = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as f64) - (*y as f64);
        diff_sq += d * d;
        a_sq += (*x as f64) * (*x as f64);
    }
    let denom = a_sq.sqrt().max(1e-12);
    (diff_sq.sqrt() / denom) as f32
}

/// Top-K Jaccard index of two ranked id lists. `k` is the truncation depth.
pub fn topk_jaccard(a: &[(u32, f32)], b: &[(u32, f32)], k: usize) -> f32 {
    let kk = k.min(a.len()).min(b.len());
    if kk == 0 {
        return 0.0;
    }
    let mut set_a: std::collections::HashSet<u32> = a.iter().take(kk).map(|p| p.0).collect();
    let set_b: std::collections::HashSet<u32> = b.iter().take(kk).map(|p| p.0).collect();
    let inter = set_a.iter().filter(|id| set_b.contains(id)).count();
    set_a.extend(set_b.iter());
    if set_a.is_empty() {
        0.0
    } else {
        inter as f32 / set_a.len() as f32
    }
}

/// Decode a contiguous BF16 buffer (little-endian u16) into f32 values.
pub fn bf16_bytes_to_f32(buf: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(buf.len() / 2);
    for c in buf.chunks_exact(2) {
        let bits = u16::from_le_bytes([c[0], c[1]]);
        out.push(f32::from_bits((bits as u32) << 16));
    }
    out
}
