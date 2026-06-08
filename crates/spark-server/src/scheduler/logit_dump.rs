// SPDX-License-Identifier: AGPL-3.0-only

//! Complete per-decode-step logit dump for Atlas↔vLLM divergence analysis.
//!
//! Gated by `ATLAS_LOGIT_DUMP=<file>`. When set, every decode step appends
//! one JSONL record capturing the FULL pre/post-processor picture:
//!   - `raw_topk`:  top-K (id, logit) of the raw model logits, BEFORE Atlas's
//!                  additive logit-bias stack (WS mask, attractor, A4 think
//!                  suppression, C4 lift, …). This is the model's own
//!                  distribution — diff it against vLLM's raw top-K to see
//!                  MODEL divergence.
//!   - `bias`:      every (id, delta) Atlas applied this step. vLLM applies
//!                  none of these — so this list IS the Atlas-only processor
//!                  divergence, itemized.
//!   - `post_argmax`: argmax after applying `bias` to the raw logits (the
//!                  additive part of post-processing; multiplicative min_p /
//!                  penalties happen inside the sampler and are not replayed
//!                  here — `sampled` reflects the true final pick).
//!   - `sampled`:   the token actually chosen.
//!   - `in_body` / `chars`: tool-param-body context, so dumps can be sliced
//!                  to the structured-content region where wandering occurs.
//!
//! vLLM's patched sampler writes the same `raw_topk` + `sampled` shape, so a
//! per-step diff localizes whether a divergence is the model (raw_topk
//! differs) or an Atlas processor (raw_topk matches but `bias` flips it).

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::{Mutex, OnceLock};

static DUMP: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();

fn writer() -> Option<&'static Mutex<BufWriter<File>>> {
    DUMP.get_or_init(|| {
        let path = std::env::var("ATLAS_LOGIT_DUMP").ok()?;
        if path.is_empty() {
            return None;
        }
        match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(f) => Some(Mutex::new(BufWriter::new(f))),
            Err(e) => {
                tracing::error!("ATLAS_LOGIT_DUMP: cannot open {path}: {e}");
                None
            }
        }
    })
    .as_ref()
}

/// Whether `ATLAS_LOGIT_DUMP` is active (used to skip the per-step bias
/// clone in the hot path when dumping is off).
pub(crate) fn enabled() -> bool {
    writer().is_some()
}

/// Returns the top-`k` (index, logit) pairs by logit, descending.
fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let kk = k.min(idx.len());
    idx.select_nth_unstable_by(kk.saturating_sub(1).max(0), |&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut top: Vec<(u32, f32)> =
        idx.into_iter().take(kk).map(|i| (i, logits[i as usize])).collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top
}

/// Append one per-step record. No-op unless `ATLAS_LOGIT_DUMP` is set.
#[allow(clippy::too_many_arguments)]
pub(crate) fn record(
    step: usize,
    in_body: bool,
    chars: usize,
    raw_logits: &[f32],
    bias: &[(u32, f32)],
    sampled: u32,
) {
    let Some(w) = writer() else {
        return;
    };
    const K: usize = 12;
    let raw_topk = top_k(raw_logits, K);
    // Post-bias argmax (additive part only).
    let mut post_best = (0u32, f32::NEG_INFINITY);
    for (i, &l) in raw_logits.iter().enumerate() {
        let mut v = l;
        for &(tok, d) in bias {
            if tok as usize == i {
                v += d;
            }
        }
        if v > post_best.1 {
            post_best = (i as u32, v);
        }
    }
    // Hand-built JSON to avoid pulling serde into the hot path for arrays.
    let mut s = String::with_capacity(256);
    s.push_str(&format!(
        "{{\"step\":{step},\"in_body\":{in_body},\"chars\":{chars},\"sampled\":{sampled},\"post_argmax\":{},\"raw_topk\":[",
        post_best.0
    ));
    for (n, (id, lg)) in raw_topk.iter().enumerate() {
        if n > 0 {
            s.push(',');
        }
        s.push_str(&format!("[{id},{lg:.4}]"));
    }
    s.push_str("],\"bias\":[");
    for (n, (id, d)) in bias.iter().enumerate() {
        if n > 0 {
            s.push(',');
        }
        s.push_str(&format!("[{id},{d:.4}]"));
    }
    s.push_str("]}\n");
    if let Ok(mut guard) = w.lock() {
        let _ = guard.write_all(s.as_bytes());
        let _ = guard.flush();
    }
}
