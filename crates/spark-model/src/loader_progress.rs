// SPDX-License-Identifier: AGPL-3.0-only

//! Loader progress bar — one global `indicatif::ProgressBar` that the
//! per-loader layer-load loops increment as each layer's weights move
//! GPU-resident.
//!
//! Why a global: every weight loader (`qwen35_dense`, `qwen35`,
//! `qwen3_vl`, `gemma4`, `minimax`, `mistral`, `nemotron`,
//! `qwen3_next`) has its own per-layer iteration but the trait
//! signature doesn't let `factory::build_model` hand a progress
//! handle in. A process-global `OnceLock` keeps the surface change
//! tiny — `start(total)` from factory before `loader.load_layers`,
//! `inc()` once per layer inside the loop, `finish()` at the end.
//! Loaders that aren't yet wired silently no-op (loaders unaware of
//! the bar continue to log their existing "Loaded layers 0..N"
//! lines).

use std::sync::{Mutex, OnceLock};

use indicatif::{ProgressBar, ProgressStyle};

static BAR: OnceLock<Mutex<Option<ProgressBar>>> = OnceLock::new();

fn slot() -> &'static Mutex<Option<ProgressBar>> {
    BAR.get_or_init(|| Mutex::new(None))
}

/// Begin a new progress bar for an `n_layers`-layer model load. Replaces
/// any prior bar (the previous one is finished). Safe to call once at the
/// start of model construction.
pub fn start(n_layers: usize) {
    let pb = ProgressBar::new(n_layers as u64);
    // Plain ASCII (no Unicode block chars) — the bar renders into the
    // container's stderr, which is often captured by Docker / journald
    // without a UTF-8 locale.
    let style = ProgressStyle::with_template(
        "  loading [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=>-");
    pb.set_style(style);
    pb.set_message("layers");
    let mut g = slot().lock().expect("loader_progress mutex poisoned");
    if let Some(old) = g.take() {
        old.finish_and_clear();
    }
    *g = Some(pb);
}

/// Advance the bar by one. No-op if `start()` was never called.
pub fn inc() {
    if let Some(pb) = slot().lock().expect("loader_progress mutex poisoned").as_ref() {
        pb.inc(1);
    }
}

/// Append a hint to the bar's right-hand message (e.g. memory remaining).
/// Truncated by the renderer if too long. No-op if the bar isn't active.
pub fn set_message(msg: impl Into<String>) {
    if let Some(pb) = slot().lock().expect("loader_progress mutex poisoned").as_ref() {
        pb.set_message(msg.into());
    }
}

/// Finish + clear the bar. Safe to call multiple times. Always called
/// at the end of `factory::build_model` so the bar doesn't outlive
/// model construction even on error.
pub fn finish() {
    if let Some(pb) = slot().lock().expect("loader_progress mutex poisoned").take() {
        pb.finish_and_clear();
    }
}
