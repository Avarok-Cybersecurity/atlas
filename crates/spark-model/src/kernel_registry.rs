// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel-selection telemetry.
//!
//! Many components in the model construction path pick between
//! precision variants of the same logical kernel (e.g. GDN forward in
//! BF16 vs FP32, attention QKV in NVFP4 / FP8-block-scaled / BF16,
//! LM head in NVFP4 vs BF16). These choices are usually correct, but
//! they ARE silent fallbacks when a preferred kernel module isn't
//! built for the active target — and silent fallbacks have caused
//! multi-day debugging sessions (e.g. dense Qwen 3.6 27B FP8 hit the
//! BF16 GDN path because `gated_delta_rule_decode_f32` was probed
//! via `try_kernel` and returned `KernelHandle(0)` on miss, producing
//! a "decode coherent through ~1500 tokens then degrades into
//! malformed CSS syntax" symptom that masqueraded as an attention
//! bug for days).
//!
//! This module gives each construction site a one-line API to
//! record "I'm using kernel `<name>` for component `<role>` because
//! `<reason>`" — and a startup-time dumper that emits the full table
//! plus loud warnings for any non-preferred-path entries. Future
//! silent fallbacks become loud at first boot.
//!
//! The registry is a process-global `OnceLock<Mutex<Vec<…>>>` and is
//! dumped once near the end of `factory::build`. Cost is negligible —
//! at most a few dozen entries per model.

use std::sync::{Mutex, OnceLock};

use comfy_table::{Cell, Color, ContentArrangement, Table, presets::UTF8_FULL_CONDENSED};

/// One kernel-selection event recorded during model construction.
#[derive(Debug, Clone)]
pub struct KernelChoice {
    /// Logical component (e.g. "GDN forward", "Attention Q proj",
    /// "LM head (main)", "MTP head lm_head"). Free-form but should be
    /// stable across runs so the table is greppable.
    pub component: String,
    /// Selected kernel module::function pair (e.g.
    /// "gated_delta_rule::gated_delta_rule_decode_f32"). For the
    /// rare cases where the choice is not a CUDA kernel but a
    /// higher-level path (e.g. "BF16 dense_gemv lm_head"), use a
    /// human-readable label here — the table is for humans first.
    pub kernel: String,
    /// "preferred" / "fallback" / "only-option" — drives both the
    /// row colour and whether a `tracing::warn!` is emitted at
    /// table-dump time.
    pub status: ChoiceStatus,
    /// Why this kernel was picked (e.g. "F32 GDN kernel registered
    /// for this target", "F32 GDN kernel not loaded — falling back
    /// to BF16; long-context (>1500 tok) decode will drift",
    /// "MODEL.toml opts out of LM-head quantization"). One short
    /// sentence; this is the row's last column.
    pub reason: String,
}

/// Status of a kernel choice — drives table colour + warn-vs-info logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChoiceStatus {
    /// The preferred path is engaged (e.g. native FP8 for an FP8
    /// checkpoint; F32 GDN for a long-context-sensitive model).
    Preferred,
    /// A silent fallback fired (e.g. F32 GDN kernel not built →
    /// dropped to BF16; native FP8 not detected → quantized to NVFP4).
    /// This is the case we most want to surface.
    Fallback,
    /// Only one path exists for this component; the row is
    /// informational only (e.g. a kernel that has no alternatives).
    OnlyOption,
}

static REGISTRY: OnceLock<Mutex<Vec<KernelChoice>>> = OnceLock::new();

fn registry() -> &'static Mutex<Vec<KernelChoice>> {
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

/// Record one kernel selection.
///
/// Insert is dedup-by-(component) — the first writer wins, so per-layer
/// records collapse into a single entry. Layer 0 is the canonical record
/// site; later layers re-record but are ignored.
pub fn record(component: impl Into<String>, kernel: impl Into<String>, status: ChoiceStatus, reason: impl Into<String>) {
    let component = component.into();
    let mut g = registry().lock().expect("kernel registry mutex poisoned");
    if g.iter().any(|c| c.component == component) {
        return;
    }
    g.push(KernelChoice {
        component,
        kernel: kernel.into(),
        status,
        reason: reason.into(),
    });
}

/// Convenience: record a preferred-path selection.
pub fn record_preferred(component: impl Into<String>, kernel: impl Into<String>, reason: impl Into<String>) {
    record(component, kernel, ChoiceStatus::Preferred, reason);
}

/// Convenience: record a silent-fallback selection. Also emits a
/// `tracing::warn!` so any single fallback shows up in logs even
/// without the startup table.
pub fn record_fallback(component: impl Into<String>, kernel: impl Into<String>, reason: impl Into<String>) {
    let component = component.into();
    let kernel = kernel.into();
    let reason = reason.into();
    tracing::warn!(
        "kernel fallback: component={component:?} kernel={kernel:?} reason={reason:?}"
    );
    record(component, kernel, ChoiceStatus::Fallback, reason);
}

/// Convenience: record a no-choice (only one path) selection.
pub fn record_only(component: impl Into<String>, kernel: impl Into<String>, reason: impl Into<String>) {
    record(component, kernel, ChoiceStatus::OnlyOption, reason);
}

/// Print the recorded kernel-selection table to stdout (one block,
/// once per process). Call at the tail of model construction.
///
/// Also emits `tracing::warn!` for every `Fallback` row so they're
/// captured by structured-logging consumers (the structured log is
/// the duplicate of the per-`record_fallback` warn — kept here so
/// the table-dump path is self-contained for ad-hoc diagnostics).
pub fn dump() {
    let g = registry().lock().expect("kernel registry mutex poisoned");
    if g.is_empty() {
        tracing::info!("kernel-selection registry is empty (no record() calls made)");
        return;
    }
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL_CONDENSED)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Component", "Selected kernel", "Status", "Reason"]);
    let mut fallback_count = 0usize;
    for c in g.iter() {
        let status_cell = match c.status {
            ChoiceStatus::Preferred => Cell::new("preferred").fg(Color::Green),
            ChoiceStatus::Fallback => {
                fallback_count += 1;
                Cell::new("FALLBACK").fg(Color::Yellow)
            }
            ChoiceStatus::OnlyOption => Cell::new("only").fg(Color::DarkGrey),
        };
        table.add_row(vec![
            Cell::new(&c.component),
            Cell::new(&c.kernel),
            status_cell,
            Cell::new(&c.reason),
        ]);
    }
    let line_count = g.len();
    drop(g);
    // Emit as a single multi-line tracing::info! so the table stays
    // visually contiguous in the structured log stream.
    tracing::info!(
        "Kernel selection table ({} entr{}, {} fallback{}):\n{}",
        line_count,
        if line_count == 1 { "y" } else { "ies" },
        fallback_count,
        if fallback_count == 1 { "" } else { "s" },
        table
    );
}
