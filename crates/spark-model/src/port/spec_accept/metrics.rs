// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the `atlas_mtp_device_accept_*` `/metrics` series (lane L12) and the fallback
//! reasons they are labeled by.
//!
//! Process-wide atomics, rendered by `spark-server`'s `api::port_metrics::render`. Every reason is
//! always printed (zeros included) so the series set does not change when a reason first fires. With
//! the `mtp_device_accept` row off nothing is ever recorded and every series reads 0.

use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

/// Why one verify call kept the host pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// A grammar is active (bitmask, forced tokens and matcher advances are host-only).
    Grammar,
    /// The request carries a `logit_bias`.
    LogitBias,
    /// An LZ or DRY penalty is live.
    LzDry,
    /// A stateful thinking stage is armed (F2 confidence stop, forced `</think>`).
    ThinkStage,
    /// The legacy pin to `<tool_call>` after `</think>` could fire.
    ToolPin,
    /// A diagnostic mode wants the host pipeline (`AVAROK_FORCE_TEMP_ZERO`, AdaDec recording).
    Diagnostic,
    /// Sampler parameters outside the kernel's set (top-n-sigma, top-k above the candidate cap, a
    /// non-finite or non-positive temperature).
    SamplerParams,
    /// More edited tokens than the shared-memory budget holds.
    Edits,
    /// The vocabulary is too large for the shared-memory bitmap.
    Vocab,
    /// More verify positions than the kernel's request holds.
    Positions,
    /// The logits buffer is FP32 (the kernel reads BF16).
    Fp32Logits,
    /// No device path (model without the port extension, kernel not built for this target).
    Unavailable,
    /// The launch or a copy failed.
    DeviceError,
    /// A sampled job found no finite logit.
    Empty,
    /// A sampled job's support did not close within the candidate cap.
    Support,
}

impl Fallback {
    /// Every reason, in label order.
    pub const ALL: [Fallback; 15] = [
        Fallback::Grammar,
        Fallback::LogitBias,
        Fallback::LzDry,
        Fallback::ThinkStage,
        Fallback::ToolPin,
        Fallback::Diagnostic,
        Fallback::SamplerParams,
        Fallback::Edits,
        Fallback::Vocab,
        Fallback::Positions,
        Fallback::Fp32Logits,
        Fallback::Unavailable,
        Fallback::DeviceError,
        Fallback::Empty,
        Fallback::Support,
    ];

    /// The `reason` label value.
    pub fn label(self) -> &'static str {
        match self {
            Fallback::Grammar => "grammar",
            Fallback::LogitBias => "logit_bias",
            Fallback::LzDry => "lz_dry",
            Fallback::ThinkStage => "think_stage",
            Fallback::ToolPin => "tool_pin",
            Fallback::Diagnostic => "diagnostic",
            Fallback::SamplerParams => "sampler_params",
            Fallback::Edits => "edits",
            Fallback::Vocab => "vocab",
            Fallback::Positions => "positions",
            Fallback::Fp32Logits => "fp32_logits",
            Fallback::Unavailable => "unavailable",
            Fallback::DeviceError => "device_error",
            Fallback::Empty => "empty",
            Fallback::Support => "support",
        }
    }

    fn slot(self) -> usize {
        Fallback::ALL.iter().position(|&r| r == self).unwrap_or(0)
    }
}

static ROWS: AtomicU64 = AtomicU64::new(0);
static FALLBACKS: [AtomicU64; 15] = [const { AtomicU64::new(0) }; 15];

/// Count `n` verify rows picked on the device.
pub fn record_rows(n: usize) {
    ROWS.fetch_add(n as u64, Ordering::Relaxed);
}

/// Count one verify call that kept the host pipeline.
pub fn record_fallback(r: Fallback) {
    FALLBACKS[r.slot()].fetch_add(1, Ordering::Relaxed);
}

/// Rows picked on the device so far.
pub fn rows() -> u64 {
    ROWS.load(Ordering::Relaxed)
}

/// Host fallbacks so far for one reason.
pub fn fallbacks(r: Fallback) -> u64 {
    FALLBACKS[r.slot()].load(Ordering::Relaxed)
}

/// Append both series in Prometheus text format.
pub fn render_prometheus(out: &mut String) {
    let _ = writeln!(
        out,
        "# HELP atlas_mtp_device_accept_rows_total Verify rows picked by device-side speculative acceptance (row mtp_device_accept).\n\
         # TYPE atlas_mtp_device_accept_rows_total counter\n\
         atlas_mtp_device_accept_rows_total {}",
        rows()
    );
    let _ = writeln!(
        out,
        "# HELP atlas_mtp_device_accept_fallback_total Verify calls that kept the host pipeline with mtp_device_accept on, by reason.\n\
         # TYPE atlas_mtp_device_accept_fallback_total counter"
    );
    for r in Fallback::ALL {
        let _ = writeln!(
            out,
            "atlas_mtp_device_accept_fallback_total{{reason=\"{}\"}} {}",
            r.label(),
            fallbacks(r)
        );
    }
}
