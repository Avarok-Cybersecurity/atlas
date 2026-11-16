// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the `atlas_mtp_device_accept_*` `/metrics` series (lane L12) and the fallback
//! reasons they are labeled by.
//!
//! Process-wide atomics, rendered by `spark-server`'s `api::port_metrics::render`. Every reason is
//! always printed (zeros included) so the series set does not change when a reason first fires. With
//! the `mtp_device_accept` row off nothing is ever recorded and every series reads 0.
//!
//! With the row on, every verify pass is accounted for: its rows are counted, or a reason is. The
//! verify pick helper consults the device path before anything else, so each of its calls records
//! one or the other, and the scheduler's per-step check ([`events`], [`Fallback::NotConsulted`])
//! covers a verify pass that never reached the helper. An earlier GPU run on this card is what
//! this prevents: 675 verify passes with 0 rows and every reason 0, because a greedy shortcut
//! answered every call ahead of the hook.

use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

/// Why one verify call was not picked by the device acceptance path.
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
    /// A verify pass made its picks without the verify pick helper at all (DFlash raw-argmax verify
    /// is one), so the device path was never asked. Counted by the scheduler's per-step check, once
    /// per such pass.
    NotConsulted,
}

impl Fallback {
    /// Every reason, in label order.
    pub const ALL: [Fallback; 16] = [
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
        Fallback::NotConsulted,
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
            Fallback::NotConsulted => "not_consulted",
        }
    }

    fn slot(self) -> usize {
        Fallback::ALL.iter().position(|&r| r == self).unwrap_or(0)
    }
}

static ROWS: AtomicU64 = AtomicU64::new(0);
static FALLBACKS: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];

/// Count `n` verify rows picked on the device.
pub fn record_rows(n: usize) {
    ROWS.fetch_add(n as u64, Ordering::Relaxed);
}

/// Count one verify call the device path did not pick.
pub fn record_fallback(r: Fallback) {
    record_fallbacks(r, 1);
}

/// Count `n` such calls at once (the per-step check, one per unaccounted verify pass).
pub fn record_fallbacks(r: Fallback, n: u64) {
    FALLBACKS[r.slot()].fetch_add(n, Ordering::Relaxed);
}

/// Everything recorded so far: device rows plus every reason. It only grows, and with the row on
/// it grows on every verify pass, which is what the scheduler's per-step check holds it to.
pub fn events() -> u64 {
    rows() + Fallback::ALL.iter().map(|&r| fallbacks(r)).sum::<u64>()
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
        "# HELP atlas_mtp_device_accept_fallback_total Verify calls the device acceptance path did not pick with mtp_device_accept on, by reason.\n\
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_reason_has_its_own_label_and_slot() {
        let mut labels: Vec<&str> = Fallback::ALL.iter().map(|r| r.label()).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), Fallback::ALL.len());
        for (i, r) in Fallback::ALL.iter().enumerate() {
            assert_eq!(r.slot(), i, "{}", r.label());
        }
        assert_eq!(Fallback::NotConsulted.label(), "not_consulted");
    }

    #[test]
    fn events_grow_with_every_row_and_every_reason() {
        // The counters are process-wide and other tests add to them, so only
        // growth is asserted.
        let before = events();
        record_rows(3);
        assert!(events() >= before + 3);
        for r in Fallback::ALL {
            let (events_before, reason_before) = (events(), fallbacks(r));
            record_fallback(r);
            assert!(events() > events_before, "{}", r.label());
            assert!(fallbacks(r) > reason_before, "{}", r.label());
        }
        let not_consulted = fallbacks(Fallback::NotConsulted);
        record_fallbacks(Fallback::NotConsulted, 5);
        assert!(fallbacks(Fallback::NotConsulted) >= not_consulted + 5);
    }

    #[test]
    fn the_rendered_series_names_every_reason() {
        let mut out = String::new();
        render_prometheus(&mut out);
        assert!(out.contains("atlas_mtp_device_accept_rows_total "));
        for r in Fallback::ALL {
            let series = format!("reason=\"{}\"", r.label());
            assert!(out.contains(&series), "{series}");
        }
    }
}
