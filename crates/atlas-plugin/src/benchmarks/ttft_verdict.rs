// SPDX-License-Identifier: AGPL-3.0-only

//! How a TTFT run is JUDGED, as opposed to how it is measured.
//!
//! Split out of `ttft.rs` for the repo's 500-line cap, along the seam bfcl
//! already uses (`bfcl/report.rs`): the state machine collects samples, and
//! this decides what they mean. A child module of `ttft`, so it reads the
//! gate's own fields rather than growing accessors for them.

use super::super::{baseline, stats};
use super::TtftGate;
use crate::result::{CellStyle, Stat, Verdict, VerdictKind};

impl TtftGate {
    /// Whether this run's numbers become the box's new baseline.
    ///
    /// ★ **A REGRESSION MUST NOT BECOME THE NEW BAR.** The stored baseline is
    /// what the NEXT run is measured against, so writing it from a run that
    /// just failed makes the same regressed build pass on the second attempt —
    /// a gate that walks itself down one re-run at a time, and the record then
    /// says PASS. An `Info` run still stores: that is the "no baseline on this
    /// box yet" case, which exists precisely to create one.
    pub(super) fn should_store(&self, verdict: &Verdict) -> bool {
        self.update_baseline && verdict.kind != VerdictKind::Fail
    }

    /// Compare against the stored baseline and decide. `None` baseline is
    /// `Info`, never a green PASS — a gate with nothing to compare against has
    /// not passed anything.
    pub(super) fn verdict(&self, median: Option<f64>, p90: Option<f64>) -> (Verdict, Vec<Stat>) {
        let store = match self.handle() {
            Ok(h) => h.artifacts().clone(),
            Err(_) => return (Verdict::info("no handle"), Vec::new()),
        };
        let id = self.mode.descriptor().id;
        // Model-keyed: a gate with several checkpoints (BENCH.toml `hw.models`)
        // otherwise has its variants overwrite one shared baseline.json, and the
        // run after every variant switch finds another target's numbers,
        // declines to compare, and reports `info` instead of a verdict.
        let model_now = self.handle().ok().map(|h| h.target().model.clone());
        let stored = baseline::load_for(&store, id, model_now.as_deref());
        let mut summary = vec![
            Stat::new("Median TTFT", stats::fmt_ms(median), "ms").with_style(CellStyle::Accent),
            Stat::new("p90 TTFT", stats::fmt_ms(p90), "ms"),
        ];
        if !median.is_some_and(super::valid_ttft_ms) || !p90.is_some_and(super::valid_ttft_ms) {
            return (
                Verdict::fail("run produced no usable median and p90 TTFT measurements"),
                summary,
            );
        }
        let Some(base) = stored else {
            summary.push(Stat::new("Baseline", "none", "").with_style(CellStyle::Dim));
            return (
                Verdict::info("no baseline on this box yet — this run is recorded as the baseline"),
                summary,
            );
        };
        let target_now = self
            .handle()
            .map(|h| h.target().base_url.clone())
            .unwrap_or_default();
        let model_now = self
            .handle()
            .map(|h| h.target().model.clone())
            .unwrap_or_default();
        if !super::ttft_target::same_box(&base.target, &target_now) || base.model != model_now {
            summary.push(Stat::new("Baseline", "other target", "").with_style(CellStyle::Warn));
            return (
                Verdict::info(format!(
                    "baseline was recorded against {} / {} — not comparable, reporting only",
                    base.target, base.model
                )),
                summary,
            );
        }
        if !base.get("median_ms").is_some_and(super::valid_ttft_ms)
            || !base.get("p90_ms").is_some_and(super::valid_ttft_ms)
        {
            summary.push(Stat::new("Baseline", "incomplete", "").with_style(CellStyle::Warn));
            return (
                Verdict::info(
                    "same-box baseline is missing usable median_ms or p90_ms — not comparable, \
                     reporting only",
                ),
                summary,
            );
        }
        let dm = stats::pct_delta(median, base.get("median_ms"));
        let dp = stats::pct_delta(p90, base.get("p90_ms"));
        summary.push(
            Stat::new(
                "vs baseline",
                dm.map(|d| format!("{d:+.1}")).unwrap_or_else(|| "—".into()),
                format!("% median · {}", base.age_text()),
            )
            .with_style(match dm {
                Some(d) if d > self.median_limit_pct => CellStyle::Bad,
                Some(d) if d < 0.0 => CellStyle::Good,
                _ => CellStyle::Neutral,
            }),
        );
        // A percentage alone cannot gate a fast endpoint. At a 30 ms TTFT the
        // +3% limit is 0.9 ms — below host scheduling jitter — so ordinary
        // noise reads as REGRESSED. On a real serve (TTFT ~1.5 s) the floor is
        // never the binding constraint; it only suppresses deltas too small to
        // have been measured. A regression must clear BOTH tests to fail.
        let over_floor = |now: Option<f64>, key: &str| {
            now.zip(base.get(key))
                .is_some_and(|(now, was)| now - was > Self::NOISE_FLOOR_MS)
        };
        let median_bad =
            dm.is_some_and(|d| d > self.median_limit_pct) && over_floor(median, "median_ms");
        let p90_bad = dp.is_some_and(|d| d > self.p90_limit_pct) && over_floor(p90, "p90_ms");
        let detail = format!(
            "median {} (limit +{:.1}%) · p90 {} (limit +{:.1}%)",
            dm.map(|d| format!("{d:+.1}%"))
                .unwrap_or_else(|| "—".into()),
            self.median_limit_pct,
            dp.map(|d| format!("{d:+.1}%"))
                .unwrap_or_else(|| "—".into()),
            self.p90_limit_pct,
        );
        if median_bad || p90_bad {
            (Verdict::fail(format!("REGRESSED — {detail}")), summary)
        } else {
            (Verdict::pass(detail), summary)
        }
    }
}
