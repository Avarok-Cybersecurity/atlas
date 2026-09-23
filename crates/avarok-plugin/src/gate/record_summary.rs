// SPDX-License-Identifier: AGPL-3.0-only

//! The one-line summary a gate record carries, and the clock it is stamped
//! with. Split out of `record.rs` at the 500-line cap — an exact piecewise
//! move, no logic changed.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::history::RunRecord;

/// The one line a future reader sees first. States the headline numbers and,
/// when the frame logged warnings, the first one — those are the observations
/// worth carrying into the next run's context.
pub(super) fn summarize(record: &RunRecord) -> String {
    let frame = &record.frame;
    let numbers: Vec<String> = frame
        .metrics
        .iter()
        .map(|(k, v)| format!("{k}={v:.2}"))
        .collect();
    let numbers = if numbers.is_empty() {
        "no metrics".to_string()
    } else {
        numbers.join(", ")
    };
    let warning = frame
        .log
        .iter()
        .find(|l| {
            matches!(
                l.level,
                crate::result::LogLevel::Warn | crate::result::LogLevel::Error
            )
        })
        .map(|l| format!(" · warning: {}", l.text));
    let verdict = frame
        .verdict
        .as_ref()
        .map(|v| format!("{:?}: {}", v.kind, v.reason))
        .unwrap_or_else(|| "no verdict".into());
    format!(
        "{} · {} · {}{}",
        record.target_model,
        numbers,
        verdict,
        warning.unwrap_or_default()
    )
}

/// Unix seconds now.
pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}
