// SPDX-License-Identifier: AGPL-3.0-only

//! Stable repository paths for default and model-variant gate records.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::gate_dir;
use super::record::{GateBaseline, GateRecord};

/// `YYYY-MM-DD` (UTC) from unix seconds.
pub fn date_of(unix_secs: u64) -> String {
    let days = (unix_secs / 86_400) as i64 + 719_468;
    let era = days.div_euclid(146_097);
    let doe = days.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

/// The default-variant filename: `YYYY-MM-DD-<sha>.json`.
pub fn record_path(root: &Path, benchmark_id: &str, unix_secs: u64, sha: &str) -> PathBuf {
    gate_dir(root, benchmark_id).join(format!("{}-{sha}.json", date_of(unix_secs)))
}

/// A readable filename-safe, deliberately lossy checkpoint slug.
pub fn variant_slug(model: &str) -> String {
    let mut out = String::with_capacity(model.len());
    for c in model.chars() {
        let mapped = if c.is_ascii_alphanumeric() || c == '.' {
            c.to_ascii_lowercase()
        } else {
            '-'
        };
        if mapped == '-' && out.ends_with('-') {
            continue;
        }
        out.push(mapped);
    }
    out.trim_matches('-').to_string()
}

/// Preserve historical slugs unless two declared checkpoints collapse to one.
fn variant_file_slug(baseline: &GateBaseline, model: &str) -> String {
    let slug = variant_slug(model);
    let collides = baseline
        .hardware
        .values()
        .flat_map(|hardware| hardware.models.keys())
        .any(|other| other != model && variant_slug(other) == slug);
    if !collides {
        return slug;
    }
    let digest = format!("{:x}", Sha256::digest(model.as_bytes()));
    format!("{slug}-{}", &digest[..16])
}

/// The record path keyed by benchmark, declared variant, day, and commit.
pub fn record_path_for(root: &Path, record: &GateRecord) -> PathBuf {
    let legacy = record_path(
        root,
        &record.benchmark_id,
        record.recorded_at,
        &record.git_sha,
    );
    let Ok(baseline) = super::bench::baseline_for(root, &record.benchmark_id) else {
        return legacy;
    };
    if baseline.hardware.is_empty() {
        return legacy;
    }
    let hardware = record.hardware.gate_key();
    let is_default = match baseline.hardware.get(&hardware) {
        Some(hw) => hw.default == record.target_model,
        None => baseline
            .hardware
            .values()
            .any(|hw| hw.default == record.target_model),
    };
    if is_default {
        legacy
    } else {
        gate_dir(root, &record.benchmark_id).join(format!(
            "{}-{}-{}.json",
            date_of(record.recorded_at),
            record.git_sha,
            variant_file_slug(&baseline, &record.target_model)
        ))
    }
}
