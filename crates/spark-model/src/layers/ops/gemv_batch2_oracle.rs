// SPDX-License-Identifier: AGPL-3.0-only
//! Kill-bar for `w4a16_gemv_batch2` prefetch candidates.
//!
//! SSOT for the 3% regression / 3% win thresholds the cold-DRAM oracle
//! prints. GPU timing stays in the example; this module is host-testable
//! so a candidate PR cannot silently change the bar.

/// STREAM read ceiling on this GB10 (GB/s). Override with `ATLAS_STREAM_GBPS`.
pub const STREAM_GBPS_DEFAULT: f64 = 230.0;
/// Datasheet LPDDR5X peak (GB/s). Override with `ATLAS_PEAK_GBPS`.
pub const PEAK_GBPS_DEFAULT: f64 = 273.0;
/// `candidate_us / batch2_us` above this is a regression. Do not default-on.
pub const FAIL_SLOWER: f64 = 1.03;
/// `batch2_us / candidate_us` at or above this is a win. Default-on still
/// requires bit-identity in a separate microtest.
pub const WIN_FASTER: f64 = 1.03;

/// Parse `ATLAS_GEMV_BATCH2_CANDIDATE=module:kernel`. Empty sides are reject.
pub fn parse_candidate(spec: &str) -> Option<(&str, &str)> {
    let (module, kernel) = spec.split_once(':')?;
    if module.is_empty() || kernel.is_empty() {
        return None;
    }
    Some((module, kernel))
}

/// Parse a positive GB/s override. Junk / zero / absent → `default`.
pub fn gbps_from(raw: Option<&str>, default: f64) -> f64 {
    raw.and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(default)
}

/// Per-shape verdict from two GPU wall times (same units).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OracleVerdict {
    /// ≥3% faster than template batch2.
    Win,
    /// Within ±3%.
    Neutral,
    /// >3% slower. The example exits 1.
    Fail,
}

pub fn verdict(candidate_us: f64, batch2_us: f64) -> OracleVerdict {
    if !candidate_us.is_finite()
        || !batch2_us.is_finite()
        || candidate_us <= 0.0
        || batch2_us <= 0.0
    {
        return OracleVerdict::Fail;
    }
    let vs = candidate_us / batch2_us;
    if vs > FAIL_SLOWER {
        OracleVerdict::Fail
    } else if batch2_us / candidate_us >= WIN_FASTER {
        OracleVerdict::Win
    } else {
        OracleVerdict::Neutral
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_candidate_requires_both_sides() {
        assert_eq!(
            parse_candidate("w4a16_gemv_batch2_dualissue:w4a16_gemv_batch2_dualissue"),
            Some(("w4a16_gemv_batch2_dualissue", "w4a16_gemv_batch2_dualissue"))
        );
        assert_eq!(parse_candidate(""), None);
        assert_eq!(parse_candidate("nocolon"), None);
        assert_eq!(parse_candidate(":kernel"), None);
        assert_eq!(parse_candidate("module:"), None);
    }

    #[test]
    fn gbps_from_rejects_junk_and_non_positive() {
        assert_eq!(gbps_from(None, STREAM_GBPS_DEFAULT), STREAM_GBPS_DEFAULT);
        assert_eq!(
            gbps_from(Some(""), STREAM_GBPS_DEFAULT),
            STREAM_GBPS_DEFAULT
        );
        assert_eq!(
            gbps_from(Some("abc"), STREAM_GBPS_DEFAULT),
            STREAM_GBPS_DEFAULT
        );
        assert_eq!(
            gbps_from(Some("0"), STREAM_GBPS_DEFAULT),
            STREAM_GBPS_DEFAULT
        );
        assert_eq!(
            gbps_from(Some("-1"), STREAM_GBPS_DEFAULT),
            STREAM_GBPS_DEFAULT
        );
        assert_eq!(gbps_from(Some("240"), STREAM_GBPS_DEFAULT), 240.0);
        assert_eq!(gbps_from(Some("273"), PEAK_GBPS_DEFAULT), 273.0);
    }

    #[test]
    fn verdict_497_in_proj_is_fail() {
        // Published #497 cold-DRAM: 72.4 us cpasync vs 62.3 us batch2.
        assert_eq!(verdict(72.4, 62.3), OracleVerdict::Fail);
    }

    #[test]
    fn verdict_exact_3pct_slower_is_neutral() {
        assert_eq!(verdict(62.3 * FAIL_SLOWER, 62.3), OracleVerdict::Neutral);
    }

    #[test]
    fn verdict_just_over_3pct_slower_is_fail() {
        assert_eq!(verdict(62.3 * 1.031, 62.3), OracleVerdict::Fail);
    }

    #[test]
    fn verdict_exact_3pct_faster_is_win() {
        assert_eq!(verdict(62.3 / WIN_FASTER, 62.3), OracleVerdict::Win);
    }

    #[test]
    fn verdict_tie_is_neutral() {
        assert_eq!(verdict(62.3, 62.3), OracleVerdict::Neutral);
    }

    #[test]
    fn verdict_non_finite_or_non_positive_is_fail() {
        assert_eq!(verdict(f64::NAN, 62.3), OracleVerdict::Fail);
        assert_eq!(verdict(62.3, 0.0), OracleVerdict::Fail);
        assert_eq!(verdict(-1.0, 62.3), OracleVerdict::Fail);
    }
}
