// SPDX-License-Identifier: AGPL-3.0-only

//! The screen's number and enum formatting.

use super::*;

/// The bug this module exists for: 20 000 000 000 bytes is `18.6 GB`, and it
/// has to be `18.6 GB` in the download line and in the Library card, because
/// it is the same file two seconds apart. A decimal divisor renders it
/// `20.0 GB` and the size appears to change when the download completes.
#[test]
fn one_file_is_one_size_wherever_it_is_shown() {
    assert_eq!(bytes(20_000_000_000), "18.6 GB");
    assert_eq!(
        bytes(20_000_000_000),
        crate::tui::data::library::human_size(20_000_000_000),
    );
}

#[test]
fn the_unit_turns_over_at_a_gibibyte_not_a_gigabyte() {
    // One byte short of the turnover must not read LARGER than the value just
    // past it, which a rounding `{:.0}` would print as "1024 MB".
    assert_eq!(bytes(1024 * 1024 * 1024 - 1), "1023 MB");
    assert_eq!(bytes(1024 * 1024 * 1024), "1.0 GB");
    // A decimal gigabyte is NOT the turnover point.
    assert_eq!(bytes(1_000_000_000), "953 MB");
}

#[test]
fn zero_is_a_size_not_an_error() {
    assert_eq!(bytes(0), "0 MB");
}

/// The rate half of the same bug: the download row printed a 1024-based size
/// and a 10⁶-based rate ON THE SAME LINE, so dividing one by the other — the
/// only reason both are there — was 7% wrong.
#[test]
fn a_rate_divides_into_the_size_printed_beside_it() {
    // A gibibyte a second against a gibibyte to go: both scale by 1024³, so
    // one second of this rate moves the size readout by exactly what it says.
    assert_eq!(rate(1024.0 * 1024.0 * 1024.0), "1.0 GB/s");
    assert_eq!(bytes(1024 * 1024 * 1024), "1.0 GB");
    // The line that carried the bug. This rate rendered as "96 MB/s" beside a
    // 1024-scaled "3.7 GB / 32.5 GB" on the same row.
    assert_eq!(rate(96_000_000.0), "92 MB/s");
    // The last digit still differs from `bytes(96_000_000)` == "91 MB",
    // because `bytes` TRUNCATES below a gibibyte on purpose (see its doc) and
    // a rate rounds. That is a rounding rule, not a second scale — which is
    // the thing that was wrong.
    assert_eq!(bytes(96_000_000), "91 MB");
}

#[test]
fn a_rate_says_what_unit_it_is_in() {
    // The Stats tile rendered "3.0M/s" — a magnitude and no unit, on a tile
    // whose other figures are request counts.
    assert_eq!(rate(3_145_728.0), "3.0 MB/s");
    assert_eq!(rate(2048.0), "2.0 KB/s");
    for r in [0.0, 512.0, 2048.0, 3e6, 4e9] {
        assert!(rate(r).ends_with("B/s"), "{r} rendered as {}", rate(r));
    }
}

#[test]
fn a_rate_turns_over_on_the_same_boundaries_as_a_size() {
    assert_eq!(rate(1023.0), "1023 B/s");
    assert_eq!(rate(1024.0), "1.0 KB/s");
    // One short of the turnover rounds to "1024 KB/s", which `bytes` goes out
    // of its way to avoid. Deliberately not copied: that rule exists because a
    // size sits beside a LARGER size on the same line and must not out-read
    // it. A rate has no such neighbour — the only thing it is compared with is
    // the same rate a frame earlier, and rates jitter across the rung anyway.
    assert_eq!(rate(1024.0 * 1024.0 - 1.0), "1024 KB/s");
    assert_eq!(rate(1024.0 * 1024.0), "1.0 MB/s");
    assert_eq!(rate(1024.0 * 1024.0 * 1024.0), "1.0 GB/s");
    // A decimal megabyte is not the turnover, exactly as for `bytes`.
    assert_eq!(rate(1_000_000.0), "977 KB/s");
}

#[test]
fn the_decimal_appears_only_where_it_says_something() {
    // Below ten, one digit shows movement; above it, the last digit is jitter
    // on a figure that is already noisy.
    assert_eq!(rate(9.9 * 1024.0 * 1024.0), "9.9 MB/s");
    assert_eq!(rate(10.4 * 1024.0 * 1024.0), "10 MB/s");
}

#[test]
fn an_idle_or_unmeasurable_rate_is_zero_rather_than_nonsense() {
    assert_eq!(rate(0.0), "0 B/s");
    // `rate_bps` is a division by an elapsed time and a byte delta; neither a
    // negative nor a NaN should reach the screen as one.
    assert_eq!(rate(-1.0), "0 B/s");
    assert_eq!(rate(f64::NAN), "0 B/s");
}

/// `{:?}` printed `Mtp`, which names a variant rather than describing a state.
/// Every label has to be something a reader who has never opened the enum can
/// act on.
#[test]
fn every_mtp_state_reads_as_english_not_as_a_variant_name() {
    for (mode, want) in [
        (MtpModeSnap::Mtp, "speculative"),
        (MtpModeSnap::Serial, "serial"),
        (MtpModeSnap::Probing, "probing"),
        (MtpModeSnap::Off, "off"),
    ] {
        assert_eq!(mtp_mode_label(mode), want);
        assert_ne!(
            mtp_mode_label(mode),
            format!("{mode:?}"),
            "a label that equals the Debug output has not been written yet"
        );
    }
}
