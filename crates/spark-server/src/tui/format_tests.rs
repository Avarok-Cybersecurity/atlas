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
