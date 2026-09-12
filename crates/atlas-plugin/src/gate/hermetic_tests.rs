// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use std::collections::BTreeMap;

fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

#[test]
fn only_the_affirmative_spelling_requests_the_regime() {
    assert!(is_requested(&map(&[("hermetic", "true")])));
    for not_yes in ["false", "1", "TRUE", "yes", ""] {
        assert!(
            !is_requested(&map(&[("hermetic", not_yes)])),
            "{not_yes:?} must not read as a request — it arrives as a string \
             from --serve-override, and guessing is how a regime gets claimed \
             that was never in force"
        );
    }
    assert!(!is_requested(&map(&[])));
}

#[test]
fn a_complete_pin_set_is_reported_complete() {
    let mut m = map(&[("hermetic", "true")]);
    for (k, v) in CLOSED_KEYS {
        m.insert((*k).to_string(), (*v).to_string());
    }
    assert!(missing_pins(&m).is_empty(), "{:?}", missing_pins(&m));
}

#[test]
fn a_missing_pin_is_named() {
    let missing = missing_pins(&map(&[("hermetic", "true")]));
    assert_eq!(missing.len(), CLOSED_KEYS.len());
    assert!(missing.iter().any(|(k, _)| *k == "enable_prefix_caching"));
}

/// A key pinned at the WRONG value is as bad as a missing one — worse, since
/// it reads as though someone considered it.
#[test]
fn a_pin_at_the_wrong_value_counts_as_missing() {
    let missing = missing_pins(&map(&[
        ("hermetic", "true"),
        ("enable_prefix_caching", "true"),
        ("mtp_gate", "force"),
    ]));
    assert_eq!(
        missing,
        vec![("enable_prefix_caching", "false")],
        "a wrong value must be reported, not accepted"
    );
}
