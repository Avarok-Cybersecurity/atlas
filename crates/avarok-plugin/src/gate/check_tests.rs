// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the gate verdict itself — see `check.rs`.

use super::*;

/// The exit code must depend on the verdicts and nothing else. This is a
/// behavioural pin; the real guarantee is the signature, which cannot accept an
/// advisory argument without a visible edit.
#[test]
fn the_exit_code_is_a_function_of_the_verdicts_alone() {
    fn m(pairs: Vec<(&str, GateStatus)>) -> BTreeMap<String, GateStatus> {
        pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    }
    assert_eq!(exit_code(&m(vec![])), 0, "nothing checked, nothing open");
    assert_eq!(exit_code(&m(vec![("a", GateStatus::Pass)])), 0);
    assert_eq!(
        exit_code(&m(vec![
            ("a", GateStatus::Pass),
            ("b", GateStatus::Missing("no record".into())),
        ])),
        1,
        "Missing is open — \"we have not measured this\" is not a pass"
    );
    assert_eq!(
        exit_code(&m(vec![("a", GateStatus::Fail(vec!["over bound".into()]))])),
        1
    );
}
