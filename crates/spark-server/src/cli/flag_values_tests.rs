// SPDX-License-Identifier: AGPL-3.0-only

//! The no-drift contract: every value [`options_for_flag`] offers must
//! survive the full parse-and-validate round trip a launch performs, and a
//! value outside the set must be refused. This is the property the option
//! picker rests on — if it fails, the dashboard is offering a config the
//! server will not start with, which is worse than offering nothing.

use super::options_for_flag;
use crate::cli::validate_serve_args;
use clap::Parser;

/// The flags with a closed value set. Listed once here so a NEW entry in
/// `options_for_flag` that is not added to this test fails the count check
/// below rather than silently going untested.
const ENUMERATED: &[&str] = &[
    "lm-head-dtype",
    "mtp-quantization",
    "scheduling-policy",
    "ssm-h-dtype",
    "mtp-gate",
    "tool-call-parser",
    "kv-cache-dtype",
];

/// Flags whose values are only self-consistent beside another flag.
/// `--ssm-h-dtype f16` alone is a validate-level error (the FP16 h-state
/// twins live on the fused-norm arm), so its round trip carries the arm.
fn companions(flag: &str) -> &'static [&'static str] {
    match flag {
        "ssm-h-dtype" => &["--gdn-fused-norm"],
        _ => &[],
    }
}

fn round_trip(flag: &str, value: &str) -> Result<(), String> {
    let mut argv: Vec<String> = ["spark", "serve", "dummy/model", "--model-name", "dummy"]
        .map(String::from)
        .to_vec();
    argv.push(format!("--{flag}"));
    argv.push(value.to_string());
    argv.extend(companions(flag).iter().map(|s| s.to_string()));
    let cli = crate::cli::Cli::try_parse_from(argv).map_err(|e| e.to_string())?;
    let crate::cli::Command::Serve(args) = cli.command else {
        unreachable!("this test parses a serve command");
    };
    validate_serve_args(&args)
}

#[test]
fn every_offered_value_is_accepted_by_parse_and_validate() {
    for flag in ENUMERATED {
        let options = options_for_flag(flag).expect("listed flags have options");
        assert!(!options.is_empty(), "--{flag} offers nothing");
        for value in &options {
            round_trip(flag, value)
                .unwrap_or_else(|e| panic!("--{flag} {value} is offered but refused: {e}"));
        }
    }
}

#[test]
fn a_value_outside_the_set_is_refused_for_every_enumerated_flag() {
    // The other half of the contract. If a flag stops being validated, its
    // picker is decoration and free text through the same flag ships typos.
    for flag in ENUMERATED {
        let err = round_trip(flag, "zzz-not-a-value")
            .expect_err(&format!("--{flag} accepted a value outside its set"));
        assert!(err.contains(flag), "the refusal names the flag: {err}");
    }
}

#[test]
fn the_enumerated_list_and_the_registry_agree_on_membership() {
    for flag in ENUMERATED {
        assert!(
            options_for_flag(flag).is_some(),
            "--{flag} is tested here but not in options_for_flag"
        );
    }
    // The reverse direction cannot be enumerated from the registry (it is a
    // match, not a table), so pin the count: extending the match without
    // extending ENUMERATED trips this and points at both lists.
    let known = crate::tui::lib_fields::serve_fields()
        .iter()
        .filter(|s| options_for_flag(&s.flag).is_some())
        .count();
    assert_eq!(
        known,
        ENUMERATED.len(),
        "options_for_flag knows a flag this test does not (or vice versa)"
    );
}
