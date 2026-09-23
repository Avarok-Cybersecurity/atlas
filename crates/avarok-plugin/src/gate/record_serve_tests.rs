// SPDX-License-Identifier: AGPL-3.0-only

use super::super::tests::{SHA, bfcl_baseline, hw, run_record};
use super::super::{GateRecord, check_record, read_record, records_newest_first};
use super::{MTP_GATE, PREFILL_CODISPATCH, SPECULATIVE, disclosure};
use crate::result::Verdict;
use std::collections::BTreeMap;

fn keys(m: &BTreeMap<String, String>) -> Vec<(&str, &str)> {
    m.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect()
}

#[test]
fn disclosure_spells_the_regime_and_omits_what_was_not_resolved() {
    assert_eq!(
        keys(&disclosure(Some(true), true, None)),
        vec![(MTP_GATE, "force"), (SPECULATIVE, "true")]
    );
    assert_eq!(
        keys(&disclosure(Some(false), true, None)),
        vec![(MTP_GATE, "auto"), (SPECULATIVE, "true")]
    );
    // No `--mtp-gate` on the rendered command: the SERVER's environment
    // decides, which this process cannot see for a leased server. Absent,
    // not "auto" — "the recipe pinned nothing" is the finding.
    assert_eq!(
        keys(&disclosure(None, false, None)),
        vec![(SPECULATIVE, "false")]
    );
    // G22: `--prefill-codispatch` as rendered, both ways; absent when the
    // flag was, exactly like `mtp_gate`.
    assert_eq!(
        keys(&disclosure(None, true, Some(true))),
        vec![(PREFILL_CODISPATCH, "true"), (SPECULATIVE, "true")]
    );
    assert_eq!(
        keys(&disclosure(None, true, Some(false))),
        vec![(PREFILL_CODISPATCH, "false"), (SPECULATIVE, "true")]
    );
}

fn passing_record() -> GateRecord {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 87.74);
    GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        Some("qwen3.6/qwen3.6-35b-a3b-fp8-bf16head".into()),
    )
    .unwrap()
}

#[test]
fn serve_resolved_round_trips_and_older_records_simply_lack_it() {
    let record = passing_record().with_serve_resolved(disclosure(Some(true), true, None));
    let json = serde_json::to_value(&record).unwrap();
    assert_eq!(json["serve_resolved"][MTP_GATE], "force");
    assert_eq!(json["serve_resolved"][SPECULATIVE], "true");
    let back: GateRecord = serde_json::from_value(json).unwrap();
    assert_eq!(back.serve_resolved, record.serve_resolved);

    // A record written before the field existed loads with it empty, and
    // is not serialised with an empty map either.
    let bare = serde_json::to_value(passing_record()).unwrap();
    assert!(bare.get("serve_resolved").is_none());
    let old: GateRecord = serde_json::from_value(bare).unwrap();
    assert!(old.serve_resolved.is_empty());

    // The committed record set: every real record still parses, and none
    // written before this claims a resolution it never made.
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace layout")
        .to_path_buf();
    let committed = records_newest_first(&root, "agentic-webserver");
    assert!(
        !committed.is_empty(),
        "no committed agentic-webserver records found"
    );
    for path in committed {
        let record = read_record(&path).unwrap_or_else(|e| panic!("{e:#}"));
        assert!(
            record.serve_resolved.is_empty() || record.serve_resolved.contains_key(SPECULATIVE),
            "{}: a disclosure without the `speculative` key is malformed",
            path.display()
        );
    }
}

/// Disclosure, not a threshold: the same record scores the same with and
/// without it, on a baseline it passes AND on one it fails. A future bound on
/// these keys would be a gate change, and this is where it would show.
#[test]
fn serve_resolved_never_reaches_check_record() {
    let baseline = bfcl_baseline();
    let without = passing_record();
    let with = passing_record().with_serve_resolved(disclosure(Some(false), true, None));
    assert_eq!(check_record(&with, &baseline), None);
    assert_eq!(
        check_record(&with, &baseline),
        check_record(&without, &baseline)
    );

    let mut failing_without = without;
    failing_without
        .metrics
        .insert("overall_accuracy".into(), 80.0);
    let failing_with =
        failing_without
            .clone()
            .with_serve_resolved(disclosure(Some(true), true, None));
    let verdict = check_record(&failing_with, &baseline);
    assert!(
        verdict.is_some(),
        "the floor must bite for the comparison to mean anything"
    );
    assert_eq!(verdict, check_record(&failing_without, &baseline));
}
