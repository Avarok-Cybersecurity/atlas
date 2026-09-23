// SPDX-License-Identifier: AGPL-3.0-only

//! What the record discloses about its server's environment (#1242).

use super::super::GateRecord;
use super::super::tests::{SHA, hw, run_record};
use crate::result::Verdict;
use std::collections::BTreeMap;

/// #1242: the record discloses the WHOLE lever set the gate applied, and the
/// co-dispatch disclosure is read through that set rather than off this
/// process — a leased child served under a declared `AVAROK_PREFILL_CODISPATCH=1`
/// the harness itself does not carry must not be recorded as `0`. Old records
/// and operator-endpoint runs simply lack the field.
#[test]
fn the_record_discloses_the_applied_serve_env_and_reads_perf_env_through_it() {
    // NEGATIVE CONTROL: without an applied set the disclosure is this
    // process's, which the test binary does not export.
    let bare = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        Some("qwen3.8/qwen3.8-27b-nvfp4-unsloth".into()),
    )
    .unwrap();
    assert!(bare.serve_env.is_empty());
    assert_eq!(
        bare.perf_env
            .get("AVAROK_PREFILL_CODISPATCH")
            .map(String::as_str),
        Some("0"),
        "the test environment must not export the control this test flips"
    );
    assert!(
        serde_json::to_value(&bare)
            .unwrap()
            .get("serve_env")
            .is_none()
    );

    let applied: BTreeMap<String, String> = [
        ("AVAROK_FP8_ROWWISE", "1"),
        ("AVAROK_MTP_DCUT_RATIO", "1.0"),
        ("AVAROK_MTP_K_LADDER", "1:3,2:1,4:2,8:2,16:1"),
        ("AVAROK_PREFILL_CODISPATCH", "1"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    let record = bare.clone().with_serve_env(applied.clone());
    assert_eq!(record.serve_env, applied);
    assert_eq!(
        record
            .perf_env
            .get("AVAROK_PREFILL_CODISPATCH")
            .map(String::as_str),
        Some("1"),
        "the applied set is what the server read, so it is what the record says"
    );
    assert_eq!(
        record
            .perf_env
            .get("AVAROK_PREFILL_CODISPATCH_WINDOW_MS")
            .map(String::as_str),
        Some("100"),
        "a control the set does not name still resolves to its default"
    );
    let json = serde_json::to_value(&record).unwrap();
    assert_eq!(json["serve_env"]["AVAROK_FP8_ROWWISE"], "1");
    assert_eq!(
        json["serve_env"]["AVAROK_MTP_K_LADDER"],
        "1:3,2:1,4:2,8:2,16:1"
    );
    let back: GateRecord = serde_json::from_value(json).unwrap();
    assert_eq!(back.serve_env, applied);
    assert_eq!(back.perf_env, record.perf_env);
}
