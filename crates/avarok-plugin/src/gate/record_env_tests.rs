// SPDX-License-Identifier: AGPL-3.0-only

//! What the record discloses about its server's environment (#1242).

use super::super::GateRecord;
use super::super::tests::{SHA, hw, run_record};
use crate::result::Verdict;
use std::collections::BTreeMap;

/// #1242: the record discloses the WHOLE lever set the gate applied, and the
/// co-dispatch disclosure is read through that set rather than off this
/// process — a leased child served under a declared
/// `AVAROK_PREFILL_CODISPATCH_WINDOW_MS=50` the harness itself does not carry
/// must not be recorded as `100`. Old records and operator-endpoint runs
/// simply lack the field.
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
            .get("AVAROK_PREFILL_CODISPATCH_WINDOW_MS")
            .map(String::as_str),
        Some("100"),
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
        ("AVAROK_PREFILL_CODISPATCH_WINDOW_MS", "50"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    let record = bare.clone().with_serve_env(applied.clone());
    assert_eq!(record.serve_env, applied);
    assert_eq!(
        record
            .perf_env
            .get("AVAROK_PREFILL_CODISPATCH_WINDOW_MS")
            .map(String::as_str),
        Some("50"),
        "the applied set is what the server read, so it is what the record says"
    );
    assert_eq!(
        record
            .perf_env
            .get("AVAROK_PREFILL_CODISPATCH_SETTLE_MS")
            .map(String::as_str),
        Some("10"),
        "a control the set does not name still resolves to its default"
    );
    // G22: the enable is a flag now. The applied legacy variable is disclosed
    // where it was applied (`serve_env`), never re-resolved into `perf_env`,
    // whose table default would contradict a `--prefill-codispatch true` serve.
    assert!(
        !record.perf_env.contains_key("AVAROK_PREFILL_CODISPATCH"),
        "{:?}",
        record.perf_env
    );
    assert_eq!(record.serve_env["AVAROK_PREFILL_CODISPATCH"], "1");
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

/// `AVAROK_NO_W4A16_TC` is a PRESENCE kill switch: `0` still disables the
/// tensor-core small-M GEMV, so the record must carry the value the server saw
/// verbatim, never a normalised on/off.
#[test]
fn a_set_tc_kill_switch_is_disclosed_verbatim() {
    for var in ["AVAROK_NO_W4A16_TC", "AVAROK_W4A16_TC_WIDE"] {
        for v in ["1", "0"] {
            let resolved = super::resolve_perf_env(|k| (k == var).then(|| v.to_string()));
            assert_eq!(resolved.get(var).map(String::as_str), Some(v));
        }
    }
}

/// The record's `unset` default is right only while the lever treats unset AND
/// exported-empty as ON (`resolve_perf_env` maps an empty value to the
/// default) and any other value as OFF. Read the lever's own source so a rule
/// change there fails here instead of silently mislabelling every record.
#[test]
fn tc_kill_switch_default_matches_the_lever() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join("crates/spark-model/src/layers/ops/gemv_tc.rs");
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let at = src
        .find("pub fn tc_enabled()")
        .expect("the tensor-core lever SSOT is gone; PERF_CONTROLS discloses a dead variable");
    let body: String = src[at..].chars().take(260).collect();
    assert!(
        body.contains("std::env::var_os(\"AVAROK_NO_W4A16_TC\").is_none_or(|v| v.is_empty())"),
        "the tc kill-switch rule changed; the record's \"unset\" default assumes unset or \
         empty means the tensor-core path ran: {body}"
    );
    let at = src
        .find("pub fn wide_rows_enabled()")
        .expect("the row-edge widening lever is gone; PERF_CONTROLS discloses a dead variable");
    let body: String = src[at..].chars().take(260).collect();
    assert!(
        body.contains("std::env::var_os(\"AVAROK_W4A16_TC_WIDE\").is_some_and(|v| !v.is_empty())"),
        "the widening opt-in rule changed; the record's \"unset\" default assumes unset or \
         empty means the widening did NOT run: {body}"
    );
    let resolved = super::resolve_perf_env(|_| None);
    for var in ["AVAROK_NO_W4A16_TC", "AVAROK_W4A16_TC_WIDE"] {
        assert_eq!(resolved.get(var).map(String::as_str), Some("unset"));
    }
}

/// `AVAROK_NO_MTP_TC` is a PRESENCE kill switch like `AVAROK_NO_W4A16_TC`.
/// The record's `unset` default is right only while the lever treats unset
/// AND exported-empty as ON and any other value, `0` included, as OFF. Read
/// the lever's own source so a rule change there fails here instead of
/// silently mislabelling every record; a set value is disclosed verbatim.
#[test]
fn tc_mtp_switch_default_matches_the_lever() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join("crates/spark-model/src/layers/ops/dense_gemv_tc.rs");
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let rule = src
        .find("pub fn mtp_tc_from(")
        .map(|at| src[at..].chars().take(200).collect::<String>())
        .expect("the drafter tc rule is gone; PERF_CONTROLS discloses a dead variable");
    assert!(
        rule.contains("kill.is_none_or(|v| v.is_empty())"),
        "the AVAROK_NO_MTP_TC rule changed; the record's \"unset\" default assumes unset or \
         empty means the tensor-core path ran: {rule}"
    );
    let read = src
        .find("pub fn mtp_tc_enabled()")
        .map(|at| src[at..].chars().take(300).collect::<String>())
        .expect("the drafter tc lever is gone");
    assert!(
        read.contains("mtp_tc_from(std::env::var_os(\"AVAROK_NO_MTP_TC\")"),
        "the lever no longer reads AVAROK_NO_MTP_TC through mtp_tc_from: {read}"
    );
    assert_eq!(
        super::resolve_perf_env(|_| None)
            .get("AVAROK_NO_MTP_TC")
            .map(String::as_str),
        Some("unset")
    );
    for v in ["1", "0"] {
        let resolved =
            super::resolve_perf_env(|k| (k == "AVAROK_NO_MTP_TC").then(|| v.to_string()));
        assert_eq!(
            resolved.get("AVAROK_NO_MTP_TC").map(String::as_str),
            Some(v)
        );
    }
}
