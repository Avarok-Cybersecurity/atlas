// SPDX-License-Identifier: AGPL-3.0-only
//! The wire contract with atlasctl, pinned from this side: the JSON shapes
//! `bench nodes/submit/attach/fetch --json` emit (atlas-recipes
//! `docs/BENCH.md`) parse into what the driver acts on, and every exit code
//! maps to its class.
use super::*;

/// A `bench nodes --json` row as atlasctl 0.5 writes it, verbatim shape.
pub(in crate::cli::bench_certify::remote) const NODE_ROW: &str = r#"{"node":"10.10.10.2","ok":true,"info":{"node":"1730e1bea1873a8a7000000000000000000000000000000000000000000000000","name":"spark-43fa","agent_version":"0.5.0","peer_version_max":3,"bench_enabled":true,"disabled_reason":null,"gpu":{"name":"NVIDIA GB10","count":1,"driver_version":"580.126.09","cuda_version":"13.0","sm_clock_mhz":{"state":"reading","value":208.0},"sm_clock_healthy_mhz":3003,"temperature_c":{"state":"reading","value":48.0},"memory_total_bytes":{"state":"reading","value":130663886848.0},"memory_used_frac":{"state":"reading","value":0.12},"memory_is_unified":true},"thermal":{"chassis_temps_c":[65.0,62.0,59.0],"throttle_thermal":false,"sm_clock_max_mhz":3003.0,"mem_total_kb":127601452},"alerts":[],"hardware_class":"gb10","atlas_repo":{"path":"/workspace/avarok","remote_name":"atlas","remote_url":"git@github.com:Avarok-Cybersecurity/atlas.git","head_sha":"1a0dc88a8c9083bb956bd84cafa2cccbdb8e6e18","fetched_at_s":1757770000},"atlas_home":"/workspace/.avarok","signer_fp":"a27dbc8ed2fc2a31","signer_pubkey_hex":"00","recipes_synced":true,"built_shas":[{"sha":"1a0dc88a8c9083bb956bd84cafa2cccbdb8e6e18","binary_sha256":"ff","built_at_s":1,"bytes":1}],"busy":false,"busy_reason":null,"running_job":null,"queued":0,"queue_depth":2,"disk_free_bytes":{"state":"reading","value":50000000000.0},"min_free_disk_bytes":21474836480,"min_free_fraction":0.85,"host_free_fraction":{"state":"reading","value":0.91},"max_run_s":10800}}"#;

#[test]
fn a_node_row_parses_into_the_facts_the_driver_reads() {
    let row: NodeRow = serde_json::from_str(NODE_ROW).expect("parses");
    assert!(row.ok);
    let info = row.info.expect("info");
    assert!(info.bench_enabled);
    assert_eq!(info.hardware_class.as_deref(), Some("gb10"));
    assert_eq!(info.signer_fp.as_deref(), Some("a27dbc8ed2fc2a31"));
    assert_eq!(
        info.gpu.as_ref().map(|g| g.driver_version.as_str()),
        Some("580.126.09")
    );
    let t = info.thermal.as_ref().expect("thermal");
    assert_eq!(t.chassis_temps_c, vec![65.0, 62.0, 59.0]);
    assert_eq!(t.throttle_thermal, Some(false));
    assert_eq!(t.sm_clock_max_mhz, Some(3003.0));
    assert_eq!(t.mem_total_kb, Some(127_601_452));
    assert_eq!(info.host_free_fraction.value(), Some(0.91));
    assert_eq!(info.built_shas.len(), 1);
    // An error row, and an unknown extra field, both parse.
    let err: NodeRow = serde_json::from_str(
        r#"{"node":"10.10.10.9","ok":false,"error":{"code":"unreachable","message":"no","node":"10.10.10.9","retryable":true},"future_field":1}"#,
    )
    .unwrap();
    assert!(!err.ok && err.info.is_none());
    assert_eq!(err.error.unwrap().code, "unreachable");
    // A node with no thermal block at all: the field is None, not a parse error.
    let bare: NodeRow =
        serde_json::from_str(r#"{"node":"x","ok":true,"info":{"bench_enabled":true}}"#).unwrap();
    assert!(bare.info.unwrap().thermal.is_none());
}

#[test]
fn stream_events_and_the_done_line_carry_what_classification_needs() {
    let ev: StreamEvent = serde_json::from_str(
        r#"{"job":"jb-1-a","seq":9,"at_ms":1,"kind":"progress","phase":"isl 512","detail":"[3/8]"}"#,
    )
    .unwrap();
    assert_eq!((ev.seq, ev.kind.as_str()), (9, "progress"));
    let done: StreamEvent = serde_json::from_str(
        r#"{"job":"jb-1-a","seq":40,"at_ms":1,"kind":"done","outcome":"completed","exit_code":2,"verdict":{"kind":"fail","text":"below floor"},"record":"2026-09-13-1a0dc88a8c.json","signature":null}"#,
    )
    .unwrap();
    assert_eq!(done.outcome.as_deref(), Some("completed"));
    assert_eq!(done.exit_code, Some(2));
    let failed: StreamEvent = serde_json::from_str(
        r#"{"job":"jb-1-a","seq":3,"at_ms":1,"kind":"done","outcome":"failed","stage":"building","reason":"cargo exited 101"}"#,
    )
    .unwrap();
    assert_eq!(failed.stage.as_deref(), Some("building"));
    assert_eq!(failed.reason.as_deref(), Some("cargo exited 101"));
    let s: Submitted = serde_json::from_str(
        r#"{"node":"10.10.10.2","node_id":"ab","job_id":"jb-1-a","job_key":"k","existing":true,"state":"queued","position":0}"#,
    )
    .unwrap();
    assert!(s.existing);
    let files: Vec<FetchedFile> = serde_json::from_str(
        r#"[{"name":"r.json","relative_path":".benchmarks/decode-floor/r.json","path":"/tmp/x/.benchmarks/decode-floor/r.json","bytes":3,"sha256":"00"}]"#,
    )
    .unwrap();
    assert_eq!(
        files[0].path,
        PathBuf::from("/tmp/x/.benchmarks/decode-floor/r.json")
    );
}

#[test]
fn every_exit_code_maps_to_its_class_and_a_signal_is_other() {
    let table = [
        (0, Exit::Done),
        (1, Exit::Usage),
        (2, Exit::Unreachable),
        (3, Exit::NotPaired),
        (4, Exit::Refused),
        (5, Exit::JobFailed),
        (6, Exit::Cancelled),
        (7, Exit::StreamLost),
        (8, Exit::Unsupported),
    ];
    for (code, exit) in table {
        assert_eq!(Exit::from_code(Some(code)), exit);
    }
    assert_eq!(Exit::from_code(None), Exit::Other(None));
    assert_eq!(Exit::from_code(Some(9)), Exit::Other(Some(9)));
    let e = ErrorObj {
        code: "refused:busy".into(),
        message: "busy".into(),
        fix: Some("retry after 60 s".into()),
        ..Default::default()
    };
    assert_eq!(e.to_string(), "refused:busy: busy (fix: retry after 60 s)");
}
