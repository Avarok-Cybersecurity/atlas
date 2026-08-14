// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use serde_json::json;

use super::*;
use crate::artifacts::ArtifactStore;
use crate::plugin::TargetEndpoint;

fn handle(root: &str) -> (PluginHandle, std::path::PathBuf) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::mem::forget(rx);
    let dir = std::env::temp_dir().join(format!("atlas-mlperf-{root}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    (
        PluginHandle::new(
            1,
            TargetEndpoint::local(8888, "test-model"),
            ArtifactStore::with_root(&dir),
            tx,
            Arc::new(AtomicBool::new(false)),
        ),
        dir,
    )
}

/// The single most load-bearing behaviour while the dataset does not exist:
/// provisioning fails LOUDLY, naming the missing artifact, the TBD upstream
/// status, and the no-proxy rule — never an empty run and never a 0.0.
#[test]
fn an_absent_dataset_fails_provisioning_with_the_specific_story() {
    let (h, _dir) = handle("absent");
    let err = provision::ensure(h.artifacts(), &h, "")
        .unwrap_err()
        .to_string();
    for needle in [
        "dataset.jsonl",
        "NOT yet published",
        "link TBD",
        "mlcommons/endpoints@7935df4",
        "refuses to substitute a proxy",
        "613 trajectories",
    ] {
        assert!(
            err.contains(needle),
            "error must mention {needle:?}:\n{err}"
        );
    }
}

#[test]
fn a_present_dataset_is_hashed_and_a_wrong_pin_refuses_to_run() {
    let (h, dir) = handle("pin");
    let plugin_dir = h.artifacts().plugin_dir(provision::PLUGIN_ID).unwrap();
    std::fs::write(plugin_dir.join("dataset.jsonl"), b"{}\n").unwrap();

    let a = provision::ensure(h.artifacts(), &h, "").unwrap();
    assert_eq!(a.file_sha256.len(), 64);
    assert!(plugin_dir.join("dataset_summary.json").is_file());

    // The right pin passes (case-insensitively), the wrong one refuses.
    provision::ensure(h.artifacts(), &h, &a.file_sha256.to_uppercase()).unwrap();
    let err = provision::ensure(h.artifacts(), &h, "deadbeef")
        .unwrap_err()
        .to_string();
    assert!(err.contains("does not match its pin"), "{err}");
    let _ = std::fs::remove_dir_all(dir);
}

/// The request carries the official immutable sampling set plus the seed and
/// NOTHING else. Pinned by exact key set: the upstream rules forbid adding
/// sampling params, and the repo's own doctrine forbids a temperature knob
/// here — a temp-0 run would look comparable to the MLPerf thresholds and
/// not be.
#[test]
fn the_request_body_is_the_official_immutable_set_plus_seed() {
    let messages = vec![json!({"role": "user", "content": "hi"})];
    let tools = json!([{"type": "function"}]);
    let body = request_body("m", &messages, Some(&tools), 42);
    let mut keys: Vec<&str> = body
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec![
            "chat_template_kwargs",
            "max_tokens",
            "messages",
            "model",
            "presence_penalty",
            "repetition_penalty",
            "seed",
            "stream",
            "temperature",
            "tools",
            "top_k",
            "top_p",
        ]
    );
    assert_eq!(body["temperature"], 1.0);
    assert_eq!(body["top_k"], 20);
    assert_eq!(body["top_p"], 0.95);
    assert_eq!(body["repetition_penalty"], 1.0);
    assert_eq!(body["presence_penalty"], 1.5);
    assert_eq!(body["max_tokens"], 8192);
    assert_eq!(body["seed"], 42);
    assert_eq!(
        body["chat_template_kwargs"],
        json!({"preserve_thinking": true})
    );

    // No tools → no tools key, not tools: null.
    let body = request_body("m", &messages, None, 42);
    assert!(body.get("tools").is_none());
}

/// No temperature (or any sampling) parameter is exposed — verified against
/// the live spec list so a later param addition trips this test and forces
/// the conversation.
#[test]
fn sampling_is_not_parameterizable() {
    let b = MlperfAgentic::new();
    for spec in b.parameters() {
        for forbidden in [
            "temperature",
            "top_p",
            "top_k",
            "penalty",
            "max_new",
            "max_tokens",
        ] {
            assert!(
                !spec.key.contains(forbidden),
                "{} must not be a parameter on this leg",
                spec.key
            );
        }
    }
}

#[test]
fn model_turns_mirror_the_upstream_message_shape() {
    let mut outcome = http::ChatOutcome {
        text: "done".into(),
        reasoning: "thinking".into(),
        ..Default::default()
    };
    let turn = model_turn(&outcome);
    assert_eq!(turn["role"], "assistant");
    assert_eq!(turn["content"], "done");
    assert_eq!(turn["reasoning_content"], "thinking");
    assert!(turn["tool_calls"].is_null(), "no calls is null, not []");

    outcome.tool_calls = vec![crate::http::ToolCall {
        id: "c1".into(),
        name: "bash".into(),
        arguments: r#"{"cmd": "ls"}"#.into(),
    }];
    let turn = model_turn(&outcome);
    assert_eq!(turn["tool_calls"][0]["function"]["name"], "bash");
    // Arguments stay the raw streamed string; the scorer parses them itself.
    assert_eq!(
        turn["tool_calls"][0]["function"]["arguments"],
        r#"{"cmd": "ls"}"#
    );
    assert_eq!(scoring::bash_actions(&turn), vec!["ls".to_string()]);
}

#[test]
fn aggregation_follows_the_upstream_denominator_semantics() {
    let t = |domain, score: Option<f64>, missing, tokens| TurnRecord {
        conversation_id: "c".into(),
        turn: 1,
        domain,
        score,
        missing,
        completion_tokens: tokens,
        model: json!({}),
    };
    // Two scored coding turns (one a missing-output 0), one scored workflow
    // turn, one excluded turn.
    let turns = vec![
        t(Domain::Coding, Some(0.5), false, 100),
        t(Domain::Coding, Some(0.0), true, 0),
        t(Domain::Workflow, Some(1.0), false, 60),
        t(Domain::Coding, None, false, 40),
    ];
    let s = report::aggregate(&turns).unwrap();
    assert_eq!(s.turns_scored, 3);
    assert_eq!(s.turns_excluded, 1);
    assert_eq!(s.turns_missing, 1);
    assert!((s.inline - 0.5).abs() < 1e-9, "mean of 0.5, 0.0, 1.0");
    assert_eq!(s.coding.unwrap(), (0.25, 2));
    assert_eq!(s.workflow.unwrap(), (1.0, 1));
    // OSL is over turns that produced output (3 of 4), missing excluded.
    assert!((s.osl_per_turn_mean - (200.0 / 3.0)).abs() < 1e-9);

    // Nothing scorable → None, which the phase machine turns into a refusal,
    // never a 0.0 score.
    assert!(report::aggregate(&[t(Domain::Coding, None, false, 10)]).is_none());
}

#[test]
fn the_descriptor_says_it_cannot_run_and_defaults_validate() {
    assert_eq!(SUBSET_DESCRIPTOR.id, "mlperf-agentic-subset");
    for text in [SUBSET_DESCRIPTOR.summary, SUBSET_DESCRIPTOR.detail] {
        assert!(
            text.to_lowercase().contains("unrunnable") || text.contains("CANNOT RUN"),
            "a reader must not mistake this for a measured leg: {text}"
        );
    }
    assert!(
        SUBSET_DESCRIPTOR.detail.contains("SWE-bench"),
        "the missing official leg is named"
    );
    let b = MlperfAgentic::new();
    let values = ParamValues::defaults(&b.parameters());
    values.validate_against(&b.parameters()).unwrap();
}

/// The verdict can never be PASS: there is no baseline for it to have passed.
#[test]
fn the_verdict_is_info_and_names_the_unmeasured_state() {
    let mut b = MlperfAgentic::new();
    let v = b.verdict();
    assert_eq!(v.kind, crate::result::VerdictKind::Info);

    b.turns.push(TurnRecord {
        conversation_id: "sim_001".into(),
        turn: 1,
        domain: Domain::Workflow,
        score: Some(1.0),
        missing: false,
        completion_tokens: 10,
        model: json!({}),
    });
    b.scores = report::aggregate(&b.turns);
    let v = b.verdict();
    assert_eq!(
        v.kind,
        crate::result::VerdictKind::Info,
        "perfect scores still do not PASS"
    );
    assert!(v.reason.contains("UNMEASURED"), "{}", v.reason);
    assert!(v.reason.contains("NOT this draw's floors"), "{}", v.reason);
}

/// The fingerprint written to the terminal frame reaches the gate record —
/// the whole point of the additive `dataset_fingerprint` field.
#[test]
fn the_dataset_fingerprint_survives_into_a_gate_record() {
    let mut frame = crate::result::BenchmarkResult::completed("done", Duration::from_secs(1));
    frame.metrics.insert("inline_accuracy".into(), 50.0);
    frame.dataset_fingerprint = Some("file-sha256:aa;draw-sha256:bb".into());
    let record = crate::history::RunRecord {
        schema: 1,
        run_id: "r".into(),
        benchmark_id: "mlperf-agentic-subset".into(),
        benchmark_name: "MLPerf agentic (subset)".into(),
        recorded_at: 1,
        target_url: "http://localhost:1".into(),
        target_model: "m".into(),
        params: Default::default(),
        source: crate::history::RunSource::Cli,
        atlas_version: "test".into(),
        frame,
    };
    let gate = crate::gate::GateRecord::from_run(
        &record,
        crate::hardware::Hardware::default(),
        "abc123".into(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
    assert_eq!(
        gate.dataset_fingerprint.as_deref(),
        Some("file-sha256:aa;draw-sha256:bb")
    );
    // And it round-trips the record file format.
    let json = serde_json::to_string(&gate).unwrap();
    let back: crate::gate::GateRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(back.dataset_fingerprint, gate.dataset_fingerprint);
    // Older records without the field still parse.
    let stripped = json.replace(
        ",\"dataset_fingerprint\":\"file-sha256:aa;draw-sha256:bb\"",
        "",
    );
    let old: crate::gate::GateRecord = serde_json::from_str(&stripped).unwrap();
    assert_eq!(old.dataset_fingerprint, None);
}
