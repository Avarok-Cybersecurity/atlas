// SPDX-License-Identifier: AGPL-3.0-only

//! The coherence probe against a real socket.
//!
//! The point of these tests is the *ordering* guarantee: a benchmark whose
//! endpoint answers nonsense must fail before it does any expensive setup, not
//! after hours of uniformly-failing samples.

// Each integration binary includes the mock separately, so the helpers this
// one does not call are dead code from its point of view only.
#[allow(dead_code)]
mod mock_endpoint;

use std::time::Duration;

use atlas_plugin::coherence::{self, CoherencePolicy};
use atlas_plugin::plugin::TargetEndpoint;

#[derive(Default)]
struct WarningReporter {
    warnings: Vec<String>,
}

impl atlas_plugin::headless::RunReporter for WarningReporter {
    fn event(&mut self, event: &atlas_plugin::PluginEvent) {
        if let atlas_plugin::PluginEvent::Log(line) = event
            && line.level == atlas_plugin::LogLevel::Warn
        {
            self.warnings.push(line.text.clone());
        }
    }
}

fn target(port: u16) -> TargetEndpoint {
    TargetEndpoint::local(port, "mock")
}

#[tokio::test]
async fn an_endpoint_answering_correctly_is_clean() {
    // One reply satisfies both checks, so a single canned string is enough.
    let mock =
        mock_endpoint::start_saying(Some("4 Paris".into()), 1, Duration::ZERO, Duration::ZERO)
            .await;
    let report = coherence::probe(&target(mock.port), Duration::from_secs(5)).await;
    assert_eq!(report.answers.len(), 2);
    assert!(report.is_clean());
    assert!(report.concern(&target(mock.port)).is_none());
}

#[tokio::test]
async fn an_endpoint_answering_nonsense_warns_and_says_what_it_said() {
    let mock = mock_endpoint::start_saying(
        Some("I am a teapot".into()),
        1,
        Duration::ZERO,
        Duration::ZERO,
    )
    .await;
    let report = coherence::probe(&target(mock.port), Duration::from_secs(5)).await;
    assert!(!report.is_clean());
    let text = report.concern(&target(mock.port)).expect("a concern");
    // Quote the answer back: "the probe failed" alone sends the reader to the
    // server logs for something the client already had in hand.
    assert!(text.contains("teapot"), "{text}");
    assert!(
        text.contains("arithmetic"),
        "names the failing check: {text}"
    );
    // And it must NOT read as a refusal — the run is still allowed.
    assert!(text.contains("still valid"), "{text}");
}

#[tokio::test]
async fn an_unreachable_endpoint_is_a_transport_error_not_a_wrong_answer() {
    // A closed port and a confused model are different diagnoses; conflating
    // them sends the reader looking at the wrong thing.
    let report = coherence::probe(&target(1), Duration::from_secs(2)).await;
    assert!(report.transport_error.is_some());
    let text = report.concern(&target(1)).expect("a concern");
    assert!(
        !text.contains("different model"),
        "should not blame the model: {text}"
    );
}

#[tokio::test]
async fn a_failed_probe_warns_but_still_runs_the_benchmark() {
    use atlas_plugin::headless::{HeadlessOptions, RunRequest, run_blocking};
    use atlas_plugin::{ArtifactStore, BenchmarkExecutor, ParamValues, registry};

    let mock = mock_endpoint::start_saying(
        Some("I am a teapot".into()),
        1,
        Duration::ZERO,
        Duration::ZERO,
    )
    .await;
    let requests = mock.requests.clone();
    let dir =
        std::env::temp_dir().join(format!("atlas-coherence-{:?}", std::thread::current().id()));
    std::fs::create_dir_all(&dir).expect("scratch");

    let descriptor = registry::find("concurrency-sweep").expect("registered");
    let specs = descriptor.build().parameters();
    let executor = BenchmarkExecutor::new(
        tokio::runtime::Handle::current(),
        ArtifactStore::with_root(&dir),
    );
    let request = RunRequest {
        descriptor,
        values: ParamValues::defaults(&specs),
        target: target(mock.port),
        options: HeadlessOptions {
            poll: Duration::from_millis(10),
            save: false,
            source: atlas_plugin::RunSource::Cli,
            atlas_version: "test".into(),
            coherence: CoherencePolicy::Probe,
        },
    };

    let (outcome, warnings) = tokio::task::spawn_blocking(move || {
        let mut reporter = WarningReporter::default();
        let outcome = run_blocking(&executor, request, &mut reporter, &|| false);
        (outcome, reporter.warnings)
    })
    .await
    .expect("join");
    let outcome = outcome.expect("drives");

    // The whole point of the change: an endpoint that answers oddly is a
    // WARNING, so the sweep still runs. Two probe questions plus the sweep's
    // own requests — far more than 2.
    assert!(
        requests.load(std::sync::atomic::Ordering::Relaxed) > 2,
        "the benchmark must not have been blocked by the probe"
    );
    assert_eq!(outcome.exit_code(), 0, "a warning is not a failure");
    assert!(
        warnings
            .iter()
            .any(|warning| warning.contains("teapot") && warning.contains("still valid")),
        "the advisory reaches the caller: {warnings:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `/v1/models` must be readable through **chunked** framing.
///
/// This is the bug that made the model check silently useless: Atlas replies
/// with `Transfer-Encoding: chunked`, so the body carries hex length prefixes
/// and a terminating `0\r\n\r\n`. A plain `from_str` from the first `{` fails
/// on those trailing bytes, `list_models` errored, and the wrong-model warning
/// never fired against a real server.
#[tokio::test]
async fn the_model_list_survives_chunked_framing() {
    let mock =
        mock_endpoint::start_saying(Some("4 Paris".into()), 1, Duration::ZERO, Duration::ZERO)
            .await;
    let models = atlas_plugin::http::list_models(&target(mock.port), Duration::from_secs(5))
        .await
        .expect("the list parses");
    assert_eq!(models, vec!["mock".to_string()]);
}

#[tokio::test]
async fn a_model_the_server_does_not_serve_is_reported() {
    let mock =
        mock_endpoint::start_saying(Some("4 Paris".into()), 1, Duration::ZERO, Duration::ZERO)
            .await;
    // The mock serves "mock"; ask for something else.
    let wrong = TargetEndpoint::local(mock.port, "does/not-exist");
    let report = coherence::probe(&wrong, Duration::from_secs(5)).await;
    assert!(!report.is_clean(), "a wrong model name is not clean");
    let concern = report.concern(&wrong).expect("a concern");
    assert!(concern.contains("mock"), "names what IS served: {concern}");
    assert!(concern.contains("does/not-exist"), "{concern}");
    // The questions still passed — which is exactly why the model list is the
    // only thing that could have caught this.
    assert!(
        report.answers.iter().all(|a| a.passed),
        "{:?}",
        report.answers
    );
}
