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

fn target(port: u16) -> TargetEndpoint {
    TargetEndpoint::local(port, "mock")
}

#[tokio::test]
async fn an_endpoint_answering_correctly_passes() {
    // One reply satisfies both checks, so a single canned string is enough.
    let mock =
        mock_endpoint::start_saying(Some("4 Paris".into()), 1, Duration::ZERO, Duration::ZERO)
            .await;
    let answers = coherence::verify(&target(mock.port), Duration::from_secs(5))
        .await
        .expect("probe passes");
    assert_eq!(answers.len(), 2);
    assert!(answers.iter().all(|a| a.passed));
}

#[tokio::test]
async fn an_endpoint_answering_nonsense_fails_and_says_what_it_said() {
    let mock = mock_endpoint::start_saying(
        Some("I am a teapot".into()),
        1,
        Duration::ZERO,
        Duration::ZERO,
    )
    .await;
    let err = coherence::verify(&target(mock.port), Duration::from_secs(5))
        .await
        .expect_err("probe fails");
    let text = format!("{err:#}");
    // The message must quote the answer back — "coherence probe failed" alone
    // sends the reader to the server logs for something the client already saw.
    assert!(text.contains("teapot"), "{text}");
    assert!(
        text.contains("arithmetic"),
        "names the failing check: {text}"
    );
    // And it must name the escape hatch, or the only way out is reading source.
    assert!(text.contains("--skip-coherence-probe"), "{text}");
}

#[tokio::test]
async fn an_unreachable_endpoint_is_a_transport_error_not_a_wrong_answer() {
    // A closed port and a confused model are different diagnoses; conflating
    // them sends the reader looking at the wrong thing.
    let err = coherence::verify(&target(1), Duration::from_secs(2))
        .await
        .expect_err("probe fails");
    let text = format!("{err:#}");
    assert!(
        !text.contains("coherence probe"),
        "should not blame the model: {text}"
    );
}

#[tokio::test]
async fn the_probe_runs_before_a_benchmark_does_any_work() {
    use atlas_plugin::headless::{HeadlessOptions, RunRequest, SilentReporter, run_blocking};
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
            coherence: CoherencePolicy::Require,
        },
    };

    let outcome = tokio::task::spawn_blocking(move || {
        run_blocking(&executor, request, &mut SilentReporter, &|| false)
    })
    .await
    .expect("join")
    .expect("drives");

    assert_ne!(outcome.exit_code(), 0, "a failed probe must fail the run");
    // The defaults are 5 concurrencies x 4 input lengths with warm-ups — many
    // hundreds of requests. Only the two probe questions may have been asked.
    assert_eq!(
        requests.load(std::sync::atomic::Ordering::Relaxed),
        2,
        "the sweep must not have started"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
