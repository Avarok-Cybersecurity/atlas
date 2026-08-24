// SPDX-License-Identifier: AGPL-3.0-only

//! The Serve Matrix's state machine, driven to completion against a real
//! socket through a fake [`ServeHost`].
//!
//! The unit tests cover the bars and the classification. This covers the thing
//! that actually breaks: a matrix that skips a round, advances its cursor
//! twice, or terminates before reaching the last checkpoint — every one of
//! which quietly shrinks coverage, which is the one property this benchmark
//! exists to guarantee.

mod mock_endpoint;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use atlas_plugin::benchmarks::serve_matrix::ServeMatrix;
use atlas_plugin::benchmarks::serve_matrix::host::{ServeCandidate, ServeHost, ServeOptions};
use atlas_plugin::{
    ArtifactStore, Benchmark, BenchmarkResult, ParamValue, ParamValues, Plugin, PluginHandle,
    RunStatus, TargetEndpoint, VerdictKind,
};
use futures::StreamExt;
use futures::future::BoxFuture;

/// A host that serves `up` at the mock's port and refuses everything else, so
/// one run exercises both the verified and the did-not-boot paths.
struct FakeHost {
    port: u16,
    up: String,
    restores: Arc<AtomicBool>,
}

impl ServeHost for FakeHost {
    fn roster(&self) -> anyhow::Result<Vec<ServeCandidate>> {
        Ok(vec![
            ServeCandidate::ready(&self.up, "nvfp4"),
            ServeCandidate::ready("org/crashes", "fp8"),
        ])
    }

    fn serve(
        &self,
        model: &str,
        _opts: ServeOptions,
    ) -> BoxFuture<'_, anyhow::Result<TargetEndpoint>> {
        let model = model.to_string();
        Box::pin(async move {
            if model != self.up {
                anyhow::bail!("CUDA out of memory");
            }
            Ok(TargetEndpoint::local(self.port, model))
        })
    }

    fn restore(&self) -> BoxFuture<'_, anyhow::Result<()>> {
        self.restores.store(true, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }
}

fn store(name: &str) -> ArtifactStore {
    let dir = std::env::temp_dir().join(format!("atlas-sm-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    ArtifactStore::with_root(dir)
}

fn handle(port: u16, store: ArtifactStore) -> PluginHandle {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || while rx.recv().is_ok() {});
    PluginHandle::new(
        1,
        TargetEndpoint::local(port, "mock"),
        store,
        tx,
        Arc::new(AtomicBool::new(false)),
    )
}

async fn collect(bench: &mut ServeMatrix) -> Vec<BenchmarkResult> {
    let stream = bench.run();
    futures::pin_mut!(stream);
    let mut frames = Vec::new();
    while let Some(item) = stream.next().await {
        frames.push(item.expect("benchmark step failed"));
    }
    frames
}

#[tokio::test]
async fn every_planned_round_is_reached_and_a_crashed_one_fails_the_matrix() {
    // The mock answers `/v1/models` with `{"id":"mock"}`, so the round whose
    // model is "mock" also passes the identity bar.
    let mock = mock_endpoint::start(8, Duration::from_millis(1), Duration::from_millis(1)).await;
    let restores = Arc::new(AtomicBool::new(false));
    let mut bench = ServeMatrix::with_host(Arc::new(FakeHost {
        port: mock.port,
        up: "mock".into(),
        restores: restores.clone(),
    }));
    bench
        .load(handle(mock.port, store("e2e")))
        .await
        .expect("load");

    let mut values = ParamValues::defaults(&bench.parameters());
    // The long-context probe would send 16k tokens through the mock for no
    // extra signal; every other probe still runs.
    values.set("long_ctx_tokens", ParamValue::Int(0));
    values.set("tps_tokens", ParamValue::Int(16));
    values.set("probe_budget", ParamValue::Int(32));
    bench.configure(&values).expect("configure");

    let frames = collect(&mut bench).await;
    let last = frames.last().expect("at least one frame");
    assert_eq!(last.status, RunStatus::Completed, "{:?}", last.verdict);
    // plan frame + one frame per planned round + the terminal frame.
    assert_eq!(frames.len(), 4, "plan, 2 rounds, done");

    let table = last.table.as_ref().expect("a results table");
    assert_eq!(table.rows.len(), 2, "one row per candidate");

    let verdict = last.verdict.as_ref().expect("a verdict");
    // The mock is a canned SSE stream, not a model, so the round that DID come
    // up fails its content probes. That is correct and not what this test is
    // about — what matters is that both rounds are in the denominator and each
    // failure is attributed to what actually happened to it.
    assert_eq!(verdict.kind, VerdictKind::Fail, "{}", verdict.reason);
    assert!(
        verdict.reason.contains("/2 planned checkpoints verified"),
        "the crashed round must stay in the denominator: {}",
        verdict.reason
    );
    assert!(
        verdict.reason.contains("org/crashes") && verdict.reason.contains("CUDA out of memory"),
        "the boot failure's reason has to reach the verdict: {}",
        verdict.reason
    );
    // Read the two rows rather than parse the prose. Rows are model-sorted, so
    // `mock` is first and `org/crashes` second.
    let cell = |row: usize| table.rows[row].last().expect("a verdict cell").text.clone();
    assert!(
        table.rows[0][0].text.starts_with("mock"),
        "{:?}",
        table.rows[0][0].text
    );
    // The round that came up was PROBED, not skipped: its failure names probes,
    // never `did-not-boot`. Confusing the two is the whole defect being fixed.
    assert!(!cell(0).contains("did-not-boot"), "{}", cell(0));
    assert!(
        !cell(0).contains("wrong-model"),
        "the endpoint reported `mock`, which is what this round loaded: {}",
        cell(0)
    );
    assert!(cell(1).contains("did-not-boot"), "{}", cell(1));

    // coherence (2) + codegen + tool call + throughput, for the ONE round that
    // came up. The crashed round must not have reached the endpoint at all.
    assert_eq!(
        mock.requests.load(Ordering::Relaxed),
        5,
        "one round's probes, and only one round's"
    );

    bench.cleanup().await.expect("cleanup");
    assert!(
        restores.load(Ordering::SeqCst),
        "the box must be put back however the run ended"
    );
}

#[tokio::test]
async fn a_round_the_endpoint_serves_under_another_name_is_not_scored_as_that_model() {
    // The fake host claims it booted `org/never-loaded`; the endpoint answers
    // every question correctly but reports `mock` in `/v1/models`. Without the
    // identity bar this is a clean PASS filed under a checkpoint that was
    // never loaded.
    let mock = mock_endpoint::start(8, Duration::from_millis(1), Duration::from_millis(1)).await;
    let mut bench = ServeMatrix::with_host(Arc::new(FakeHost {
        port: mock.port,
        up: "org/never-loaded".into(),
        restores: Arc::new(AtomicBool::new(false)),
    }));
    bench
        .load(handle(mock.port, store("identity")))
        .await
        .expect("load");

    let mut values = ParamValues::defaults(&bench.parameters());
    values.set("include", ParamValue::Text("never-loaded".into()));
    values.set("long_ctx_tokens", ParamValue::Int(0));
    values.set("tps_tokens", ParamValue::Int(16));
    values.set("probe_budget", ParamValue::Int(32));
    bench.configure(&values).expect("configure");

    let frames = collect(&mut bench).await;
    let verdict = frames
        .last()
        .and_then(|f| f.verdict.as_ref())
        .expect("a verdict");
    assert_eq!(frames.len(), 3, "plan, selected round, done");
    let table = frames
        .last()
        .and_then(|frame| frame.table.as_ref())
        .expect("a results table");
    let selected = table
        .rows
        .iter()
        .find(|row| row[0].text.starts_with("org/never-loaded"))
        .expect("the selected round remains visible in the roster table");
    assert!(
        selected
            .last()
            .expect("a verdict cell")
            .text
            .contains("wrong-model"),
        "the identity bar belongs to the requested round: {}",
        selected.last().expect("a verdict cell").text
    );
    assert_eq!(verdict.kind, VerdictKind::Fail, "{}", verdict.reason);
    assert!(
        verdict.reason.contains("0/1 planned checkpoints verified")
            && verdict.reason.contains("org/never-loaded")
            && verdict.reason.contains("wrong-model"),
        "the round must be attributed to the identity check: {}",
        verdict.reason
    );
    assert_eq!(
        mock.requests.load(Ordering::Relaxed),
        5,
        "only the selected round reaches the endpoint"
    );
}
