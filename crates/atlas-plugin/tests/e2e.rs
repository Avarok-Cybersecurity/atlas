// SPDX-License-Identifier: AGPL-3.0-only

//! End-to-end: real benchmarks, driven through the real executor, against a
//! real (if tiny) HTTP endpoint.
//!
//! The unit tests cover the arithmetic; these cover the thing that actually
//! breaks — a benchmark that streams, measures and terminates correctly over a
//! socket, including the chunked framing the mock deliberately splits mid-line.

mod mock_endpoint;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use atlas_plugin::benchmarks::{concurrency::ConcurrencySweep, ttft::TtftGate};
use atlas_plugin::{
    ArtifactStore, Benchmark, BenchmarkResult, ParamValue, ParamValues, Plugin, PluginHandle,
    RunStatus, TargetEndpoint, VerdictKind,
};
use futures::StreamExt;

fn temp_store(name: &str) -> ArtifactStore {
    let dir = std::env::temp_dir().join(format!("atlas-e2e-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    ArtifactStore::with_root(dir)
}

/// A handle whose event receiver is kept alive for the test's duration.
fn handle(port: u16, store: ArtifactStore) -> (PluginHandle, Arc<AtomicBool>) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || while rx.recv().is_ok() {});
    let cancel = Arc::new(AtomicBool::new(false));
    (
        PluginHandle::new(
            1,
            TargetEndpoint::local(port, "mock"),
            store,
            tx,
            cancel.clone(),
        ),
        cancel,
    )
}

async fn collect<B: Benchmark + Send>(bench: &mut B) -> Vec<BenchmarkResult> {
    let stream = bench.run();
    futures::pin_mut!(stream);
    let mut frames = Vec::new();
    while let Some(item) = stream.next().await {
        frames.push(item.expect("benchmark step failed"));
    }
    frames
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrency_sweep_measures_a_real_stream_end_to_end() {
    let mock = mock_endpoint::start(8, Duration::from_millis(40), Duration::from_millis(5)).await;
    let (h, _cancel) = handle(mock.port, temp_store("sweep"));

    let mut bench = ConcurrencySweep::default();
    bench.load(h).await.expect("load");
    let mut values = ParamValues::defaults(&bench.parameters());
    values.set("concurrencies", ParamValue::IntList(vec![1, 4]));
    values.set("isls", ParamValue::IntList(vec![64]));
    values.set("osl", ParamValue::Int(8));
    values.set("warmup", ParamValue::Int(0));
    bench.configure(&values).expect("configure");

    let frames = collect(&mut bench).await;
    let last = frames.last().expect("at least one frame");
    assert_eq!(last.status, RunStatus::Completed, "{:?}", last.verdict);

    // probe frame + one frame per cell + the terminal frame
    assert_eq!(frames.len(), 4, "probe, 2 cells, done");

    let table = last.table.as_ref().expect("a results table");
    assert_eq!(table.rows.len(), 2, "one row per (isl x conc) cell");
    // 1 + 4 requests, and the mock counts every one of them.
    assert_eq!(mock.requests.load(Ordering::Relaxed), 5);

    // TTFT was measured through the chunk split, and the token count survived
    // it: 8 deltas were streamed and the usage chunk agrees.
    let ttft_p50 = &table.rows[0][2].text;
    assert_ne!(ttft_p50, "—", "TTFT must be measured, not missing");
    let throughput: f64 = table.rows[0][8]
        .text
        .parse()
        .expect("throughput is numeric");
    assert!(throughput > 0.0, "throughput was {throughput}");

    assert_eq!(
        last.verdict.as_ref().map(|v| v.kind),
        Some(VerdictKind::Info),
        "a sweep measures, it does not gate"
    );
    assert!(
        last.verdict
            .as_ref()
            .unwrap()
            .reason
            .contains("no request errors"),
        "{:?}",
        last.verdict
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_sweep_against_a_dead_endpoint_fails_at_the_probe() {
    // The port is closed: the sweep must stop at the probe rather than produce
    // a whole table of suspiciously fast empty cells.
    let dead = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = dead.local_addr().unwrap().port();
    drop(dead);

    let (h, _cancel) = handle(port, temp_store("dead"));
    let mut bench = ConcurrencySweep::default();
    bench.load(h).await.expect("load");
    let values = ParamValues::defaults(&bench.parameters());
    bench.configure(&values).expect("configure");

    let stream = bench.run();
    futures::pin_mut!(stream);
    let first = stream.next().await.expect("one item");
    let err = first.expect_err("a closed port must not look like a fast run");
    assert!(format!("{err:#}").contains("probe failed"), "{err:#}");
    assert!(
        stream.next().await.is_none(),
        "the stream ends on the error"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_warm_gate_records_a_baseline_then_gates_against_it() {
    let mock = mock_endpoint::start(4, Duration::from_millis(30), Duration::from_millis(2)).await;
    let store = temp_store("warmgate");

    // Leg 1: no baseline exists, so the verdict reports rather than passes,
    // and the run becomes the baseline.
    let (h, _c1) = handle(mock.port, store.clone());
    let mut bench = TtftGate::new(atlas_plugin::benchmarks::ttft::Mode::Warm);
    bench.load(h).await.expect("load");
    let mut values = ParamValues::defaults(&bench.parameters());
    values.set("prompt_lengths", ParamValue::IntList(vec![64]));
    values.set("repeats", ParamValue::Int(3));
    bench.configure(&values).expect("configure");

    let first = collect(&mut bench).await;
    let last = first.last().unwrap();
    assert_eq!(last.status, RunStatus::Completed);
    assert_eq!(
        last.verdict.as_ref().map(|v| v.kind),
        Some(VerdictKind::Info),
        "with nothing to compare against, this is not a PASS"
    );
    let table = last.table.as_ref().expect("a table");
    assert_eq!(table.rows.len(), 1);
    // The mock reports 40 cached prompt tokens, so the warm gate can prove the
    // cache actually hit — a warm leg reading 0 measured a cold path.
    assert_eq!(table.rows[0][5].text, "40");

    // Leg 2: the baseline is now on disk and the endpoint has not changed, so
    // the same latency must pass.
    let (h2, _c2) = handle(mock.port, store.clone());
    let mut second = TtftGate::new(atlas_plugin::benchmarks::ttft::Mode::Warm);
    second.load(h2).await.expect("load");
    second.configure(&values).expect("configure");
    let frames = collect(&mut second).await;
    let verdict = frames.last().unwrap().verdict.as_ref().expect("a verdict");
    assert_eq!(
        verdict.kind,
        VerdictKind::Pass,
        "an unchanged endpoint must not regress: {}",
        verdict.reason
    );
    assert!(verdict.reason.contains("limit +3.0%"), "{}", verdict.reason);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_cold_gate_never_reuses_a_prompt() {
    // 6 samples, no priming request: the cold gate must issue exactly one
    // request per sample. (The warm gate issues two — prime then measure.)
    let mock = mock_endpoint::start(2, Duration::from_millis(5), Duration::from_millis(1)).await;
    let (h, _cancel) = handle(mock.port, temp_store("coldgate"));
    let mut bench = TtftGate::new(atlas_plugin::benchmarks::ttft::Mode::Cold);
    bench.load(h).await.expect("load");
    let mut values = ParamValues::defaults(&bench.parameters());
    values.set("prompt_lengths", ParamValue::IntList(vec![32]));
    values.set("repeats", ParamValue::Int(6));
    values.set("update_baseline", ParamValue::Bool(false));
    bench.configure(&values).expect("configure");

    let frames = collect(&mut bench).await;
    assert_eq!(frames.last().unwrap().status, RunStatus::Completed);
    assert_eq!(
        mock.requests.load(Ordering::Relaxed),
        6,
        "cold mode measures once per sample, with no priming request"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn cancellation_stops_a_run_between_steps() {
    // Slow enough that the run is still going when it is cancelled.
    let mock = mock_endpoint::start(40, Duration::from_millis(20), Duration::from_millis(30)).await;
    let (h, cancel) = handle(mock.port, temp_store("cancel"));
    let mut bench = ConcurrencySweep::default();
    bench.load(h).await.expect("load");
    let mut values = ParamValues::defaults(&bench.parameters());
    values.set("concurrencies", ParamValue::IntList(vec![1]));
    values.set("isls", ParamValue::IntList(vec![32, 64, 128, 256]));
    values.set("osl", ParamValue::Int(40));
    values.set("warmup", ParamValue::Int(0));
    bench.configure(&values).expect("configure");

    let stream = bench.run();
    futures::pin_mut!(stream);
    // Consume the probe frame, then cancel.
    let _probe = stream.next().await.expect("probe");
    cancel.store(true, Ordering::Relaxed);
    let next = stream.next().await.expect("one more item");
    let err = next.expect_err("cancellation surfaces as an error, not a clean result");
    assert!(format!("{err:#}").contains("cancelled"), "{err:#}");
    assert!(stream.next().await.is_none());
}
