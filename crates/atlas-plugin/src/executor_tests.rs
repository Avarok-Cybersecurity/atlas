// SPDX-License-Identifier: AGPL-3.0-only

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::benchmark::Benchmark;
use crate::metadata::PluginMetadata;
use crate::params::ParamSpec;
use crate::plugin::Plugin;

/// A benchmark that yields `steps` running frames, then one terminal frame.
struct Fake {
    steps: usize,
    seen: usize,
    fail_at: Option<usize>,
    cleanups: Arc<AtomicUsize>,
}

// STATIC, DELIBERATELY — compile-time data, test fixture. `BenchmarkDescriptor`
// and `PluginMetadata` are borrowed as `&'static` by the traits under test,
// so a test double has to have static storage to be passed at all. Both are
// const-constructed literals with no interior mutability.
const FAKE_DESC: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "fake",
    name: "Fake",
    summary: "test double",
    detail: "test double",
    duration_hint: "instant",
    updated: "2026-07-31",
    needs_confirmation: false,
    intended_for: None,
    ctor: || Box::new(Fake::new(1, None, Arc::new(AtomicUsize::new(0)))),
};

impl Fake {
    fn new(steps: usize, fail_at: Option<usize>, cleanups: Arc<AtomicUsize>) -> Self {
        Self {
            steps,
            seen: 0,
            fail_at,
            cleanups,
        }
    }
}

// Same category as `FAKE_DESC`: compile-time test data that the trait
// borrows as `&'static`.
const FAKE_META: PluginMetadata = PluginMetadata::atlas("test double");

impl Plugin for Fake {
    fn metadata(&self) -> &'static PluginMetadata {
        &FAKE_META
    }
    async fn load(&mut self, _handle: PluginHandle) -> Result<()> {
        Ok(())
    }
}

impl Benchmark for Fake {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &FAKE_DESC
    }
    fn parameters(&self) -> Vec<ParamSpec> {
        Vec::new()
    }
    fn configure(&mut self, _values: &ParamValues) -> Result<()> {
        Ok(())
    }
    fn next(&mut self) -> impl Future<Output = Result<BenchmarkResult>> + Send {
        self.seen += 1;
        let n = self.seen;
        let steps = self.steps;
        let fail_at = self.fail_at;
        async move {
            if fail_at == Some(n) {
                anyhow::bail!("step {n} exploded");
            }
            Ok(if n > steps {
                BenchmarkResult::completed("done", Duration::from_millis(n as u64))
            } else {
                BenchmarkResult::running(format!("step {n}"), Duration::from_millis(n as u64))
            })
        }
    }
    fn cleanup(&mut self) -> impl Future<Output = Result<()>> + Send {
        self.cleanups.fetch_add(1, Ordering::Relaxed);
        async { Ok(()) }
    }
}

async fn collect(b: &mut Fake) -> Vec<Result<BenchmarkResult>> {
    let stream = b.run();
    futures::pin_mut!(stream);
    let mut out = Vec::new();
    while let Some(x) = stream.next().await {
        out.push(x);
    }
    out
}

#[tokio::test]
async fn run_stops_at_the_first_terminal_status() {
    let cleanups = Arc::new(AtomicUsize::new(0));
    let mut b = Fake::new(3, None, cleanups.clone());
    let frames = collect(&mut b).await;
    assert_eq!(frames.len(), 4, "3 running + 1 terminal");
    assert!(
        frames[..3]
            .iter()
            .all(|f| f.as_ref().unwrap().status == RunStatus::Running)
    );
    assert_eq!(frames[3].as_ref().unwrap().status, RunStatus::Completed);
    // `next()` is not called again after the terminal frame.
    assert_eq!(b.seen, 4);
}

#[tokio::test]
async fn an_error_terminates_the_stream() {
    let mut b = Fake::new(9, Some(2), Arc::new(AtomicUsize::new(0)));
    let frames = collect(&mut b).await;
    assert_eq!(frames.len(), 2);
    assert!(frames[1].is_err());
    assert_eq!(b.seen, 2, "no further steps after the error");
}

#[tokio::test]
async fn executor_runs_the_lifecycle_and_always_cleans_up() {
    let cleanups = Arc::new(AtomicUsize::new(0));
    let counter = cleanups.clone();
    // A descriptor whose ctor closes over the shared counter is not possible
    // with a plain `fn` pointer, so drive the dyn object directly here — the
    // executor's own teardown is covered by `cleanup_runs_after_a_setup_error`.
    let mut b: Box<dyn DynBenchmark> = Box::new(Fake::new(2, None, counter));
    let mut n = 0;
    {
        let stream = drive(b.as_mut());
        futures::pin_mut!(stream);
        while stream.next().await.is_some() {
            n += 1;
        }
    }
    b.cleanup().await.unwrap();
    assert_eq!(n, 3);
    assert_eq!(cleanups.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn registry_dispatch_goes_through_the_same_driver() {
    let mut b = FAKE_DESC.build();
    let mut statuses = Vec::new();
    let stream = drive(b.as_mut());
    futures::pin_mut!(stream);
    while let Some(f) = stream.next().await {
        statuses.push(f.unwrap().status);
    }
    assert_eq!(
        statuses,
        vec![RunStatus::Running, RunStatus::Completed],
        "the ctor's 1-step Fake must drive identically to a typed run"
    );
}
