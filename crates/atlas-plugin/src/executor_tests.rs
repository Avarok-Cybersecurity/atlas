// SPDX-License-Identifier: AGPL-3.0-only

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

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
    threshold_params: &[],
    // Speed, so the hardware pre-check's REFUSE path is the one exercised
    // here — the correctness path can never refuse and would prove nothing.
    sensitivity: crate::hardware::Sensitivity::Speed,
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

static LIFECYCLE: LazyLock<Mutex<Vec<&'static str>>> = LazyLock::new(|| Mutex::new(Vec::new()));

struct LifecycleFake {
    seen: usize,
}

const LIFECYCLE_DESC: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "lifecycle-fake",
    name: "Lifecycle Fake",
    summary: "test double",
    detail: "test double",
    duration_hint: "instant",
    updated: "2026-08-24",
    needs_confirmation: false,
    intended_for: None,
    threshold_params: &[],
    sensitivity: crate::hardware::Sensitivity::Correctness,
    ctor: || Box::new(LifecycleFake { seen: 0 }),
};

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

impl Plugin for LifecycleFake {
    fn metadata(&self) -> &'static PluginMetadata {
        &FAKE_META
    }

    async fn load(&mut self, _handle: PluginHandle) -> Result<()> {
        LIFECYCLE.lock().unwrap().push("load");
        Ok(())
    }
}

impl Benchmark for LifecycleFake {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &LIFECYCLE_DESC
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        Vec::new()
    }

    fn configure(&mut self, _values: &ParamValues) -> Result<()> {
        LIFECYCLE.lock().unwrap().push("configure");
        Ok(())
    }

    fn next(&mut self) -> impl Future<Output = Result<BenchmarkResult>> + Send {
        self.seen += 1;
        let seen = self.seen;
        LIFECYCLE.lock().unwrap().push("next");
        async move {
            Ok(if seen == 1 {
                BenchmarkResult::running("working", Duration::from_millis(1))
            } else {
                BenchmarkResult::completed("done", Duration::from_millis(2))
            })
        }
    }

    fn cleanup(&mut self) -> impl Future<Output = Result<()>> + Send {
        LIFECYCLE.lock().unwrap().push("cleanup");
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
    assert_eq!(
        frames[1].as_ref().unwrap_err().to_string(),
        "step 2 exploded"
    );
    assert_eq!(b.seen, 2, "no further steps after the error");
}

#[tokio::test]
async fn executor_runs_the_lifecycle_and_always_cleans_up() {
    LIFECYCLE.lock().unwrap().clear();
    let store = ArtifactStore::with_root(
        std::env::temp_dir().join(format!("atlas-executor-lifecycle-{}", std::process::id())),
    );
    let executor = BenchmarkExecutor::new(tokio::runtime::Handle::current(), store);
    let run = executor.start(
        &LIFECYCLE_DESC,
        ParamValues::default(),
        TargetEndpoint::local(1, "unused"),
        CoherencePolicy::Skip,
    );

    tokio::time::timeout(Duration::from_secs(10), async {
        while !run.is_finished() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the real executor completes and tears down");

    let statuses = run
        .drain()
        .into_iter()
        .filter_map(|message| match message {
            ExecutorMessage::Frame(frame) => Some(frame.status),
            ExecutorMessage::Event(_) => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(statuses, [RunStatus::Running, RunStatus::Completed]);
    assert_eq!(
        *LIFECYCLE.lock().unwrap(),
        ["load", "configure", "next", "next", "cleanup"]
    );
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
