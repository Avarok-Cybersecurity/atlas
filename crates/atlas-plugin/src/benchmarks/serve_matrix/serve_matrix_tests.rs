// SPDX-License-Identifier: AGPL-3.0-only

//! Configuration, planning against a fake host, and the restore contract.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::benchmarks::serve_matrix::host::{Absence, ServeCandidate};
use crate::params::ParamValue;
use futures::future::BoxFuture;

#[derive(Default)]
struct FakeHost {
    roster: Vec<ServeCandidate>,
    restores: AtomicUsize,
}

impl ServeHost for FakeHost {
    fn roster(&self) -> Result<Vec<ServeCandidate>> {
        Ok(self.roster.clone())
    }
    fn serve(
        &self,
        _model: &str,
        _opts: ServeOptions,
    ) -> BoxFuture<'_, Result<TargetEndpointAlias>> {
        Box::pin(async { anyhow::bail!("fake host does not serve") })
    }
    fn restore(&self) -> BoxFuture<'_, Result<()>> {
        self.restores.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }
}

type TargetEndpointAlias = crate::plugin::TargetEndpoint;

fn host_with(roster: Vec<ServeCandidate>) -> Arc<FakeHost> {
    Arc::new(FakeHost {
        roster,
        restores: AtomicUsize::new(0),
    })
}

fn configured(b: &mut ServeMatrix, edit: impl FnOnce(&mut ParamValues)) -> Result<()> {
    let mut v = ParamValues::defaults(&b.parameters());
    edit(&mut v);
    b.configure(&v)
}

#[test]
fn the_defaults_run_everything_the_box_can_serve() {
    let mut b = ServeMatrix::default();
    configured(&mut b, |_| {}).unwrap();
    assert_eq!(b.include, "", "`all` means no filter");
    assert!(!b.options().unwrap().speculative);
    assert_eq!(b.options().unwrap().max_seq_len, 32_768);
}

#[test]
fn a_long_context_probe_that_cannot_fit_is_rejected_before_the_run() {
    let mut b = ServeMatrix::default();
    let err = configured(&mut b, |v| {
        v.set("max_seq_len", ParamValue::Int(4096));
        v.set("long_ctx_tokens", ParamValue::Int(16_384));
    })
    .unwrap_err()
    .to_string();
    assert!(err.contains("does not fit"), "{err}");
}

#[test]
fn turning_the_long_context_probe_off_is_allowed() {
    let mut b = ServeMatrix::default();
    configured(&mut b, |v| {
        v.set("long_ctx_tokens", ParamValue::Int(0));
    })
    .unwrap();
    assert_eq!(b.long_ctx_tokens, 0);
}

#[test]
fn the_plan_comes_from_the_box_not_from_a_list_in_this_file() {
    let host = host_with(vec![
        ServeCandidate::ready("org/a", "nvfp4"),
        ServeCandidate::absent("org/b", "fp8", Absence::NoKernels),
    ]);
    let mut b = ServeMatrix::with_host(host);
    configured(&mut b, |_| {}).unwrap();
    b.build_plan().unwrap();
    assert_eq!(b.plan.planned_count(), 1);
    assert_eq!(b.plan.planned().next().unwrap().label(), "org/a · nvfp4");
    assert_eq!(b.plan.skipped().count(), 1);
}

#[test]
fn reconfiguring_discards_a_previous_plan_and_its_results() {
    let host = host_with(vec![ServeCandidate::ready("org/a", "nvfp4")]);
    let mut b = ServeMatrix::with_host(host);
    configured(&mut b, |_| {}).unwrap();
    b.build_plan().unwrap();
    b.results.push(RoundResult {
        label: "org/a · nvfp4".into(),
        outcome: Outcome::NotReached,
        baseline_tps: None,
    });
    configured(&mut b, |_| {}).unwrap();
    assert!(
        b.results.is_empty() && b.plan.planned_count() == 0 && !b.planned_built && b.cursor == 0
    );
}

#[tokio::test]
async fn cleanup_restores_the_box_once_a_plan_exists() {
    let host = host_with(vec![ServeCandidate::ready("org/a", "nvfp4")]);
    let mut b = ServeMatrix::with_host(host.clone());
    configured(&mut b, |_| {}).unwrap();
    // Nothing was booted yet, so there is nothing to put back.
    b.cleanup().await.unwrap();
    assert_eq!(host.restores.load(Ordering::SeqCst), 0);

    b.build_plan().unwrap();
    b.cleanup().await.unwrap();
    assert_eq!(
        host.restores.load(Ordering::SeqCst),
        1,
        "a cancelled matrix must not leave the box on whatever round four loaded"
    );
}

#[test]
fn without_a_host_the_benchmark_says_what_is_missing() {
    // `Plugin::load`'s contract: the message lands where the Start button
    // would be, so it has to name the thing that is absent.
    assert!(host::NO_HOST.contains("Atlas server"));
    assert!(ServeMatrix::default().host().is_err());
}

#[test]
fn the_descriptor_warns_that_a_run_replaces_the_serving_model() {
    const { assert!(DESCRIPTOR.needs_confirmation) };
    assert!(DESCRIPTOR.detail.contains("replaces whatever model"));
}
