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
    assert_eq!(
        b.options().unwrap(),
        ServeOptions {
            max_seq_len: 32_768,
            speculative: false,
        }
    );
    assert_eq!(b.long_ctx_tokens, 16_384);
    assert_eq!(b.tps_tokens, 256);
    assert_eq!(b.probe_budget, 512);
    assert_eq!(b.timeout, Duration::from_secs(300));
    assert!(!b.update_baselines);
}

#[test]
fn a_long_context_probe_that_cannot_fit_is_rejected_before_the_run() {
    let mut exact_fit = ServeMatrix::default();
    configured(&mut exact_fit, |v| {
        v.set("max_seq_len", ParamValue::Int(4096));
        v.set("long_ctx_tokens", ParamValue::Int(3840));
    })
    .unwrap();

    let mut overflow = ServeMatrix::default();
    let err = configured(&mut overflow, |v| {
        v.set("max_seq_len", ParamValue::Int(4096));
        v.set("long_ctx_tokens", ParamValue::Int(3841));
    })
    .unwrap_err();
    assert_eq!(
        err.to_string(),
        "Long-context probe: 3841 tokens plus 256 tokens of needle, question and answer does not fit in a 4096-token context"
    );
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
    assert_eq!(
        b.plan,
        Plan {
            rounds: vec![
                Round {
                    model: "org/a".into(),
                    quant: "nvfp4".into(),
                    skipped: None,
                    excluded: false,
                },
                Round {
                    model: "org/b".into(),
                    quant: "fp8".into(),
                    skipped: Some(Absence::NoKernels),
                    excluded: false,
                },
            ],
        }
    );
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
    b.cursor = 7;
    configured(&mut b, |v| {
        v.set("include", ParamValue::Text("  ORG/B  ".into()));
        v.set("max_seq_len", ParamValue::Int(8192));
        v.set("long_ctx_tokens", ParamValue::Int(1024));
        v.set("tps_tokens", ParamValue::Int(128));
        v.set("probe_budget", ParamValue::Int(64));
        v.set("speculative", ParamValue::Bool(true));
        v.set("request_timeout_s", ParamValue::Int(15));
        v.set("update_baselines", ParamValue::Bool(true));
    })
    .unwrap();
    assert_eq!(b.include, "ORG/B");
    assert_eq!(
        b.options().unwrap(),
        ServeOptions {
            max_seq_len: 8192,
            speculative: true,
        }
    );
    assert_eq!(b.long_ctx_tokens, 1024);
    assert_eq!(b.tps_tokens, 128);
    assert_eq!(b.probe_budget, 64);
    assert_eq!(b.timeout, Duration::from_secs(15));
    assert!(b.update_baselines);
    assert_eq!(b.plan, Plan::default());
    assert!(!b.planned_built);
    assert_eq!(b.cursor, 0);
    assert!(b.results.is_empty());
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
    let err = ServeMatrix::default()
        .host()
        .err()
        .expect("a default matrix has no installed host");
    assert_eq!(err.to_string(), host::NO_HOST);
}

#[test]
fn the_descriptor_warns_that_a_run_replaces_the_serving_model() {
    const { assert!(DESCRIPTOR.needs_confirmation) };
    assert!(DESCRIPTOR.detail.contains("replaces whatever model"));
}
