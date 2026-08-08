// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the gate's self-start.
//!
//! Everything here runs without a GPU: the branching (which box class, which
//! model, whether a recipe is bound), the two process-wide refusals, the
//! headroom threshold, and the teardown.
//!
//! ★ Nothing here may trip the process-global shutdown latch. It has no reset,
//! and `model_swap::swap` reads it directly — so a test that requested a
//! shutdown would make two of `model_swap`'s refusal tests fail depending on
//! which ran first. That is why `SelfServed::shutdown` (which does request one)
//! has no case here and `Drop` (which does not) has two.

use super::*;
use atlas_plugin::gate::{GateBaseline, HardwareBaseline, ModelBaseline};

fn baseline(entries: &[(&str, &str, Option<&str>)]) -> GateBaseline {
    let mut hardware = BTreeMap::new();
    for (hw, model, recipe) in entries {
        let e = hardware
            .entry(hw.to_string())
            .or_insert_with(|| HardwareBaseline {
                default: model.to_string(),
                models: BTreeMap::new(),
            });
        e.models.insert(
            model.to_string(),
            ModelBaseline {
                recipe: recipe.map(str::to_string),
                note: String::new(),
                metrics: BTreeMap::new(),
            },
        );
    }
    GateBaseline {
        schema: 2,
        hardware,
    }
}

#[test]
fn a_single_box_class_is_inferred() {
    let b = baseline(&[("gb10", "unsloth/Qwen3.6-27B-NVFP4", Some("qwen3.6/x"))]);
    let r = resolve(&b, "bfcl-subset", None).expect("inferred");
    assert_eq!(r.model, "unsloth/Qwen3.6-27B-NVFP4");
    assert_eq!(r.recipe_id, "qwen3.6/x");
}

#[test]
fn several_box_classes_refuse_to_guess() {
    // Guessing here would serve one box's config and score it against the
    // other's thresholds — TTFT ceilings are box-local.
    let b = baseline(&[("gb10", "m", Some("r")), ("mi300x", "m", Some("r2"))]);
    let err = resolve(&b, "ttft-warm-gate", None).expect_err("refused");
    let msg = format!("{err:#}");
    assert!(msg.contains("gb10"), "{msg}");
    assert!(msg.contains("mi300x"), "{msg}");
    assert!(msg.contains("--hardware"), "names the fix: {msg}");
}

#[test]
fn an_explicit_box_class_picks_its_entry() {
    let b = baseline(&[
        ("gb10", "a", Some("recipe-a")),
        ("mi300x", "b", Some("recipe-b")),
    ]);
    let r = resolve(&b, "ttft-warm-gate", Some("mi300x")).expect("picked");
    assert_eq!(r.recipe_id, "recipe-b");
}

#[test]
fn an_unknown_box_class_names_what_exists() {
    let b = baseline(&[("gb10", "m", Some("r"))]);
    let err = resolve(&b, "bfcl-subset", Some("h100")).expect_err("refused");
    let msg = format!("{err:#}");
    assert!(msg.contains("h100"), "{msg}");
    assert!(msg.contains("gb10"), "lists what it has: {msg}");
}

#[test]
fn a_baseline_without_a_recipe_cannot_self_start() {
    // The honest failure: this gate has thresholds but nothing says how to
    // serve them, so it must refuse rather than invent a config.
    let b = baseline(&[("gb10", "m", None)]);
    let err = resolve(&b, "bfcl-subset", None).expect_err("refused");
    let msg = format!("{err:#}");
    assert!(msg.contains("no recipe is bound"), "{msg}");
    assert!(
        msg.contains("--url/--model"),
        "offers the alternative: {msg}"
    );
}

#[test]
fn an_empty_baseline_is_an_error_not_a_default() {
    let b = baseline(&[]);
    assert!(resolve(&b, "bfcl-subset", None).is_err());
}

// ── The one-server-per-process invariant ──

#[test]
fn the_start_slot_is_claimable_exactly_once() {
    // The refusal is what keeps a second self-start from hanging: teardown
    // tripped the one-way shutdown latch, so the second server would come up
    // into a draining process and never begin serving. A LOCAL latch, so the
    // real `STARTED` is not spent by this test.
    let started = AtomicBool::new(false);
    claim_start_slot(&started, false).expect("the first claim takes the slot");
    let err = claim_start_slot(&started, false).expect_err("the second is refused");
    let msg = format!("{err:#}");
    assert!(msg.contains("already started a server"), "{msg}");
    assert!(
        msg.contains("one benchmark per invocation"),
        "says what to do instead: {msg}"
    );
}

#[test]
fn an_already_requested_shutdown_refuses_before_the_wait() {
    // Same outcome as a spent slot, different cause — and "run one benchmark
    // per invocation" would be the wrong advice, so it is a distinct message.
    // Refusing HERE is the point: the alternative is fifteen minutes of polling
    // a listener that is not coming.
    let started = AtomicBool::new(false);
    let err = claim_start_slot(&started, true).expect_err("refused");
    let msg = format!("{err:#}");
    assert!(msg.contains("shutdown has already been requested"), "{msg}");
    assert!(
        !started.load(Ordering::SeqCst),
        "and the slot is not spent by a claim that never started anything"
    );
}

// ── The co-tenancy preflight ──

#[test]
fn a_clean_box_serves_at_the_recipes_utilisation() {
    // ~0.94 available is what a clean GB10 reads. The line must repeat the
    // recipe's utilisation VERBATIM: this check exists to refuse co-tenants,
    // never to second-guess the config the thresholds were measured under.
    let line = headroom_verdict(121.0, 114.0, 0.90, "qwen3.6/27b").expect("a clean box passes");
    assert!(line.contains("0.90"), "{line}");
    assert!(line.contains("94 %"), "{line}");
}

#[test]
fn a_co_tenanted_box_is_refused_with_the_remedies() {
    // 16 GB of co-tenants on a 121 GB unified pool: measured to cost Atlas 32 %
    // at C=16 while costing vLLM ~0, so this corrupts the measurement long
    // before it OOM-freezes the box.
    let err = headroom_verdict(121.0, 98.0, 0.90, "qwen3.6/27b").expect_err("refused");
    let msg = format!("{err:#}");
    assert!(msg.contains("qwen3.6/27b"), "names the recipe: {msg}");
    assert!(msg.contains("docker ps"), "names a remedy: {msg}");
    assert!(msg.contains("nvidia-smi"), "and the other one: {msg}");
    assert!(
        msg.contains("not a judgement on the recipe"),
        "and says what it is NOT refusing: {msg}"
    );
}

#[test]
fn the_threshold_itself_is_inclusive() {
    // Exactly at the line passes; a hair under does not. Stated because the
    // constant is the whole of the check.
    let total = 100.0;
    assert!(headroom_verdict(total, total * MIN_FREE_FRACTION, 0.9, "r").is_ok());
    assert!(headroom_verdict(total, total * MIN_FREE_FRACTION - 0.1, 0.9, "r").is_err());
}

// ── Teardown ──

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("a current-thread runtime")
}

/// A `SelfServed` around a task that never finishes on its own, plus a receiver
/// that resolves with `Err` once that task has actually been destroyed.
///
/// The sender lives INSIDE the task, so the channel closing is proof the task
/// was dropped — not merely that a flag was set beside it.
fn served_forever() -> (SelfServed, tokio::sync::oneshot::Receiver<()>) {
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        let _tx = tx;
        std::future::pending::<()>().await;
        Ok(())
    });
    let served = SelfServed {
        target: TargetEndpoint::local(1, "m"),
        recipe_id: "r".to_string(),
        server: Some(server),
    };
    (served, rx)
}

#[test]
fn dropping_a_self_served_tears_the_server_down() {
    // The leak this exists to prevent: a dropped `JoinHandle` DETACHES its
    // task, so every early return between the spawn and an explicit shutdown
    // used to leave a ~100 GB model resident on a unified-memory box.
    runtime().block_on(async {
        let (served, rx) = served_forever();
        drop(served);
        let waited = tokio::time::timeout(Duration::from_secs(5), rx).await;
        assert!(
            matches!(waited, Ok(Err(_))),
            "the server task must be aborted, not detached: {waited:?}"
        );
    });
}

#[test]
fn a_torn_down_server_is_not_torn_down_twice() {
    // `Drop` still runs after an explicit teardown took the handle. It must
    // find nothing and do nothing — an abort on a spent handle would be
    // harmless, but a second teardown that PRINTS one is a false report that a
    // path leaked when it did not.
    runtime().block_on(async {
        let (mut served, rx) = served_forever();
        let handle = served.server.take().expect("constructed as Some");
        handle.abort();
        drop(served);
        let waited = tokio::time::timeout(Duration::from_secs(5), rx).await;
        assert!(matches!(waited, Ok(Err(_))), "{waited:?}");
    });
}
