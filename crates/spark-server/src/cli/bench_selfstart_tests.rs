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
        overrides: Default::default(),
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

/// The ordinary case: KEY=VALUE reaches the recipe as an override.
///
/// The motivating one, specifically — every gate recipe pins `kv_cache_dtype:
/// bf16`, so a change to the fp8-KV attention kernel could not be exercised by
/// any gate at all without this.
#[test]
fn a_key_value_pair_becomes_a_recipe_override() {
    let parsed = parse_serve_overrides(&[
        "kv_cache_dtype=fp8".to_string(),
        "fp8_kv_calibration_tokens=512".to_string(),
    ])
    .unwrap();
    assert_eq!(
        parsed.get("kv_cache_dtype").map(String::as_str),
        Some("fp8")
    );
    assert_eq!(
        parsed.get("fp8_kv_calibration_tokens").map(String::as_str),
        Some("512")
    );
}

/// A value containing `=` keeps it — only the FIRST `=` separates.
///
/// Recipe values are rendered into a CLI, and flags whose value carries an `=`
/// exist. Splitting on every `=` would silently truncate one.
#[test]
fn only_the_first_equals_separates() {
    let parsed = parse_serve_overrides(&["extra_args=--foo=bar".to_string()]).unwrap();
    assert_eq!(
        parsed.get("extra_args").map(String::as_str),
        Some("--foo=bar")
    );
}

/// An empty value is a value, not an omission — some recipe keys render as a
/// bare flag, and `key=` is how you ask for that.
#[test]
fn an_empty_value_is_kept() {
    let parsed = parse_serve_overrides(&["disable_thinking=".to_string()]).unwrap();
    assert_eq!(parsed.get("disable_thinking").map(String::as_str), Some(""));
}

/// Missing `=` is refused, and the message shows the shape it wanted.
///
/// The alternative — treating a bare word as a flag — would silently accept
/// `kv_cache_dtype fp8` (two argv words) as a key with no value, and serve the
/// recipe unchanged while the operator believed otherwise.
#[test]
fn a_pair_without_an_equals_is_refused() {
    let e = parse_serve_overrides(&["kv_cache_dtype".to_string()]).unwrap_err();
    assert!(e.to_string().contains("KEY=VALUE"), "{e}");
}

#[test]
fn an_empty_key_is_refused() {
    assert!(parse_serve_overrides(&["=fp8".to_string()]).is_err());
}

/// ★ `port` is refused rather than accepted-and-dropped.
///
/// `serve_for` binds a free port and passes its own override, so an operator's
/// `port` would lose — but losing SILENTLY means the gate serves somewhere the
/// operator is not looking, and the failure surfaces as a confusing connection
/// error instead of a sentence explaining it.
#[test]
fn overriding_the_port_is_refused_with_a_reason() {
    let e = parse_serve_overrides(&["port=8888".to_string()]).unwrap_err();
    let msg = e.to_string();
    assert!(msg.contains("port"), "{msg}");
    assert!(
        msg.contains("free port"),
        "the refusal must say who owns the port: {msg}"
    );
}

/// A repeated key takes the LAST value, matching every other CLI on the box.
#[test]
fn a_repeated_key_takes_the_last_value() {
    let parsed = parse_serve_overrides(&[
        "kv_cache_dtype=bf16".to_string(),
        "kv_cache_dtype=fp8".to_string(),
    ])
    .unwrap();
    assert_eq!(
        parsed.get("kv_cache_dtype").map(String::as_str),
        Some("fp8")
    );
}

/// No overrides is the normal case and produces an empty map, which is what
/// keeps `serve_overrides` absent from an unmodified run's gate record.
#[test]
fn no_overrides_is_empty_not_an_error() {
    assert!(parse_serve_overrides(&[]).unwrap().is_empty());
}
