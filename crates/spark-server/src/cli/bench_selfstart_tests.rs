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
        baseline_entry: Default::default(),
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

// ── Baseline-declared serve pins ──

/// Baseline pins land on the resolved serve, and a CLI clash takes the
/// operator: `[benchmarks.serve_overrides]` states what the gate needs, but an
/// operator typing `--serve-override` is stating intent for THIS run. Both end
/// up disclosed in the record either way, so precedence never hides anything.
#[test]
fn baseline_pins_are_applied_and_the_operator_wins_a_clash() {
    let baseline = BTreeMap::from([
        ("ssm_cache_slots".to_string(), "256".to_string()),
        ("kv_cache_dtype".to_string(), "bf16".to_string()),
    ]);
    let requested = BTreeMap::from([("kv_cache_dtype".to_string(), "fp8".to_string())]);
    let merged = atlas_plugin::gate::merge_serve_overrides(baseline, requested);
    assert_eq!(
        merged.get("ssm_cache_slots").map(String::as_str),
        Some("256"),
        "an unclashed pin survives the merge"
    );
    assert_eq!(
        merged.get("kv_cache_dtype").map(String::as_str),
        Some("fp8"),
        "the operator's value wins the clash"
    );
}

/// ★ The 2026-08-15 repro, end to end over the REAL tree: a serve pin the
/// BASELINE declares — with NO `--serve-override` typed — must reach the
/// rendered serve args. The concurrency gate's pin was committed ABOVE its own
/// `[[benchmarks]]` header, so TOML attached it to the preceding decode-floor
/// entry; the gate then served the recipe's `max_batch: 1` (C=1..16 all
/// ~19 tok/s, serial by construction) and printed no OVERRIDES disclosure,
/// because `serve_for`'s warn only fires on a non-empty merged set. Nothing in
/// the code path was wrong — which is exactly why this walks `serve_for`'s
/// pure prefix (read_baseline → resolve → merge with an EMPTY CLI map →
/// `Recipe::serve_args`) against the committed BENCH.toml, where the bug
/// lived.
#[test]
fn a_baseline_declared_serve_pin_reaches_the_rendered_serve_args_without_cli_flags() {
    let root = crate::cli::bench_run::repo_root().expect("inside the repo");
    let baseline = gate::read_baseline(&root, "concurrency-sweep").expect("baseline assembles");
    let resolved = crate::cli::bench_resolve::resolve(&baseline, "concurrency-sweep", None, None)
        .expect("the default variant resolves");
    assert_eq!(resolved.recipe_id, "qwen3.8/qwen3.8-27b-nvfp4-unsloth");

    // No CLI overrides — the whole point of the repro.
    let merged =
        gate::merge_serve_overrides(resolved.entry.serve_overrides.clone(), BTreeMap::new());
    assert!(
        !merged.is_empty(),
        "the baseline-declared serve pin was lost before the merge: with no CLI flags the \
         gate would serve the recipe verbatim and print no OVERRIDES line"
    );

    // Render through a committed recipe fixture that pins the same `defaults:`
    // keys the real qwen3.8 agentic recipe does (the real yaml lives in the
    // atlas-recipes repo, not in this tree; the fixture stands in for the
    // RENDERING only — which entry carries the pin is asserted on the real
    // BENCH.toml above and in gate::bench_tests).
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recipes/qwen3.6/qwen3.6-27b-nvfp4.yaml");
    let text = std::fs::read_to_string(&fixture).expect("fixture recipe");
    let recipe = crate::recipe::Recipe::parse("qwen3.6/qwen3.6-27b-nvfp4", &text).expect("parses");
    let args = recipe
        .serve_args(&merged)
        .expect("pins render to valid serve args");
    // 128 since the ladder widened to C=128: the cap must cover the widest
    // measured rung or that rung measures the cap rather than the engine.
    // Read from the committed pin above rather than re-typed, so a future
    // change to the instrument moves this assertion with it instead of
    // failing it.
    assert_eq!(
        args.max_batch_size.to_string(),
        merged["max_batch_size"],
        "the batching pin reached the serve"
    );
    assert_eq!(args.kv_cache_dtype.as_deref(), Some("fp8"));
    // Marconi pinned at 8 slots (2026-08-16): 1/4 the 32-slot reserve at
    // 151.5 MiB/slot — see the BENCH.toml serve_overrides comment.
    assert_eq!(args.ssm_cache_slots, 8);
    assert_eq!(args.max_seq_len, 4096);
}
