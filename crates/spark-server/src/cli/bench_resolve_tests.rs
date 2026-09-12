// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the gate's pure resolution: box class, model variant, recipe
//! binding, serve-override parsing, and baseline-defined run parameters.
//! Moved verbatim from `bench_selfstart_tests.rs` alongside the code they
//! test; everything here runs without a GPU.

use std::collections::BTreeMap;

use super::*;
use atlas_plugin::gate::{Bound, GateBaseline, HardwareBaseline, ModelBaseline};

pub(super) fn baseline(entries: &[(&str, &str, Option<&str>)]) -> GateBaseline {
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
                label: String::new(),
                note: String::new(),
                metrics: BTreeMap::new(),
                serve_overrides: BTreeMap::new(),
                param_overrides: BTreeMap::new(),
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
    let r = resolve(&b, "bfcl-subset", None, None).expect("inferred");
    assert_eq!(r.model, "unsloth/Qwen3.6-27B-NVFP4");
    assert_eq!(r.recipe_id, "qwen3.6/x");
}

#[test]
fn several_box_classes_refuse_to_guess() {
    // Guessing here would serve one box's config and score it against the
    // other's thresholds — TTFT ceilings are box-local.
    let b = baseline(&[("gb10", "m", Some("r")), ("mi300x", "m", Some("r2"))]);
    let err = resolve(&b, "ttft-warm-gate", None, None).expect_err("refused");
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
    let r = resolve(&b, "ttft-warm-gate", Some("mi300x"), None).expect("picked");
    assert_eq!(r.recipe_id, "recipe-b");
}

/// Oracle: `bench_resolve::resolve`'s own contract — "every refusal names both
/// what was asked for and what exists". `h800` is a real NVIDIA part that Atlas
/// has never registered as a box class, so it exercises the branch a typo
/// reaches. (It used to be spelled `h100`; that id is a registered Hopper slot
/// now, so it tests the OTHER refusal below.)
#[test]
fn an_unknown_box_class_names_what_exists() {
    let b = baseline(&[("gb10", "m", Some("r"))]);
    let err = resolve(&b, "bfcl-subset", Some("h800"), None).expect_err("refused");
    let msg = format!("{err:#}");
    assert!(msg.contains("h800"), "{msg}");
    assert!(msg.contains("gb10"), "lists what it has: {msg}");
    assert!(
        msg.contains("not a box class Atlas knows"),
        "says WHY it is unresolvable, so a typo is distinguishable from an \
         unmeasured box: {msg}"
    );
}

#[test]
fn a_baseline_without_a_recipe_cannot_self_start() {
    // The honest failure: this gate has thresholds but nothing says how to
    // serve them, so it must refuse rather than invent a config.
    let b = baseline(&[("gb10", "m", None)]);
    let err = resolve(&b, "bfcl-subset", None, None).expect_err("refused");
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
    assert!(resolve(&b, "bfcl-subset", None, None).is_err());
}

// ── Model variants ──

/// Two variants on one box: the committed `default = true` subject serves when
/// none is named. Not a guess — `bench::baseline_for` refuses to assemble a
/// baseline with zero or two defaults, so this is a declaration being read.
#[test]
fn no_checkpoint_takes_the_declared_default_variant() {
    let b = two_variant_baseline();
    let r = resolve(&b, "agentic-webserver", None, None).expect("default");
    assert_eq!(r.model, "Qwen/Qwen3.6-35B-A3B-FP8");
    assert_eq!(r.recipe_id, "qwen3.6/qwen3.6-35b-a3b-fp8-bf16head");
}

/// `--checkpoint` selects the non-default variant, and the resolved entry
/// carries THAT variant's thresholds — the serve recipe and the bounds travel
/// together or the whole exercise is a model-id swap.
#[test]
fn an_explicit_checkpoint_picks_its_variant_with_its_thresholds() {
    let b = two_variant_baseline();
    let r = resolve(
        &b,
        "agentic-webserver",
        None,
        Some("unsloth/Qwen3.8-27B-NVFP4"),
    )
    .expect("picked");
    assert_eq!(r.model, "unsloth/Qwen3.8-27B-NVFP4");
    assert_eq!(r.recipe_id, "qwen3.8/qwen3.8-27b-nvfp4-unsloth");
    assert_eq!(
        r.entry.metrics.get("sum_wall_s").and_then(|b| b.max),
        Some(2500.0),
        "the dense variant's own ceiling, not the default's"
    );
}

/// An unknown checkpoint is an error naming what exists — never a fallback to
/// the default, which would serve a different model than the one asked for.
#[test]
fn an_unknown_checkpoint_names_what_exists() {
    let b = two_variant_baseline();
    let err = resolve(&b, "agentic-webserver", None, Some("nvidia/NoSuch")).expect_err("refused");
    let msg = format!("{err:#}");
    assert!(msg.contains("nvidia/NoSuch"), "{msg}");
    assert!(msg.contains("Qwen3.6-35B"), "lists the variants: {msg}");
    assert!(msg.contains("Qwen3.8-27B"), "lists the variants: {msg}");
}

fn two_variant_baseline() -> GateBaseline {
    let mut models = BTreeMap::new();
    models.insert(
        "Qwen/Qwen3.6-35B-A3B-FP8".to_string(),
        ModelBaseline {
            recipe: Some("qwen3.6/qwen3.6-35b-a3b-fp8-bf16head".to_string()),
            label: "35B MoE flagship".to_string(),
            note: String::new(),
            metrics: BTreeMap::from([("sum_wall_s".to_string(), max_bound(1000.0))]),
            serve_overrides: BTreeMap::new(),
            param_overrides: BTreeMap::new(),
        },
    );
    models.insert(
        "unsloth/Qwen3.8-27B-NVFP4".to_string(),
        ModelBaseline {
            recipe: Some("qwen3.8/qwen3.8-27b-nvfp4-unsloth".to_string()),
            label: "dense 27B".to_string(),
            note: String::new(),
            metrics: BTreeMap::from([("sum_wall_s".to_string(), max_bound(2500.0))]),
            serve_overrides: BTreeMap::new(),
            param_overrides: BTreeMap::new(),
        },
    );
    GateBaseline {
        schema: 2,
        hardware: BTreeMap::from([(
            "gb10".to_string(),
            HardwareBaseline {
                default: "Qwen/Qwen3.6-35B-A3B-FP8".to_string(),
                models,
            },
        )]),
    }
}

fn max_bound(max: f64) -> Bound {
    Bound {
        min: None,
        max: Some(max),
        noise: None,
    }
}

// ── Baseline-defined run parameters ──

/// The selected variant's committed ceiling replaces the schema default; an
/// explicit `--param` outranks it; and a variant with no bound leaves the
/// schema default standing. All three precedence arms in one place.
#[test]
fn threshold_params_derive_from_the_variant_and_yield_to_an_explicit_param() {
    let descriptor = atlas_plugin::registry::find("agentic-webserver").expect("registered");
    let specs = descriptor.build().parameters();
    let entry = two_variant_baseline().hardware["gb10"].models["unsloth/Qwen3.8-27B-NVFP4"].clone();

    // Derived: no explicit --param, so the dense ceiling lands.
    let mut values = atlas_plugin::ParamValues::from_overrides(&specs, vec![]).unwrap();
    let applied =
        apply_threshold_params(descriptor, &specs, &mut values, &entry, &[]).expect("applies");
    assert_eq!(applied, vec![("wall_budget_s".to_string(), 2500.0)]);
    assert_eq!(values.float("wall_budget_s").unwrap(), 2500.0);

    // Explicit --param wins untouched.
    let explicit = vec![("wall_budget_s".to_string(), "1234".to_string())];
    let mut values =
        atlas_plugin::ParamValues::from_overrides(&specs, vec![("wall_budget_s", "1234")]).unwrap();
    let applied = apply_threshold_params(descriptor, &specs, &mut values, &entry, &explicit)
        .expect("applies");
    assert!(applied.is_empty(), "stated intent is never overridden");
    assert_eq!(values.float("wall_budget_s").unwrap(), 1234.0);

    // No bound on the paired metric: schema default stands.
    let mut bare = entry.clone();
    bare.metrics.clear();
    let mut values = atlas_plugin::ParamValues::from_overrides(&specs, vec![]).unwrap();
    let applied =
        apply_threshold_params(descriptor, &specs, &mut values, &bare, &[]).expect("applies");
    assert!(applied.is_empty());
    assert_eq!(values.float("wall_budget_s").unwrap(), 1000.0);
}

/// The `min` arm, through the REAL bfcl-subset wiring: a floor metric (BFCL's
/// accuracies carry `min` bounds, no `max`) substitutes the variant's floor
/// into the paired verdict param — this is what lets a non-MLPerf checkpoint
/// that clears its own BENCH.toml bars get the PASS run verdict the gate
/// machinery requires (review C1).
#[test]
fn threshold_params_substitute_a_min_bound_when_the_metric_is_a_floor() {
    let descriptor = atlas_plugin::registry::find("bfcl-subset").expect("registered");
    let specs = descriptor.build().parameters();
    let mut entry =
        two_variant_baseline().hardware["gb10"].models["unsloth/Qwen3.8-27B-NVFP4"].clone();
    entry.metrics = BTreeMap::from([
        ("overall_accuracy".to_string(), min_bound(83.82)),
        ("normalized_single_turn_score".to_string(), min_bound(83.72)),
    ]);
    let mut values = atlas_plugin::ParamValues::from_overrides(&specs, vec![]).unwrap();
    let applied =
        apply_threshold_params(descriptor, &specs, &mut values, &entry, &[]).expect("applies");
    assert_eq!(
        applied,
        vec![
            ("min_overall".to_string(), 83.82),
            ("min_normalized".to_string(), 83.72),
        ]
    );
    assert_eq!(values.float("min_overall").unwrap(), 83.82);
    assert_eq!(values.float("min_normalized").unwrap(), 83.72);
}

/// The driver self-verdicts on RAW values; the gate judge allows the bound's
/// noise band. The bar handed to the driver must therefore be the
/// noise-adjusted one (min - noise, max + noise), or a run inside the band
/// passes `gate::scoring::compare` and still records a FAIL verdict — the
/// exact split the 2026-08-16 echolp record hit (86.35 vs min 86.50,
/// noise 0.4).
#[test]
fn threshold_params_hand_the_driver_the_noise_adjusted_bar() {
    let descriptor = atlas_plugin::registry::find("bfcl-subset").expect("registered");
    let specs = descriptor.build().parameters();
    let mut entry =
        two_variant_baseline().hardware["gb10"].models["unsloth/Qwen3.8-27B-NVFP4"].clone();
    entry.metrics = BTreeMap::from([
        (
            "overall_accuracy".to_string(),
            Bound {
                min: Some(86.50),
                max: None,
                noise: Some(0.4),
            },
        ),
        (
            "normalized_single_turn_score".to_string(),
            Bound {
                min: Some(86.90),
                max: None,
                noise: Some(0.4),
            },
        ),
    ]);
    let mut values = atlas_plugin::ParamValues::from_overrides(&specs, vec![]).unwrap();
    let applied =
        apply_threshold_params(descriptor, &specs, &mut values, &entry, &[]).expect("applies");
    assert_eq!(
        applied,
        vec![
            ("min_overall".to_string(), 86.10),
            ("min_normalized".to_string(), 86.50),
        ]
    );
    let sample = 86.35;
    assert!(
        sample >= values.float("min_overall").unwrap(),
        "a run the gate judge passes must clear the driver's bar too"
    );
}

/// A paired metric declaring BOTH bounds is ambiguous — which one the driver
/// should self-verdict against cannot be inferred — so the substitution errors
/// loudly instead of guessing a direction.
#[test]
fn a_paired_metric_with_both_bounds_is_a_loud_error_not_a_guess() {
    let descriptor = atlas_plugin::registry::find("agentic-webserver").expect("registered");
    let specs = descriptor.build().parameters();
    let mut entry =
        two_variant_baseline().hardware["gb10"].models["unsloth/Qwen3.8-27B-NVFP4"].clone();
    entry.metrics = BTreeMap::from([(
        "sum_wall_s".to_string(),
        Bound {
            min: Some(900.0),
            max: Some(2500.0),
            noise: None,
        },
    )]);
    let mut values = atlas_plugin::ParamValues::from_overrides(&specs, vec![]).unwrap();
    let err = apply_threshold_params(descriptor, &specs, &mut values, &entry, &[])
        .expect_err("ambiguous bounds must not be resolved silently");
    let msg = format!("{err:#}");
    assert!(msg.contains("BOTH min"), "{msg}");
    assert!(msg.contains("sum_wall_s"), "{msg}");
    // An explicit --param still wins over the ambiguity — stated intent needs
    // no bound at all.
    let explicit = vec![("wall_budget_s".to_string(), "1234".to_string())];
    let mut values =
        atlas_plugin::ParamValues::from_overrides(&specs, vec![("wall_budget_s", "1234")]).unwrap();
    let applied = apply_threshold_params(descriptor, &specs, &mut values, &entry, &explicit)
        .expect("explicit param sidesteps the ambiguous bound");
    assert!(applied.is_empty());
}

fn min_bound(min: f64) -> Bound {
    Bound {
        min: Some(min),
        max: None,
        noise: None,
    }
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

#[test]
fn resolve_carries_baseline_serve_overrides() {
    // The pin travels inside the resolved entry, verbatim — `serve_for` reads
    // it from there, so a dropped clone here would silently serve the recipe's
    // default pool while the record claimed the pinned one.
    let mut b = baseline(&[("gb10", "m", Some("r"))]);
    b.hardware
        .get_mut("gb10")
        .unwrap()
        .models
        .get_mut("m")
        .unwrap()
        .serve_overrides
        .insert("ssm_cache_slots".to_string(), "256".to_string());
    let r = resolve(&b, "bfcl-subset", None, None).expect("resolved");
    assert_eq!(
        r.entry
            .serve_overrides
            .get("ssm_cache_slots")
            .map(String::as_str),
        Some("256")
    );
}
