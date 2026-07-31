// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use std::collections::BTreeMap;

fn recipe(id: &str, model: &str, runtime: &str) -> Recipe {
    Recipe {
        id: id.into(),
        version: "2".into(),
        model: model.into(),
        runtime: Some(runtime.into()),
        container: "c".into(),
        min_nodes: 1,
        description: "d".into(),
        maintainer: "m".into(),
        category: "agent".into(),
        model_params: "27B".into(),
        quantization: "nvfp4".into(),
        kv_dtype: "bf16".into(),
        defaults: BTreeMap::new(),
    }
}

fn catalogue() -> Vec<Recipe> {
    vec![
        recipe("q/27b", "nvidia/Qwen3.6-27B-NVFP4", "atlas"),
        recipe("q/35b", "Qwen/Qwen3.6-35B-A3B-FP8", "atlas"),
        recipe("d/vllm", "org/vllm-only", "vllm"),
    ]
}

const LIVE: &str = "nvidia/Qwen3.6-27B-NVFP4";

#[test]
fn a_different_known_model_swaps() {
    assert_eq!(
        decide("Qwen/Qwen3.6-35B-A3B-FP8", LIVE, &catalogue()),
        Decision::SwapTo("q/35b".into())
    );
}

#[test]
fn the_live_model_does_not_swap() {
    assert_eq!(decide(LIVE, LIVE, &catalogue()), Decision::ServeCurrent);
}

/// Clients send arbitrary strings here; the benchmark harness sends whatever
/// `--model` was typed. Erroring or swapping on those would break callers that
/// work today.
#[test]
fn an_unknown_model_is_ignored_not_an_error() {
    assert_eq!(
        decide("does/not-exist", LIVE, &catalogue()),
        Decision::ServeCurrent
    );
}

#[test]
fn an_absent_or_blank_model_is_ignored() {
    assert_eq!(decide("", LIVE, &catalogue()), Decision::ServeCurrent);
    assert_eq!(decide("   ", LIVE, &catalogue()), Decision::ServeCurrent);
}

/// A vLLM recipe names a real model but cannot be launched here, so it must not
/// be a swap target — that would tear down a working model for a load that
/// cannot succeed.
#[test]
fn a_vllm_recipe_is_never_a_swap_target() {
    assert_eq!(
        decide("org/vllm-only", LIVE, &catalogue()),
        Decision::ServeCurrent
    );
}

/// Matching is exact. A prefix or substring match would swap to something the
/// caller did not ask for.
#[test]
fn matching_is_exact_not_fuzzy() {
    assert_eq!(
        decide("Qwen/Qwen3.6-35B", LIVE, &catalogue()),
        Decision::ServeCurrent,
        "a prefix of a known id is not that id"
    );
    assert_eq!(
        decide("nvidia/Qwen3.6-27B-NVFP4-extra", LIVE, &catalogue()),
        Decision::ServeCurrent
    );
}

#[test]
fn an_empty_catalogue_never_swaps() {
    assert_eq!(decide("anything", LIVE, &[]), Decision::ServeCurrent);
}
