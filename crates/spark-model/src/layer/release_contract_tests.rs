// SPDX-License-Identifier: AGPL-3.0-only

//! The per-sequence release contract — PR-L L1/L2 invariants, enforced at source level.
//!
//! These are cheap and they are the only thing standing between a future refactor and a
//! silent reintroduction of the two defects this lane closed: a second release owner, and an
//! environment variable that turns the release off.

const SEQUENCE_RS: &str = include_str!("../model/trait_impl/sequence.rs");
const TRANSFORMER_LAYER_RS: &str = include_str!("transformer_layer.rs");

/// L1: exactly ONE release owner on `TransformerLayer`.
///
/// The campaign carried `free_state(gpu, state)`; upstream #821 defines
/// `release_state(state, gpu)`. Two hooks meant two call sites, two sets of impls, and no
/// single place a reviewer could check. `DraftProposer::free_state` is a DIFFERENT trait on a
/// different type and is deliberately untouched — this pin is scoped to `transformer_layer.rs`.
#[test]
fn transformer_layer_has_exactly_one_release_hook() {
    assert_eq!(
        TRANSFORMER_LAYER_RS.matches("fn release_state").count(),
        1,
        "TransformerLayer must declare release_state exactly once"
    );
    assert_eq!(
        TRANSFORMER_LAYER_RS.matches("fn free_state").count(),
        0,
        "TransformerLayer::free_state was retired in favour of #821's release_state; \
         reintroducing it recreates the two-owner split"
    );
}

/// L2: no environment variable may gate the release chokepoint.
///
/// `ATLAS_GLM_DSA_STATE_LEAK` wrapped the whole loop and restored the leak when present. A
/// model-named env switch that disables an engine invariant is the anti-pattern upstream has
/// been deleting; the paired A/B it existed for is recorded in the commit history, and a
/// future control arm is built from the parent commit, not from a runtime flag.
#[test]
fn the_release_chokepoint_has_no_env_escape_hatch() {
    let dispatch = SEQUENCE_RS
        .split_once("fn free_sequence_dispatch")
        .expect("free_sequence_dispatch exists")
        .1;
    assert!(
        !dispatch.contains("env::var"),
        "free_sequence_dispatch must not read an environment variable: a release that can be \
         switched off is not an invariant"
    );
    assert_eq!(
        SEQUENCE_RS.matches("ATLAS_GLM_DSA_STATE_LEAK").count(),
        0,
        "the DSA leak switch must stay deleted"
    );
}

/// L2: the chokepoint calls the one hook, unconditionally, with no call-site type filter.
///
/// The campaign's loop skipped `LinearAttention && uses_ssm_pool()` at the call site — a
/// second spelling of "is this pooled?" that can drift out of agreement with the type refusal
/// inside the impls. #821 loops over every layer and lets each impl refuse by type.
#[test]
fn the_chokepoint_releases_every_layer_without_a_site_filter() {
    let dispatch = SEQUENCE_RS
        .split_once("fn free_sequence_dispatch")
        .expect("free_sequence_dispatch exists")
        .1;
    assert!(
        dispatch.contains("release_state("),
        "the chokepoint must call release_state"
    );
    assert_eq!(
        dispatch.matches(".free_state(").count(),
        1,
        "exactly one .free_state( call may remain in the chokepoint: the proposer's, which is \
         DraftProposer::free_state and a different trait"
    );
    assert!(
        !dispatch.contains("uses_ssm_pool()"),
        "no call-site pooled-ness filter: impls refuse by type"
    );
}
