// SPDX-License-Identifier: AGPL-3.0-only

//! The resident-expert plan.
//!
//! Two readers consult this — the weight loaders and the router mask — and a
//! disagreement between them is a null-pointer dereference in a kernel, not
//! a wrong answer. So the cases here are about the plan meaning exactly one
//! thing.

use super::BelPlan;

fn plan() -> BelPlan {
    // 4 layers, 8 experts; layers 1 and 3 restricted, 0 and 2 untouched
    // (a hybrid model's dense layers).
    BelPlan::new(
        "code-python",
        0.9,
        4,
        8,
        vec![(1usize, vec![0u16, 3, 7]), (3usize, vec![2u16])],
    )
    .expect("valid plan")
}

// ---------------------------------------------------------------- Path A

#[test]
fn a_restricted_layer_admits_only_its_listed_experts() {
    let p = plan();
    assert!(p.is_loaded(1, 0));
    assert!(p.is_loaded(1, 3));
    assert!(p.is_loaded(1, 7));
    assert!(!p.is_loaded(1, 1));
    assert!(!p.is_loaded(1, 6));
    assert_eq!(p.layer_count(1), Some(3));
}

#[test]
fn an_unlisted_layer_is_unrestricted() {
    // A dense layer of a hybrid model has no experts to restrict; treating
    // "not in the table" as "load nothing" would strand it.
    let p = plan();
    assert!(!p.restricts_layer(0));
    assert!(p.is_loaded(0, 5), "unlisted layers load in full");
    assert_eq!(p.layer_count(0), None);
    assert!(p.router_mask(0).is_none(), "nothing to mask");
}

#[test]
fn totals_count_only_restricted_layers() {
    // 3 of 8 in layer 1, 1 of 8 in layer 3.
    assert_eq!(plan().totals(), (4, 16));
    assert_eq!(plan().restricted_layers(), vec![1, 3]);
}

// ---------------------------------------------------------------- Path B

#[test]
fn the_router_mask_is_minus_infinity_exactly_where_weights_are_absent() {
    // This is the pairing that keeps the serve alive: every expert the mask
    // leaves selectable MUST be one the loader kept.
    let p = plan();
    let mask = p.router_mask(1).expect("layer 1 is restricted");
    assert_eq!(mask.len(), 8);
    for (e, m) in mask.iter().enumerate() {
        if p.is_loaded(1, e) {
            assert_eq!(*m, 0.0, "expert {e} is loaded, must stay selectable");
        } else {
            assert_eq!(
                *m,
                f32::NEG_INFINITY,
                "expert {e} was never loaded, must be unselectable"
            );
        }
    }
}

#[test]
fn a_category_listing_every_expert_masks_nothing() {
    // The negative control a BEL run is judged against: output must be
    // byte-identical to a no-flag run, which requires an all-zero additive
    // mask.
    let all = BelPlan::new("everything", 1.0, 2, 4, vec![(0usize, vec![0u16, 1, 2, 3])]).unwrap();
    assert_eq!(all.router_mask(0).unwrap(), vec![0.0, 0.0, 0.0, 0.0]);
    assert_eq!(all.totals(), (4, 4));
}

#[test]
fn an_expert_id_beyond_the_model_is_refused() {
    // The table was measured on a different checkpoint. Silently ignoring
    // the id would produce a plan that masks the wrong experts.
    let err = BelPlan::new("c", 0.9, 2, 4, vec![(0usize, vec![9u16])]).unwrap_err();
    assert!(err.contains("names expert 9"), "got: {err}");
    assert!(err.contains("different checkpoint"), "got: {err}");
}

#[test]
fn a_layer_beyond_the_model_is_refused() {
    let err = BelPlan::new("c", 0.9, 2, 4, vec![(5usize, vec![1u16])]).unwrap_err();
    assert!(err.contains("names layer 5"), "got: {err}");
}

// ---------------------------------------------------------------- Path C

#[test]
fn an_out_of_range_query_reads_as_not_loaded_not_as_loaded() {
    // A caller asking about an expert id past the end must not get `true`:
    // the loader would keep a tensor the mask does not protect, or worse,
    // the reverse.
    let p = plan();
    assert!(!p.is_loaded(1, 99), "past the expert count is not loaded");
    // A layer past the end is unrestricted, which is the safe direction:
    // the loader keeps everything and the mask is absent, so nothing can be
    // selected without weights behind it.
    assert!(p.is_loaded(9, 0));
    assert!(!p.restricts_layer(9));
}

#[test]
fn an_empty_layer_list_admits_nothing_in_that_layer() {
    // The MODEL.toml parser rejects an empty array, so this can only arise
    // in-process; pin the direction anyway, because "empty means everything"
    // would silently disable BEL for that layer.
    let p = BelPlan::new("c", 0.9, 1, 4, vec![(0usize, Vec::new())]).unwrap();
    assert!(p.restricts_layer(0));
    assert_eq!(p.layer_count(0), Some(0));
    assert!(!p.is_loaded(0, 0));
}
