// SPDX-License-Identifier: AGPL-3.0-only

//! The restore gate's completeness predicate. Its failure mode is SILENT — a
//! partial aux set that passes restores some layers at the snapshot position
//! and leaves the rest at zero, behind a KV image that assumes all of them
//! populated — so every way a set can be wrong is pinned here.

use super::aux_set_covers;

/// GLM-5.3's shape: 3 KDA then 1 DSA, repeating, 45 layers. Only the DSA
/// layers carry aux; KDA rides the SSM pool slot.
fn glm_carries() -> Vec<bool> {
    (0..45).map(|i| i % 4 == 3).collect()
}

fn glm_dsa_layers() -> Vec<u32> {
    (0..45u32).filter(|i| i % 4 == 3).collect()
}

#[test]
fn the_full_dsa_set_is_complete_in_any_order() {
    let c = glm_carries();
    assert_eq!(glm_dsa_layers().len(), 11);
    assert!(aux_set_covers(&c, &glm_dsa_layers()));
    let mut rev = glm_dsa_layers();
    rev.reverse();
    assert!(aux_set_covers(&c, &rev), "save order is not a contract");
}

#[test]
fn a_model_with_no_carriers_is_complete_only_when_empty() {
    let none = vec![false; 8];
    assert!(aux_set_covers(&none, &[]));
    assert!(
        !aux_set_covers(&none, &[0]),
        "a blob for a non-carrier is a corrupted set"
    );
}

/// The case the gate exists for: one DSA layer's blob missing. Restoring the
/// other ten would leave layer 7 at zero rows and — on a model without
/// `decode_k`'s lockstep bail — select over an empty context there.
#[test]
fn one_missing_carrier_is_incomplete() {
    let c = glm_carries();
    let partial: Vec<u32> = glm_dsa_layers().into_iter().filter(|&i| i != 7).collect();
    assert_eq!(partial.len(), 10);
    assert!(!aux_set_covers(&c, &partial));
    assert!(
        !aux_set_covers(&c, &[]),
        "an empty set on an aux-carrying model declines"
    );
}

#[test]
fn a_blob_on_a_kda_layer_is_a_corrupted_set() {
    let c = glm_carries();
    let mut with_kda = glm_dsa_layers();
    with_kda.push(4); // layer 4 is KDA (4 % 4 == 0)
    assert!(!aux_set_covers(&c, &with_kda));
}

#[test]
fn a_duplicate_index_is_refused() {
    let c = glm_carries();
    let mut dup = glm_dsa_layers();
    dup.push(3);
    assert!(!aux_set_covers(&c, &dup));
}

#[test]
fn an_out_of_range_index_is_refused_not_ignored() {
    let c = glm_carries();
    let mut oob = glm_dsa_layers();
    oob.push(45);
    assert!(!aux_set_covers(&c, &oob));
    oob.pop();
    oob.push(u32::MAX);
    assert!(!aux_set_covers(&c, &oob));
}
