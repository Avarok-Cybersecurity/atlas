// SPDX-License-Identifier: AGPL-3.0-only

use super::responses_file;
use crate::benchmarks::bfcl::dataset::Shard;

fn shard(index: usize) -> Option<Shard> {
    Some(Shard { index, count: 4 })
}

/// THE PROPERTY SHARDING NEEDS, stated the way the bug actually presents.
///
/// ★ The first version of this test passed distinct benchmark IDS and proved
/// nothing: `Bfcl::descriptor()` returns the VARIANT's descriptor, so every
/// shard of `bfcl-subset` reports the id `bfcl-subset`. The five legs share an
/// id, and it is the SHARD that distinguishes them. Testing the function with
/// inputs the caller never produces is how a fix ships broken.
#[test]
fn every_leg_of_a_group_writes_its_own_responses_file() {
    let names: Vec<String> = std::iter::once(responses_file("bfcl-subset", None))
        .chain((0..4).map(|i| responses_file("bfcl-subset", shard(i))))
        .collect();
    let unique: std::collections::BTreeSet<&String> = names.iter().collect();
    assert_eq!(
        unique.len(),
        5,
        "all five legs share the id `bfcl-subset`; the shard must distinguish them: {names:?}"
    );
}

/// The whole draw and a shard of it are different measurements and must not
/// share a file — this is the pair that actually collided in production.
#[test]
fn the_group_and_its_shard_are_distinct_files() {
    assert_ne!(
        responses_file("bfcl-subset", None),
        responses_file("bfcl-subset", shard(0))
    );
}

/// And the name must SAY which leg wrote it, so a directory of them is readable
/// without opening each file.
#[test]
fn the_file_name_identifies_the_leg_that_wrote_it() {
    assert_eq!(
        responses_file("bfcl-subset", None),
        "responses-bfcl-subset.jsonl"
    );
    assert_eq!(
        responses_file("bfcl-subset", shard(2)),
        "responses-bfcl-subset-2of4.jsonl"
    );
}

/// A different draw with the same shape stays distinct too.
#[test]
fn two_groups_do_not_collide() {
    assert_ne!(
        responses_file("bfcl-subset", shard(0)),
        responses_file("bfcl-subset-echolp", shard(0))
    );
}
