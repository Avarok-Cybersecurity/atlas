// SPDX-License-Identifier: AGPL-3.0-only

use super::responses_file;

/// THE PROPERTY SHARDING NEEDS. Five legs of one group must not write one file.
#[test]
fn every_leg_of_a_group_writes_its_own_responses_file() {
    let names: Vec<String> = [
        "bfcl-subset",
        "bfcl-subset-a",
        "bfcl-subset-b",
        "bfcl-subset-c",
        "bfcl-subset-d",
    ]
    .iter()
    .map(|id| responses_file(id))
    .collect();
    let unique: std::collections::BTreeSet<&String> = names.iter().collect();
    assert_eq!(
        unique.len(),
        names.len(),
        "two legs would overwrite each other: {names:?}"
    );
}

/// And the name must SAY which leg it is, so a directory of them is readable
/// without opening each file. A hash would be unique and useless here.
#[test]
fn the_file_name_identifies_the_leg_that_wrote_it() {
    assert_eq!(
        responses_file("bfcl-subset-c"),
        "responses-bfcl-subset-c.jsonl"
    );
    assert!(responses_file("bfcl-echolp-a").contains("bfcl-echolp-a"));
}

/// A prefix must not be mistaken for a whole id: `bfcl-subset` and
/// `bfcl-subset-a` differ, and neither name may be a prefix-collision of the
/// other in a way that a glob would confuse.
#[test]
fn the_group_and_its_shard_are_distinct_files() {
    assert_ne!(
        responses_file("bfcl-subset"),
        responses_file("bfcl-subset-a")
    );
}
