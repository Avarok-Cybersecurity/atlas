// SPDX-License-Identifier: AGPL-3.0-only

//! The one promise of `record_write.rs`: a failing record never disappears
//! without a human having said so.

use std::path::Path;

use super::super::record_path::{rerun_path, time_of_day};
use super::super::tests::{SHA, hw, run_record, tempdir};
use super::super::{GateRecord, read_record, records_newest_first, write_record};
use crate::result::Verdict;
use std::collections::BTreeMap;

const ID: &str = "bfcl-subset";
/// 2026-08-05 00:56:22 UTC — the fixture's `recorded_at`.
const T0: u64 = 1_785_891_382;

fn record(recorded_at: u64, verdict: Verdict) -> GateRecord {
    let mut run = run_record(BTreeMap::new(), verdict);
    run.recorded_at = recorded_at;
    GateRecord::from_run(&run, hw(), SHA.into(), Vec::new(), None).unwrap()
}

fn verdict_at(path: &Path) -> Option<String> {
    read_record(path).unwrap().verdict
}

fn files(root: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(root.join(".benchmarks").join(ID))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[test]
fn time_of_day_is_the_utc_clock_reading() {
    assert_eq!(time_of_day(0), "000000");
    assert_eq!(time_of_day(86_399), "235959");
    assert_eq!(time_of_day(86_400), "000000");
    assert_eq!(time_of_day(T0), "005622");
}

#[test]
fn the_rerun_name_carries_the_time_and_keeps_every_suffix_in_place() {
    let dir = Path::new("/r/.benchmarks/bfcl-subset");
    assert_eq!(
        rerun_path(&dir.join("2026-08-05-b72dad1893.json"), T0),
        dir.join("2026-08-05T005622Z-b72dad1893.json")
    );
    assert_eq!(
        rerun_path(
            &dir.join("2026-08-05-b72dad1893-unsloth-qwen3.8-27b-nvfp4-s1of4.json"),
            T0 + 3_600
        ),
        dir.join("2026-08-05T015622Z-b72dad1893-unsloth-qwen3.8-27b-nvfp4-s1of4.json")
    );
    // A lexical sort — what `bench_card` relies on — still puts the re-run
    // after the record it follows: `-` sorts before `T`.
    assert!("2026-08-05-b72dad1893.json" < "2026-08-05T005622Z-b72dad1893.json");
}

/// The 2026-08-29 sequence: 9/10, then 10/10 on a same-day re-run at the same
/// commit. The 9/10 used to vanish. Now both records exist, the gate still
/// reads the newest, and the failure is on disk for anyone counting.
#[test]
fn a_same_day_rerun_never_erases_a_failing_record() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    let failed =
        write_record(root, &record(T0, Verdict::fail("followed_directions 9/10"))).unwrap();
    assert_eq!(
        failed,
        root.join(".benchmarks/bfcl-subset/2026-08-05-b72dad1893.json")
    );
    let passed = write_record(root, &record(T0 + 3_600, Verdict::pass("10/10"))).unwrap();
    assert_eq!(
        passed,
        root.join(".benchmarks/bfcl-subset/2026-08-05T015622Z-b72dad1893.json"),
        "the re-run lands beside the failure, not on top of it"
    );
    assert_eq!(
        verdict_at(&failed).as_deref(),
        Some("FAIL"),
        "the failure survived"
    );
    assert_eq!(verdict_at(&passed).as_deref(), Some("PASS"));
    assert_eq!(
        records_newest_first(root, ID),
        vec![passed.clone(), failed.clone()],
        "gating still reads the newest record first"
    );
}

/// The guard is specific to failures. A passing record replaced by a
/// same-day re-run is the behaviour every consumer was built on — one file
/// per (day, commit) — and replacing a PASS hides nothing a reader needs.
#[test]
fn a_passing_record_is_still_replaced_by_a_same_day_rerun() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    let first = write_record(root, &record(T0, Verdict::pass("10/10"))).unwrap();
    let again = write_record(root, &record(T0 + 60, Verdict::pass("10/10"))).unwrap();
    assert_eq!(first, again);
    assert_eq!(files(root), vec!["2026-08-05-b72dad1893.json"]);
    assert_eq!(read_record(&again).unwrap().recorded_at, T0 + 60);
    // ...and in the strict direction too: a FAIL may replace a PASS.
    let failed = write_record(root, &record(T0 + 120, Verdict::fail("9/10"))).unwrap();
    assert_eq!(failed, first);
    assert_eq!(verdict_at(&first).as_deref(), Some("FAIL"));
}

#[test]
fn a_second_failure_is_kept_beside_the_first_and_a_rewrite_of_one_is_refused() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    write_record(root, &record(T0, Verdict::fail("9/10"))).unwrap();
    let second = write_record(root, &record(T0 + 60, Verdict::fail("8/10"))).unwrap();
    assert_eq!(
        files(root),
        vec![
            "2026-08-05-b72dad1893.json",
            "2026-08-05T005722Z-b72dad1893.json"
        ]
    );
    // Same second, same commit, both failing: that is one run being rewritten,
    // and the rule refuses rather than picks which failure to lose.
    let err = write_record(root, &record(T0 + 60, Verdict::pass("10/10"))).unwrap_err();
    assert!(
        format!("{err:#}").contains("both hold FAILING records"),
        "{err:#}"
    );
    assert_eq!(verdict_at(&second).as_deref(), Some("FAIL"));
}

#[test]
fn a_file_that_cannot_be_read_as_a_record_is_never_overwritten() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    let path = root.join(".benchmarks/bfcl-subset/2026-08-05-b72dad1893.json");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "{ not a record").unwrap();
    let err = write_record(root, &record(T0, Verdict::pass("10/10"))).unwrap_err();
    assert!(
        format!("{err:#}").contains("is not a readable gate record"),
        "{err:#}"
    );
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "{ not a record");
}

/// A sharded group: the shard suffix stays on the re-run name, so the second
/// record is still recognisably shard 0 of 2 — and `GateRecord::shard()`
/// reads the metrics, not the name, so both parse identically.
#[test]
fn a_failing_shard_is_preserved_under_its_shard_suffix() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    let shard = |at: u64, v: Verdict| {
        let mut m = BTreeMap::new();
        m.insert("shard.index".to_string(), 0.0);
        m.insert("shard.count".to_string(), 2.0);
        let mut run = run_record(m, v);
        run.recorded_at = at;
        GateRecord::from_run(&run, hw(), SHA.into(), Vec::new(), None).unwrap()
    };
    let failed = write_record(root, &shard(T0, Verdict::fail("low"))).unwrap();
    let passed = write_record(root, &shard(T0 + 5, Verdict::pass("ok"))).unwrap();
    assert!(
        failed.ends_with("2026-08-05-b72dad1893-s0of2.json"),
        "{failed:?}"
    );
    assert!(
        passed.ends_with("2026-08-05T005627Z-b72dad1893-s0of2.json"),
        "{passed:?}"
    );
    assert_eq!(read_record(&failed).unwrap().shard(), Some((0, 2)));
    assert_eq!(read_record(&passed).unwrap().shard(), Some((0, 2)));
}
