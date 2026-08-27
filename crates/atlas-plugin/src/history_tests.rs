// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for run-history persistence.

use super::*;
use crate::registry;
use crate::result::{RunStatus, Verdict};

fn store() -> (ArtifactStore, tempdir::Dir) {
    let dir = tempdir::Dir::new();
    (ArtifactStore::with_root(dir.path()), dir)
}

/// Minimal scratch directory — the crate has no dev-dep on `tempfile`, and one
/// removed on drop is four lines.
mod tempdir {
    use std::path::{Path, PathBuf};
    pub struct Dir(PathBuf);
    impl Dir {
        pub fn new() -> Self {
            let n = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default();
            let p = std::env::temp_dir().join(format!(
                "atlas-history-{n}-{:?}",
                std::thread::current().id()
            ));
            std::fs::create_dir_all(&p).expect("scratch dir");
            Self(p)
        }
        pub fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn frame(phase: &str) -> BenchmarkResult {
    BenchmarkResult {
        status: RunStatus::Completed,
        phase: phase.into(),
        progress: None,
        summary: Vec::new(),
        table: None,
        verdict: Some(Verdict::pass("all good")),
        metrics: std::collections::BTreeMap::new(),
        dataset_fingerprint: None,
        log: Vec::new(),
        elapsed: std::time::Duration::from_secs(1),
        hardware_state: None,
    }
}

fn descriptor() -> &'static BenchmarkDescriptor {
    registry::find("concurrency-sweep").expect("registered")
}

fn record(f: BenchmarkResult) -> RunRecord {
    let d = descriptor();
    let specs = d.build().parameters();
    let values = ParamValues::defaults(&specs);
    let target = TargetEndpoint::new("http://127.0.0.1:8888", "m");
    RunRecord::new(d, &values, &target, RunSource::Cli, "1.0.0-beta-preview", f)
}

#[test]
fn a_saved_record_reads_back_through_the_public_reader() {
    let (store, _d) = store();
    let mut r = record(frame("done"));
    let path = save(&store, &mut r).expect("saves");

    assert!(path.ends_with(format!("{}.json", r.run_id)));
    assert_eq!(
        r.run_id.len(),
        4 + 19,
        "run- plus 19 fixed digits, so lexical order is time order: {}",
        r.run_id
    );

    let back = load(&store, "concurrency-sweep");
    assert_eq!(back.len(), 1);
    let b = &back[0];
    assert_eq!(b.run_id, r.run_id, "the stem addresses the record");
    assert_eq!(b.schema, SCHEMA);
    assert_eq!(b.source, RunSource::Cli);
    assert_eq!(b.benchmark_name, "Concurrency Sweep");
    assert_eq!(b.atlas_version, "1.0.0-beta-preview");
    assert_eq!(
        b.target(),
        TargetEndpoint::new("http://127.0.0.1:8888", "m")
    );
    assert!(!b.is_legacy());
}

#[test]
fn two_runs_in_the_same_instant_do_not_overwrite_each_other() {
    // The regression test for the bug this module replaces: the old writer
    // named files by whole seconds, so a second run inside the same second
    // silently destroyed the first.
    let (store, _d) = store();
    let mut a = record(frame("first"));
    let mut b = record(frame("second"));
    let instant = 1_777_777_777_777_777_777;
    let pa = save_at(&store, &mut a, instant, |_| {}).expect("saves");
    let pb = save_at(&store, &mut b, instant, |_| {}).expect("saves");

    assert_ne!(pa, pb, "distinct paths");
    assert_ne!(a.run_id, b.run_id, "distinct ids");
    assert_eq!(load(&store, "concurrency-sweep").len(), 2, "both survive");
}

#[test]
fn params_are_recorded_whole_and_rehydrate() {
    let (store, _d) = store();
    let d = descriptor();
    let specs = d.build().parameters();
    let values = ParamValues::from_overrides(&specs, [("osl", "8")]).expect("parses");
    let target = TargetEndpoint::new("http://127.0.0.1:9", "m");
    let mut r = RunRecord::new(d, &values, &target, RunSource::Cli, "v", frame("done"));
    save(&store, &mut r).expect("saves");

    let back = &load(&store, "concurrency-sweep")[0];
    assert_eq!(
        back.params.len(),
        specs.len(),
        "every parameter, not just the override: {:?}",
        back.params
    );
    assert_eq!(back.params["osl"], "8");
    assert_eq!(
        back.values(&specs).expect("rehydrates"),
        values,
        "a stored run is re-runnable"
    );
}

#[test]
fn a_legacy_bare_frame_still_appears_in_history() {
    // Files written before this module exist as a bare BenchmarkResult. They
    // must keep showing up, or upgrading Atlas would appear to delete history.
    let (store, _d) = store();
    let dir = store.runs_dir("concurrency-sweep").expect("dir");
    let legacy = serde_json::to_string(&frame("legacy")).expect("serializes");
    std::fs::write(dir.join("run-1730000000.json"), legacy).expect("writes");

    let back = load(&store, "concurrency-sweep");
    assert_eq!(back.len(), 1);
    let b = &back[0];
    assert!(b.is_legacy());
    assert_eq!(b.recorded_at, 1_730_000_000, "timestamp from the stem");
    assert_eq!(b.run_id, "run-1730000000");
    assert!(b.params.is_empty());
    assert_eq!(b.source, RunSource::Unknown);
    assert_eq!(b.frame.phase, "legacy");
}

#[test]
fn newest_sorts_first_and_a_legacy_file_sorts_by_its_own_time() {
    let (store, _d) = store();
    let dir = store.runs_dir("concurrency-sweep").expect("dir");
    let legacy = serde_json::to_string(&frame("old")).expect("serializes");
    std::fs::write(dir.join("run-1730000000.json"), legacy).expect("writes");
    let mut fresh = record(frame("new"));
    save(&store, &mut fresh).expect("saves");

    let back = load(&store, "concurrency-sweep");
    assert_eq!(back.len(), 2);
    assert_eq!(back[0].frame.phase, "new", "newest first");
    assert_eq!(back[1].frame.phase, "old");
}

#[test]
fn a_corrupt_file_is_skipped_rather_than_hiding_the_rest() {
    let (store, _d) = store();
    let dir = store.runs_dir("concurrency-sweep").expect("dir");
    std::fs::write(dir.join("run-1730000001.json"), "{ not json").expect("writes");
    let mut ok = record(frame("good"));
    save(&store, &mut ok).expect("saves");

    let back = load(&store, "concurrency-sweep");
    assert_eq!(back.len(), 1, "the readable one still loads");
    assert_eq!(back[0].frame.phase, "good");
}

#[test]
fn baseline_json_is_not_mistaken_for_a_run() {
    // ttft writes baseline.json into the same directory.
    let (store, _d) = store();
    let dir = store.runs_dir("concurrency-sweep").expect("dir");
    let baseline_shaped_like_a_run = serde_json::to_string(&record(frame("not a run")))
        .expect("serializes a valid record shape");
    std::fs::write(dir.join("baseline.json"), baseline_shaped_like_a_run).expect("writes");
    assert!(load(&store, "concurrency-sweep").is_empty());
}

#[test]
fn find_addresses_a_run_by_its_id() {
    let (store, _d) = store();
    let mut r = record(frame("done"));
    save(&store, &mut r).expect("saves");
    assert_eq!(find(&store, &r.run_id).expect("found").frame.phase, "done");
    let mut other = record(frame("other benchmark"));
    other.benchmark_id = "quick-speed-bench".into();
    other.benchmark_name = "Quick Speed".into();
    save(&store, &mut other).expect("saves another benchmark");
    assert_eq!(
        find(&store, &other.run_id)
            .expect("found across registry")
            .frame
            .phase,
        "other benchmark"
    );
    assert!(find(&store, "run-0000000000000000000").is_none());
}

#[test]
fn concurrent_writers_never_lose_a_run() {
    // The dashboard and `spark benchmark` are separate processes writing the
    // same tree. `exists()` then `write` lets both conclude the same name is
    // free, and one silently overwrites the other's run.
    let (store, _d) = store();
    const WRITERS: usize = 8;
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(WRITERS));
    let store = std::sync::Arc::new(store);

    let handles: Vec<_> = (0..WRITERS)
        .map(|_| {
            let (store, barrier) = (store.clone(), barrier.clone());
            std::thread::spawn(move || {
                barrier.wait();
                let mut r = record(frame("done"));
                save(&store, &mut r).expect("saves")
            })
        })
        .collect();
    let paths: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("no panic"))
        .collect();

    let unique: std::collections::HashSet<_> = paths.iter().collect();
    assert_eq!(unique.len(), WRITERS, "every writer got its own file");
    assert_eq!(
        load(&store, descriptor().id).len(),
        WRITERS,
        "and every run is readable afterwards"
    );
}

#[test]
fn a_published_record_is_never_half_written() {
    // Readers cache, so a torn read is not merely retried — it can hide a run
    // until something else invalidates. Publishing by rename makes the file
    // appear complete or not at all.
    let (store, _d) = store();
    let mut r = record(frame("done"));
    let mut claimed = None;
    let path = save_at(&store, &mut r, 1_777_777_777_777_777_777, |path| {
        claimed = Some(std::fs::File::open(path).expect("claimed placeholder"));
    })
    .expect("saves");
    let mut placeholder = String::new();
    std::io::Read::read_to_string(&mut claimed.expect("captured claim"), &mut placeholder)
        .expect("reads original inode");
    assert!(
        placeholder.is_empty(),
        "publication replaces the claim instead of filling the visible inode"
    );
    let text = std::fs::read_to_string(&path).expect("readable");
    serde_json::from_str::<serde_json::Value>(&text).expect("complete JSON");
    assert!(
        !path.with_extension("json.tmp").exists(),
        "no temp file left behind"
    );
}
