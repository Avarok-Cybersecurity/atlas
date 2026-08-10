// SPDX-License-Identifier: AGPL-3.0-only

//! The Library's disk half, against a real HuggingFace cache tree.
//!
//! Built in a temp directory and handed to the scanner through `--cache-dir`,
//! so every answer here is the TREE's and not this box's own cache — the same
//! reason `model_resolver`'s tests build their own mock. The shapes covered are
//! the ones a cache actually reaches: a finished checkpoint, a download that
//! stopped before `refs/main`, a snapshot with no `config.json`, the
//! symlink-into-`blobs/` layout `huggingface-cli` writes, and a directory the
//! process cannot read.

use super::*;
use crate::tui::data::library::scan;
use std::path::{Path, PathBuf};

/// A flat (non-nested) config the real parser accepts, so the metadata columns
/// are proved end to end rather than against a stub.
const CONFIG: &str = r#"{
  "model_type": "llama",
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "max_position_embeddings": 8192,
  "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}
}"#;

fn repo_dir(root: &Path, id: &str) -> PathBuf {
    root.join(format!("models--{}", id.replace('/', "--")))
}

/// A checkpoint written the way Atlas's own downloader writes it: files
/// directly in the snapshot, no `blobs/`.
fn checkpoint(root: &Path, id: &str, config: Option<&str>, shard_bytes: usize) -> PathBuf {
    let dir = repo_dir(root, id);
    let snap = dir.join("snapshots/rev0");
    std::fs::create_dir_all(&snap).expect("snapshot dir");
    std::fs::write(snap.join("model.safetensors"), vec![7u8; shard_bytes]).expect("shard");
    if let Some(json) = config {
        std::fs::write(snap.join("config.json"), json).expect("config");
    }
    publish(&dir);
    snap
}

/// Write the `refs/main` the resolver keys on — the marker a download only
/// gets once it has finished.
fn publish(dir: &Path) {
    std::fs::create_dir_all(dir.join("refs")).expect("refs dir");
    std::fs::write(dir.join("refs/main"), "rev0").expect("refs/main");
}

fn ids(entries: &[LibraryEntry]) -> Vec<String> {
    entries.iter().map(|e| e.id.clone()).collect()
}

#[test]
fn a_finished_checkpoint_is_listed_with_its_config_metadata() {
    let tmp = tempfile::tempdir().expect("tempdir");
    checkpoint(tmp.path(), "org/complete", Some(CONFIG), 4096);

    let found = scan(Some(tmp.path()));
    assert_eq!(ids(&found), ["org/complete"], "the id is un-mangled");
    let e = &found[0];
    assert!(e.has_weights, "a shard and a published refs/main");
    assert_eq!(e.model_type, "llama");
    assert_eq!(e.layers, 32);
    assert_eq!(e.hidden, 4096);
    assert_eq!(e.heads, 32);
    assert_eq!(e.context, 8192);
    assert_eq!(e.quant, "nvfp4", "quant_algo, lowercased");
    assert_eq!(e.size_bytes, 4096 + CONFIG.len() as u64);
    assert!(e.snapshot_dir.ends_with("snapshots/rev0"));
}

#[test]
fn a_download_that_never_published_refs_main_is_listed_but_not_ready() {
    // The whole point of keying on `refs/main`: a shard on disk is not a
    // finished download, and "ready" has to mean "the loader would accept it".
    let tmp = tempfile::tempdir().expect("tempdir");
    checkpoint(tmp.path(), "org/partial", Some(CONFIG), 512);
    std::fs::remove_file(repo_dir(tmp.path(), "org/partial").join("refs/main")).expect("unpublish");

    let found = scan(Some(tmp.path()));
    assert_eq!(ids(&found), ["org/partial"], "still listed");
    assert!(!found[0].has_weights, "but not loadable");
}

#[test]
fn a_snapshot_with_no_config_json_is_listed_with_unknown_metadata() {
    // A config that will not parse must not remove the row: the weights are
    // there and the user can still see what the cache holds.
    let tmp = tempfile::tempdir().expect("tempdir");
    checkpoint(tmp.path(), "org/no-config", None, 256);

    let found = scan(Some(tmp.path()));
    assert_eq!(ids(&found), ["org/no-config"]);
    assert_eq!(found[0].model_type, "?", "unknown, not invented");
    assert_eq!(found[0].quant, "-");
    assert_eq!(found[0].layers, 0);
    assert!(!found[0].optimized, "no config means no kernel match");
    assert!(found[0].has_weights, "the shard is still there");
}

#[test]
fn an_unparseable_config_json_leaves_the_row_with_unknown_metadata() {
    let tmp = tempfile::tempdir().expect("tempdir");
    checkpoint(tmp.path(), "org/bad-config", Some("{not json"), 256);

    let found = scan(Some(tmp.path()));
    assert_eq!(found[0].model_type, "?");
    assert!(found[0].has_weights);
}

#[test]
#[cfg(unix)]
fn a_symlinked_snapshot_is_measured_by_the_blob_it_points_at() {
    // `huggingface-cli` puts the bytes in `blobs/` and symlinks them into the
    // snapshot, so `len()` of a snapshot entry measures the LINK. Every model
    // fetched that way reported a few hundred bytes.
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = repo_dir(tmp.path(), "org/linked");
    let snap = dir.join("snapshots/rev0");
    let blobs = dir.join("blobs");
    std::fs::create_dir_all(&snap).expect("snapshot dir");
    std::fs::create_dir_all(&blobs).expect("blobs dir");
    let blob = blobs.join("deadbeef");
    std::fs::write(&blob, vec![3u8; 64 * 1024]).expect("blob");
    std::os::unix::fs::symlink(&blob, snap.join("model.safetensors")).expect("symlink");
    std::fs::write(snap.join("config.json"), CONFIG).expect("config");
    publish(&dir);

    let found = scan(Some(tmp.path()));
    assert_eq!(ids(&found), ["org/linked"]);
    assert_eq!(found[0].size_bytes, 64 * 1024, "the blob, not the link");
    assert!(found[0].has_weights, "a symlinked shard is a shard");
}

#[test]
fn an_empty_cache_scans_to_an_empty_list() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(scan(Some(tmp.path())).is_empty());
}

#[test]
fn a_cache_root_that_does_not_exist_scans_to_an_empty_list() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(scan(Some(&tmp.path().join("never-created"))).is_empty());
}

#[test]
fn directories_that_are_not_models_are_skipped() {
    // A real hub cache also holds `.locks/`, `datasets--*` and `version.txt`.
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join(".locks")).expect("locks");
    std::fs::create_dir_all(tmp.path().join("datasets--org--set/snapshots/rev0")).expect("dataset");
    std::fs::write(tmp.path().join("version.txt"), "1").expect("version");
    checkpoint(tmp.path(), "org/real", Some(CONFIG), 128);

    assert_eq!(ids(&scan(Some(tmp.path()))), ["org/real"]);
}

#[test]
fn a_model_directory_with_no_snapshots_is_skipped_rather_than_listed_empty() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(repo_dir(tmp.path(), "org/husk")).expect("husk");
    assert!(scan(Some(tmp.path())).is_empty());
}

#[test]
fn the_biggest_checkpoint_is_listed_first() {
    // The list is what a user scans for "what is eating the disk", so size
    // order is the useful one — and it is what the selection anchoring in
    // `LibState::rebuild` has to survive.
    let tmp = tempfile::tempdir().expect("tempdir");
    checkpoint(tmp.path(), "org/small", Some(CONFIG), 1024);
    checkpoint(tmp.path(), "org/big", Some(CONFIG), 512 * 1024);
    checkpoint(tmp.path(), "org/medium", Some(CONFIG), 8 * 1024);

    assert_eq!(
        ids(&scan(Some(tmp.path()))),
        ["org/big", "org/medium", "org/small"]
    );
}

#[test]
#[cfg(unix)]
fn an_unreadable_cache_root_scans_to_an_empty_list_rather_than_failing() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().expect("tempdir");
    checkpoint(tmp.path(), "org/hidden", Some(CONFIG), 128);
    let locked = tmp.path().join("locked");
    std::fs::create_dir_all(&locked).expect("dir");
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).expect("chmod");
    // Root ignores the mode bits, so the assertion below would be vacuous.
    let denied = std::fs::read_dir(&locked).is_err();
    let found = scan(Some(&locked));
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).expect("restore");

    if denied {
        assert!(found.is_empty(), "a refusal is an empty list, not a panic");
    }
}

#[test]
#[cfg(unix)]
fn one_unreadable_model_does_not_cost_the_rest_of_the_list() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().expect("tempdir");
    checkpoint(tmp.path(), "org/readable", Some(CONFIG), 4096);
    let snap = checkpoint(tmp.path(), "org/denied", Some(CONFIG), 4096);
    std::fs::set_permissions(&snap, std::fs::Permissions::from_mode(0o000)).expect("chmod");
    let denied = std::fs::read_dir(&snap).is_err();

    let found = scan(Some(tmp.path()));
    std::fs::set_permissions(&snap, std::fs::Permissions::from_mode(0o755)).expect("restore");

    assert!(
        ids(&found).contains(&"org/readable".to_string()),
        "a model that cannot be read must not empty the list: {:?}",
        ids(&found)
    );
    if denied {
        let hidden = found.iter().find(|e| e.id == "org/denied");
        assert!(
            hidden.is_none_or(|e| !e.has_weights),
            "an unreadable snapshot cannot claim weights"
        );
    }
}

#[test]
fn the_background_scan_reports_what_the_direct_scan_finds() {
    // The render thread only ever `try_recv`s, so the scan has to arrive on a
    // channel; this is that whole round trip against a real tree.
    let tmp = tempfile::tempdir().expect("tempdir");
    checkpoint(tmp.path(), "org/one", Some(CONFIG), 2048);
    checkpoint(tmp.path(), "org/two", Some(CONFIG), 1024);

    let mut s = LibState::default();
    assert!(!s.attached(), "no store yet");
    s.start_scan(Some(tmp.path()));

    let mut got = None;
    for _ in 0..200 {
        if let Some(found) = s.poll_scan() {
            got = Some(found);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(ids(&got.expect("the scan lands")), ["org/one", "org/two"]);
    assert!(s.poll_scan().is_none(), "and it is collected once");
}

#[test]
fn a_second_scan_request_while_one_is_running_is_ignored() {
    // `mark_dirty` is set by the reducer and can be set on consecutive frames;
    // without this guard that is a thread per frame.
    let tmp = tempfile::tempdir().expect("tempdir");
    checkpoint(tmp.path(), "org/would-appear", Some(CONFIG), 128);
    let (tx, rx) = std::sync::mpsc::channel();
    let mut s = LibState::default();
    s.pending_scan = Some(rx);

    s.start_scan(Some(tmp.path()));
    assert!(
        tx.send(Vec::new()).is_ok(),
        "the in-flight receiver was dropped — a second scan replaced it"
    );
    let found = s.poll_scan().expect("the in-flight scan is what is polled");
    assert!(
        found.is_empty(),
        "a second request must not start a scan of its own: {:?}",
        ids(&found)
    );
}

#[test]
fn a_dead_scanner_stops_being_polled_and_leaves_the_list_alone() {
    let mut s = LibState::default();
    let (tx, rx) = std::sync::mpsc::channel::<Vec<LibraryEntry>>();
    drop(tx);
    s.pending_scan = Some(rx);

    assert!(s.poll_scan().is_none(), "nothing to show");
    assert!(s.pending_scan.is_none(), "and it is not polled forever");
}

#[test]
fn attaching_a_store_makes_the_library_count_as_attached() {
    // `attached` is what separates "first entry into the Library" — which also
    // starts a GitHub fetch — from a plain rescan.
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut s = LibState::default();
    s.attach(tmp.path().to_path_buf(), &[]);
    assert!(s.attached());
}
