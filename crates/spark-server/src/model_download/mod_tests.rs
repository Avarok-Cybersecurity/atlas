// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use std::path::Path;

/// A temp cache root that cleans up after itself.
struct Cache(PathBuf);

impl Cache {
    fn new(name: &str) -> Self {
        let p = std::env::temp_dir().join(format!("atlas-dl-{name}"));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("temp cache");
        Self(p)
    }
}

impl Drop for Cache {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Lay out a snapshot by hand, as a finished download would.
fn place(cache: &Path, repo: &str, rev: &str, files: &[(&str, &[u8])]) -> PathBuf {
    let snap = hf::repo_dir(cache, repo).join("snapshots").join(rev);
    std::fs::create_dir_all(&snap).unwrap();
    for (name, body) in files {
        std::fs::write(snap.join(name), body).unwrap();
    }
    snap
}

#[test]
fn refs_main_is_not_written_for_a_partial_download() {
    // The ordering the module exists to get right. `snapshot_has_weights` is
    // true as soon as ONE shard lands, so publishing early would give the
    // Library a green tick on a model the loader cannot open.
    let c = Cache::new("partial");
    place(&c.0, "org/m", "rev1", &[("config.json", b"{}")]); // no weights yet
    assert!(
        hf::publish(&c.0, "org/m", "rev1").is_err(),
        "a snapshot without weights must not be published"
    );
    assert!(
        hf::local_revision(&c.0, "org/m").is_none(),
        "refs/main must not exist after a refused publish"
    );
}

#[test]
fn refs_main_is_not_written_without_a_config() {
    let c = Cache::new("noconfig");
    place(&c.0, "org/m", "rev1", &[("model.safetensors", b"w")]);
    assert!(hf::publish(&c.0, "org/m", "rev1").is_err());
    assert!(hf::local_revision(&c.0, "org/m").is_none());
}

#[test]
fn a_complete_snapshot_publishes_and_reads_back() {
    let c = Cache::new("complete");
    place(
        &c.0,
        "org/m",
        "rev1",
        &[("config.json", b"{}"), ("model.safetensors", b"weights")],
    );
    hf::publish(&c.0, "org/m", "rev1").expect("a complete snapshot publishes");
    assert_eq!(hf::local_revision(&c.0, "org/m").as_deref(), Some("rev1"));
}

#[test]
fn a_published_download_is_one_the_resolver_accepts() {
    // The whole point of writing the cache layout by hand rather than taking a
    // Hub client: this asserts the two agree. If `model_resolver` ever changes
    // what it requires, this fails here rather than on someone's first
    // download.
    let c = Cache::new("resolves");
    let snap = place(
        &c.0,
        "org/m",
        "rev1",
        &[("config.json", b"{}"), ("model.safetensors", b"weights")],
    );
    hf::publish(&c.0, "org/m", "rev1").unwrap();

    let resolved = crate::model_resolver::resolve_model_dir("org/m", Some(&c.0))
        .expect("the resolver must accept what we just wrote");
    assert_eq!(resolved, snap);
}

#[test]
fn the_resolver_refuses_a_model_whose_publish_never_happened() {
    // The other half: an interrupted download must not be loadable.
    let c = Cache::new("unpublished");
    place(
        &c.0,
        "org/m",
        "rev1",
        &[("config.json", b"{}"), ("model.safetensors", b"w")],
    );
    // Files are all present, but refs/main was never written.
    assert!(
        crate::model_resolver::resolve_model_dir("org/m", Some(&c.0)).is_err(),
        "without refs/main the model is not finished and must not load"
    );
}

#[test]
fn repo_dir_matches_the_hub_naming_the_resolver_expects() {
    let c = Cache::new("naming");
    assert_eq!(
        hf::repo_dir(&c.0, "nvidia/Qwen3.6-27B-NVFP4")
            .file_name()
            .unwrap(),
        "models--nvidia--Qwen3.6-27B-NVFP4"
    );
}

#[test]
fn every_error_names_an_action() {
    // A failure the reader cannot act on is the thing this whole change is
    // meant to stop producing.
    let errs = [
        DownloadError::Offline("no route".into()),
        DownloadError::Gated {
            repo: "org/m".into(),
            had_token: false,
        },
        DownloadError::Gated {
            repo: "org/m".into(),
            had_token: true,
        },
        DownloadError::NotFound {
            repo: "org/m".into(),
        },
        DownloadError::RateLimited,
        DownloadError::DiskFull,
        DownloadError::NotEnoughSpace {
            need: 28_100_000_000,
            free: 6_400_000_000,
        },
        DownloadError::NoSafetensors {
            repo: "org/m".into(),
        },
        DownloadError::Http {
            repo: "org/m".into(),
            status: 500,
        },
        DownloadError::Io("disk on fire".into()),
    ];
    for e in errs {
        let h = e.hint();
        assert!(!h.is_empty(), "{e:?} has no hint");
        assert!(h.len() > 15, "{e:?} hint is too terse to act on: {h}");
    }
}

#[test]
fn a_gated_repo_reads_differently_with_and_without_a_token() {
    // 401 and 403 are different problems: one is "log in", the other is
    // "accept the licence". Telling someone with a valid token to log in
    // sends them in a circle.
    let without = DownloadError::Gated {
        repo: "org/m".into(),
        had_token: false,
    }
    .hint();
    let with = DownloadError::Gated {
        repo: "org/m".into(),
        had_token: true,
    }
    .hint();
    assert_ne!(without, with);
    assert!(
        without.contains("HF_TOKEN") || without.contains("login"),
        "{without}"
    );
    assert!(
        with.contains("licence") || with.contains("accepted"),
        "{with}"
    );
}

#[test]
fn not_enough_space_states_both_numbers() {
    let h = DownloadError::NotEnoughSpace {
        need: 28_100_000_000,
        free: 6_400_000_000,
    }
    .hint();
    assert!(h.contains("28.1"), "{h}");
    assert!(h.contains("6.4"), "{h}");
}

#[test]
fn free_bytes_reports_something_for_a_real_directory() {
    let c = Cache::new("statvfs");
    let free = hf::free_bytes(&c.0).expect("temp dir is on a real filesystem");
    assert!(free > 0, "a writable filesystem has some space");
}

#[test]
fn free_bytes_is_none_for_a_path_that_does_not_exist() {
    assert_eq!(
        hf::free_bytes(Path::new("/definitely/not/here/at/all")),
        None
    );
}

// ---- network tests, run by hand ----

/// The end-to-end proof: download a real (tiny) model and load-resolve it.
#[test]
#[ignore = "network"]
fn a_real_download_produces_a_model_the_resolver_accepts() {
    let c = Cache::new("real");
    let repo = "hf-internal-testing/tiny-random-gpt2";
    let h = start(repo, c.0.clone());

    let mut planned = None;
    let mut done = None;
    for msg in h.rx.iter() {
        match msg {
            DownloadMsg::Planned {
                files, total_bytes, ..
            } => {
                eprintln!("planned {files} files, {total_bytes} bytes");
                planned = Some(files);
            }
            DownloadMsg::Done { snapshot, revision } => {
                eprintln!("done {revision} -> {}", snapshot.display());
                done = Some(snapshot);
            }
            DownloadMsg::Failed(e) => panic!("download failed: {}", e.hint()),
            _ => {}
        }
    }
    assert!(planned.unwrap_or(0) > 0, "something was planned");
    let snap = done.expect("the download completed");
    assert!(snap.join("config.json").exists());

    let resolved = crate::model_resolver::resolve_model_dir(repo, Some(&c.0))
        .expect("a freshly downloaded model must resolve");
    assert_eq!(resolved, snap);
}

/// Cancellation must stop promptly and leave resume credit, not a model.
#[test]
#[ignore = "network"]
fn a_cancelled_download_leaves_no_refs_main() {
    let c = Cache::new("cancel");
    let repo = "hf-internal-testing/tiny-random-gpt2";
    let h = start(repo, c.0.clone());
    h.cancel();
    let mut cancelled = false;
    for msg in h.rx.iter() {
        match msg {
            DownloadMsg::Cancelled { .. } => cancelled = true,
            DownloadMsg::Done { .. } => panic!("a cancelled download must not publish"),
            _ => {}
        }
    }
    assert!(cancelled, "the worker reported the cancellation");
    assert!(
        hf::local_revision(&c.0, repo).is_none(),
        "a cancelled download must not be loadable"
    );
}

#[test]
fn part_files_cannot_collide_between_siblings() {
    // `with_extension("part")` maps `foo.safetensors` and `foo.json` onto the
    // SAME `foo.part`. Since resume reads a byte count from that file, a
    // collision appends one file's body onto another's and yields a corrupt
    // weight file that still exists, still has a plausible size, and still
    // passes every check the loader makes before reading tensors.
    let dir = Path::new("/tmp/x");
    let a = super::hf::part_path(&dir.join("model-00001-of-00002.safetensors"));
    let b = super::hf::part_path(&dir.join("model-00001-of-00002.json"));
    assert_ne!(a, b, "sibling files must not share a .part");
    assert!(a.to_string_lossy().ends_with(".safetensors.part"), "{a:?}");

    // And the real-world name that prompted the check.
    let c = super::hf::part_path(&dir.join("model.safetensors-00001-of-00001.safetensors"));
    let d = super::hf::part_path(&dir.join("model.safetensors.index.json"));
    assert_ne!(c, d);
}

#[test]
fn an_index_without_its_shards_is_not_publishable() {
    // `model_resolver::snapshot_has_weights` counts the INDEX as weights, and
    // the index is a small file that lands first. Trusting it here would
    // publish a model whose shards had not arrived — the resolver would then
    // accept it, and the failure would surface deep inside the loader.
    let c = Cache::new("indexonly");
    place(
        &c.0,
        "org/m",
        "rev1",
        &[
            ("config.json", b"{}"),
            ("model.safetensors.index.json", b"{}"),
        ],
    );
    // The resolver's own helper is satisfied...
    assert!(crate::model_resolver::snapshot_has_weights(
        &hf::repo_dir(&c.0, "org/m").join("snapshots").join("rev1")
    ));
    // ...and publishing must still refuse.
    assert!(
        hf::publish(&c.0, "org/m", "rev1").is_err(),
        "an index naming absent shards is not a downloaded model"
    );
    assert!(hf::local_revision(&c.0, "org/m").is_none());
}

#[test]
fn a_part_file_does_not_count_as_a_shard() {
    let c = Cache::new("partonly");
    place(
        &c.0,
        "org/m",
        "rev1",
        &[
            ("config.json", b"{}"),
            ("model-00001-of-00001.safetensors.part", b"half"),
        ],
    );
    assert!(hf::publish(&c.0, "org/m", "rev1").is_err());
}
