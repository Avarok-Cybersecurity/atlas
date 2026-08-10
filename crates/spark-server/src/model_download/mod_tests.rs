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
fn free_bytes_measures_the_cache_root_even_before_it_exists() {
    // A first download creates the cache directory, so at pre-flight time the
    // path is usually MISSING — and `statvfs` on a missing path fails. Falling
    // back to None skipped the space check exactly when it mattered most.
    let c = Cache::new("statvfs-missing");
    let not_yet = c.0.join("models--org--m/snapshots/rev1");
    assert!(!not_yet.exists());
    let free = hf::free_bytes(&not_yet).expect("walks up to a real ancestor");
    assert!(free > 0);
    // And it agrees with the root it will actually be created under.
    let root_free = hf::free_bytes(&c.0).expect("root exists");
    assert!(
        free.abs_diff(root_free) < root_free / 100,
        "the answer must describe the filesystem the files will land on"
    );
}

#[test]
fn free_bytes_is_none_only_when_nothing_up_the_tree_can_be_measured() {
    // "/" always exists, so an absolute nonsense path still resolves — which
    // is correct: that IS the filesystem the write would be attempted on.
    assert!(hf::free_bytes(Path::new("/definitely/not/here/at/all")).is_some());
}

#[test]
fn the_space_check_refuses_only_when_it_genuinely_will_not_fit() {
    use super::fits;
    // Exactly enough is enough.
    assert!(fits(100, 0, 100).is_ok());
    // One byte short is not.
    match fits(101, 0, 100) {
        Err(DownloadError::NotEnoughSpace { need, free }) => {
            assert_eq!((need, free), (101, 100));
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    // A resumed download only needs the REMAINDER — refusing on the full size
    // would block a download that is 90% done and fits easily.
    assert!(fits(1_000, 950, 100).is_ok());
    // And an over-complete `.part` must not underflow into a huge `need`.
    assert!(fits(100, 250, 0).is_ok());
}

// ---- network tests, run by hand ----

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

#[test]
fn the_token_precedence_rule_prefers_the_environment_then_the_login_file() {
    use super::hf::pick_token;
    let some = |s: &str| Some(s.to_string());

    // Nothing anywhere: public models need no token, so this is normal.
    assert_eq!(pick_token(&[None, None], None), None);

    // The login file is the fallback — what `hf auth login` writes,
    // which a user who has "logged in" expects to work.
    assert_eq!(
        pick_token(&[None, None], Some("from-file")).as_deref(),
        Some("from-file")
    );

    // HF_TOKEN wins over the file...
    assert_eq!(
        pick_token(&[some("env-a"), None], Some("from-file")).as_deref(),
        Some("env-a")
    );
    // ...and over the second variable.
    assert_eq!(
        pick_token(&[some("env-a"), some("env-b")], None).as_deref(),
        Some("env-a")
    );
    // The second variable is still honoured when the first is unset.
    assert_eq!(
        pick_token(&[None, some("env-b")], None).as_deref(),
        Some("env-b")
    );
}

#[test]
fn a_blank_token_reads_as_absent_rather_than_as_a_credential() {
    use super::hf::pick_token;
    // `export HF_TOKEN=` is a common shell accident, and sending
    // `Authorization: Bearer ` turns a PUBLIC model into a 401 — a download
    // that would have worked failing with "requires credentials".
    assert_eq!(pick_token(&[Some(String::new()), None], None), None);
    assert_eq!(pick_token(&[Some("   ".into()), None], None), None);
    assert_eq!(pick_token(&[None, None], Some("\n")), None);
    // And a real token with the trailing newline every file has is usable.
    assert_eq!(
        pick_token(&[None, None], Some("hf_realtoken\n")).as_deref(),
        Some("hf_realtoken")
    );
    // A blank env var must not shadow a good file.
    assert_eq!(
        pick_token(&[Some("  ".into()), None], Some("hf_realtoken")).as_deref(),
        Some("hf_realtoken")
    );
}

#[test]
fn hub_statuses_map_to_causes_a_reader_can_act_on() {
    use super::hf::classify;
    // 401 and 403 are the same *class* of problem but different actions, and
    // the token is what tells them apart — so the flag, not the code, decides
    // the wording.
    assert_eq!(
        classify("org/m", 401, false),
        DownloadError::Gated {
            repo: "org/m".into(),
            had_token: false
        }
    );
    assert_eq!(
        classify("org/m", 403, true),
        DownloadError::Gated {
            repo: "org/m".into(),
            had_token: true
        }
    );
    assert_eq!(
        classify("org/m", 404, false),
        DownloadError::NotFound {
            repo: "org/m".into()
        }
    );
    assert_eq!(classify("org/m", 429, false), DownloadError::RateLimited);
    // Anything else keeps the number rather than inventing a cause.
    assert_eq!(
        classify("org/m", 502, false),
        DownloadError::Http {
            repo: "org/m".into(),
            status: 502
        }
    );
}

/// A genuinely gated repo, asked for WITHOUT credentials.
///

#[test]
fn a_full_disk_is_reported_as_a_full_disk() {
    use super::hf::write_error;
    // ENOSPC is 28 on Linux. Asserting through `from_raw_os_error` rather than
    // through `ErrorKind::StorageFull` directly is the point: if that mapping
    // is ever not what the OS actually returns, `DiskFull` would never fire
    // and a full disk would surface as a generic write failure — which reads
    // like corruption and sends the reader after the wrong thing.
    assert_eq!(
        write_error(std::io::Error::from_raw_os_error(28)),
        DownloadError::DiskFull
    );
    // Anything else keeps its own message rather than being guessed at.
    match write_error(std::io::Error::from_raw_os_error(13)) {
        DownloadError::Io(m) => assert!(!m.is_empty(), "permission denied keeps its text"),
        other => panic!("EACCES is not a full disk: {other:?}"),
    }
    // EFBIG (a file-size ulimit or quota) is NOT ENOSPC and must not claim to
    // be — the fix is different.
    match write_error(std::io::Error::from_raw_os_error(27)) {
        DownloadError::Io(_) => {}
        other => panic!("EFBIG is not a full disk: {other:?}"),
    }
}
