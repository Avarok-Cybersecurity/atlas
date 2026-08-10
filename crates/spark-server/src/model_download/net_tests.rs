// SPDX-License-Identifier: AGPL-3.0-only

//! Download tests that need the real Hub.
//!
//! Split from `mod_tests.rs` at the LoC cap, which is also the honest
//! boundary: everything here is `#[ignore]`d and run by hand
//! (`cargo test -p spark-server --bin spark -- --ignored`), because it depends
//! on huggingface.co being reachable and on what it serves today.
//!
//! They are worth having anyway. The offline tests can prove the cache layout
//! and the error mapping; only these can prove the layout AGREES with the Hub,
//! which is the assumption the whole module rests on.

use super::*;

/// A temp cache root that cleans up after itself.
struct Cache(PathBuf);

impl Cache {
    fn new(name: &str) -> Self {
        let p = std::env::temp_dir().join(format!("atlas-dlnet-{name}"));
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

/// The interesting discovery this pins: HuggingFace serves a gated model's
/// METADATA to anyone (200) and gates only the FILES (401). So the failure
/// necessarily arrives after planning, from `fetch_file` — not from
/// `repo_info` — and the pre-flight cannot catch it. If HF ever changes that,
/// this test says so rather than the behaviour quietly shifting.
#[test]
#[ignore = "network"]
fn a_gated_repo_is_reported_as_gated_and_not_as_a_network_fault() {
    const GATED: &str = "meta-llama/Llama-3.2-1B";

    // Metadata is public even when the weights are not.
    let (revision, files) =
        hf::repo_info(GATED, None).expect("a gated repo still describes itself");
    assert!(!revision.is_empty());
    assert!(!files.is_empty());

    // The files are where the gate is. No token passed, so this is the
    // 401-without-credentials case.
    let c = Cache::new("gated");
    let dest = c.0.join("config.json");
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let err = hf::fetch_file(
        GATED,
        &revision,
        "config.json",
        &dest,
        None,
        &cancel,
        &mut |_| {},
    )
    .expect_err("without a token the weights must be refused");

    match &err {
        DownloadError::Gated { repo, had_token } => {
            assert_eq!(repo, GATED);
            assert!(!had_token, "no token was supplied");
        }
        other => panic!("a gated repo must not read as {other:?}"),
    }
    // And the message must send the reader somewhere useful.
    let hint = err.hint();
    assert!(
        hint.contains("HF_TOKEN") || hint.contains("login"),
        "gated hint must name how to authenticate: {hint}"
    );
    assert!(!dest.exists(), "nothing should have been written");
}
