// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for `model_resolver`, split out at the 500-LoC cap (`#[path]`).

use super::*;
use std::fs;

/// Create a mock HF cache structure for testing.
fn setup_mock_cache(tmp: &Path, org: &str, name: &str, hash: &str) -> PathBuf {
    let model_id = format!("{org}/{name}");
    let dir_name = format!("models--{}", model_id.replace('/', "--"));
    let model_cache = tmp.join(&dir_name);

    let snapshot_dir = model_cache.join("snapshots").join(hash);
    fs::create_dir_all(&snapshot_dir).unwrap();
    fs::create_dir_all(model_cache.join("refs")).unwrap();
    fs::write(model_cache.join("refs/main"), hash).unwrap();
    fs::write(snapshot_dir.join("config.json"), "{}").unwrap();
    // Default mock includes weights; tests that need a metadata-only
    // snapshot must remove them explicitly via `setup_mock_cache_no_weights`.
    fs::write(snapshot_dir.join("model.safetensors"), b"weights").unwrap();

    snapshot_dir
}

/// Mock HF cache where the snapshot has only metadata (config + tokenizer)
/// but no weight files — mirrors the real-world failure where refs/main
/// pointed to a partial sync revision of nvidia/Gemma-4-31B-IT-NVFP4.
fn setup_mock_cache_no_weights(tmp: &Path, org: &str, name: &str, hash: &str) -> PathBuf {
    let model_id = format!("{org}/{name}");
    let dir_name = format!("models--{}", model_id.replace('/', "--"));
    let model_cache = tmp.join(&dir_name);
    let snapshot_dir = model_cache.join("snapshots").join(hash);
    fs::create_dir_all(&snapshot_dir).unwrap();
    fs::create_dir_all(model_cache.join("refs")).unwrap();
    fs::write(model_cache.join("refs/main"), hash).unwrap();
    fs::write(snapshot_dir.join("config.json"), "{}").unwrap();
    fs::write(snapshot_dir.join("tokenizer.json"), "{}").unwrap();
    snapshot_dir
}

#[test]
fn resolve_local_directory() {
    let tmp = tempfile::tempdir().unwrap();
    fs::write(tmp.path().join("config.json"), "{}").unwrap();

    let result = resolve_model_dir(tmp.path().to_str().unwrap(), None);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), tmp.path());
}

#[test]
fn resolve_hf_model_id() {
    let tmp = tempfile::tempdir().unwrap();
    let expected = setup_mock_cache(
        tmp.path(),
        "nvidia",
        "Qwen3-Next-80B-A3B-Instruct-NVFP4",
        "abc123",
    );

    let result = resolve_model_dir("nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4", Some(tmp.path()));
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), expected);
}

#[test]
fn resolve_missing_model_gives_download_hint() {
    let tmp = tempfile::tempdir().unwrap();
    let result = resolve_model_dir("nonexistent/model", Some(tmp.path()));
    let err = result.unwrap_err().to_string();
    assert!(err.contains("not found in HF cache"));
    assert!(err.contains("hf download"), "{err}");
    // The old name stays NAMED, not recommended: a box on
    // huggingface_hub < 1.0 has only `huggingface-cli`.
    assert!(err.contains("huggingface-cli"), "{err}");
}

#[test]
fn resolve_missing_config_json() {
    let tmp = tempfile::tempdir().unwrap();
    let dir_name = "models--org--model";
    let snapshot_dir = tmp.path().join(dir_name).join("snapshots/abc");
    fs::create_dir_all(&snapshot_dir).unwrap();
    fs::create_dir_all(tmp.path().join(dir_name).join("refs")).unwrap();
    fs::write(tmp.path().join(dir_name).join("refs/main"), "abc").unwrap();

    let result = resolve_model_dir("org/model", Some(tmp.path()));
    let err = result.unwrap_err().to_string();
    assert!(err.contains("missing config.json"));
}

#[test]
fn cache_dir_arg_takes_precedence() {
    let custom = tempfile::tempdir().unwrap();
    let _expected = setup_mock_cache(custom.path(), "org", "model", "hash1");

    let result = resolve_model_dir("org/model", Some(custom.path()));
    assert!(result.is_ok());
    assert!(result.unwrap().starts_with(custom.path()));
}

#[test]
fn falls_back_to_sibling_snapshot_with_weights() {
    // refs/main points at a metadata-only snapshot; a sibling snapshot
    // has the actual weights. Resolver must return the sibling instead
    // of bailing out (the dual-DGX sweep can't pass --model-from-path).
    let tmp = tempfile::tempdir().unwrap();
    let bad = setup_mock_cache_no_weights(tmp.path(), "nvidia", "Gemma-4-31B-IT-NVFP4", "05fa17");
    // Pre-existing sibling snapshot with safetensors.
    let model_cache = bad.parent().unwrap().parent().unwrap();
    let good_hash = "1365cf";
    let good = model_cache.join("snapshots").join(good_hash);
    fs::create_dir_all(&good).unwrap();
    fs::write(good.join("config.json"), "{}").unwrap();
    fs::write(good.join("model-00001-of-00004.safetensors"), b"shard").unwrap();
    // Touch the good dir so its mtime is newer than the bad one.
    std::thread::sleep(std::time::Duration::from_millis(20));
    fs::write(
        good.join("model-00001-of-00004.safetensors"),
        b"shard-newer",
    )
    .unwrap();

    let result = resolve_model_dir("nvidia/Gemma-4-31B-IT-NVFP4", Some(tmp.path()));
    assert!(result.is_ok(), "expected fallback to succeed: {:?}", result);
    assert_eq!(result.unwrap(), good);
}

#[test]
fn bails_when_all_snapshots_lack_weights() {
    // Resolver should still surface a clear error if the entire cache
    // entry is metadata-only with no usable sibling.
    let tmp = tempfile::tempdir().unwrap();
    setup_mock_cache_no_weights(tmp.path(), "org", "model", "h1");
    let result = resolve_model_dir("org/model", Some(tmp.path()));
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("no weight files") || err.contains("metadata-only"),
        "expected weight-files error, got: {err}"
    );
    assert!(err.contains("hf download"), "{err}");
}

// ── Revisions (serve presets pin an HF branch) ──

/// Mock a multi-branch repo the way `hf download --revision` lays it out:
/// one cache entry, one `refs/<branch>` per branch, one snapshot per branch.
fn setup_branch(tmp: &Path, repo: &str, branch: &str, hash: &str) -> PathBuf {
    let model_cache = tmp.join(format!("models--{}", repo.replace('/', "--")));
    let snap = model_cache.join("snapshots").join(hash);
    fs::create_dir_all(&snap).unwrap();
    fs::create_dir_all(model_cache.join("refs")).unwrap();
    fs::write(model_cache.join("refs").join(branch), hash).unwrap();
    fs::write(snap.join("config.json"), "{}").unwrap();
    fs::write(snap.join("model-00001-of-00009.safetensors"), b"w").unwrap();
    snap
}

#[test]
fn explicit_revision_resolves_its_own_branch_not_main() {
    // turboderp's layout: no refs/main at all, three bit-width branches.
    let tmp = tempfile::tempdir().unwrap();
    let repo = "turboderp/Qwen3.8-Flash-Next-exl3";
    let _two = setup_branch(tmp.path(), repo, "2.05bpw_h4_ng4", "65c895");
    let four = setup_branch(tmp.path(), repo, "4.05bpw_h6_ng6", "55a732");

    let got = resolve_model_dir_at(repo, Some("4.05bpw_h6_ng6"), Some(tmp.path())).unwrap();
    assert_eq!(got, four);
    // Without a revision the SAME cache entry is unusable: refs/main is
    // absent, and the error must say so rather than guess a branch.
    let err = resolve_model_dir(repo, Some(tmp.path()))
        .unwrap_err()
        .to_string();
    assert!(err.contains("refs/main"), "{err}");
}

#[test]
fn missing_revision_names_available_refs_and_the_download_command() {
    let tmp = tempfile::tempdir().unwrap();
    let repo = "turboderp/Qwen3.8-Flash-Next-exl3";
    setup_branch(tmp.path(), repo, "2.05bpw_h4_ng4", "65c895");

    let err = resolve_model_dir_at(repo, Some("4.05bpw_h6_ng6"), Some(tmp.path()))
        .unwrap_err()
        .to_string();
    assert!(err.contains("4.05bpw_h6_ng6"), "{err}");
    assert!(err.contains("Available refs: 2.05bpw_h4_ng4"), "{err}");
    assert!(err.contains("--revision 4.05bpw_h6_ng6"), "{err}");
}

#[test]
fn revision_may_be_a_snapshot_hash() {
    let tmp = tempfile::tempdir().unwrap();
    let repo = "org/model";
    let snap = setup_branch(tmp.path(), repo, "somebranch", "abc123");
    let got = resolve_model_dir_at(repo, Some("abc123"), Some(tmp.path())).unwrap();
    assert_eq!(got, snap);
}

#[test]
fn pinned_revision_never_falls_back_to_a_sibling_snapshot() {
    // The pinned branch's snapshot is metadata-only; a sibling branch has
    // weights. For an UNPINNED lookup the sibling fallback is the documented
    // Gemma-4 recovery; for a pinned revision it would serve another quant.
    let tmp = tempfile::tempdir().unwrap();
    let repo = "turboderp/Qwen3.8-Flash-Next-exl3";
    setup_branch(tmp.path(), repo, "2.05bpw_h4_ng4", "65c895");
    let four = setup_branch(tmp.path(), repo, "4.05bpw_h6_ng6", "55a732");
    fs::remove_file(four.join("model-00001-of-00009.safetensors")).unwrap();

    let err = resolve_model_dir_at(repo, Some("4.05bpw_h6_ng6"), Some(tmp.path()))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("Not falling back to a sibling snapshot"),
        "{err}"
    );
    assert!(err.contains("--revision 4.05bpw_h6_ng6"), "{err}");
}

#[test]
fn local_directory_ignores_revision() {
    let tmp = tempfile::tempdir().unwrap();
    fs::write(tmp.path().join("config.json"), "{}").unwrap();
    let got = resolve_model_dir_at(tmp.path().to_str().unwrap(), Some("whatever"), None).unwrap();
    assert_eq!(got, tmp.path());
}
