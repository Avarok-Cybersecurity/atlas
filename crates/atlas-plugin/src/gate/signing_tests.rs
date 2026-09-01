// SPDX-License-Identifier: AGPL-3.0-only
//! Signature tests. Every one of these is a negative control or a round trip —
//! a signature check that cannot fail is worse than no signature check, because
//! it reports provenance it never verified.
use super::signing::*;
use super::tests::tempdir;
use std::path::Path;

/// Write a record + registry entry for a fresh identity, and return the temp
/// root, the record path and the identity.
fn fixture(recorded_at_note: &str) -> (tempdir::Dir, std::path::PathBuf, Identity) {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    let home = root.join("atlas-home");
    std::fs::create_dir_all(&home).unwrap();
    let identity = load_or_create(&home).expect("identity");
    register(root, &identity).expect("register");

    let rec_dir = root.join(".benchmarks").join("decode-floor");
    std::fs::create_dir_all(&rec_dir).unwrap();
    let record = rec_dir.join("2026-09-02-abcdef1234.json");
    std::fs::write(&record, format!("{{\"note\":\"{recorded_at_note}\"}}\n")).unwrap();
    (dir, record, identity)
}

const AFTER: u64 = SIGNATURE_REQUIRED_AFTER + 1;

#[test]
fn a_signed_record_verifies() {
    let (dir, record, id) = fixture("ok");
    sign_record(&id, &record, "abcdef1234").unwrap();
    assert_eq!(
        verify_record(dir.path(), &record, "abcdef1234", AFTER).unwrap(),
        Verified::Signed {
            fingerprint: id.fingerprint().to_string()
        }
    );
}

#[test]
fn editing_the_record_breaks_the_signature() {
    let (dir, record, id) = fixture("ok");
    sign_record(&id, &record, "abcdef1234").unwrap();
    // The exact attack: a plausible number, changed after the run.
    std::fs::write(&record, "{\"note\":\"ok\",\"tok_s\":9999}\n").unwrap();
    let err = verify_record(dir.path(), &record, "abcdef1234", AFTER).unwrap_err();
    assert!(
        err.to_string().contains("does not match its signature"),
        "expected a tamper message, got: {err}"
    );
}

#[test]
fn a_record_cannot_be_repointed_at_another_commit() {
    // The signature covers the sha, so lifting a passing record onto a different
    // commit fails even though the file itself is untouched.
    let (dir, record, id) = fixture("ok");
    sign_record(&id, &record, "abcdef1234").unwrap();
    assert!(verify_record(dir.path(), &record, "999999beef", AFTER).is_err());
}

#[test]
fn a_signature_from_another_record_does_not_transfer() {
    let (dir, record, id) = fixture("ok");
    let other = record.with_file_name("2026-09-02-abcdef1234-other.json");
    std::fs::write(&other, "{\"note\":\"other\"}\n").unwrap();
    sign_record(&id, &other, "abcdef1234").unwrap();
    // Move the valid sidecar onto the first record.
    std::fs::copy(sig_path(&other), sig_path(&record)).unwrap();
    assert!(verify_record(dir.path(), &record, "abcdef1234", AFTER).is_err());
}

#[test]
fn an_unregistered_signer_is_refused() {
    let (dir, record, id) = fixture("ok");
    sign_record(&id, &record, "abcdef1234").unwrap();
    // A real signature, a real key — but nobody put it in the registry, which is
    // the review step the whole design leans on.
    std::fs::remove_file(
        dir.path()
            .join(REGISTRY_DIR)
            .join(format!("{}.pub", id.fingerprint())),
    )
    .unwrap();
    let err = verify_record(dir.path(), &record, "abcdef1234", AFTER).unwrap_err();
    assert!(
        err.to_string().contains("not in .github/record-signers"),
        "expected a registry message, got: {err}"
    );
}

#[test]
fn a_post_cutover_record_with_no_signature_fails() {
    let (dir, record, _id) = fixture("ok");
    let err = verify_record(dir.path(), &record, "abcdef1234", AFTER).unwrap_err();
    assert!(err.to_string().contains("has no signature"), "got: {err}");
}

#[test]
fn a_pre_cutover_record_with_no_signature_is_exempt() {
    let (dir, record, _id) = fixture("ok");
    assert_eq!(
        verify_record(
            dir.path(),
            &record,
            "abcdef1234",
            SIGNATURE_REQUIRED_AFTER - 1
        )
        .unwrap(),
        Verified::Exempt
    );
}

#[test]
fn the_identity_is_created_once_and_reused() {
    let dir = tempdir::Dir::new();
    let home = dir.path().join("h");
    std::fs::create_dir_all(&home).unwrap();
    let a = load_or_create(&home).unwrap();
    let b = load_or_create(&home).unwrap();
    assert_eq!(
        a.fingerprint(),
        b.fingerprint(),
        "a second run minted a new identity; every box would churn the registry"
    );
}

#[cfg(unix)]
#[test]
fn the_private_key_is_not_world_readable() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempdir::Dir::new();
    let home = dir.path().join("h");
    std::fs::create_dir_all(&home).unwrap();
    load_or_create(&home).unwrap();
    let mode = std::fs::metadata(home.join("identity").join("ed25519.pk8"))
        .unwrap()
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o600, "key mode is {mode:o}, expected 600");
}

#[test]
fn registering_twice_writes_once() {
    let dir = tempdir::Dir::new();
    let home = dir.path().join("h");
    std::fs::create_dir_all(&home).unwrap();
    let id = load_or_create(&home).unwrap();
    assert!(register(dir.path(), &id).unwrap(), "first register writes");
    assert!(
        !register(dir.path(), &id).unwrap(),
        "second register must be a no-op, or every run nags about a new signer"
    );
}

#[test]
fn the_sidecar_path_appends_and_does_not_eat_a_versioned_model_name() {
    // `variant_slug` keeps `.` in a model name, so `with_extension` would turn
    // `...-qwen3.8-27b.json` into `...-qwen3.sig`.
    let p = Path::new(".benchmarks/x/2026-08-05-abc-qwen3.8-27b.json");
    assert_eq!(
        sig_path(p).to_string_lossy(),
        ".benchmarks/x/2026-08-05-abc-qwen3.8-27b.json.sig"
    );
}

/// ★ The cutover cannot drift into "signatures optional forever".
///
/// This asserts the invariant directly rather than pinning a magic number: no
/// record committed to this repo may sit at or after the cutover without a
/// sidecar. Move the constant forward to excuse a new unsigned record and this
/// fails, which is the whole point.
#[test]
fn no_committed_record_escapes_the_cutover_unsigned() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let benchmarks = root.join(".benchmarks");
    if !benchmarks.exists() {
        return; // not a full checkout
    }
    let mut naked = Vec::new();
    for gate in std::fs::read_dir(&benchmarks).unwrap().flatten() {
        if !gate.path().is_dir() {
            continue;
        }
        for entry in std::fs::read_dir(gate.path()).unwrap().flatten() {
            let p = entry.path();
            if p.extension().is_none_or(|e| e != "json")
                || p.file_name().is_some_and(|n| n == "BASELINE.json")
            {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
                continue;
            };
            let at = v.get("recorded_at").and_then(serde_json::Value::as_u64);
            if at.is_some_and(|t| t >= SIGNATURE_REQUIRED_AFTER) && !sig_path(&p).exists() {
                naked.push(p.display().to_string());
            }
        }
    }
    assert!(
        naked.is_empty(),
        "these records are at/after the signing cutover but carry no .sig — either \
         sign them or they will fail their gate:\n  {}",
        naked.join("\n  ")
    );
}
