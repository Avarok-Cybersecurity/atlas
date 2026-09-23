// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

fn lease() -> Lease {
    Lease {
        pid: 4242,
        port: 40001,
        model: "Qwen/Qwen3.8-27B".into(),
        recipe_id: "qwen/qwen3.8-27b".into(),
        argv_sha256: "a".repeat(64),
        binary_sha256: "b".repeat(64),
        owner_pid: 1,
        started_at: 0,
    }
}

fn reported() -> ServeIdentity {
    ServeIdentity {
        argv_sha256: "a".repeat(64),
        binary_sha256: "b".repeat(64),
        engine_env_sha256: "e".repeat(64),
        pid: 4242,
    }
}

/// The reuse decision: same pid, same binary, same rendering, same model —
/// and each NEGATIVE CONTROL flips exactly one and is refused by name.
#[test]
fn a_server_is_reused_only_when_every_digest_matches() {
    let want = ("a".repeat(64), "b".repeat(64), "e".repeat(64));
    assert_eq!(
        mismatch(&lease(), &reported(), &want, "Qwen/Qwen3.8-27B"),
        None
    );

    let mut r = reported();
    r.pid = 4243;
    assert!(
        mismatch(&lease(), &r, &want, "Qwen/Qwen3.8-27B")
            .unwrap()
            .contains("pid 4243")
    );

    let mut r = reported();
    r.binary_sha256 = "c".repeat(64);
    assert!(
        mismatch(&lease(), &r, &want, "Qwen/Qwen3.8-27B")
            .unwrap()
            .contains("another binary")
    );

    // The rendering differs: a hermetic kat server is not an open bfcl one.
    let hermetic = ("d".repeat(64), "b".repeat(64), "e".repeat(64));
    assert!(
        mismatch(&lease(), &reported(), &hermetic, "Qwen/Qwen3.8-27B")
            .unwrap()
            .contains("another rendering")
    );

    assert!(
        mismatch(&lease(), &reported(), &want, "Qwen/Qwen3.6-35B")
            .unwrap()
            .contains("this run needs")
    );
}

/// ★ THE ENVIRONMENT HALF, owner 2026-09-22: "we only allow server re-use IF
/// the recipes the bench uses are the SAME." The recipe is enforced by the
/// rendering digest above — a recipe becomes flags. These are the cases that
/// digest CANNOT see, because a lever like `AVAROK_PREFILL_CODISPATCH` never
/// reaches argv: argv and binary match exactly and the server is still wrong.
#[test]
fn a_server_under_another_engine_environment_is_refused() {
    let want = ("a".repeat(64), "b".repeat(64), "e".repeat(64));

    // Same argv, same binary, same model, DIFFERENT levers. Before the env
    // digest this returned None and the run measured an undeclared config.
    let mut r = reported();
    r.engine_env_sha256 = "f".repeat(64);
    let why = mismatch(&lease(), &r, &want, "Qwen/Qwen3.8-27B").expect("refused");
    assert!(why.contains("engine environment"), "{why}");
    assert!(
        why.contains("env-only"),
        "the message must say argv and binary matched, so a reader is not sent \
         looking for a recipe difference that does not exist: {why}"
    );

    // A server predating the digest reports "", which must read as UNKNOWN and
    // be refused — "cannot tell" is not "matches".
    let mut r = reported();
    r.engine_env_sha256 = String::new();
    let why = mismatch(&lease(), &r, &want, "Qwen/Qwen3.8-27B").expect("refused");
    assert!(
        why.contains("does not report its engine environment"),
        "{why}"
    );
}

/// The digest itself: what it separates, and what it deliberately ignores.
#[test]
fn the_engine_env_digest_covers_avarok_vars_only_and_cannot_collide() {
    let f = |v: &[(&str, &str)]| {
        avarok_plugin::serve_identity::engine_env_fingerprint(
            v.iter().map(|(k, x)| ((*k).to_string(), (*x).to_string())),
        )
    };
    // The lever that motivated this: 1 and 0 must differ.
    assert_ne!(
        f(&[("AVAROK_PREFILL_CODISPATCH", "1")]),
        f(&[("AVAROK_PREFILL_CODISPATCH", "0")])
    );
    // ...and so must the three levers PERF_CONTROLS does NOT disclose, which is
    // why this digest is not built from that list.
    for k in [
        "AVAROK_FP8_ROWWISE",
        "AVAROK_MTP_DCUT_RATIO",
        "AVAROK_MTP_K_LADDER",
    ] {
        assert_ne!(f(&[(k, "1")]), f(&[(k, "2")]), "{k} must be covered");
    }
    // Set-but-empty is not unset: an operator who exports a lever to "" has
    // configured something different from one who never exported it.
    assert_ne!(f(&[("AVAROK_X", "")]), f(&[]));
    // Order of enumeration must not matter — `std::env::vars()` gives no order.
    assert_eq!(
        f(&[("AVAROK_A", "1"), ("AVAROK_B", "2")]),
        f(&[("AVAROK_B", "2"), ("AVAROK_A", "1")])
    );
    // NUL separation: these must not collide on concatenation.
    assert_ne!(
        f(&[("AVAROK_A", "1"), ("AVAROK_B", "")]),
        f(&[("AVAROK_A", "1B")])
    );
    // Non-engine variables are ignored: PATH churn between two runs on one box
    // is not a config change, and refusing on it would make reuse never happen.
    assert_eq!(
        f(&[("AVAROK_A", "1"), ("PATH", "/x")]),
        f(&[("AVAROK_A", "1")])
    );
}

/// The lease round-trips through its file, and a file that is not a lease is
/// an error rather than "no lease".
#[test]
fn the_lease_file_round_trips_and_a_bad_one_is_refused() {
    let dir = std::env::temp_dir().join(format!("serve-lease-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let store = ArtifactStore::with_root(dir.clone());
    assert_eq!(read(&store).unwrap(), None);
    write(&store, &lease()).unwrap();
    assert_eq!(read(&store).unwrap(), Some(lease()));
    std::fs::write(lease_path(&store), "not json").unwrap();
    assert!(read(&store).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A lease whose owner is dead is released; one whose owner lives is kept.
/// Pid 1 is always alive; a pid no process has is not.
#[test]
#[cfg(target_os = "linux")]
fn an_orphaned_lease_is_released_and_a_live_one_kept() {
    let dir = std::env::temp_dir().join(format!("serve-lease-orphan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let store = ArtifactStore::with_root(dir.clone());
    // The server pid is dead too, so release does not signal anything real.
    let dead_server = Lease {
        pid: 4_000_000_000 - 7,
        owner_pid: 1,
        ..lease()
    };
    write(&store, &dead_server).unwrap();
    assert_eq!(release_if_orphaned(&store).unwrap(), None);
    assert!(lease_path(&store).exists());
    let orphan = Lease {
        owner_pid: 4_000_000_000 - 9,
        ..dead_server
    };
    write(&store, &orphan).unwrap();
    assert_eq!(release_if_orphaned(&store).unwrap(), Some(orphan));
    assert!(!lease_path(&store).exists());
    let _ = std::fs::remove_dir_all(&dir);
}
