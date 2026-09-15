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
        pid: 4242,
    }
}

/// The reuse decision: same pid, same binary, same rendering, same model —
/// and each NEGATIVE CONTROL flips exactly one and is refused by name.
#[test]
fn a_server_is_reused_only_when_every_digest_matches() {
    let want = ("a".repeat(64), "b".repeat(64));
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
    let hermetic = ("d".repeat(64), "b".repeat(64));
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
